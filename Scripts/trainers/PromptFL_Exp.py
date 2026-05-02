import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import os, os.path as osp, json

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from Dassl.dassl.data import DataManager
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler
from Dassl.dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)

# from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
# from sampling import cifar_iid, cifar_noniid

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'PROMPTFL_Exp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PROMPTFL_Exp.N_CTX
        ctx_init = cfg.TRAINER.PROMPTFL_Exp.CTX_INIT #   (XXXXX) class with the  sylte of XXXXXXX 16
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        device = clip_model.token_embedding.weight.device


        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.PROMPTFL_Exp.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.PROMPTFL_Exp.CLASS_TOKEN_POSITION

        self.prompt_prefix = prompt_prefix
        self.clip_model = clip_model
        self.dtype = dtype

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

    @torch.no_grad()
    def _build_tokens_from_classnames(self, classnames_override):
        """
        Build token_prefix/token_suffix/tokenized_prompts/name_lens for given classnames_override.
        classnames_override: list[str] length = n_cls
        """
        device = self.ctx.device

        names = [n.replace("_", " ") for n in classnames_override]
        name_lens = [len(_tokenizer.encode(n)) for n in names]
        prompts = [self.prompt_prefix + " " + n + "." for n in names]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

        embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.dtype)
        token_prefix = embedding[:, :1, :]                # SOS
        token_suffix = embedding[:, 1 + self.n_ctx:, :]   # (CLS tokens + '.' + EOS + padding)

        return token_prefix, token_suffix, tokenized_prompts, name_lens

    def forward_with_classnames(self, classnames_override):
        """
        Return:
          prompts (Tensor[n_cls, 77, dim]), tokenized_prompts (Tensor[n_cls, 77])
        """
        token_prefix, token_suffix, tokenized_prompts, name_lens = \
            self._build_tokens_from_classnames(classnames_override)

        # assemble prompts exactly like your forward(), but using dynamic prefix/suffix/name_lens
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = token_prefix
        suffix = token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts_all = []
            for i in range(self.n_cls):
                name_len = name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts_all.append(prompt)
            prompts = torch.cat(prompts_all, dim=0)

        elif self.class_token_position == "front":
            prompts_all = []
            for i in range(self.n_cls):
                name_len = name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts_all.append(prompt)
            prompts = torch.cat(prompts_all, dim=0)

        else:
            raise ValueError

        return prompts, tokenized_prompts

    def forward_with_classnames_batch(self, classnames_overrides):
        """
        Vectorized multi-view prompt building.

        Args:
            classnames_overrides: List[List[str]]
                length = K (views), each is a list of length C (classes).

        Returns:
            prompts_flat: Tensor[(K*C), 77, dim]
            tokenized_flat: Tensor[(K*C), 77]
        """
        assert isinstance(classnames_overrides, (list, tuple)) and len(classnames_overrides) > 0
        K = len(classnames_overrides)
        C = self.n_cls

        for k in range(K):
            assert len(classnames_overrides[k]) == C, f"View {k}: expected {C} classnames, got {len(classnames_overrides[k])}"

        # ctx: [C, n_ctx, dim] (either class-specific or shared expanded)
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(C, -1, -1)

        # collect dynamic tokens for each view
        prefix_list, suffix_list, tok_list = [], [], []
        # name_lens not needed for "end"
        for k in range(K):
            token_prefix, token_suffix, tokenized_prompts, name_lens = \
                self._build_tokens_from_classnames(classnames_overrides[k])

            prefix_list.append(token_prefix)
            suffix_list.append(token_suffix)
            tok_list.append(tokenized_prompts)

        # stack: [K, C, ...]
        prefix = torch.stack(prefix_list, dim=0)   # [K,C,1,dim]
        suffix = torch.stack(suffix_list, dim=0)   # [K,C,*,dim]
        tokenized = torch.stack(tok_list, dim=0)   # [K,C,77]

        # align device to ctx (learnable prompt)
        dev = ctx.device
        prefix = prefix.to(dev)
        suffix = suffix.to(dev)
        tokenized = tokenized.to(dev)

        # expand ctx to [K,C,n_ctx,dim]
        ctx_k = ctx.unsqueeze(0).expand(K, -1, -1, -1)

        if self.class_token_position != "end":
            raise NotImplementedError(
                f"forward_with_classnames_batch currently supports CLASS_TOKEN_POSITION='end' only, "
                f"got '{self.class_token_position}'"
            )

        # prompts: [K,C,77,dim]
        prompts = torch.cat([prefix, ctx_k, suffix], dim=2)

        # flatten to [K*C,77,dim] and [K*C,77]
        prompts_flat = prompts.reshape(K * C, prompts.size(2), prompts.size(3))
        tokenized_flat = tokenized.reshape(K * C, tokenized.size(2))

        return prompts_flat, tokenized_flat

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features,_= self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits,image_features,text_features

    def forward_with_classnames(self, image, classnames_override):
        """
        classnames_override: list[str] length C
        Return: logits, image_features, text_features
        """
        image_features, _ = self.image_encoder(image.type(self.dtype))

        prompts, tokenized_prompts = self.prompt_learner.forward_with_classnames(classnames_override)

        if not hasattr(self, "_dbg_dyn_once"):
            print("[DYN] forward_with_classnames enabled:",
                  "n_cls_override=", len(classnames_override),
                  "prompts=", tuple(prompts.shape), "tokenized=", tuple(tokenized_prompts.shape),
                  "device(prompts)=", prompts.device, "device(tok)=", tokenized_prompts.device)
            print("[DYN] example override[0]:", classnames_override[0][:120])
            self._dbg_dyn_once = True

        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits, image_features, text_features


    def forward_with_classnames_batch(self, image, classnames_overrides):
        """
        Batch multi-view forward.
        Args:
            image: Tensor[B, ...]
            classnames_overrides: List[List[str]] length K, each length C

        Returns:
            logits: Tensor[K, B, C]
            image_features: Tensor[B, D]
            text_features: Tensor[K, C, D]
        """
        # image features (same as your normal forward)
        image_features, _  = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # build prompts for all views at once: [K*C,77,dim], [K*C,77]
        prompts_flat, tokenized_flat = self.prompt_learner.forward_with_classnames_batch(classnames_overrides)

        # encode text once
        text_features_flat = self.text_encoder(prompts_flat, tokenized_flat)
        text_features_flat = F.normalize(text_features_flat, dim=-1)

        # reshape to [K,C,D]
        K = len(classnames_overrides)
        C = self.prompt_learner.n_cls
        D = text_features_flat.size(-1)
        text_features = text_features_flat.view(K, C, D)

        # logits: [K,B,C]
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.einsum("bd,kcd->kbc", image_features, text_features)

        return logits, image_features, text_features

# @TRAINER_REGISTRY.register("PROMPTFL_Exp")
class PROMPTFL_Exp(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTFL_Exp.PREC in ["fp16", "fp32", "amp"]

    def _load_llm_expansion(self):
        """
        Selection (reading) only:
          - USE_ALL=True  -> read all expansions per class (K varies)
          - USE_ALL=False -> read exactly one expansion at IDX (K=1)

        Output structures are consistent for SINGLE/ALL.
        No pooling (mean) is done here; pooling belongs to computation strategy.
        """
        cfg = self.cfg
        dataset = cfg.DATASET.NAME.lower()
        classnames = list(self.dm.dataset.classnames)
        C = len(classnames)

        inject_mode = cfg.INJECT.MODE.lower()     # concat | kl | mse
        form = cfg.SEMANTIC.FORM.lower()          # attr | desc
        root = cfg.SEMANTIC.ROOT
        use_all = bool(cfg.SEMANTIC.USE_ALL)
        idx_one = int(getattr(cfg.SEMANTIC, "IDX", 0))
        sep = getattr(cfg.INJECT, "CONCAT_SEP", "; ")

        def choose_indices(n_items: int):
            if n_items <= 0:
                return []
            if use_all:
                return list(range(n_items))
            ii = idx_one
            if ii < 0:
                ii = n_items + ii
            ii = max(0, min(ii, n_items - 1))
            return [ii]

        print("=" * 80)
        print("[LLM-EXP] Loading expansions (READING ONLY)")
        print(f"[LLM-EXP] dataset={dataset} | num_classes={C}")
        print(f"[LLM-EXP] inject_mode={inject_mode} | form={form}")
        print(f"[LLM-EXP] select={'ALL' if use_all else 'SINGLE'} | IDX={idx_one} | sep={repr(sep)}")
        print("=" * 80)

        # ==================================================
        # CONCAT: read raw texts (always produce llm_exp as list[list[str]])
        # ==================================================
        if inject_mode == "concat":
            subdir = "ClassAttr" if form in ["attr", "attribute", "attributes"] else "ClassDescribe"
            base_dir = os.path.join(root, "CLS_Exp", subdir, dataset)
            print(f"[LLM-EXP][CONCAT] txt_base_dir: {base_dir}")

            llm_exp = []
            sel_indices = []
            total_lines_stats = []

            for cn in classnames:
                txt_path = os.path.join(base_dir, f"{cn}.txt")
                print(txt_path)
                if not os.path.isfile(txt_path):
                    alt = cn.replace(" ", "_")
                    print(alt)
                    txt_path2 = os.path.join(base_dir, f"{alt}.txt")
                    if os.path.isfile(txt_path2):
                        txt_path = txt_path2
                    else:
                        raise FileNotFoundError(f"LLM txt not found for class='{cn}': {txt_path}")

                with open(txt_path, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f.readlines()]
                lines = [ln for ln in lines if len(ln) > 0]

                idxs = choose_indices(len(lines))
                picked = [lines[i] for i in idxs]

                llm_exp.append(picked)        # list[str], K=1 or K=N
                sel_indices.append(idxs)      # list[int]
                total_lines_stats.append(len(lines))

            self.llm_exp = llm_exp
            self.llm_exp_joined = [sep.join(x) for x in llm_exp]
            self.sel_indices = sel_indices

            # logging
            sel_lens = [len(x) for x in llm_exp]
            print(f"[LLM-EXP][CONCAT] per-class candidate lines: min={min(total_lines_stats)}, max={max(total_lines_stats)}")
            print(f"[LLM-EXP][CONCAT] selected K per class: min={min(sel_lens)}, max={max(sel_lens)}")

            show_n = min(2, C)
            for i in range(show_n):
                print(f"[LLM-EXP][CONCAT][EX] class_id={i} '{classnames[i]}' idx={sel_indices[i]}")
                for j, s in enumerate(self.llm_exp[i][:5]):
                    print(f"  - text[{j}]: {s}")

            print("[LLM-EXP] Done loading (CONCAT).")
            return

        # ==================================================
        # KL/MSE: read embeddings (always produce embedding as [C,Kmax,D] + mask)
        # ==================================================
        if inject_mode in ["kl", "mse"]:

            # 目录名：保持你训练代码里的 dataset（一般是 "fedisic"/"fedcamelyon17md"）
            emb_dataset_dir = dataset

            # 文件名前缀：按你磁盘实际命名（驼峰/规范名）
            FILE_PREFIX_MAP = {
                "fedisic": "FedISIC",
                "fedcamelyon17md": "FedCamelyon17MD",
                # 如果你 covid 也是这种命名，就按实际补上：
                "covidflmd": "COVIDFLMD",  # 或 "CovidFLMD" 取决于你文件名
                "whu":"WHU",
                "pacs":"PACS"
            }

            dataset_key = str(dataset).lower()
            ds_prefix = FILE_PREFIX_MAP.get(dataset_key, dataset)  # 找不到就退化用 dataset 本身

            if form in ["attr", "attribute", "attributes"]:
                emb_path = os.path.join(
                    root, "embeddings", emb_dataset_dir, "text",
                    f"{ds_prefix}_class_attributes.pt"
                )
            else:
                emb_path = os.path.join(
                    root, "embeddings", emb_dataset_dir, "text",
                    f"{ds_prefix}_class_descriptions.pt"
                )

            print(f"[LLM-EXP][{inject_mode.upper()}] embedding_path: {emb_path}")
            obj = torch.load(emb_path, map_location="cpu")
            emb_dict = obj["embeddings"]  # class_name -> Tensor[N,D] (already normalized in your pipeline)

            picked_list = []
            sel_indices = []
            candidate_counts = []
            D = None

            for cn in classnames:
                key = cn if cn in emb_dict else cn.lower().replace(" ", "_")
                if key not in emb_dict:
                    raise KeyError(f"Class '{cn}' not found in embedding dict keys")

                embs = emb_dict[key].float()  # [N,D]
                if embs.dim() != 2:
                    raise ValueError(f"Embedding for class '{cn}' must be [N,D], got {tuple(embs.shape)}")

                if D is None:
                    D = embs.shape[1]

                idxs = choose_indices(embs.shape[0])
                picked = embs[idxs] if len(idxs) > 0 else embs[:1]  # [K,D]

                picked_list.append(picked)
                sel_indices.append(idxs)
                candidate_counts.append(int(embs.shape[0]))

            Kmax = max(x.shape[0] for x in picked_list)
            emb_tensor = torch.zeros(C, Kmax, D, dtype=torch.float32)
            emb_mask = torch.zeros(C, Kmax, dtype=torch.bool)

            for i, x in enumerate(picked_list):
                kk = x.shape[0]
                emb_tensor[i, :kk] = x
                emb_mask[i, :kk] = True

            self.embedding = emb_tensor.to(self.device)      # [C,Kmax,D]
            self.embedding_mask = emb_mask.to(self.device)   # [C,Kmax]
            self.sel_indices = sel_indices

            # logging
            sel_lens = [len(x) for x in picked_list]
            print(f"[LLM-EXP][{inject_mode.upper()}] D={D} | Kmax={Kmax}")
            print(f"[LLM-EXP][{inject_mode.upper()}] per-class candidates: min={min(candidate_counts)}, max={max(candidate_counts)}")
            print(f"[LLM-EXP][{inject_mode.upper()}] selected K per class: min={min(sel_lens)}, max={max(sel_lens)}")

            show_n = min(2, C)
            for i in range(show_n):
                print(f"[LLM-EXP][{inject_mode.upper()}][EX] class_id={i} '{classnames[i]}' idx={sel_indices[i]} picked={tuple(picked_list[i].shape)}")

            print(f"[LLM-EXP] Done loading ({inject_mode.upper()}).")
            return

        raise ValueError(f"Unknown INJECT.MODE={inject_mode}")

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.PROMPTFL_Exp.PREC in ["fp32", "amp"]:
            clip_model.float()

        # 先构建一次模型（classnames 用原始类名即可）
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # freeze 非 prompt 部分
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        self.model.to(self.device)

        # optimizer 只给 prompt_learner
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PROMPTFL_Exp.PREC == "amp" else None

        # 读取 LLM 扩展（给 concat/kl/mse forward 用）
        self._load_llm_expansion()

        print("="*90)
        print("[RUN] dataset=", cfg.DATASET.NAME, "| trainer=", type(self).__name__, "| nctx=", cfg.TRAINER.PROMPTFL_Exp.N_CTX)
        print("[RUN] inject=", cfg.INJECT.MODE, "| form=", cfg.SEMANTIC.FORM,
              "| use_all=", cfg.SEMANTIC.USE_ALL, "| idx=", cfg.SEMANTIC.IDX,
              "| multi_reduce=", cfg.SEMANTIC.MULTI_REDUCE)
        print("[RUN] lambda_kl=", cfg.INJECT.LAMBDA_KL, "| lambda_mse=", cfg.INJECT.LAMBDA_MSE, "| T=", cfg.INJECT.T)

        if cfg.INJECT.MODE.lower() == "concat":
            # llm_exp: List[List[str]]
            lens = [len(x) for x in self.llm_exp]
            print("[RUN][CONCAT] per-class K: min=", min(lens), "max=", max(lens), "example lens=", lens[:3])
            for cid in range(min(2, len(self.llm_exp))):
                print(f"[RUN][CONCAT][EX] class[{cid}]='{self.dm.dataset.classnames[cid]}'")
                for j, t in enumerate(self.llm_exp[cid][:min(2, len(self.llm_exp[cid]))]):
                    print(f"  - exp[{j}]: {t[:120]}")
        else:
            # embedding: [C,Kmax,D]
            print("[RUN][EMB] embedding=", tuple(self.embedding.shape),
                  "mask_valid=", int(self.embedding_mask.sum().item()))
        print("="*90)


    def forward_backward(self, idx, batch_idx, batch, **kwargs):
        """
        Full 3-branch forward_backward for:
          - Concat (LLM text expansion): multi-view training by K prompts, aggregate by:
              cfg.SEMANTIC.MULTI_REDUCE = "calc_then_mean" | "mean_then_calc"
          - KL (embedding alignment): align prompt logits with LLM-embedding logits, aggregate multi by:
              cfg.SEMANTIC.MULTI_REDUCE = "calc_then_mean" | "mean_then_calc"
          - MSE (embedding alignment): align prompt text_features with LLM embeddings, aggregate multi by:
              cfg.SEMANTIC.MULTI_REDUCE = "calc_then_mean" | "mean_then_calc"

        Assumptions:
          1) For concat mode, you have added:
               - PromptLearner.forward_with_classnames(classnames_override) -> (prompts, tokenized_prompts)
               - CustomCLIP.forward_with_classnames(image, classnames_override) -> (logits, image_features, text_features)
             and you have self.llm_exp: List[List[str]] aligned to class order.
          2) For kl/mse, you have self.embedding: [C,Kmax,D] and self.embedding_mask: [C,Kmax]
          3) self.model(image) returns (logits_prompt, image_features, text_features)
          4) self.model.logit_scale exists (CLIP) and image/text feats are L2-normalized in model forward.
        """
        cfg = self.cfg
        mode = cfg.INJECT.MODE.lower()  # "concat" | "kl" | "mse"
        prec = cfg.TRAINER.PROMPTFL_Exp.PREC

        image, label = self.parse_batch_train(batch)

        multi_reduce = getattr(cfg.SEMANTIC, "MULTI_REDUCE", "calc_then_mean").lower()
        use_all = bool(getattr(cfg.SEMANTIC, "USE_ALL", False))

        # ------------------------- helpers -------------------------
        def _ce_from_probs(probs, y):
            # probs: [B,C], y: [B]
            return -torch.log(probs.gather(1, y.view(-1, 1)).clamp_min(1e-12)).mean()

        def _kl_student_teacher(logits_student, logits_teacher):
            T = float(getattr(cfg.INJECT, "T", 1.0))
            log_p_student = F.log_softmax(logits_student / T, dim=1)
            p_teacher = F.softmax((logits_teacher / T).detach(), dim=1)
            return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)

        def _build_proto_from_embedding():
            # proto: mean over valid k for each class -> [C,D]
            emb = self.embedding            # [C,K,D]
            mask = self.embedding_mask      # [C,K]
            denom = mask.sum(dim=1).clamp(min=1).unsqueeze(1).float()  # [C,1]
            proto = (emb * mask.unsqueeze(-1)).sum(dim=1) / denom      # [C,D]
            return proto

        def _valid_ks_all_classes():
            # ks where every class has a valid embedding (mask all True for that k)
            mask = self.embedding_mask  # [C,K]
            valid_k = mask.all(dim=0)   # [K]
            ks = torch.nonzero(valid_k, as_tuple=False).view(-1)
            return ks

        def _compute_losses_concat():
            """
            Multi-view concat using (classname + expansion) prompts.
            Parallel encode K views in ONE text-encoder forward.

            Returns:
              loss_total, loss_ce, loss_kl(None), loss_mse(None), logits_for_acc
            """
            classnames = [n.replace("_", " ") for n in list(self.dm.dataset.classnames)]
            C = len(classnames)

            K_valid = min(len(self.llm_exp[c]) for c in range(C))
            if K_valid <= 0:
                raise RuntimeError("Concat mode: no LLM expansions found (K_valid <= 0)")

            # -------------------- low-noise debug gates --------------------
            client_id = int(idx) if "idx" in locals() else -1
            if not hasattr(self, "_dbg_concat_parallel_once"):
                self._dbg_concat_parallel_once = set()  # {(client_id,)}
            key_c = (client_id,)

            # Build K views of classnames_override
            classnames_overrides = []
            for k in range(K_valid):
                cn_k = [f"{classnames[c]} {self.llm_exp[c][k]}" for c in range(C)]
                classnames_overrides.append(cn_k)

            # -------------------- debug: print once per client --------------------
            if key_c not in self._dbg_concat_parallel_once:
                lens = [len(x) for x in self.llm_exp]
                print(f"[CONCAT-P] client={client_id} K_valid={K_valid} (minK={min(lens)}, maxK={max(lens)}) "
                      f"multi_reduce={multi_reduce}")
                # show what the override looks like (k=0/1, class0)
                print(f"[CONCAT-P][EX] k=0 cn0='{classnames_overrides[0][0][:180]}'")
                if K_valid > 1:
                    print(f"[CONCAT-P][EX] k=1 cn0='{classnames_overrides[1][0][:180]}'")
                self._dbg_concat_parallel_once.add(key_c)

            # One forward: logits[K,B,C]
            logits_kbc, _, _ = self.model.forward_with_classnames_batch(image, classnames_overrides)

            K, B, C2 = logits_kbc.shape
            assert K == K_valid and C2 == C, f"logits shape mismatch: got {tuple(logits_kbc.shape)} expected ({K_valid},B,{C})"

            # debug: logits sanity (once per client)
            if (client_id, "logits") not in getattr(self, "_dbg_concat_parallel_once2", set()):
                if not hasattr(self, "_dbg_concat_parallel_once2"):
                    self._dbg_concat_parallel_once2 = set()
                print(f"[CONCAT-P] logits shape={tuple(logits_kbc.shape)} "
                      f"mean={logits_kbc.mean().item():.4f} std={logits_kbc.std().item():.4f} "
                      f"device={logits_kbc.device}")
                self._dbg_concat_parallel_once2.add((client_id, "logits"))

            if multi_reduce == "calc_then_mean":
                # vectorized CE over K views: flatten to [K*B, C]
                logits_flat = logits_kbc.reshape(K * B, C)
                labels_rep = label.unsqueeze(0).expand(K, B).reshape(K * B)
                ce_all = F.cross_entropy(logits_flat, labels_rep, reduction="none")  # [K*B]
                loss_ce = ce_all.mean()

                # probs_avg for acc/logging
                probs_avg = F.softmax(logits_kbc, dim=-1).mean(dim=0)  # [B,C]
                logits_for_acc = torch.log(probs_avg.clamp_min(1e-12))

                # debug: aggregation summary (once per client)
                if (client_id, "reduce") not in self._dbg_concat_parallel_once2:
                    # show first few ce values (scalar summary)
                    print(f"[CONCAT-P][REDUCE] mode=calc_then_mean ce_mean={loss_ce.detach().cpu().item():.4f}")
                    self._dbg_concat_parallel_once2.add((client_id, "reduce"))

            elif multi_reduce == "mean_then_calc":
                probs_avg = F.softmax(logits_kbc, dim=-1).mean(dim=0)  # [B,C]
                loss_ce = _ce_from_probs(probs_avg, label)
                logits_for_acc = torch.log(probs_avg.clamp_min(1e-12))

                if (client_id, "reduce") not in self._dbg_concat_parallel_once2:
                    print(f"[CONCAT-P][REDUCE] mode=mean_then_calc "
                          f"avg_prob[min,max]=({probs_avg.min().item():.4f},{probs_avg.max().item():.4f}) "
                          f"ce={loss_ce.detach().cpu().item():.4f}")
                    self._dbg_concat_parallel_once2.add((client_id, "reduce"))

            else:
                raise ValueError(f"Concat mode: unknown SEMANTIC.MULTI_REDUCE={multi_reduce}")

            loss_total = loss_ce
            return loss_total, loss_ce, None, None, logits_for_acc

        def _compute_losses_mse(model_out):
            """
            MSE alignment between prompt text_features [C,D] and LLM embeddings.

            Aggregation for multi embeddings:
              - mean_then_calc: proto mean -> MSE(text_features, proto)
              - calc_then_mean: mean over (class,k) of mse(text_features[c], emb[c,k]) (masked)

            NOTE:
              Align teacher (proto/emb) dtype+device to text_features to avoid AMP dtype mismatch.
            """
            logits_prompt, _, text_features = model_out
            loss_ce = F.cross_entropy(logits_prompt, label)
            lam = float(getattr(cfg.INJECT, "LAMBDA_MSE", 1.0))

            # ---- dtype/device alignment reference (student side) ----
            ref_dev = text_features.device
            ref_dtype = text_features.dtype  # AMP: likely torch.float16

            # SINGLE behaves fine in both paths; use_all decides whether multi matters.
            if (not use_all) or (multi_reduce == "mean_then_calc"):
                proto = _build_proto_from_embedding()  # [C,D] (likely float32)
                proto = proto.to(device=ref_dev, dtype=ref_dtype)  # <-- 关键：对齐
                loss_mse = F.mse_loss(text_features, proto, reduction="mean")

            elif multi_reduce == "calc_then_mean":
                emb = self.embedding.to(device=ref_dev, dtype=ref_dtype)       # [C,K,D] <-- 关键：对齐
                mask = self.embedding_mask.to(device=ref_dev)                  # [C,K] bool

                # [C,1,D] - [C,K,D] -> [C,K,D] -> mean(D) -> [C,K]
                diff = (text_features.unsqueeze(1) - emb).pow(2).mean(dim=-1)  # [C,K]

                m = mask.float()
                loss_mse = (diff * m).sum() / m.sum().clamp(min=1.0)

            else:
                raise ValueError(f"MSE mode: unknown SEMANTIC.MULTI_REDUCE={multi_reduce}")

            loss_total = loss_ce + lam * loss_mse
            return loss_total, loss_ce, None, loss_mse, logits_prompt


        def _compute_losses_kl(model_out):
            """
            KL alignment between prompt logits (student) and LLM-embedding logits (teacher).
            Teacher logits:
              logits_llm = scale * image_features @ proto^T
            Aggregation for multi embeddings:
              - mean_then_calc: proto mean -> logits_teacher -> KL
              - calc_then_mean: KL per valid k (where all classes exist) -> mean
            """
            logits_prompt, image_features, _ = model_out
            loss_ce = F.cross_entropy(logits_prompt, label)
            lam = float(getattr(cfg.INJECT, "LAMBDA_KL", 1.0))

            logit_scale = self.model.logit_scale.exp()

            if (not use_all) or (multi_reduce == "mean_then_calc"):
                proto = _build_proto_from_embedding()  # [C,D]

                proto = proto.to(device=image_features.device, dtype=image_features.dtype)
                logit_scale = logit_scale.to(dtype=image_features.dtype)

                logits_teacher = logit_scale * image_features @ proto.t()  # [B,C]
                loss_kl = _kl_student_teacher(logits_prompt, logits_teacher)

            elif multi_reduce == "calc_then_mean":
                ks = _valid_ks_all_classes()
                if ks.numel() == 0:
                    # fallback
                    proto = _build_proto_from_embedding()

                    proto = proto.to(device=image_features.device, dtype=image_features.dtype)
                    logit_scale = logit_scale.to(dtype=image_features.dtype)

                    logits_teacher = logit_scale * image_features @ proto.t()
                    loss_kl = _kl_student_teacher(logits_prompt, logits_teacher)
                else:
                    loss_list = []

                    logit_scale = logit_scale.to(dtype=image_features.dtype)

                    for k in ks.tolist():
                        emb_k = self.embedding[:, k, :]  # [C,D]

                        emb_k = emb_k.to(device=image_features.device, dtype=image_features.dtype)

                        logits_teacher_k = logit_scale * image_features @ emb_k.t()  # [B,C]
                        loss_list.append(_kl_student_teacher(logits_prompt, logits_teacher_k))
                    loss_kl = torch.stack(loss_list).mean()

            else:
                raise ValueError(f"KL mode: unknown SEMANTIC.MULTI_REDUCE={multi_reduce}")

            loss_total = loss_ce + lam * loss_kl
            return loss_total, loss_ce, loss_kl, None, logits_prompt

        # ------------------------- compute (amp/non-amp) -------------------------
        if prec == "amp":
            with autocast():
                if mode == "concat":
                    loss_total, loss_ce, loss_kl, loss_mse, logits_for_acc = _compute_losses_concat()
                else:
                    model_out = self.model(image)
                    if mode == "mse":
                        loss_total, loss_ce, loss_kl, loss_mse, logits_for_acc = _compute_losses_mse(model_out)
                    elif mode == "kl":
                        loss_total, loss_ce, loss_kl, loss_mse, logits_for_acc = _compute_losses_kl(model_out)
                    else:
                        raise ValueError(f"Unknown INJECT.MODE={mode}")

            # ✅ 就贴在这里（autocast结束后，zero_grad前）
            if not torch.isfinite(loss_total):
                self.optim.zero_grad(set_to_none=True)
                self._skip = getattr(self, "_skip", 0) + 1
                print(
                    f"[WARN] skip step: non-finite loss_total={loss_total.item()} "
                    f"(epoch={self.epoch}, batch={self.batch_idx}, skips={self._skip}, mode={mode}, prec={prec})"
                )
                return {"loss_total": 0.0, "loss_ce": 0.0, "loss_kl": 0.0, "loss_mse": 0.0, "acc": 0.0}

            self.optim.zero_grad()
            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        else:
            if mode == "concat":
                loss_total, loss_ce, loss_kl, loss_mse, logits_for_acc = _compute_losses_concat()
            else:
                model_out = self.model(image)
                if mode == "mse":
                    loss_total, loss_ce, loss_kl, loss_mse, logits_for_acc = _compute_losses_mse(model_out)
                elif mode == "kl":
                    loss_total, loss_ce, loss_kl, loss_mse, logits_for_acc = _compute_losses_kl(model_out)
                else:
                    raise ValueError(f"Unknown INJECT.MODE={mode}")

            if not torch.isfinite(loss_total):
                self.optim.zero_grad(set_to_none=True)
                self._skip = getattr(self, "_skip", 0) + 1
                print(
                    f"[WARN] skip step: non-finite loss_total={loss_total.item()} "
                    f"(epoch={self.epoch}, batch={self.batch_idx}, skips={self._skip}, mode={mode}, prec={prec})"
                )
                return {"loss_total": 0.0, "loss_ce": 0.0, "loss_kl": 0.0, "loss_mse": 0.0, "acc": 0.0}

            self.model_backward_and_update(loss_total)

        # ------------------------- lr update -------------------------
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        # ------------------------- return dict -------------------------
        loss_summary = {
            "loss_total": float(loss_total.item()),
            "loss_ce": float(loss_ce.item()),
            "loss_kl": float(loss_kl.item()) if loss_kl is not None else 0.0,
            "loss_mse": float(loss_mse.item()) if loss_mse is not None else 0.0,
            "acc": compute_accuracy(logits_for_acc, label)[0].item(),
        }
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model bash main.sh caltech101 rn50_ep50 end 16 1 Falsenot found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def maybe_log_step(self, client_id, global_epoch, batch_idx, loss_summary):
        """
        Save per-step loss curve for each (client, round) as JSONL.
        One line = one training step (can be full or sampled by LOG_EVERY).

        File:
          <run_dir>/metrics_step/client_{id}/round_{round:04d}.jsonl
        """
        # only for federated local train
        if client_id < 0 or global_epoch < 0:
            return

        # log every N steps (1 = save all steps)
        log_every = int(getattr(self.cfg.TRAIN, "LOG_EVERY", 1))
        if (int(batch_idx) % log_every) != 0:
            return

        round_id = int(global_epoch)

        # local epoch and step within this round (avoid repeated batch_idx across local epochs)
        local_epoch = int(getattr(self, "epoch", 0))
        num_batches = int(getattr(self, "num_batches", 1))
        local_step = local_epoch * num_batches + int(batch_idx)

        # run_dir: use the cached readable run_dir you already implemented
        run_dir = self._get_prompt_embed_run_dir()
        metrics_dir = osp.join(run_dir, "metrics_step")
        os.makedirs(metrics_dir, exist_ok=True)

        out_dir = osp.join(metrics_dir, f"client_{int(client_id)}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = osp.join(out_dir, f"round_{round_id:04d}.jsonl")

        rec = {
            "round": round_id,
            "client": int(client_id),
            "local_epoch": local_epoch,
            "batch_idx": int(batch_idx),
            "local_step": int(local_step),
            "lr": float(self.get_current_lr()),
        }

        # only numeric fields (loss_total/loss_ce/loss_kl/loss_mse/acc...)
        for k, v in loss_summary.items():
            if isinstance(v, (int, float)):
                rec[k] = float(v)

        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # =========================
    # Run directory (human readable, no hash)
    # =========================
    def _run_tag(self):
        cfg = self.cfg
        seed = int(getattr(cfg, "SEED", -1))
        lk = float(getattr(cfg.INJECT, "LAMBDA_KL", 0.0))
        lm = float(getattr(cfg.INJECT, "LAMBDA_MSE", 0.0))
        T = float(getattr(cfg.INJECT, "T", 1.0))

        # keep short: use :g formatting
        return f"seed{seed}_lk{lk:g}_lm{lm:g}_T{T:g}"

    def _alloc_rep_id(self, run_dir_base: str):
        rep = 0
        while osp.exists(f"{run_dir_base}_rep{rep}"):
            rep += 1
        return rep

    def _get_prompt_embed_run_dir(self):
        """
        Cache run_dir so it doesn't change across rounds within the same run.
        """
        if hasattr(self, "_prompt_run_dir") and self._prompt_run_dir is not None:
            return self._prompt_run_dir

        cfg = self.cfg
        dataset = cfg.DATASET.NAME
        nctx = cfg.TRAINER.PROMPTFL_Exp.N_CTX
        trainer_name = type(self).__name__

        form = getattr(cfg.SEMANTIC, "FORM", "unknown")
        inject = getattr(cfg.INJECT, "MODE", "unknown")
        use_all = bool(getattr(cfg.SEMANTIC, "USE_ALL", False))
        idx = int(getattr(cfg.SEMANTIC, "IDX", 0))
        select = "all" if use_all else f"idx{idx}"
        reduce_mode = getattr(cfg.SEMANTIC, "MULTI_REDUCE", "na")

        base = "./embeddings/prompt_text_embeddings"

        run_tag = self._run_tag()
        run_dir_base = osp.join(
            base,
            dataset,
            f"nctx_{nctx}",
            f"trainer_{trainer_name}",
            f"form_{form}",
            f"inject_{inject}",
            f"select_{select}",
            f"reduce_{reduce_mode}",
            f"run_{run_tag}",
        )

        rep_id = self._alloc_rep_id(run_dir_base)
        run_dir = f"{run_dir_base}_rep{rep_id}"

        os.makedirs(run_dir, exist_ok=True)
        self._prompt_run_dir = run_dir
        self._prompt_rep_id = rep_id
        return run_dir

    def _get_prompt_embed_save_dir(self, round_id: int):
        run_dir = self._get_prompt_embed_run_dir()
        return osp.join(run_dir, f"round_{round_id:04d}")

    def _maybe_dump_manifest(self):
        run_dir = self._get_prompt_embed_run_dir()
        manifest_path = osp.join(run_dir, "manifest.json")
        if osp.isfile(manifest_path):
            return

        cfg = self.cfg
        payload = {
            "dataset": cfg.DATASET.NAME,
            "trainer": type(self).__name__,
            "backbone": cfg.MODEL.BACKBONE.NAME if hasattr(cfg.MODEL, "BACKBONE") else "unknown",
            "nctx": int(cfg.TRAINER.PROMPTFL_Exp.N_CTX),

            "semantic_root": getattr(cfg.SEMANTIC, "ROOT", ""),
            "semantic_form": getattr(cfg.SEMANTIC, "FORM", ""),
            "semantic_use_all": bool(getattr(cfg.SEMANTIC, "USE_ALL", False)),
            "semantic_idx": int(getattr(cfg.SEMANTIC, "IDX", 0)),
            "multi_reduce": getattr(cfg.SEMANTIC, "MULTI_REDUCE", ""),

            "inject_mode": getattr(cfg.INJECT, "MODE", ""),
            "lambda_kl": float(getattr(cfg.INJECT, "LAMBDA_KL", 0.0)),
            "lambda_mse": float(getattr(cfg.INJECT, "LAMBDA_MSE", 0.0)),
            "T": float(getattr(cfg.INJECT, "T", 1.0)),

            "seed": int(getattr(cfg, "SEED", -1)),
            "run_tag": self._run_tag(),
            "rep_id": int(getattr(self, "_prompt_rep_id", -1)),
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"🧾 Saved manifest: {manifest_path}")

    # =========================
    # Prompt text embedding dumps
    # =========================
    @torch.no_grad()
    def _dump_client_text_embedding(self, client_id: int, round_id: int):
        self.set_model_mode("eval")
        self._maybe_dump_manifest()

        save_dir = self._get_prompt_embed_save_dir(round_id)
        os.makedirs(save_dir, exist_ok=True)

        dataset_name = self.cfg.DATASET.NAME
        classnames = list(self.dm.dataset.classnames)

        prompts = self.model.prompt_learner()
        tokenized_prompts = self.model.prompt_learner.tokenized_prompts.to(self.device)

        text_features = self.model.text_encoder(prompts, tokenized_prompts)
        text_features = F.normalize(text_features, dim=-1)  # [C,D]

        payload = {
            "dataset": dataset_name,
            "round": int(round_id),
            "client_id": int(client_id),
            "classnames": classnames,
            "text_features": text_features.detach().cpu(),
        }

        out_path = osp.join(save_dir, f"client_{client_id}.pt")
        torch.save(payload, out_path)
        print(f"💾 Saved client text embedding: {out_path}")

    @torch.no_grad()
    def save_global_text_embedding(self, round_id: int):
        self.set_model_mode("eval")
        self._maybe_dump_manifest()

        save_dir = self._get_prompt_embed_save_dir(round_id)
        os.makedirs(save_dir, exist_ok=True)

        dataset_name = self.cfg.DATASET.NAME
        classnames = list(self.dm.dataset.classnames)

        prompts = self.model.prompt_learner()
        tokenized_prompts = self.model.prompt_learner.tokenized_prompts.to(self.device)

        text_features = self.model.text_encoder(prompts, tokenized_prompts)
        text_features = F.normalize(text_features, dim=-1)  # [C,D]

        payload = {
            "dataset": dataset_name,
            "round": int(round_id),
            "classnames": classnames,
            "text_features": text_features.detach().cpu(),
        }

        out_path = osp.join(save_dir, "global.pt")
        torch.save(payload, out_path)
        print(f"💾 Saved global text embedding: {out_path}")
        self.save_global_ctx(round_id)

    # =========================
    # Learnable prompt dumps (ctx only)
    # =========================
    @torch.no_grad()
    def _dump_client_ctx(self, client_id: int, round_id: int):
        self.set_model_mode("eval")
        self._maybe_dump_manifest()

        save_dir = self._get_prompt_embed_save_dir(round_id)
        os.makedirs(save_dir, exist_ok=True)

        pl = getattr(self.model, "prompt_learner", None)
        if pl is None or not hasattr(pl, "ctx"):
            print("[WARN] model.prompt_learner.ctx not found; skip saving ctx.")
            return

        ctx = pl.ctx.detach().float().cpu()  # ctx is nn.Parameter
        payload = {
            "dataset": self.cfg.DATASET.NAME,
            "round": int(round_id),
            "client_id": int(client_id),
            "ctx": ctx,
        }

        out_path = osp.join(save_dir, f"client_{client_id}_ctx.pt")
        torch.save(payload, out_path)
        print(f"💾 Saved client ctx: {out_path} | ctx_shape={tuple(ctx.shape)}")


    @torch.no_grad()
    def save_global_ctx(self, round_id: int):
        self.set_model_mode("eval")
        self._maybe_dump_manifest()

        save_dir = self._get_prompt_embed_save_dir(round_id)
        os.makedirs(save_dir, exist_ok=True)

        pl = getattr(self.model, "prompt_learner", None)
        if pl is None or not hasattr(pl, "ctx"):
            print("[WARN] model.prompt_learner.ctx not found; skip saving global ctx.")
            return

        ctx = pl.ctx.detach().float().cpu()
        payload = {
            "dataset": self.cfg.DATASET.NAME,
            "round": int(round_id),
            "ctx": ctx,
        }

        out_path = osp.join(save_dir, "global_ctx.pt")
        torch.save(payload, out_path)
        print(f"💾 Saved global ctx: {out_path} | ctx_shape={tuple(ctx.shape)}")

    def after_train(self, idx=-1, epoch=0, is_fed=False):
        super().after_train(idx=idx, epoch=epoch, is_fed=is_fed)
        if idx >= 0 and epoch >= 0:
            self._dump_client_text_embedding(client_id=idx, round_id=epoch)
            self._dump_client_ctx(client_id=idx, round_id=epoch)
