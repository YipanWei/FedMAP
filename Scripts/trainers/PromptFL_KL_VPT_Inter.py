import os.path as osp
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

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

import json

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
    design_details = {"trainer": 'VPT',
                      "vision_depth": cfg.TRAINER.PROMPTFL_KL_VPT_Inter.PROMPT_DEPTH_VISION,
                      "language_depth": 0, "vision_ctx": cfg.TRAINER.PROMPTFL_KL_VPT_Inter.N_CTX_VISION,
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
        n_ctx = cfg.TRAINER.PROMPTFL_KL_VPT_Inter.N_CTX
        ctx_init = cfg.TRAINER.PROMPTFL_KL_VPT_Inter.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.PROMPTFL_KL_VPT_Inter.CSC:
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

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
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
        self.class_token_position = cfg.TRAINER.PROMPTFL_KL_VPT_Inter.CLASS_TOKEN_POSITION

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

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # [新增] Meta-Net (Projector): 用于将视觉特征映射为文本特征的偏移量
        # 结构: Linear -> ReLU -> Linear (瓶颈结构以减少参数量)
        self.vis_dim = clip_model.visual.output_dim
        self.text_dim = clip_model.ln_final.weight.shape[0] # 通常与 vis_dim 相同

        self.meta_net = nn.Sequential(
            nn.Linear(self.vis_dim, self.vis_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(self.vis_dim // 16, self.text_dim)
        ).type(self.dtype) # 确保精度与 CLIP 一致 (fp16/fp32)

    def forward(self, image):
        # 1. 获取视觉特征 (由 VPT 增强)
        # image_features: [Batch, Dim]
        image_features, _ = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 2. 获取基础文本特征 (由 LPT 生成)
        # text_features_base: [Class, Dim]
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features_base = self.text_encoder(prompts, tokenized_prompts)
        text_features_base = text_features_base / text_features_base.norm(dim=-1, keepdim=True)

        # 3. [核心机制] 视觉引导的文本校准
        # 计算偏移量 bias: [Batch, Dim]
        bias = self.meta_net(image_features)

        # 将 bias 加到 base text features 上
        # text_features_base: [1, C, D]
        # bias: [B, 1, D]
        # text_features_inst: [B, C, D] (Instance-specific Text Features)
        text_features_inst = text_features_base.unsqueeze(0) + bias.unsqueeze(1)

        # 对校准后的特征再次归一化
        text_features_inst = text_features_inst / text_features_inst.norm(dim=-1, keepdim=True)

        # 4. 计算 Logits
        logit_scale = self.logit_scale.exp()

        # 使用 einsum 进行批量矩阵乘法:
        # b: Batch, c: Class, d: Dim
        # [B, D] x [B, C, D] -> [B, C]
        logits = logit_scale * torch.einsum("bd,bcd->bc", image_features, text_features_inst)

        # 返回:
        # logits: 用于 CE Loss (利用了视觉校准)
        # image_features: 原始视觉特征
        # text_features_base: 原始文本特征 (用于 KL Loss，去逼近 LLM)
        return logits, image_features, text_features_base

class PROMPTFL_KL_VPT_Inter(TrainerX):
    """
    Strict extraction of:
      PROMPTFL_KL_VPT_Inter with INJECT.MODE=kl, SEMANTIC.FORM=attr, SEMANTIC.USE_ALL=True,
      SEMANTIC.MULTI_REDUCE=mean_then_calc

    Goal: with same seed & same dataloader order, the loss/acc logits logs should match PROMPTFL_KL_VPT_Inter exactly.
    """

    def check_cfg(self, cfg):
        # keep identical precision constraint
        assert cfg.TRAINER.PROMPTFL_KL_VPT_Inter.PREC in ["fp16", "fp32", "amp"]
    # -------------------------
    # LLM embedding loader (KL + attr only)
    # -------------------------
    def _load_llm_attr_embeddings_kl(self):
        """
        This is a strict subset of PROMPTFL_KL_VPT_Inter._load_llm_expansion()
        for inject_mode in ['kl'] and form == 'attr'.

        It produces:
          self.embedding: [C, Kmax, D] (float32, on self.device)
          self.embedding_mask: [C, Kmax] (bool, on self.device)
          self.sel_indices
        """
        cfg = self.cfg
        dataset = cfg.DATASET.NAME.lower()
        classnames = list(self.dm.dataset.classnames)
        C = len(classnames)

        root = cfg.SEMANTIC.ROOT
        use_all = True
        idx_one = int(getattr(cfg.SEMANTIC, "IDX", 0))

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

        # 目录名：保持你训练代码里的 dataset（一般是 "fedisic"/"fedcamelyon17md"）
        emb_dataset_dir = dataset

        # 文件名前缀：按你磁盘实际命名（驼峰/规范名）
        FILE_PREFIX_MAP = {
            "fedisic": "FedISIC",
            "fedcamelyon17md": "FedCamelyon17MD",
            "covidflmd": "COVIDFLMD",  # 依你实际文件名可调整
            "whu": "WHU",
            "pacs": "PACS",
        }

        dataset_key = str(dataset).lower()
        ds_prefix = FILE_PREFIX_MAP.get(dataset_key, dataset)

        # KL + attr -> attributes
        emb_path = os.path.join(
            root, "embeddings", emb_dataset_dir, "text",
            f"{ds_prefix}_class_attributes.pt"
        )

        print("=" * 80)
        print("[LLM-EXP][KL-ATTR] Loading embeddings (STRICT)")
        print(f"[LLM-EXP][KL-ATTR] embedding_path: {emb_path}")
        print("=" * 80)

        obj = torch.load(emb_path, map_location="cpu",weights_only=True)
        emb_dict = obj["embeddings"]  # class_name -> Tensor[N,D]

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

        # optional logs (doesn't affect computation)
        sel_lens = [len(x) for x in picked_list]
        print(f"[LLM-EXP][KL-ATTR] D={D} | Kmax={Kmax}")
        print(f"[LLM-EXP][KL-ATTR] per-class candidates: min={min(candidate_counts)}, max={max(candidate_counts)}")
        print(f"[LLM-EXP][KL-ATTR] selected K per class: min={min(sel_lens)}, max={max(sel_lens)}")
        print("[LLM-EXP][KL-ATTR] Done.")

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.PROMPTFL_KL_VPT_Inter.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name or 'VPT' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")


        self.model.to(self.device)

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PROMPTFL_KL_VPT_Inter.PREC == "amp" else None

        # ONLY load KL attr embeddings
        self._load_llm_attr_embeddings_kl()

        print("=" * 90)
        print("[RUN] dataset=", cfg.DATASET.NAME, "| trainer=", type(self).__name__, "| nctx=", cfg.TRAINER.PROMPTFL_KL_VPT_Inter.N_CTX)
        print("[RUN] inject=kl | form=attr | use_all=True | multi_reduce=mean_then_calc")
        print("[RUN] lambda_kl=", cfg.INJECT.LAMBDA_KL, "| T=", cfg.INJECT.T)
        print("[RUN][EMB] embedding=", tuple(self.embedding.shape),
              "mask_valid=", int(self.embedding_mask.sum().item()))
        print("=" * 90)

    # -------------------------
    # exact KL pieces (copied from PROMPTFL_KL_VPT_Inter branch)
    # -------------------------
    def forward_backward(self, idx, batch_idx, batch, **kwargs):
        cfg = self.cfg
        prec = cfg.TRAINER.PROMPTFL_KL_VPT_Inter.PREC

        image, label = self.parse_batch_train(batch)

        def _kl_student_teacher(logits_student, logits_teacher):
            T = float(getattr(cfg.INJECT, "T", 2.0))
            log_p_student = F.log_softmax(logits_student / T, dim=1)
            p_teacher = F.softmax((logits_teacher / T).detach(), dim=1)
            return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)

        def _build_proto_from_embedding():
            emb = self.embedding            # [C,K,D]
            mask = self.embedding_mask      # [C,K]
            denom = mask.sum(dim=1).clamp(min=1).unsqueeze(1).float()  # [C,1]
            proto = (emb * mask.unsqueeze(-1)).sum(dim=1) / denom      # [C,D]
            return proto

        def _compute_losses_kl_mean_then_calc(model_out):
            """
            Modified Behavior for Visual-Guided Text Refinement:
              - CE Loss: Calculated on the 'Refined' logits (Vision + Text + Bias)
              - KL Loss: Calculated on the 'Base' logits (Vision + Base Text) vs LLM
            """
            # [修改] 解包返回值，注意第三个参数是 text_features_base
            logits_inst, image_features, text_features_base = model_out

            # 1. CE Loss: 使用校准后的 logits (Instance-specific)
            loss_ce = F.cross_entropy(logits_inst, label)

            lam = float(getattr(cfg.INJECT, "LAMBDA_KL", 0.5))

            # 2. 准备 Teacher (LLM Expansion)
            proto = _build_proto_from_embedding()  # [C,D]
            proto = proto.to(device=image_features.device, dtype=image_features.dtype)
            logit_scale = self.model.logit_scale.exp()
            logit_scale = logit_scale.to(dtype=image_features.dtype)

            logits_teacher = logit_scale * image_features @ proto.t()  # [B,C]

            # 3. [关键修改] 准备 Student for KL (Base LPT)
            # 我们希望 Base LPT 学到通用的 LLM 语义，而不是让 Bias 去拟合 LLM
            # 所以这里重新计算 image vs base_text 的 logits
            logits_student_base = logit_scale * image_features @ text_features_base.t() # [B,C]

            # 4. KL Loss: 约束 Base LPT 逼近 LLM
            loss_kl = _kl_student_teacher(logits_student_base, logits_teacher)

            loss_total = loss_ce + lam * loss_kl
            return loss_total, loss_ce, loss_kl, logits_inst

        if prec == "amp":
            with autocast():
                model_out = self.model(image)
                loss_total, loss_ce, loss_kl, logits_for_acc = _compute_losses_kl_mean_then_calc(model_out)

            # same non-finite skip logic
            if not torch.isfinite(loss_total):
                self.optim.zero_grad(set_to_none=True)
                self._skip = getattr(self, "_skip", 0) + 1
                print(
                    f"[WARN] skip step: non-finite loss_total={loss_total.item()} "
                    f"(epoch={self.epoch}, batch={self.batch_idx}, skips={self._skip}, mode=kl, prec={prec})"
                )
                return {"loss_total": 0.0, "loss_ce": 0.0, "loss_kl": 0.0, "acc": 0.0}

            self.optim.zero_grad()
            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        else:
            model_out = self.model(image)
            loss_total, loss_ce, loss_kl, logits_for_acc = _compute_losses_kl_mean_then_calc(model_out)

            if not torch.isfinite(loss_total):
                self.optim.zero_grad(set_to_none=True)
                self._skip = getattr(self, "_skip", 0) + 1
                print(
                    f"[WARN] skip step: non-finite loss_total={loss_total.item()} "
                    f"(epoch={self.epoch}, batch={self.batch_idx}, skips={self._skip}, mode=kl, prec={prec})"
                )
                return {"loss_total": 0.0, "loss_ce": 0.0, "loss_kl": 0.0, "acc": 0.0}

            self.model_backward_and_update(loss_total)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        loss_summary = {
            "loss_total": float(loss_total.item()),
            "loss_ce": float(loss_ce.item()),
            "loss_kl": float(loss_kl.item()),
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
        # keep identical to PROMPTFL_KL_VPT_Inter (you can reuse yours verbatim)
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch_ckpt = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch_ckpt})')
            self._models[name].load_state_dict(state_dict, strict=False)