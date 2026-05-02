import os
import os.path as osp
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler
from Dassl.dassl.utils import count_num_param, load_checkpoint, load_pretrained_weights

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

ANCHOR_ROOT = "./CLS_Exp/ClassAnchor"
ANCHOR_NUM = 4


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": "PROMPTFL_KL_Anchor",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0
    }
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


# ---------------------- Anchor2 diversity (your version) ----------------------
def branch_diversity_loss(text_feat_acd: torch.Tensor) -> torch.Tensor:
    """
    text_feat_acd: (A, C, D)
    """
    A, C, D = text_feat_acd.shape
    proto = text_feat_acd.mean(dim=1)      # (A, D)
    proto = F.normalize(proto, dim=-1)     # (A, D)
    sim = proto @ proto.t()                # (A, A)
    eye = torch.eye(A, device=sim.device, dtype=sim.dtype)
    off = sim - eye
    return (off ** 2).sum() / (A * (A - 1))


# ---------------------- Anchor2 PromptLearner (your multi-branch prompt) ----------------------
class Anchor2PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.n_cls = len(classnames)

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        n_ctx = int(cfg.TRAINER.PROMPTFL_KL_Anchor.N_CTX)
        self.n_ctx_cfg = n_ctx
        self.anchor_num = ANCHOR_NUM
        self.n_per_anchor = int(math.ceil(n_ctx / float(self.anchor_num)))
        self.n_ctx_total = int(self.anchor_num * self.n_per_anchor)

        dataset_name = cfg.DATASET.NAME.lower()
        anchor_path = osp.join(ANCHOR_ROOT, f"{dataset_name}.txt")
        if not osp.isfile(anchor_path):
            raise FileNotFoundError(
                f"[Anchor] File not found: {anchor_path}\n"
                f"Expected: {ANCHOR_ROOT}/{dataset_name}.txt"
            )

        with open(anchor_path, "r", encoding="utf-8") as f:
            anchors = [line.strip() for line in f if line.strip()]

        if len(anchors) != self.anchor_num:
            raise ValueError(f"[Anchor] Expect {self.anchor_num} anchors, got {len(anchors)} in {anchor_path}")

        for a in anchors:
            n_words = len(a.split())
            if n_words < 1 or n_words > 2:
                print(f"[Anchor][Warn] Anchor not 1-2 words: '{a}' (words={n_words})")
        self.anchors = anchors

        ctx_vectors = torch.empty(self.anchor_num, self.n_per_anchor, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        x_block = " ".join(["X"] * self.n_per_anchor)
        classnames_ = [c.replace("_", " ") for c in classnames]

        prompts_text = []
        for a in self.anchors:
            prompts_text.append([f"{a} {x_block} {name}." for name in classnames_])

        tokenized_list = []
        for a_idx in range(self.anchor_num):
            tokenized_a = torch.cat([clip.tokenize(p) for p in prompts_text[a_idx]])  # (C, 77)
            tokenized_list.append(tokenized_a)
        tokenized_prompts = torch.stack(tokenized_list, dim=0)  # (A, C, 77)

        with torch.no_grad():
            embedding_list = []
            for a_idx in range(self.anchor_num):
                emb = clip_model.token_embedding(tokenized_prompts[a_idx]).type(dtype)  # (C, 77, dim)
                embedding_list.append(emb)
            embedding_template = torch.stack(embedding_list, dim=0)

        self.register_buffer("tokenized_prompts", tokenized_prompts)       # (A, C, 77)
        self.register_buffer("embedding_template", embedding_template)     # (A, C, 77, dim)

        x_token_id = clip.tokenize("X")[0][1].item()
        x_pos_mask = (tokenized_prompts == x_token_id)   # (A, C, 77)
        x_count = x_pos_mask.sum(dim=-1)                 # (A, C)
        bad = (x_count != self.n_per_anchor).nonzero(as_tuple=False)
        if bad.numel() > 0:
            a0, c0 = bad[0].tolist()
            raise ValueError(
                f"[Anchor] Each branch prompt must contain exactly {self.n_per_anchor} 'X' tokens.\n"
                f"Found mismatch at branch={a0}, class={c0}, count={int(x_count[a0, c0])}"
            )
        self.register_buffer("x_pos_mask", x_pos_mask)

        print("========== [Anchor2 PromptLearner Debug] ==========")
        print(f"[Anchor2] dataset={dataset_name} | anchors={self.anchors}")
        print(f"[Anchor2] cfg N_CTX={self.n_ctx_cfg} | n_per_anchor={self.n_per_anchor} | total={self.n_ctx_total}")
        print(f"[Anchor2] tokenized={tuple(tokenized_prompts.shape)} template={tuple(embedding_template.shape)}")
        print("===================================================")

    def forward(self):
        A = self.anchor_num
        C = self.n_cls
        prompts = self.embedding_template.clone()  # (A, C, 77, dim)

        for a_idx in range(A):
            mask = self.x_pos_mask[a_idx]  # (C, 77)
            x_pos_idx = mask.nonzero(as_tuple=False)  # (C*n_per_anchor, 2)
            rows = x_pos_idx[:, 0]
            cols = x_pos_idx[:, 1]
            ctx_rep = self.ctx[a_idx].unsqueeze(0).expand(C, -1, -1).reshape(-1, prompts.size(-1))
            prompts[a_idx, rows, cols, :] = ctx_rep

        return prompts  # (A, C, 77, dim)


# ---------------------- KL PromptLearner: EXACTLY reuse your version ----------------------
class KLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PROMPTFL_KL_Anchor.N_CTX
        ctx_init = cfg.TRAINER.PROMPTFL_KL_Anchor.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        device = clip_model.token_embedding.weight.device

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.PROMPTFL_KL_Anchor.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.PROMPTFL_KL_Anchor.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)

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
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
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
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts


# ---------------------- Combined CLIP ----------------------
class CombinedCLIP(nn.Module):
    """
    Shared image/text encoder.
    Anchor2 branch -> mean logits over A
    KL branch -> logits_kl
    Fusion via image-conditioned gate (init prefers KL).
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_anchor2 = Anchor2PromptLearner(cfg, classnames, clip_model)
        self.prompt_kl = KLPromptLearner(cfg, classnames, clip_model)

        self.tokenized_anchor2 = self.prompt_anchor2.tokenized_prompts  # (A,C,77)
        self.tokenized_kl = self.prompt_kl.tokenized_prompts            # (C,77)

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        d = clip_model.ln_final.weight.shape[0]
        h = max(1, d // 4)
        self.gate = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, 1),
        )
        with torch.no_grad():
            # Start close to KL-only to avoid A+B<A at early stage
            self.gate[-1].bias.fill_(2.2)  # sigmoid ~0.9

    def forward(self, image):
        """
        returns:
          logits_mix: (B,C)
          logits_anchor: (B,C)
          logits_kl: (B,C)
          image_features: (B,D)
          text_features_acd: (A,C,D)
          w: (B,1)
        """
        image_features, _ = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        scale = self.logit_scale.exp()

        # ---- Anchor2 branch ----
        prompts_ac77d = self.prompt_anchor2()  # (A,C,77,dim)
        A, C, _, dim = prompts_ac77d.shape

        prompts_flat = prompts_ac77d.reshape(A * C, 77, dim)
        tokenized_flat = self.tokenized_anchor2.reshape(A * C, 77)

        text_flat = self.text_encoder(prompts_flat, tokenized_flat)  # (A*C,D)
        text_features_acd = text_flat.reshape(A, C, -1)
        text_features_acd = text_features_acd / text_features_acd.norm(dim=-1, keepdim=True)

        logits_abc = scale * torch.einsum("bd,acd->abc", image_features, text_features_acd)  # (A,B,C)
        logits_anchor = logits_abc.mean(dim=0)  # (B,C)

        # ---- KL branch ----
        prompts_k = self.prompt_kl()  # (C,77,dim)
        text_k = self.text_encoder(prompts_k, self.tokenized_kl)  # (C,D)
        text_k = text_k / text_k.norm(dim=-1, keepdim=True)
        logits_kl = scale * (image_features @ text_k.t())

        gate_dtype = next(self.gate.parameters()).dtype
        w = torch.sigmoid(self.gate(image_features.to(dtype=gate_dtype))).clamp(0.01, 0.99)
        w = w.to(dtype=image_features.dtype)  # 可选：让 w 回到 half，后面融合更一致

        logits_mix = w * logits_kl + (1.0 - w) * logits_anchor

        return logits_mix, logits_anchor, logits_kl, image_features, text_features_acd, w


# ---------------------- Trainer ----------------------
class PROMPTFL_KL_Anchor(TrainerX):
    def check_cfg(self, cfg):
        # reuse Anchor2 precision if exists else KL
        if hasattr(cfg.TRAINER, "PROMPTFL_KL_Anchor") and hasattr(cfg.TRAINER.PROMPTFL_KL_Anchor, "PREC"):
            prec = cfg.TRAINER.PROMPTFL_KL_Anchor.PREC
        else:
            prec = cfg.TRAINER.PROMPTFL_KL_Anchor.PREC
        assert prec in ["fp16", "fp32", "amp"]

    # ---- teacher embeddings for KL ----
    def _load_llm_attribute_embeddings_all(self):
        cfg = self.cfg
        dataset = cfg.DATASET.NAME.lower()
        classnames = list(self.dm.dataset.classnames)
        C = len(classnames)

        root = cfg.SEMANTIC.ROOT
        emb_dataset_dir = dataset

        FILE_PREFIX_MAP = {
            "fedisic": "FedISIC",
            "fedcamelyon17md": "FedCamelyon17MD",
            "covidflmd": "COVIDFLMD",
            "whu": "WHU",
            "pacs": "PACS",
        }
        ds_prefix = FILE_PREFIX_MAP.get(dataset, dataset)

        emb_path = osp.join(root, "embeddings", emb_dataset_dir, "text",
                            f"{ds_prefix}_class_attributes.pt")

        print("=" * 80)
        print("[KL-TEACHER] Loading ATTRIBUTE embeddings (ALL)")
        print(f"[KL-TEACHER] emb_path={emb_path}")
        print("=" * 80)

        obj = torch.load(emb_path, map_location="cpu", weights_only=True)
        emb_dict = obj["embeddings"]

        picked_list = []
        counts = []
        D = None

        for cn in classnames:
            key = cn if cn in emb_dict else cn.lower().replace(" ", "_")
            if key not in emb_dict:
                raise KeyError(f"[KL-TEACHER] class '{cn}' not found in embedding dict keys")

            embs = emb_dict[key].float()  # [K,D]
            if D is None:
                D = embs.size(1)
            picked_list.append(embs)
            counts.append(int(embs.size(0)))

        Kmax = max(x.size(0) for x in picked_list)
        emb_tensor = torch.zeros(C, Kmax, D, dtype=torch.float32)
        emb_mask = torch.zeros(C, Kmax, dtype=torch.bool)

        for i, x in enumerate(picked_list):
            kk = x.size(0)
            emb_tensor[i, :kk] = x
            emb_mask[i, :kk] = True

        self.embedding = emb_tensor.to(self.device)      # [C,K,D]
        self.embedding_mask = emb_mask.to(self.device)   # [C,K]

        print(f"[KL-TEACHER] D={D} | Kmax={Kmax} | K(min,max)=({min(counts)},{max(counts)})")
        print("=" * 80)

    def _build_proto_mean(self, ref_dev, ref_dtype):
        emb = self.embedding
        mask = self.embedding_mask
        denom = mask.sum(dim=1).clamp(min=1).unsqueeze(1).float()
        proto = (emb * mask.unsqueeze(-1)).sum(dim=1) / denom
        return proto.to(device=ref_dev, dtype=ref_dtype)

    @staticmethod
    def _kl_student_teacher(logits_student, logits_teacher, T: float):
        log_p = F.log_softmax(logits_student / T, dim=1)
        p_t = F.softmax((logits_teacher / T).detach(), dim=1)
        return F.kl_div(log_p, p_t, reduction="batchmean") * (T * T)

    def _get_prec(self):
        cfg = self.cfg
        if hasattr(cfg.TRAINER, "PROMPTFL_KL_Anchor") and hasattr(cfg.TRAINER.PROMPTFL_KL_Anchor, "PREC"):
            return cfg.TRAINER.PROMPTFL_KL_Anchor.PREC
        return cfg.TRAINER.PROMPTFL_KL_Anchor.PREC

    def _get_lambda_div(self):
        cfg = self.cfg
        if hasattr(cfg.TRAINER, "PROMPTFL_KL_Anchor") and hasattr(cfg.TRAINER.PROMPTFL_KL_Anchor, "LAMBDA_DIVERSE"):
            return float(cfg.TRAINER.PROMPTFL_KL_Anchor.LAMBDA_DIVERSE)
        return 0.0

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if self._get_prec() in ["fp32", "amp"]:
            clip_model.float()

        print("Building CombinedCLIP (KL PromptLearner REUSED + Anchor2)")
        self.model = CombinedCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            # train only prompt learners + gate
            if ("prompt_anchor2" not in name) and ("prompt_kl" not in name) and ("gate" not in name):
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        self.lambda_div = self._get_lambda_div()
        self.lambda_kl = float(getattr(cfg.INJECT, "LAMBDA_KL", 0.5))
        self.T = float(getattr(cfg.INJECT, "T", 2.0))
        self.lambda_aux_anchor = float(getattr(cfg.INJECT, "LAMBDA_AUX_ANCHOR", 0.1))  # optional

        print(f"# params: {count_num_param(self.model):,}")
        print(f"# anchor2 prompt params: {count_num_param(self.model.prompt_anchor2):,}")
        print(f"# KL prompt params: {count_num_param(self.model.prompt_kl):,}")
        print(f"# gate params: {count_num_param(self.model.gate):,}")
        print(f"[Hyper] lambda_div={self.lambda_div} | lambda_kl={self.lambda_kl} T={self.T} | lambda_aux_anchor={self.lambda_aux_anchor}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_anchor2, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if self._get_prec() == "amp" else None

        self._load_llm_attribute_embeddings_all()

    def forward_backward(self, idx, batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self._get_prec()

        def compute_losses():
            logits_mix, logits_anchor, logits_kl, img_feat, text_feat_acd, w = self.model(image)

            loss_ce = F.cross_entropy(logits_mix, label)
            loss_aux = F.cross_entropy(logits_anchor, label) if self.lambda_aux_anchor > 0 else torch.tensor(0.0, device=image.device)
            loss_div = branch_diversity_loss(text_feat_acd) if self.lambda_div > 0 else torch.tensor(0.0, device=image.device)

            proto = self._build_proto_mean(ref_dev=img_feat.device, ref_dtype=img_feat.dtype)
            scale = self.model.logit_scale.exp().to(device=img_feat.device, dtype=img_feat.dtype)
            logits_teacher = scale * (img_feat @ proto.t())
            loss_kl = self._kl_student_teacher(logits_kl, logits_teacher, T=self.T)

            loss = loss_ce + self.lambda_aux_anchor * loss_aux + self.lambda_div * loss_div + self.lambda_kl * loss_kl
            return loss, loss_ce, loss_aux, loss_div, loss_kl, logits_mix, w

        if prec == "amp":
            with autocast():
                loss, loss_ce, loss_aux, loss_div, loss_kl, logits_mix, w = compute_losses()
            if not torch.isfinite(loss):
                self.optim.zero_grad(set_to_none=True)
                return {"loss": 0.0, "loss_ce": 0.0, "loss_aux": 0.0, "loss_div": 0.0, "loss_kl": 0.0, "acc": 0.0}

            self.optim.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss, loss_ce, loss_aux, loss_div, loss_kl, logits_mix, w = compute_losses()
            if not torch.isfinite(loss):
                self.optim.zero_grad(set_to_none=True)
                return {"loss": 0.0, "loss_ce": 0.0, "loss_aux": 0.0, "loss_div": 0.0, "loss_kl": 0.0, "acc": 0.0}
            self.model_backward_and_update(loss)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        acc = float(compute_accuracy(logits_mix, label)[0].item())
        return {
            "loss": float(loss.item()),
            "loss_ce": float(loss_ce.item()),
            "loss_aux": float(loss_aux.item()) if torch.is_tensor(loss_aux) else float(loss_aux),
            "loss_div": float(loss_div.item()) if torch.is_tensor(loss_div) else float(loss_div),
            "loss_kl": float(loss_kl.item()),
            "gate_w_mean": float(w.detach().mean().item()),
            "acc": acc,
        }

    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else ("model.pth.tar-" + str(epoch))

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch_ = checkpoint["epoch"]

            # Anchor2 buffers depend on anchors + classnames
            for k in ["tokenized_prompts", "embedding_template", "x_pos_mask"]:
                kk = f"prompt_anchor2.{k}"
                if kk in state_dict:
                    del state_dict[kk]

            # KL buffers depend on classnames / ctx_init tokenization
            for k in ["token_prefix", "token_suffix", "tokenized_prompts"]:
                kk = f"prompt_kl.{k}"
                if kk in state_dict:
                    del state_dict[kk]

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch_})')
            self._models[name].load_state_dict(state_dict, strict=False)
