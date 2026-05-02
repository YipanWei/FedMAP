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
from Dassl.dassl.utils import (
    count_num_param, load_checkpoint, load_pretrained_weights
)

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# ------------------------------------------------------------
# Fixed anchor root (no cfg params)
# dataset file: {cfg.DATASET.NAME.lower()}.txt
# each file contains EXACTLY 4 anchors (your setting)
# learnable per anchor = ceil(n_ctx / anchor_num)
# prompt text: A1 X.. A2 X.. A3 X.. A4 X.. {Class}.
# ------------------------------------------------------------
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
        "trainer": "PROMPTFL_Anchor",
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


class PromptLearner(nn.Module):
    """
    Always uses anchor-based prompt:
      Anchor1 X... Anchor2 X... Anchor3 X... Anchor4 X... {ClassName}.
    where:
      n_ctx = cfg.TRAINER.PROMPTFL_Anchor.N_CTX
      n_per_anchor = ceil(n_ctx / ANCHOR_NUM)
      n_ctx_total = ANCHOR_NUM * n_per_anchor   (>= n_ctx)
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.n_cls = len(classnames)

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # ---- compute per-anchor learnable count from cfg n_ctx ----
        n_ctx = int(cfg.TRAINER.PROMPTFL_Anchor.N_CTX)
        self.n_ctx_cfg = n_ctx
        self.anchor_num = ANCHOR_NUM
        self.n_per_anchor = int(math.ceil(n_ctx / float(self.anchor_num)))
        self.n_ctx_total = int(self.anchor_num * self.n_per_anchor)

        # ---- resolve anchor file ----
        dataset_name = cfg.DATASET.NAME.lower()
        anchor_path = osp.join(ANCHOR_ROOT, f"{dataset_name}.txt")
        if not osp.isfile(anchor_path):
            raise FileNotFoundError(
                f"[Anchor] File not found: {anchor_path}\n"
                f"Expected: {ANCHOR_ROOT}/{dataset_name}.txt"
            )

        # ---- load anchors (must be 4 lines) ----
        with open(anchor_path, "r", encoding="utf-8") as f:
            anchors = [line.strip() for line in f if line.strip()]

        if len(anchors) != self.anchor_num:
            raise ValueError(f"[Anchor] Expect {self.anchor_num} anchors, got {len(anchors)} in {anchor_path}")

        # debug: anchor word count
        for a in anchors:
            n_words = len(a.split())
            if n_words < 1 or n_words > 2:
                print(f"[Anchor][Warn] Anchor not 1-2 words: '{a}' (words={n_words})")

        self.anchors = anchors

        # ---- learnable ctx (anchor_num*n_per_anchor, dim) ----
        ctx_vectors = torch.empty(self.n_ctx_total, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        # ---- build prompt text with X placeholders ----
        x_block = " ".join(["X"] * self.n_per_anchor)
        anchor_prefix = " ".join([f"{a} {x_block}" for a in self.anchors]).strip()

        classnames_ = [c.replace("_", " ") for c in classnames]
        prompts_text = [f"{anchor_prefix} {name}." for name in classnames_]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts_text])  # (n_cls, 77)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # (n_cls, 77, dim)

        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.register_buffer("embedding_template", embedding)

        # ---- locate X positions ----
        x_token_id = clip.tokenize("X")[0][1].item()
        self.x_token_id = x_token_id

        x_pos_mask = (tokenized_prompts == x_token_id)
        x_count = x_pos_mask.sum(dim=1)

        if not torch.all(x_count == self.n_ctx_total):
            bad = (x_count != self.n_ctx_total).nonzero(as_tuple=False).view(-1).tolist()
            raise ValueError(
                f"[Anchor] Each prompt must contain exactly {self.n_ctx_total} 'X' tokens.\n"
                f"Bad class indices: {bad}\n"
                f"Counts: {[int(x_count[i]) for i in bad]}\n"
                f"Example prompt (bad idx): {prompts_text[bad[0]] if bad else 'N/A'}"
            )

        self.register_buffer("x_pos_mask", x_pos_mask)

        # ---- debug prints ----
        print("========== [PromptLearner Debug] ==========")
        print(f"[Anchor] dataset          = {dataset_name}")
        print(f"[Anchor] file             = {anchor_path}")
        print(f"[Anchor] anchors           = {self.anchors}")
        print(f"[Anchor] cfg N_CTX          = {self.n_ctx_cfg}")
        print(f"[Anchor] anchor_num         = {self.anchor_num}")
        print(f"[Anchor] n_per_anchor(ceil) = {self.n_per_anchor}")
        print(f"[Anchor] n_ctx_total        = {self.n_ctx_total} (>= cfg N_CTX)")
        print(f"[Anchor] prompt example     = {prompts_text[0]}")
        print(f"[Anchor] tokenized shape    = {tuple(tokenized_prompts.shape)} (expect (n_cls, 77))")
        print(f"[Anchor] template shape     = {tuple(embedding.shape)}")
        print("==========================================")

    def forward(self):
        prompts = self.embedding_template.clone()  # (n_cls, 77, dim)

        # Debug: check shapes
        if prompts.dim() != 3 or prompts.size(0) != self.n_cls:
            raise RuntimeError(f"[PromptLearner] Bad prompts shape: {tuple(prompts.shape)}")
        if self.ctx.shape[0] != self.n_ctx_total:
            raise RuntimeError(f"[PromptLearner] ctx length mismatch: {self.ctx.shape[0]} vs {self.n_ctx_total}")

        # X positions (row-major order): (n_cls*n_ctx_total, 2)
        x_pos_idx = self.x_pos_mask.nonzero(as_tuple=False)
        expect = self.n_cls * self.n_ctx_total
        if x_pos_idx.size(0) != expect:
            raise RuntimeError(f"[PromptLearner] X positions mismatch: {x_pos_idx.size(0)} vs {expect}")

        rows = x_pos_idx[:, 0]
        cols = x_pos_idx[:, 1]

        # Repeat ctx for each class and flatten
        ctx_rep = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1).reshape(-1, prompts.size(-1))
        prompts[rows, cols, :] = ctx_rep

        # Debug: exact replacement check for class 0
        with torch.no_grad():
            first_cols = cols[rows == 0]
            if first_cols.numel() != self.n_ctx_total:
                raise RuntimeError("[PromptLearner] class0 X count mismatch in forward()")
            diff = (prompts[0, first_cols, :] - self.ctx).abs().max().item()
            if diff > 1e-6:
                raise RuntimeError(f"[PromptLearner] replacement check failed, max diff={diff:.2e}")

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

    def forward(self, image):
        image_features, _ = self.image_encoder(image.type(self.dtype))
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = self.logit_scale.exp() * image_features @ text_features.t()
        return logits, image_features, text_features


class PROMPTFL_Anchor(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTFL_Anchor.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTFL_Anchor.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        print(f"# params: {count_num_param(self.model):,}")
        print(f"# prompt learner params: {count_num_param(self.model.prompt_learner):,}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTFL_Anchor.PREC == "amp" else None

    def forward_backward(self, idx, batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PROMPTFL_Anchor.PREC

        if prec == "amp":
            with autocast():
                output, _, _ = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, _, _ = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": float(loss.item()),
            "acc": float(compute_accuracy(output, label)[0].item()),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

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

            # Rebuild tokenized_prompts/embedding_template/x_pos_mask from current anchors & classnames
            for k in ["embedding_template", "tokenized_prompts", "x_pos_mask"]:
                if k in state_dict:
                    del state_dict[k]

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch_})')
            self._models[name].load_state_dict(state_dict, strict=False)
