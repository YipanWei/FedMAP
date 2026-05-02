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

# ------------------------------------------------------------
# Fixed anchor root (no cfg params)
# dataset file: {cfg.DATASET.NAME.lower()}.txt
# each file contains EXACTLY 4 anchors (your setting)
# We now build ANCHOR_NUM independent prompts:
#   Prompt_i: [Anchor_i] [X...X] [Class].
# where:
#   n_ctx = cfg.TRAINER.PROMPTFL_Anchor2.N_CTX
#   n_per_anchor = ceil(n_ctx / ANCHOR_NUM)
#   ctx shape = (ANCHOR_NUM, n_per_anchor, dim)  (branch-specific)
#
# Training:
#   loss = CE + LAMBDA_DIVERSE * branch_diversity_loss(text_features_acd)
# Inference:
#   logits = mean over branches
# ------------------------------------------------------------
ANCHOR_ROOT = "./CLS_Exp/ClassAnchor"
ANCHOR_NUM = 4

# diversity strength (no cfg param by your requirement)


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
        "trainer": "PROMPTFL",
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
        """
        prompts: (N, 77, dim)
        tokenized_prompts: (N, 77)
        return: (N, D)
        """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


def branch_diversity_loss(text_feat_acd: torch.Tensor) -> torch.Tensor:
    """
    Encourage different anchor-branches to be decorrelated (repulsive).
    text_feat_acd: (A, C, D)  (A=ANCHOR_NUM)
    Method: prototype per branch (mean over classes) + squared off-diagonal cosine similarity.
    """
    A, C, D = text_feat_acd.shape
    proto = text_feat_acd.mean(dim=1)      # (A, D)
    proto = F.normalize(proto, dim=-1)     # (A, D)
    sim = proto @ proto.t()                # (A, A)

    eye = torch.eye(A, device=sim.device, dtype=sim.dtype)
    off = sim - eye
    loss = (off ** 2).sum() / (A * (A - 1))
    return loss


class PromptLearner(nn.Module):
    """
    Build ANCHOR_NUM independent prompts:

      Prompt_i:  Anchor_i  X...X  {ClassName}.

    where:
      n_ctx = cfg.TRAINER.PROMPTFL_Anchor2.N_CTX
      n_per_anchor = ceil(n_ctx / ANCHOR_NUM)
      ctx shape = (ANCHOR_NUM, n_per_anchor, dim)
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.n_cls = len(classnames)

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        n_ctx = int(cfg.TRAINER.PROMPTFL_Anchor2.N_CTX)
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

        # Debug: anchor word count
        for a in anchors:
            n_words = len(a.split())
            if n_words < 1 or n_words > 2:
                print(f"[Anchor][Warn] Anchor not 1-2 words: '{a}' (words={n_words})")

        self.anchors = anchors

        # ---- learnable ctx per branch: (A, n_per_anchor, dim) ----
        ctx_vectors = torch.empty(self.anchor_num, self.n_per_anchor, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        # ---- build per-branch prompt text ----
        x_block = " ".join(["X"] * self.n_per_anchor)
        classnames_ = [c.replace("_", " ") for c in classnames]

        prompts_text = []
        for a in self.anchors:
            prompts_text.append([f"{a} {x_block} {name}." for name in classnames_])
        # prompts_text: list length A, each is list length C

        # ---- tokenize per branch ----
        tokenized_list = []
        for a_idx in range(self.anchor_num):
            tokenized_a = torch.cat([clip.tokenize(p) for p in prompts_text[a_idx]])  # (C, 77)
            tokenized_list.append(tokenized_a)
        tokenized_prompts = torch.stack(tokenized_list, dim=0)  # (A, C, 77)

        with torch.no_grad():
            # embedding_template: (A, C, 77, dim)
            embedding_list = []
            for a_idx in range(self.anchor_num):
                emb = clip_model.token_embedding(tokenized_prompts[a_idx]).type(dtype)  # (C, 77, dim)
                embedding_list.append(emb)
            embedding_template = torch.stack(embedding_list, dim=0)

        self.register_buffer("tokenized_prompts", tokenized_prompts)       # (A, C, 77)
        self.register_buffer("embedding_template", embedding_template)     # (A, C, 77, dim)

        # ---- locate X positions ----
        x_token_id = clip.tokenize("X")[0][1].item()
        self.x_token_id = x_token_id
        x_pos_mask = (tokenized_prompts == x_token_id)   # (A, C, 77)
        x_count = x_pos_mask.sum(dim=-1)                 # (A, C)

        # each prompt must contain exactly n_per_anchor X tokens
        bad = (x_count != self.n_per_anchor).nonzero(as_tuple=False)
        if bad.numel() > 0:
            a0, c0 = bad[0].tolist()
            raise ValueError(
                f"[Anchor] Each branch prompt must contain exactly {self.n_per_anchor} 'X' tokens.\n"
                f"Found mismatch at branch={a0}, class={c0}, count={int(x_count[a0, c0])}\n"
                f"Example prompt: {prompts_text[a0][c0]}"
            )

        self.register_buffer("x_pos_mask", x_pos_mask)

        # ---- debug prints ----
        print("========== [PromptLearner Debug: Multi-Branch] ==========")
        print(f"[Anchor] dataset                 = {dataset_name}")
        print(f"[Anchor] file                    = {anchor_path}")
        print(f"[Anchor] anchors                  = {self.anchors}")
        print(f"[Anchor] cfg N_CTX                = {self.n_ctx_cfg}")
        print(f"[Anchor] anchor_num               = {self.anchor_num}")
        print(f"[Anchor] n_per_anchor(ceil)       = {self.n_per_anchor}")
        print(f"[Anchor] implied total ctx tokens = {self.n_ctx_total} (>= cfg N_CTX)")
        print(f"[Anchor] branch0 prompt example   = {prompts_text[0][0]}")
        print(f"[Anchor] tokenized shape          = {tuple(tokenized_prompts.shape)} (A,C,77)")
        print(f"[Anchor] template shape           = {tuple(embedding_template.shape)} (A,C,77,dim)")
        print("=========================================================")

    def forward(self):
        """
        return prompts: (A, C, 77, dim)
        """
        A = self.anchor_num
        C = self.n_cls
        prompts = self.embedding_template.clone()  # (A, C, 77, dim)

        if self.ctx.shape[:2] != (A, self.n_per_anchor):
            raise RuntimeError(f"[PromptLearner] ctx shape mismatch: {tuple(self.ctx.shape)}")

        # replace X tokens per branch
        for a_idx in range(A):
            mask = self.x_pos_mask[a_idx]  # (C, 77)
            x_pos_idx = mask.nonzero(as_tuple=False)  # (C*n_per_anchor, 2) -> (row, col)

            expect = C * self.n_per_anchor
            if x_pos_idx.size(0) != expect:
                raise RuntimeError(
                    f"[PromptLearner] X positions mismatch at branch {a_idx}: "
                    f"{x_pos_idx.size(0)} vs {expect}"
                )

            rows = x_pos_idx[:, 0]
            cols = x_pos_idx[:, 1]

            # ctx for this branch: (n_per_anchor, dim)
            # repeat for each class: (C, n_per_anchor, dim) -> (C*n_per_anchor, dim)
            ctx_rep = self.ctx[a_idx].unsqueeze(0).expand(C, -1, -1).reshape(-1, prompts.size(-1))
            prompts[a_idx, rows, cols, :] = ctx_rep

            # debug: exact replacement check for branch a_idx, class 0
            with torch.no_grad():
                first_cols = cols[rows == 0]
                if first_cols.numel() != self.n_per_anchor:
                    raise RuntimeError("[PromptLearner] class0 X count mismatch in forward()")
                diff = (prompts[a_idx, 0, first_cols, :] - self.ctx[a_idx]).abs().max().item()
                if diff > 1e-6:
                    raise RuntimeError(
                        f"[PromptLearner] replacement check failed (branch={a_idx}), max diff={diff:.2e}"
                    )

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts  # (A, C, 77)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        """
        returns:
          logits: (B, C)  (mean over branches)
          image_features: (B, D)
          text_features_acd: (A, C, D)
        """
        image_features, _ = self.image_encoder(image.type(self.dtype))  # (B, D_img)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts_ac77d = self.prompt_learner()  # (A, C, 77, dim)
        A, C, _, dim = prompts_ac77d.shape

        # flatten (A*C, 77, dim) + (A*C, 77)
        prompts_flat = prompts_ac77d.reshape(A * C, 77, dim)
        tokenized_flat = self.tokenized_prompts.reshape(A * C, 77)

        text_features_flat = self.text_encoder(prompts_flat, tokenized_flat)  # (A*C, D)
        text_features_acd = text_features_flat.reshape(A, C, -1)              # (A, C, D)
        text_features_acd = text_features_acd / text_features_acd.norm(dim=-1, keepdim=True)

        # logits per branch: (A, B, C)
        # einsum: image (B,D) with text (A,C,D) -> (A,B,C)
        logits_abc = self.logit_scale.exp() * torch.einsum("bd,acd->abc", image_features, text_features_acd)

        # mean over branches -> (B, C)
        logits = logits_abc.mean(dim=0)

        return logits, image_features, text_features_acd


class PROMPTFL_Anchor2(TrainerX):
    """
    Same name kept to align with your existing configs/usages,
    but internally upgraded to multi-branch + diversity loss.
    """
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTFL_Anchor2.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTFL_Anchor2.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP (Multi-Branch + Diversity)")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        self.LAMBDA_DIVERSE = cfg.TRAINER.PROMPTFL_Anchor2.LAMBDA_DIVERSE
        print(f"# params: {count_num_param(self.model):,}")
        print(f"# prompt learner params: {count_num_param(self.model.prompt_learner):,}")
        print(f"[Diversity] LAMBDA_DIVERSE = {self.LAMBDA_DIVERSE}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTFL_Anchor2.PREC == "amp" else None

    def forward_backward(self, idx, batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PROMPTFL_Anchor2.PREC

        if prec == "amp":
            with autocast():
                logits, _, text_features_acd = self.model(image)
                loss_ce = F.cross_entropy(logits, label)
                loss_div = branch_diversity_loss(text_features_acd)
                loss = loss_ce + self.LAMBDA_DIVERSE * loss_div

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        else:
            logits, _, text_features_acd = self.model(image)
            loss_ce = F.cross_entropy(logits, label)
            loss_div = branch_diversity_loss(text_features_acd)
            loss = loss_ce + self.LAMBDA_DIVERSE * loss_div
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": float(loss.item()),
            "loss_ce": float(loss_ce.item()),
            "loss_div": float(loss_div.item()),
            "acc": float(compute_accuracy(logits, label)[0].item()),
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

            # Rebuild buffers from current anchors & classnames:
            # embedding_template/tokenized_prompts/x_pos_mask are buffers, dependent on dataset & anchors.
            for k in ["embedding_template", "tokenized_prompts", "x_pos_mask"]:
                if k in state_dict:
                    del state_dict[k]

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch_})')
            self._models[name].load_state_dict(state_dict, strict=False)
