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
    design_details = {"trainer": 'FedMVP',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

class AttributeBank(nn.Module):
    """
    Embedding-only Attribute Bank.
    Loads precomputed CLIP text embeddings for class attributes.

    Expects .pt file format:
      obj = torch.load(path)
      emb_dict = obj["embeddings"]  # class_name -> Tensor[K, D]
    Output:
      A_cls: Tensor[C, D]
    """

    def __init__(self, cfg, classnames, device):
        super().__init__()
        self.classnames = [c.replace("_", " ") for c in classnames]
        self.device = device

        # semantic cfg
        self.root =  cfg.SEMANTIC.ROOT
        self.dataset = cfg.DATASET.NAME.lower()
        self.use_all =True
        self.idx_one = 0
        self.normalize = True
        # allow overriding prefix if you want
        self.file_prefix_override = str(getattr(cfg.SEMANTIC, "FILE_PREFIX", "")).strip()

        # buffers
        self.register_buffer("A_cls", None, persistent=False)

        # build
        self._build()

    def _get_prefix(self):
        if self.file_prefix_override:
            return self.file_prefix_override

        FILE_PREFIX_MAP = {
            "fedisic": "FedISIC",
            "fedcamelyon17md": "FedCamelyon17MD",
            "covidflmd": "COVIDFLMD",
            "whu": "WHU",
        }
        return FILE_PREFIX_MAP.get(self.dataset, self.dataset)

    def _choose_indices(self, n_items: int):
        if n_items <= 0:
            return []
        if self.use_all:
            return list(range(n_items))
        ii = self.idx_one
        if ii < 0:
            ii = n_items + ii
        ii = max(0, min(ii, n_items - 1))
        return [ii]

    def _build(self):
        prefix = self._get_prefix()
        emb_path = os.path.join(
            self.root, "embeddings", self.dataset, "text",
            f"{prefix}_class_attributes.pt"
        )

        if not os.path.isfile(emb_path):
            raise FileNotFoundError(f"[AttributeBank] embedding file not found: {emb_path}")

        obj = torch.load(emb_path, map_location="cpu",weights_only=True)
        if "embeddings" not in obj:
            raise KeyError(f"[AttributeBank] expected key 'embeddings' in {emb_path}")

        emb_dict = obj["embeddings"]  # class_name -> Tensor[K,D]

        feats = []
        for cn in self.classnames:
            # be tolerant to key variants
            key = cn if cn in emb_dict else cn.lower().replace(" ", "_")
            if key not in emb_dict:
                # fallback try replacing spaces with underscores without lower
                key2 = cn.replace(" ", "_")
                if key2 in emb_dict:
                    key = key2
                else:
                    raise KeyError(f"[AttributeBank] class '{cn}' not found in embedding dict of {emb_path}")

            embs = emb_dict[key].float()  # [K,D]
            if embs.dim() != 2:
                raise ValueError(f"[AttributeBank] embedding for '{cn}' must be [K,D], got {tuple(embs.shape)}")

            idxs = self._choose_indices(embs.shape[0])
            picked = embs[idxs] if len(idxs) > 0 else embs[:1]  # [k,D]

            # pool -> [D]
            v = picked.mean(dim=0)

            feats.append(v)

        A_cls = torch.stack(feats, dim=0)  # [C, D]

        if self.normalize:
            A_cls = F.normalize(A_cls, dim=-1)

        self.A_cls = A_cls.to(self.device)

        print(f"[AttributeBank] loaded: {emb_path}")
        print(f"[AttributeBank] A_cls shape: {tuple(self.A_cls.shape)} | use_all={self.use_all} | idx={self.idx_one}")

    def get_A(self):
        """
        Returns:
          A_cls: Tensor[C, D] on self.device
        """
        return self.A_cls

class PromptFormerLite(nn.Module):
    """
    Memory-friendly PromptFormer:
      Step-1: learn m query tokens, attend over visual patch tokens E -> Qv  [B,m,width]
      Step-2: cross-attend Qv over class attribute bank A -> P             [B,m,width]
      Output P is injected into ViT input tokens.

    Inputs:
      A: [C, D_attr]      (attribute bank, precomputed & normalized is OK)
      E: [B, b, width]    (visual patch tokens in *width* space, pre-transformer preferred)

    Output:
      P: [B, m, width]
    """

    def __init__(self, width: int, d_attr: int, m: int = 8, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert width % heads == 0, "width must be divisible by heads"
        self.width = width
        self.d_attr = d_attr
        self.m = m
        self.heads = heads
        self.dropout = dropout

        # learnable m queries (like DETR object queries)
        self.q_tokens = nn.Parameter(torch.empty(m, width))
        nn.init.normal_(self.q_tokens, std=0.02)

        # --- Step 1: queries read from visual E (self-attn style: Q from q_tokens, K/V from E) ---
        self.q1 = nn.Linear(width, width, bias=False)
        self.k1 = nn.Linear(width, width, bias=False)
        self.v1 = nn.Linear(width, width, bias=False)
        self.o1 = nn.Linear(width, width, bias=False)

        # --- Step 2: cross-attn from attributes (Q from visual summary, K/V from projected attributes) ---
        self.attr_proj = nn.Linear(d_attr, width, bias=False)
        self.q2 = nn.Linear(width, width, bias=False)
        self.k2 = nn.Linear(width, width, bias=False)
        self.v2 = nn.Linear(width, width, bias=False)
        self.o2 = nn.Linear(width, width, bias=False)

        # small FFN
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

        # optional prompt positional (helpful since we did not add CLIP positional for P)
        self.prompt_pos = nn.Parameter(torch.zeros(m, width))
        nn.init.normal_(self.prompt_pos, std=0.02)

        self.ln_qv = nn.LayerNorm(width)
        self.ln_p = nn.LayerNorm(width)

    def _attn(self, Q, K, V):
        """
        Q: [B, m, width]
        K,V: [B, n, width]  (n=b or n=C)
        return: [B, m, width]
        """
        B, m, W = Q.shape
        n = K.shape[1]
        h = self.heads
        d = W // h

        Qh = Q.view(B, m, h, d).transpose(1, 2)  # [B,h,m,d]
        Kh = K.view(B, n, h, d).transpose(1, 2)  # [B,h,n,d]
        Vh = V.view(B, n, h, d).transpose(1, 2)  # [B,h,n,d]

        attn = (Qh @ Kh.transpose(-2, -1)) / (d ** 0.5)  # [B,h,m,n]
        attn = attn.softmax(dim=-1)
        if self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = attn @ Vh  # [B,h,m,d]
        out = out.transpose(1, 2).contiguous().view(B, m, W)  # [B,m,W]
        return out

    def forward(self, A: torch.Tensor, E: torch.Tensor):
        B = E.shape[0]
        device = E.device

        # 关键：PromptFormer 内部统一用 fp32
        compute_dtype = torch.float32
        out_dtype = E.dtype

        E32 = E.to(device=device, dtype=compute_dtype)
        A32 = A.to(device=device, dtype=compute_dtype)

        Q0 = self.q_tokens.to(device=device, dtype=compute_dtype).unsqueeze(0).expand(B, -1, -1)

        # Step1
        Q = self.q1(Q0)
        K = self.k1(E32)
        V = self.v1(E32)
        Qv = self._attn(Q, K, V)
        Qv = self.o1(Qv)
        Qv = self.ln_qv(Qv + Q0)

        # Step2
        A_ = self.attr_proj(A32)              # [C,W]
        A_ = A_.unsqueeze(0).expand(B, -1, -1)

        Q2 = self.q2(Qv)
        K2 = self.k2(A_)
        V2 = self.v2(A_)
        P = self._attn(Q2, K2, V2)
        P = self.o2(P)

        P = self.ffn(P) + P
        P = self.ln_p(P)
        P = P + self.prompt_pos.to(device=device, dtype=compute_dtype).unsqueeze(0)

        # 输出再转回 vision dtype（通常 fp16）
        return P.to(dtype=out_dtype)


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

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = [c.replace("_", " ") for c in classnames]

        # vision (FedMVP needs these APIs)
        self.image_encoder = clip_model.visual

        # text modules (fixed prompts, no PromptLearner)
        self.token_embedding = clip_model.token_embedding
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # fixed template
        template = getattr(cfg.TRAINER.FedMVP, "TEXT_TEMPLATE", "a photo of a {}.")
        texts = [template.format(c) for c in self.classnames]
        tokenized = torch.cat([clip.tokenize(t) for t in texts])  # [C,77]
        self.register_buffer("tokenized_prompts", tokenized, persistent=False)

        # attribute bank (embedding-only)
        self.attr_bank = AttributeBank(cfg, self.classnames, device=torch.device("cpu"))
        d_attr = int(self.attr_bank.get_A().shape[1])

        # prompt former lite
        width = getattr(self.image_encoder, "width", None)
        if width is None:
            width = int(self.image_encoder.class_embedding.numel())

        m = int(cfg.TRAINER.FedMVP.M)
        heads = int(cfg.TRAINER.FedMVP.HEADS)
        dropout = float(getattr(cfg.TRAINER.FedMVP, "DROPOUT", 0.0))
        self.prompt_former = PromptFormerLite(width=width, d_attr=d_attr, m=m, heads=heads, dropout=dropout)

        # cache A
        self._A_cached = None
        self._A_cached_device = None
        self._A_cached_dtype = None

    def _get_A_on_device(self, device, dtype):
        if (self._A_cached is None) or (self._A_cached_device != device) or (self._A_cached_dtype != dtype):
            A = self.attr_bank.get_A()  # cpu
            self._A_cached = A.to(device=device, dtype=dtype, non_blocking=True)
            self._A_cached_device = device
            self._A_cached_dtype = dtype
        return self._A_cached

    def forward(self, image):
        device = image.device
        dtype = self.dtype

        # ---- text features (fixed) ----
        tokenized = self.tokenized_prompts.to(device)
        # prompts embedding from token_embedding (no grad)
        with torch.no_grad():
            prompt_emb = self.token_embedding(tokenized).type(dtype)  # [C,77,dim]
        text_features = self.text_encoder(prompt_emb, tokenized)      # [C,D]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # ---- attribute bank ----
        A_cls = self._get_A_on_device(device=device, dtype=dtype)  # [C,D_attr]

        # ---- vision tokens + dynamic prompts ----
        z, E = self.image_encoder.get_z_E(image.type(dtype))       # [B,1,W], [B,b,W]
        P = self.prompt_former(A_cls, E)                           # [B,m,W]
        image_features, _ = self.image_encoder.forward_with_prompts(z, E, P)  # [B,D]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # ---- logits ----
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features, text_features



class FedMVP(TrainerX):
    """
    Simplified FedMVP Trainer (Dassl style):
      - Trainable: model.prompt_former only
      - Frozen: all CLIP (vision+text) + token embeddings + logit_scale
      - Text: fixed template (inside CustomCLIP)
      - Vision: inject dynamic prompts generated by PromptFormerLite

    Works with your federated outer loop:
      local_trainer.model.load_state_dict(global_weights, strict=False)
      local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True, ...)
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.FedMVP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        # precision control
        if cfg.TRAINER.FedMVP.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building FedMVP CustomCLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # ============ freeze everything by default ============
        print("Freezing all parameters")
        for _, p in self.model.named_parameters():
            p.requires_grad_(False)

        # ============ unfreeze prompt_former only ============
        if not hasattr(self.model, "prompt_former"):
            raise AttributeError("CustomCLIP must have `prompt_former` for FedMVP.")
        for p in self.model.prompt_former.parameters():
            p.requires_grad_(True)

        print("\n" + "="*24 + " Trainable Params Breakdown " + "="*24)

        trainable_params = 0
        all_trainable = []

        for name, p in self.model.named_parameters():
            if p.requires_grad:
                n = p.numel()
                trainable_params += n
                all_trainable.append((name, tuple(p.shape), p.dtype, n))

        # 按参数量从大到小排序，方便你看“哪些占大头”
        all_trainable.sort(key=lambda x: x[-1], reverse=True)

        print(f"Trainable params total: {trainable_params:,} ({trainable_params/1e6:.4f} M)")
        print("Top trainable tensors:")
        for name, shape, dtype, n in all_trainable[:20]:  # 只打印前20个，避免刷屏
            print(f"- {name:45s} shape={str(shape):18s} dtype={str(dtype):10s} numel={n:,}")

        # ================== Communication Cost (FP16, trainable-only) ==================
        # Assumption: each round communicates ONLY trainable parameters (no backbone, no buffers)
        # Per round per client includes:
        #   - download global trainable params (server -> client)
        #   - upload updated trainable params (client -> server)
        # So: bytes/round/client = 2 * trainable_params * 2 (FP16 bytes)
        bytes_per_param = 2  # FP16
        bytes_per_round_per_client = 2 * trainable_params * bytes_per_param
        mb_per_round_per_client = bytes_per_round_per_client / (1024 ** 2)

        print("\n" + "-"*24 + " Communication Cost (FP16) " + "-"*24)
        print("Assumption: communicate trainable params only (download + upload).")
        print(f"Comm/round/client: {mb_per_round_per_client:.4f} MB")

        # Optional: if you know N clients per round and R total rounds, compute totals
        # You can manually set these two numbers, or plug in from your FL loop (m and total epochs).
        N = getattr(self.cfg, "NUM_CLIENTS_PER_ROUND", None)  # e.g., m = max(int(args.frac * cfg.DATASET.USERS), 1)
        R = getattr(self.cfg, "MAX_EPOCH", None)              # e.g., total global rounds

        if N is not None:
            mb_per_round_total = mb_per_round_per_client * N
            print(f"Comm/round total (N={N} clients): {mb_per_round_total:.4f} MB")

            if R is not None:
                gb_total_training = (mb_per_round_total * R) / 1024.0  # MB -> GB
                print(f"Comm total training (R={R} rounds): {gb_total_training:.4f} GB")

        print("-"*72 + "\n")
        print("="*72 + "\n")


# (可选) 如果你想训 logit_scale，也可以打开；论文通常不训
        # self.model.logit_scale.requires_grad_(True)

        # 检查视觉backbone能力（必须支持注入）
        if not (hasattr(self.model.image_encoder, "get_z_E") and hasattr(self.model.image_encoder, "forward_with_prompts")):
            raise AttributeError(
                "clip_model.visual must support get_z_E() and forward_with_prompts(). "
                "Please use VisionTransformer_fedmvp as visual backbone."
            )

        print(f"# params (total): {count_num_param(self.model):,}")
        print(f"# params (trainable prompt_former): {count_num_param(self.model.prompt_former):,}")

        # init weights（一般不需要，但给你保留接口；这里仅当你真有初始化文件）
        if cfg.MODEL.INIT_WEIGHTS:
            # 你可以把 INIT_WEIGHTS 用在 prompt_former 上（如果你有预训练prompt_former）
            load_pretrained_weights(self.model.prompt_former, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # ============ optimizer/scheduler ============
        # 只优化 prompt_former
        self.optim = build_optimizer(self.model.prompt_former, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        # register for checkpointing
        self.register_model("prompt_former", self.model.prompt_former, self.optim, self.sched)

        # amp scaler
        self.scaler = GradScaler() if cfg.TRAINER.FedMVP.PREC == "amp" else None

    def forward_backward(self, idx, batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.FedMVP.PREC

        if prec == "amp":
            with autocast():
                logits, _, _ = self.model(image)
                loss = F.cross_entropy(logits, label)

            self.optim.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        else:
            # fp16: 你如果不用 amp，一般 CLIP 是 fp16，loss 可能不稳；建议用 amp
            logits, _, _ = self.model(image)
            loss = F.cross_entropy(logits, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": float(loss.item()),
            "acc": float(compute_accuracy(logits, label)[0].item()),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        img = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return img, label

    def load_model(self, directory, epoch=None):
        """
        Loads only registered models: "prompt_former" (by default).
        """
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()  # should include "prompt_former"
        model_file = "model-best.pth.tar" if epoch is None else ("model.pth.tar-" + str(epoch))

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            ep = checkpoint.get("epoch", "unknown")

            print(f'Loading weights to {name} from "{model_path}" (epoch = {ep})')
            self._models[name].load_state_dict(state_dict, strict=True)