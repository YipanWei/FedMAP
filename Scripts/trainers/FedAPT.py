import os.path as osp
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.utils import Registry
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from Dassl.dassl.data import DataManager
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler
from Dassl.dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)


_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:

        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'FedAPT',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def load_cls_exp(dataset_name, classnames, mode="global", client_id=None):
    """
    从 TXT 文件读取语义扩展描述。
    支持结构：
        ./ClsExp/{dataset_name}/global/{cls}.txt
        ./ClsExp/{dataset_name}/local/client_{i}/{cls}.txt

    返回：
        {
            "covid_chest_x-ray": ["exp1", "exp2", ...],
            "pneumonia_chest_x-ray": ["exp1", "exp2", ...],
            ...
        }
    """
    base_root = f"./ClsExp/output/{dataset_name}"

    if mode == "global":
        base_path = os.path.join(base_root, "global")
    else:
        assert client_id is not None, "⚠️ local 模式必须提供 client_id"
        base_path = os.path.join(base_root, "local", f"client_{client_id}")

    semantic_dict = {}
    print(f"🧠 Loading semantic expansions ({mode}) for {dataset_name} (client={client_id})")

    for cname in classnames:
        txt_path = os.path.join(base_path, f"{cname}.txt")
        expansions = []

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    exp = line.strip()
                    if not exp:
                        continue
                    if exp.startswith('"') or exp.startswith("'"):
                        exp = exp[1:]
                    if exp.endswith('"') or exp.endswith("'"):
                        exp = exp[:-1]
                    expansions.append(exp)
        else:
            print(f"⚠️ Missing semantic file: {txt_path}")

        semantic_dict[cname] = expansions

    total_count = sum(len(v) for v in semantic_dict.values())
    print(f"✅ Loaded {len(classnames)} classes with {total_count} total expansions.")
    return semantic_dict

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
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class AdaptiveNetwork(nn.Module):
    """小型 MLP，用于根据图像特征预测 domain key 权重向量 Q(I(x))"""

    def __init__(self, in_dim=512, num_keys=6):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_keys)

    def forward(self, img_feat):
        q = self.fc(img_feat)  # [B, K]
        q = F.softmax(q, dim=-1)  # 权重归一化
        return q


class KeyBank(nn.Module):
    """存储每个 domain 的固定 key（论文中由服务器随机初始化并分配）"""

    def __init__(self, num_keys=6, n_ctx=16, ctx_dim=512):
        super().__init__()
        # e_k ∈ R^{s×d}
        keys = torch.empty(num_keys, n_ctx, ctx_dim)
        nn.init.normal_(keys, std=0.02)
        self.register_buffer("keys", keys)  # 冻结，不更新

    def forward(self, idx=None):
        if idx is None:
            return self.keys
        return self.keys[idx]

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.FedAPT.N_CTX_TEXT
        ctx_init = cfg.TRAINER.FedAPT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.FedAPT.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent Language design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])

        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        return prompts

class FedAPTPromptLearner(PromptLearner):
    """继承自原 PromptLearner，加入 FedAPT 的自适应 prompt 组合逻辑"""
    def __init__(self, cfg, classnames, clip_model, num_keys=6):
        super().__init__(cfg, classnames, clip_model)
        self.num_keys = num_keys
        ctx_dim = clip_model.ln_final.weight.shape[0]
        n_ctx = cfg.TRAINER.FedAPT.N_CTX_TEXT
        self.keybank = KeyBank(num_keys, n_ctx, ctx_dim)

        # ---- 新增的 dtype 对齐行 ----
        self.dtype = clip_model.dtype

        self.keybank = KeyBank(num_keys, n_ctx, ctx_dim)
        self.adapt_net = AdaptiveNetwork(in_dim=512, num_keys=num_keys)

        # 让所有组件与 CLIP dtype 一致
        self.keybank = self.keybank.to(dtype=self.dtype)
        self.adapt_net = self.adapt_net.to(dtype=self.dtype)
        self.ctx = nn.Parameter(self.ctx.to(dtype=self.dtype))

    def adaptive_prompt(self, image_features):
        """
        实现论文公式 (5):
        P(I(x)) = p_g + p_g ⊙ Σ_k q_k e′_k
        """
        q = self.adapt_net(image_features)           # [B, K]
        e = self.keybank().unsqueeze(0)              # [1, K, s, d]
        e = e.repeat(image_features.size(0), 1, 1, 1)

        # 计算加权 key: [B, s, d]
        q = q.unsqueeze(-1).unsqueeze(-1)
        weighted_key = torch.sum(q * e, dim=1)

        ctx = self.ctx.unsqueeze(0).expand(image_features.size(0), -1, -1)
        ctx = ctx + ctx * weighted_key.mean(dim=0, keepdim=True)  # 元提示 + 自适应扰动
        return ctx

    def forward(self, image_features=None):
        if image_features is None:
            return super().forward()
        # 自适应生成 prompt
        adaptive_ctx = self.adaptive_prompt(image_features)

        adaptive_ctx = adaptive_ctx.mean(dim=0, keepdim=True)
        adaptive_ctx = adaptive_ctx.expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = self.construct_prompts(adaptive_ctx, prefix, suffix)
        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        num_keys = getattr(cfg.DATASET, "USERS", None)
        if num_keys is None or num_keys <= 0:
            num_keys = getattr(cfg.TRAINER.FedAPT, "NUM_KEYS", 1)  # 兜底逻辑
        print(f"🧩 FedAPT using {num_keys} domain keys (from cfg.DATASET.USERS={cfg.DATASET.USERS})")

        self.prompt_learner = FedAPTPromptLearner(cfg, classnames, clip_model, num_keys=num_keys)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features, _ = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 自适应 prompt 生成
        prompts = self.prompt_learner(image_features)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()
        return logits, image_features, text_features


class FedAPT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.FedAPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.FedAPT.PREC == "fp32" or cfg.TRAINER.FedAPT.PREC == "amp":
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

        # ================== FedAPT Params & Communication (FP16, trainable + KeyBank.keys) ==================

        print("\n" + "="*20 + " FedAPT Params & Comm (FP16) " + "="*20)

        # 1) Total params & Trainable params
        all_params = 0
        trainable_params = 0
        all_trainable = []

        for name, p in self.model.named_parameters():
            n = p.numel()
            all_params += n
            if p.requires_grad:
                trainable_params += n
                all_trainable.append((name, tuple(p.shape), p.dtype, n))

        print(f"Total Params:              {all_params:,}")
        print(f"Trainable Params:          {trainable_params:,}")
        print(f"Trainable Ratio:           {trainable_params / max(all_params,1):.4%}")
        print(f"Trainable Params (M):      {trainable_params / 1e6:.4f} M")

        # (optional) show top trainable tensors
        all_trainable.sort(key=lambda x: x[-1], reverse=True)
        print("\nTop trainable tensors:")
        for name, shape, dtype, n in all_trainable[:20]:
            print(f"- {name:60s} shape={str(shape):18s} dtype={str(dtype):10s} numel={n:,}")

        # 2) Include KeyBank.keys in communication (buffer in state_dict)
        sd = self.model.state_dict()

        keybank_key = "prompt_learner.keybank.keys"  # <-- 这个是最可能的 key 名
        keybank_elems = 0
        keybank_bytes_one_way = 0

        print("\n[Extra communicated buffer: KeyBank.keys]")
        if keybank_key in sd and torch.is_tensor(sd[keybank_key]):
            t = sd[keybank_key]
            keybank_elems = int(t.numel())

            # FP16 comm assumption:
            # - float -> 2 bytes
            # - bool  -> 1 byte
            if t.dtype == torch.bool:
                bpe = 1
            elif t.is_floating_point():
                bpe = 2
            else:
                bpe = t.element_size()

            keybank_bytes_one_way = keybank_elems * bpe
            print(f"- found: {keybank_key}")
            print(f"  shape={tuple(t.shape)} dtype={t.dtype} numel={keybank_elems:,} bytes/elem={bpe}")
        else:
            # 如果你的 key 名不同，这里给你一个兜底：模糊查找包含 'keybank' 和 'keys' 的项
            candidates = [k for k in sd.keys() if ("keybank" in k.lower() and k.lower().endswith("keys"))]
            if len(candidates) > 0:
                k0 = candidates[0]
                t = sd[k0]
                keybank_elems = int(t.numel())
                if t.dtype == torch.bool:
                    bpe = 1
                elif t.is_floating_point():
                    bpe = 2
                else:
                    bpe = t.element_size()
                keybank_bytes_one_way = keybank_elems * bpe
                print(f"- key '{keybank_key}' not found, using candidate: {k0}")
                print(f"  shape={tuple(t.shape)} dtype={t.dtype} numel={keybank_elems:,} bytes/elem={bpe}")
            else:
                print(f"- NOT FOUND: {keybank_key}")
                print("  (No candidate key matching '*keybank*keys' found in state_dict.)")

        # 3) Communication cost (FP16): trainable params + KeyBank.keys
        # one-way bytes = trainable_params * 2 + keybank_bytes_one_way
        bytes_per_param_fp16 = 2
        trainable_bytes_one_way = trainable_params * bytes_per_param_fp16

        total_bytes_one_way = trainable_bytes_one_way + keybank_bytes_one_way

        # per round per client = download + upload = 2 * one-way
        bytes_per_round_per_client = 2 * total_bytes_one_way
        mb_per_round_per_client = bytes_per_round_per_client / (1024 ** 2)

        print("\n[Communication Cost]")
        print("Assumption: FP16 transmission; communicate (trainable params + KeyBank.keys) only.")
        print(f"Trainable elems:          {trainable_params:,}")
        print(f"KeyBank.keys elems:       {keybank_elems:,}")
        print(f"Comm/round/client:        {mb_per_round_per_client:.4f} MB  (down + up)")

        # 4) Optional totals (match your FL outer loop)
        # N = m = max(int(args.frac * cfg.DATASET.USERS), 1)
        # R = total global rounds
        N = getattr(self.cfg, "NUM_CLIENTS_PER_ROUND", None)  # or set manually
        R = getattr(self.cfg, "MAX_EPOCH", None)              # or set manually

        if N is not None:
            mb_per_round_total = mb_per_round_per_client * N
            print(f"Comm/round total (N={N}): {mb_per_round_total:.4f} MB")
            if R is not None:
                gb_total_training = (mb_per_round_total * R) / 1024.0
                print(f"Comm total training (R={R}): {gb_total_training:.4f} GB")

        print("="*72 + "\n")
        # ==========================================================================================


        if cfg.MODEL.INIT_WEIGHTS:
                    load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.FedAPT.PREC == "amp" else None

    def forward_backward(self, idx, batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.FedAPT.PREC

        if prec == "amp":
            with autocast():
                output, image_features, text_features = self.model(image)
                Lc = F.cross_entropy(output, label)

                # --- FedAPT domain loss ---
                Lq = 0.0
                if hasattr(self.model.prompt_learner, "adapt_net"):
                    q_pred = self.model.prompt_learner.adapt_net(image_features.detach())
                    domain_label = torch.full(
                        (image.size(0),), idx, dtype=torch.long, device=self.device
                    )  # client id 直接作为 domain id
                    Lq = F.cross_entropy(q_pred, domain_label)
                    loss = Lc + self.cfg.TRAINER.FedAPT.BETA * Lq
                else:
                    loss = Lc

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        else:
            output, image_features, text_features = self.model(image)
            Lc = F.cross_entropy(output, label)

            Lq = 0.0
            if hasattr(self.model.prompt_learner, "adapt_net"):
                q_pred = self.model.prompt_learner.adapt_net(image_features.detach())
                domain_label = torch.full(
                    (image.size(0),), idx, dtype=torch.long, device=self.device
                )
                Lq = F.cross_entropy(q_pred, domain_label)
                loss = Lc + self.cfg.TRAINER.FedAPT.BETA * Lq
            else:
                loss = Lc

            self.model_backward_and_update(loss)

        # logging
        loss_summary = {
            "Lc": Lc.item(),
            "Lq": float(Lq) if isinstance(Lq, torch.Tensor) else 0.0,
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

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


            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))

            self._models[name].load_state_dict(state_dict, strict=False)
