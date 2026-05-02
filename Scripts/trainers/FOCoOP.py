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
    design_details = {"trainer": 'FOCoOP',
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
    """
    Simplified FOCoOp PromptLearner:
      - ctx_l: local prompt context (trainable)
      - ctx_g: global prompt context (trainable, will be federated)
      - ctx_o: U OOD prompts (trainable, will be federated)
    forward() returns ID prompts by default (to keep compatibility).
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)

        # ===== cfg =====
        n_ctx = cfg.TRAINER.FOCoOP.N_CTX
        ctx_init = cfg.TRAINER.FOCoOP.CTX_INIT
        csc = cfg.TRAINER.FOCoOP.CSC
        self.class_token_position = cfg.TRAINER.FOCoOP.CLASS_TOKEN_POSITION

        # Optional FOCoOp hyper-params
        self.rho = getattr(cfg.TRAINER.FOCoOP, "RHO", 0.5)   # mix ratio
        self.U = getattr(cfg.TRAINER.FOCoOP, "U", 16)        # #OOD prompts

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # ===== 1) init a base ctx (from words or random) =====
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            base_ctx = embedding[0, 1 : 1 + n_ctx, :]   # [n_ctx, dim]
            prompt_prefix = ctx_init
        else:
            if csc:
                # class-specific base ctx
                base_ctx = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                base_ctx = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(base_ctx, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = csc

        # ===== 2) define 3 trainable ctx =====
        # If ctx_init provides [n_ctx, dim] but CSC=True, repeat it for each class
        if ctx_init and csc:
            base_ctx = base_ctx.unsqueeze(0).expand(n_cls, -1, -1).contiguous()

        # local/global have same shape
        self.ctx_l = nn.Parameter(base_ctx.clone())  # local
        self.ctx_g = nn.Parameter(base_ctx.clone())  # global

        # ood prompts: U of them (generic)
        ctx_o = torch.empty(self.U, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_o, std=0.02)
        self.ctx_o = nn.Parameter(ctx_o)

        # ===== 3) build token buffers for ID prompts =====
        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        id_prompts_str = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_id = torch.cat([clip.tokenize(p) for p in id_prompts_str])  # [C, 77]
        with torch.no_grad():
            emb_id = clip_model.token_embedding(tokenized_id).type(dtype)

        self.register_buffer("id_token_prefix", emb_id[:, :1, :])           # SOS
        self.register_buffer("id_token_suffix", emb_id[:, 1 + n_ctx :, :])  # CLS...EOS
        self.tokenized_prompts = tokenized_id  # keep the old attribute name for compatibility

        # ===== 4) build token buffers for OOD prompts =====
        # Simplified: use fixed word "unknown" as the pseudo OOD "class"
        ood_word = getattr(cfg.TRAINER.FOCoOP, "OOD_WORD", "unknown")
        self.ood_name_len = len(_tokenizer.encode(ood_word.replace("_", " ")))

        ood_prompts_str = [prompt_prefix + f" {ood_word}." for _ in range(self.U)]
        tokenized_ood = torch.cat([clip.tokenize(p) for p in ood_prompts_str])  # [U, 77]
        with torch.no_grad():
            emb_ood = clip_model.token_embedding(tokenized_ood).type(dtype)

        self.register_buffer("ood_token_prefix", emb_ood[:, :1, :])           # SOS
        self.register_buffer("ood_token_suffix", emb_ood[:, 1 + n_ctx :, :])  # "unknown" ... EOS
        self.tokenized_ood_prompts = tokenized_ood

    # --------------------------
    # Helpers to build prompts
    # --------------------------
    def _build_id_prompts(self, ctx):
        """ctx should be [C, n_ctx, dim]"""
        prefix = self.id_token_prefix
        suffix = self.id_token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts_list = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts_list.append(prompt)
            prompts = torch.cat(prompts_list, dim=0)

        elif self.class_token_position == "front":
            prompts_list = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts_list.append(prompt)
            prompts = torch.cat(prompts_list, dim=0)

        else:
            raise ValueError

        return prompts

    def _build_ood_prompts(self, ctx_o):
        """ctx_o should be [U, n_ctx, dim]"""
        prefix = self.ood_token_prefix
        suffix = self.ood_token_suffix
        U = ctx_o.shape[0]

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx_o, suffix], dim=1)

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts_list = []
            for i in range(U):
                name_len = self.ood_name_len
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx_o[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx_o[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts_list.append(prompt)
            prompts = torch.cat(prompts_list, dim=0)

        elif self.class_token_position == "front":
            prompts_list = []
            for i in range(U):
                name_len = self.ood_name_len
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx_o[i : i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts_list.append(prompt)
            prompts = torch.cat(prompts_list, dim=0)

        else:
            raise ValueError

        return prompts

    # --------------------------
    # Public APIs
    # --------------------------
    def get_id_prompts(self):
        """
        ID prompts use mixed ctx: (1-rho)*ctx_l + rho*ctx_g
        """
        ctx_l = self.ctx_l
        ctx_g = self.ctx_g

        # generic -> expand to [C, n_ctx, dim]
        if ctx_l.dim() == 2:
            ctx_l = ctx_l.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx_g = ctx_g.unsqueeze(0).expand(self.n_cls, -1, -1)

        ctx_mix = (1.0 - self.rho) * ctx_l + self.rho * ctx_g
        return self._build_id_prompts(ctx_mix)

    def get_ood_prompts(self):
        """
        OOD prompts: U prompts, each is 'X ... X unknown.'
        """
        return self._build_ood_prompts(self.ctx_o)

    def forward(self):
        # Keep compatibility: default returns ID prompts
        return self.get_id_prompts()

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)

        # ID tokenized prompts（保持兼容）
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        # OOD tokenized prompts（新增）
        self.tokenized_ood_prompts = getattr(self.prompt_learner, "tokenized_ood_prompts", None)

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        # ---- image encoder ----
        image_features, _ = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # ---- ID prompts/text ----
        prompts = self.prompt_learner()  # 默认返回 ID prompts
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()  # [B, C]

        # ---- OOD prompts/text (新增) ----
        # 兼容：如果 PromptLearner 还没实现 ood，就返回 None
        logits_ood, ood_text_features = None, None
        if hasattr(self.prompt_learner, "get_ood_prompts") and self.tokenized_ood_prompts is not None:
            prompts_ood = self.prompt_learner.get_ood_prompts()  # [U, 77, dim]
            tokenized_ood = self.tokenized_ood_prompts           # [U, 77]
            ood_text_features = self.text_encoder(prompts_ood, tokenized_ood)
            ood_text_features = ood_text_features / ood_text_features.norm(dim=-1, keepdim=True)

            logits_ood = logit_scale * image_features @ ood_text_features.t()  # [B, U]

        # 返回：保持原有输出，同时加上 logits_ood / ood_text_features
        return logits, logits_ood, image_features, text_features, ood_text_features


# @TRAINER_REGISTRY.register("FOCoOP")
class FOCoOP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.FOCoOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.FOCoOP.PREC == "fp32" or cfg.TRAINER.FOCoOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            # print(name,":",param.size())
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        print(f"# params: {count_num_param(self.model):,}")
        print(f"# prompt learner params: {count_num_param(self.model.prompt_learner):,}")

        # ================== FOCoOP Params & Communication (FP16, trainable-only) ==================

        print("\n" + "="*22 + " FOCoOP Params & Comm (FP16) " + "="*22)

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

        # 2) Communication per round (FP16, trainable-only)
        # Assumption: only trainable parameters are communicated each round (download + upload).
        bytes_per_param = 2  # FP16
        bytes_per_round_per_client = 2 * trainable_params * bytes_per_param
        mb_per_round_per_client = bytes_per_round_per_client / (1024 ** 2)

        print("\n" + "-"*22 + " Communication Cost " + "-"*22)
        print("Assumption: communicate trainable params only (download + upload), FP16.")
        print(f"Comm/round/client:         {mb_per_round_per_client:.4f} MB")

        # 3) Optional totals (match your FL outer loop)
        # N = m = max(int(args.frac * cfg.DATASET.USERS), 1)
        # R = total global rounds
        N = getattr(self.cfg, "NUM_CLIENTS_PER_ROUND", None)  # if you stored it; else set manually
        R = getattr(self.cfg, "MAX_EPOCH", None)              # if equals global rounds; else set manually

        if N is not None:
            mb_per_round_total = mb_per_round_per_client * N
            print(f"Comm/round total (N={N}):  {mb_per_round_total:.4f} MB")
            if R is not None:
                gb_total_training = (mb_per_round_total * R) / 1024.0
                print(f"Comm total training (R={R}): {gb_total_training:.4f} GB")

        print("="*72 + "\n")
        # ==========================================================================================


        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.FOCoOP.PREC == "amp" else None

    def forward_backward(self, idx, batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.FOCoOP.PREC

        # 简化 BOS 超参
        lambda_ood = getattr(self.cfg.TRAINER.FOCoOP, "LAMBDA_OOD", 0.1)

        if prec == "amp":
            with autocast():
                logits, logits_ood, *_ = self.model(image)

                loss_ce = F.cross_entropy(logits, label)
                loss = loss_ce

                # OOD 分离（简化 BOS）
                if logits_ood is not None:
                    loss_ood = torch.logsumexp(logits_ood, dim=1).mean()
                    loss = loss + lambda_ood * loss_ood

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        else:
            logits, logits_ood, *_ = self.model(image)

            loss_ce = F.cross_entropy(logits, label)
            loss = loss_ce

            if logits_ood is not None:
                loss_ood = torch.logsumexp(logits_ood, dim=1).mean()
                loss = loss + lambda_ood * loss_ood

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_ce": loss_ce.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        # 记录一下 OOD loss 便于看训练是否起作用
        if logits_ood is not None:
            loss_summary["loss_ood"] = loss_ood.item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

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

            # # Ignore fixed token vectors
            # if "token_prefix" in state_dict:
            #     del state_dict["token_prefix"]
            #
            # if "token_suffix" in state_dict:
            #     del state_dict["token_suffix"]


            for k in ["token_prefix", "token_suffix",
                      "id_token_prefix", "id_token_suffix",
                      "ood_token_prefix", "ood_token_suffix"]:
                if k in state_dict:
                    del state_dict[k]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)