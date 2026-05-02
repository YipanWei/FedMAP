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
    design_details = {"trainer": 'PROMPTFL_OB',
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
        n_ctx = cfg.TRAINER.PROMPTFL_OB.N_CTX
        ctx_init = cfg.TRAINER.PROMPTFL_OB.CTX_INIT #   (XXXXX) class with the  sylte of XXXXXXX 16
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
            if cfg.TRAINER.PROMPTFL_OB.CSC:
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
        self.class_token_position = cfg.TRAINER.PROMPTFL_OB.CLASS_TOKEN_POSITION

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


# @TRAINER_REGISTRY.register("PROMPTFL_OB")
class PROMPTFL_OB(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTFL_OB.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTFL_OB.PREC == "fp32" or cfg.TRAINER.PROMPTFL_OB.PREC == "amp":
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


        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PROMPTFL_OB.PREC == "amp" else None


    def forward_backward(self, idx,batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.PROMPTFL_OB.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output,_,_ = self.model(image)
            loss = F.cross_entropy(output, label)
            # if fedprox:
            #     model_weight = self.model.state_dict()
            #     fed_prox_reg = ((mu / 2) * torch.norm((model_weight['prompt_learner.ctx'] - global_weight['prompt_learner.ctx'])) ** 2)
            #     loss += fed_prox_reg
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

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

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    # ---------- run dir helpers ----------
    def _run_tag(self):
        cfg = self.cfg
        seed = int(getattr(cfg, "SEED", -1))

        prec = str(getattr(cfg.TRAINER.PROMPTFL_KL_Global, "PREC", "unknown"))

        return f"seed{seed}_prec{prec}"

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
        trainer_name = type(self).__name__

        # If your KL trainer also has nctx (most prompt learners do), we try to read it safely.
        nctx = None
        try:
            nctx = int(getattr(cfg.TRAINER.PROMPTFL_KL_Global, "N_CTX"))
        except Exception:
            # fallback: if not present, store as "na"
            nctx = "na"

        # Base dir: keep consistent with your old structure
        base = "./embeddings/prompt_text_embeddings"

        run_tag = self._run_tag()
        run_dir_base = osp.join(
            base,
            dataset,
            f"nctx_{nctx}",
            f"trainer_{trainer_name}",
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
        return osp.join(run_dir, f"round_{int(round_id):04d}")

    def _maybe_dump_manifest(self):
        run_dir = self._get_prompt_embed_run_dir()
        manifest_path = osp.join(run_dir, "manifest.json")
        if osp.isfile(manifest_path):
            return

        cfg = self.cfg

        # Try best-effort reading backbone info
        backbone = "unknown"
        try:
            backbone = cfg.MODEL.BACKBONE.NAME
        except Exception:
            pass

        # nctx best-effort
        try:
            nctx = int(getattr(cfg.TRAINER.PROMPTFL_KL_Global, "N_CTX"))
        except Exception:
            nctx = None

        payload = {
            "dataset": cfg.DATASET.NAME,
            "trainer": type(self).__name__,
            "backbone": backbone,
            "nctx": nctx,

            # KL-global specific hyperparams
            "kd_T": float(getattr(cfg.TRAINER.PROMPTFL_KL_Global, "KD_T", 1.0)),
            "lam_kl": float(getattr(cfg.TRAINER.PROMPTFL_KL_Global, "LAM_KL", 1.0)),
            "prec": str(getattr(cfg.TRAINER.PROMPTFL_KL_Global, "PREC", "unknown")),

            "seed": int(getattr(cfg, "SEED", -1)),
            "run_tag": self._run_tag(),
            "rep_id": int(getattr(self, "_prompt_rep_id", -1)),
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"🧾 Saved manifest: {manifest_path}")

    # =========================
    # Prompt text embedding dumps (keep interface)
    # =========================
    @torch.no_grad()
    def _dump_client_text_embedding(self, client_id: int, round_id: int):
        """
        KEEP INTERFACE (client_id, round_id).
        Save: client_{id}.pt with normalized text_features [C,D].
        """
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

        out_path = osp.join(save_dir, f"client_{int(client_id)}.pt")
        torch.save(payload, out_path)
        print(f"💾 Saved client text embedding: {out_path}")

    @torch.no_grad()
    def save_global_text_embedding(self, round_id: int):
        """
        KEEP INTERFACE (round_id).
        Save: global.pt with normalized text_features [C,D],
        and ALSO save global ctx (calls save_global_ctx).
        """
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

        # keep your previous behavior
        self.save_global_ctx(round_id)

    # =========================
    # Learnable prompt dumps (ctx only) (keep interface)
    # =========================
    @torch.no_grad()
    def _dump_client_ctx(self, client_id: int, round_id: int):
        """
        KEEP INTERFACE (client_id, round_id).
        Save: client_{id}_ctx.pt with ctx tensor.
        """
        self.set_model_mode("eval")
        self._maybe_dump_manifest()

        save_dir = self._get_prompt_embed_save_dir(round_id)
        os.makedirs(save_dir, exist_ok=True)

        pl = getattr(self.model, "prompt_learner", None)
        if pl is None or not hasattr(pl, "ctx"):
            print("[WARN] model.prompt_learner.ctx not found; skip saving ctx.")
            return

        ctx = pl.ctx.detach().float().cpu()
        payload = {
            "dataset": self.cfg.DATASET.NAME,
            "round": int(round_id),
            "client_id": int(client_id),
            "ctx": ctx,
        }

        out_path = osp.join(save_dir, f"client_{int(client_id)}_ctx.pt")
        torch.save(payload, out_path)
        print(f"💾 Saved client ctx: {out_path} | ctx_shape={tuple(ctx.shape)}")

    @torch.no_grad()
    def save_global_ctx(self, round_id: int):
        """
        KEEP INTERFACE (round_id).
        Save: global_ctx.pt with ctx tensor.
        """
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

    # =========================
    # Optional: if your framework calls after_train for local client end
    # Keep same behavior as your old trainer
    # =========================
    def after_train(self, idx=-1, epoch=0, is_fed=False):
        super().after_train(idx=idx, epoch=epoch, is_fed=is_fed)
        if idx >= 0 and epoch >= 0:
            # keep your old external call compatibility
            self._dump_client_text_embedding(client_id=idx, round_id=epoch)
            self._dump_client_ctx(client_id=idx, round_id=epoch)