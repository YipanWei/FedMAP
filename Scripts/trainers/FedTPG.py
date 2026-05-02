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

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat


def exists(val):
    return val is not None


def load_default_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'FedTPG',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x_q, x_kv=None, **kwargs):
        x_q = self.norm(x_q)

        if exists(x_kv):
            x_kv = self.norm_context(x_kv)
        else:
            x_kv = x_q

        return self.fn(x_q, x_kv, x_kv, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(
            self,
            latent_dim,
            kv_dim,
            cross_heads=4,
            seq_dropout_prob=0.
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim,
                    nn.MultiheadAttention(latent_dim, num_heads=cross_heads, kdim=kv_dim, vdim=kv_dim,
                                          dropout=seq_dropout_prob, batch_first=True),
                    context_dim=kv_dim),
            FeedForward(latent_dim)])

    def forward(
            self,
            data,
            soft_prompt,
            mask=None,
    ):
        b, *_, device = *data.shape, data.device
        x = repeat(soft_prompt, 'n d -> b n d', b=b)
        cross_attn, cross_ff = self.cross_attend_blocks
        x, _ = cross_attn(x, data, key_padding_mask=mask)
        x = cross_ff(x) + x

        return x


class SelfAttention(nn.Module):
    def __init__(
            self,
            depth,
            latent_dim,
            latent_heads=4,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(latent_dim, nn.MultiheadAttention(latent_dim, num_heads=latent_heads, batch_first=True)),
                FeedForward(latent_dim)
            ]))

    def forward(
            self,
            x,
            mask=None
    ):
        # layers

        for self_attn, self_ff in self.layers:
            x = self_attn(x, key_padding_mask=mask)[0] + x
            x = self_ff(x) + x
        return x


class PromptTranslator(nn.Module):
    def __init__(
            self,
            prompt_len,
            prompt_depth,
            prompt_dim=512,
            depth=2,
            self_heads=2,
            cross_heads=2,
            textemb_dim=512,
            device='cuda'
    ):
        super().__init__()
        self.device = device
        self.prompt_len = prompt_len
        self.prompt_depth = prompt_depth
        prompt_dim = prompt_dim
        soft_prompt = torch.empty(prompt_len * prompt_depth, prompt_dim)
        nn.init.normal_(soft_prompt, std=0.02)
        self.soft_prompt = nn.Parameter(soft_prompt)

        self.encoder = CrossAttention(
            latent_dim=prompt_dim,
            kv_dim=textemb_dim,
            cross_heads=cross_heads,
        )
        if depth > 0:
            self.transformer = SelfAttention(depth=depth, latent_dim=prompt_dim, latent_heads=self_heads)

        # self.vis_linear = nn.Linear(512,768)
        self.depth = depth

    def forward(
            self,
            text_emb,
    ):
        prompt = self.encoder(text_emb, self.soft_prompt)
        if self.depth > 0:
            prompt = self.transformer(prompt)
        prompt = prompt.reshape(self.prompt_depth, self.prompt_len, -1)
        # vis_prompt = self.vis_linear(prompt)

        return prompt, prompt


# class PromptLearner(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#
#         n_ctx, ctx_depth = cfg.MODEL.N_CTX, cfg.MODEL.D_CTX
#         self.meta_net = PromptTranslator(n_ctx, ctx_depth, depth=cfg.MODEL.DEPTH)
#         self.meta_net.half()
#
#         self.ctx_depth = ctx_depth
#         self.n_ctx = n_ctx
#
#     def forward(self, context_emb):
#         text_ctx, vis_ctx = self.meta_net(context_emb.unsqueeze(0))  # (n_ctx, ctx_dim) # self.ctx
#
#         return text_ctx, vis_ctx
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
    design_details = {"trainer": 'FedTPG',
                      "vision_depth": 0,
                      "language_depth": cfg.TRAINER.FedTPG.PROMPT_DEPTH_TEXT, "vision_ctx": 0,
                      "language_ctx": cfg.TRAINER.FedTPG.N_CTX_TEXT}


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

    def forward(self, prompts, tokenized_prompts, text_ctx):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, text_ctx, True)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()

        temp = "a photo of a"
        prompts = [temp + ' ' + c for c in classnames]
        print(f"Prompts: {prompts}")
        self.prompts = torch.cat([clip.tokenize(p) for p in prompts])
        # clip_model.float()

        # self.text_features = text_features
        self.clip_model = clip_model

    def visual_feature(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        return image_features


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, zs_clip_model):
        super().__init__()
        self.classnames = classnames
        # self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(zs_clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.clip_model = clip_model

        self.token_embedding = clip_model.token_embedding

        self.zs_model = CLIP(classnames, zs_clip_model)

        ctx_depth = cfg.TRAINER.FedTPG.D_CTX
        n_ctx = cfg.TRAINER.FedTPG.N_CTX_TEXT
        self.meta_net = PromptTranslator(n_ctx, ctx_depth, depth=cfg.TRAINER.FedTPG.PROMPT_DEPTH_TEXT)
        self.meta_net.half()

        self.prompt_prefix = " ".join(["X"] * n_ctx)

    def get_tokenized_classnames(self,device):

        prompts = [self.prompt_prefix + " " + name + "." for name in self.classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts.to(device)).type(self.dtype)
        # token_prefix = embedding[:, :1, :]  # SOS
        # token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS
        return embedding, tokenized_prompts

    def forward(self, image):
        # tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        tokenized_prompts = self.zs_model.prompts.to(logit_scale.device)
        with torch.no_grad():
            text_features_ = self.clip_model.encode_text(tokenized_prompts)
            text_features_ = text_features_ / text_features_.norm(dim=-1, keepdim=True)

        text_features, vis_ctx = self.encode_text(text_features_)

        image_features,_ = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ text_features.t()

        return logits

    def encode_text(self, text_features_):

        context_emb = text_features_
        prompt_vectors, tokenized_prompts = self.get_tokenized_classnames(text_features_.device)

        text_ctx, vis_ctx = self.meta_net(context_emb.unsqueeze(0))

        text_ctx = text_ctx.half()
        prompt_vectors = torch.cat(
            [
                prompt_vectors[:, :1],  # (dim0, 1, dim)
                text_ctx[0].unsqueeze(0).expand(prompt_vectors.shape[0], -1, -1),  # (dim0, n_ctx, dim)
                prompt_vectors[:, 1 + text_ctx.shape[1]:],  # (dim0, *, dim)
            ],
            dim=1,
        )
        prompt_vectors = prompt_vectors.half()
        if len(text_ctx) > 1:
            text_ctx = text_ctx[1:]
        else:
            text_ctx = []
        text_features = self.text_encoder(prompt_vectors, tokenized_prompts, text_ctx)
        return text_features, vis_ctx


# @TRAINER_REGISTRY.register("FedTPG")
class FedTPG(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.FedTPG.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        zs_clip_model = load_default_clip_to_cpu(cfg)
        if cfg.TRAINER.FedTPG.PREC == "fp32" or cfg.TRAINER.FedTPG.PREC == "amp":
            clip_model.float()
            zs_clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, zs_clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "meta_net" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("FedTPG", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.FedTPG.PREC == "amp" else None

    def set_device(self, gpu_id):
        """切换 Trainer 使用的 GPU，并同步优化器与 scheduler"""
        import torch
        assert torch.cuda.is_available(), "CUDA not available!"
        device = torch.device(f"cuda:{gpu_id}")
        self.device = device

        # 模型迁移
        self.model.to(device)

        # 重新构建优化器和调度器
        self.optim = build_optimizer(self.model, self.cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)

        # ⚡ 更新 Dassl 内部注册结构
        if hasattr(self, "_models") and "FedTPG" in self._models:
            self._models["FedTPG"] = self.model
            self._optims["FedTPG"] = self.optim
            self._scheds["FedTPG"] = self.sched
        else:
            self.register_model("FedTPG", self.model, self.optim, self.sched)

        # AMP scaler 同步
        if self.cfg.TRAINER.FedTPG.PREC == "amp":
            self.scaler = torch.cuda.amp.GradScaler()

        print(f"✅ Switched {self.__class__.__name__} to GPU {gpu_id} ({device}), optimizer refreshed.")

    def forward_backward(self, idx,batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.FedTPG.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
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
