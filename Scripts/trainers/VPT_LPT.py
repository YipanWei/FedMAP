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

from tqdm import tqdm

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
    design_details = {"trainer": 'VPT',
                      "vision_depth": cfg.TRAINER.VPT_LPT.PROMPT_DEPTH_VISION,
                      "language_depth": 0, "vision_ctx": cfg.TRAINER.VPT_LPT.N_CTX_VISION,
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
        x = x.permute(1, 0, 2) 
        x = self.transformer(x)
        x = x.permute(1, 0, 2) 
        x = self.ln_final(x).type(self.dtype)

       
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.VPT_LPT.N_CTX_TEXT
        ctx_init = cfg.TRAINER.VPT_LPT.CTX_INIT
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
            if cfg.TRAINER.VPT_LPT.CSC:
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
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features,_ = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        return logits



class VPT_LPT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.VPT_LPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.VPT_LPT.PREC == "fp32" or cfg.TRAINER.VPT_LPT.PREC == "amp":
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

        # ================== Parameter & Communication Cost ==================
        trainable_params = 0
        all_params = 0

        for name, param in self.model.named_parameters():
            n = param.numel()
            all_params += n
            if param.requires_grad:
                trainable_params += n
                # print(f"Trainable layer: {name} | Params: {n:,}")

        trainable_ratio = trainable_params / max(all_params, 1)

        # ---- Communication cost per round (per client) ----
        # Assumption: each round communicates ALL trainable params (server->client + client->server)
        # FP32: 4 bytes/param, FP16: 2 bytes/param
        bytes_fp32 = 4
        bytes_fp16 = 2

        comm_bytes_per_round_fp32 = 2 * trainable_params * bytes_fp32
        comm_bytes_per_round_fp16 = 2 * trainable_params * bytes_fp16

        comm_mb_per_round_fp32 = comm_bytes_per_round_fp32 / (1024 ** 2)
        comm_mb_per_round_fp16 = comm_bytes_per_round_fp16 / (1024 ** 2)

        # Optional: total communication across clients in one round (if you want)
        # NOTE: set num_clients_per_round properly (e.g., all clients or sampled clients)
        num_clients_per_round = getattr(self.cfg, "NUM_CLIENTS_PER_ROUND", None)  # or set manually
        if num_clients_per_round is not None:
            total_comm_mb_fp32 = comm_mb_per_round_fp32 * num_clients_per_round
            total_comm_mb_fp16 = comm_mb_per_round_fp16 * num_clients_per_round

        print("\n" + "="*28 + " Params & Communication " + "="*28)
        print(f"Total Parameters:      {all_params:,}")
        print(f"Trainable Parameters:  {trainable_params:,}")
        print(f"Trainable Ratio:       {trainable_ratio:.4%}")
        print(f"Trainable Params (M):  {trainable_params / 1e6:.4f} M")

        print("\n[Per-client Communication per Round] (upload + download)")
        print(f"FP32 (4B):  {comm_mb_per_round_fp32:.4f} MB/round/client")
        print(f"FP16 (2B):  {comm_mb_per_round_fp16:.4f} MB/round/client")

        if num_clients_per_round is not None:
            print("\n[Total Communication per Round] (all participating clients)")
            print(f"Clients per round: {num_clients_per_round}")
            print(f"FP32 (4B):  {total_comm_mb_fp32:.4f} MB/round")
            print(f"FP16 (2B):  {total_comm_mb_fp16:.4f} MB/round")

        print("="*75 + "\n")
        # ================================================================

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.VPT_LPT.PREC == "amp" else None

     

    def forward_backward(self, idx,batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.VPT_LPT.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output= self.model(image)
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


    @torch.no_grad()
    def save_test_image_embeddings_after_test(
            self,
            round_idx: int,                      # 传入的是 0-based epoch/round
            base_root: str = "./embeddings_test_img",
            split: str = "test",
    ):
        """
        用“当前 self.model（全局模型）”提取 test 图像 embedding 并落盘。
        保存策略：epoch 在函数内 +1（变成 1-based），只在 1,10,20,30,40,50... 保存。
        - 默认按 client 保存：self.fed_test_loader_x_dict
        - 如果没有该 dict，则退化为用 self.test_loader 保存一个 global 文件
        """

        # -----------------------------
        # 0) Save schedule: 1,10,20...
        # -----------------------------
        raw_epoch = int(round_idx)      # 0-based
        epoch = raw_epoch + 1           # 1-based（你希望 epoch 加 1）

        should_save = (epoch == 1) or (epoch % 10 == 0)
        print(f"[SAVE-DBG] raw_epoch={raw_epoch}, epoch(+1)={epoch}, should_save={should_save}", flush=True)

        if not should_save:
            return

        # -----------------------------
        # 1) Prepare model / encoder
        # -----------------------------
        device = self.device
        model = self.model

        # CustomCLIP 里定义的是 self.image_encoder
        visual = model.image_encoder
        visual_was_training = visual.training
        visual.eval()

        classnames = [c.replace("_", " ") for c in self.dm.dataset.classnames]

        dataset_name = getattr(self.cfg.DATASET, "NAME", "unknown_dataset")
        trainer_name = getattr(self.cfg.TRAINER, "NAME", "VPT_LPT")
        dataset_dir_name = dataset_name
        beta = getattr(self.cfg.DATASET, "BETA", None)
        if beta is not None and "cifar100" in str(dataset_name).lower():
            dataset_dir_name = f"{dataset_name}_beta_{float(beta):.1f}"

        # 你也可以把 depth / ctx 等写进路径，方便区分实验
        depth = getattr(self.cfg.TRAINER.VPT_LPT, "PROMPT_DEPTH_VISION", "d?")
        nctx = getattr(self.cfg.TRAINER.VPT_LPT, "N_CTX_VISION", "ctx?")

        # -----------------------------
        # 2) Build save path
        # -----------------------------
        save_root = osp.join(
            base_root,
            dataset_dir_name,
            trainer_name,
            f"depth_{depth}_ctx_{nctx}",
            f"round_{epoch:03d}",   # ✅ 用 1-based epoch 命名：001/010/020/...
            split
        )
        os.makedirs(save_root, exist_ok=True)

        # -----------------------------
        # 3) Pick loaders
        # -----------------------------
        loaders_dict = getattr(self, "fed_test_loader_x_dict", None)
        if loaders_dict is None:
            # 退化：用全局 test_loader
            loaders_dict = {"global": getattr(self, "test_loader", None)}
            if loaders_dict["global"] is None:
                print("❌ No fed_test_loader_x_dict and no test_loader found. Skip saving embeddings.", flush=True)
                return

        print(f"🧪 Saving {split} image embeddings (epoch={epoch}, raw_epoch={raw_epoch}) → {save_root}", flush=True)

        # -----------------------------
        # 4) Extract & save per client
        # -----------------------------
        for client_id, dl in tqdm(loaders_dict.items(), desc=f"🔹 Clients ({split})"):
            if dl is None:
                continue

            all_embeds = []
            all_labels = []
            all_paths = []  # 可选

            for batch in tqdm(dl, desc=f"  ↳ Client {client_id}", leave=False):
                imgs = batch["img"].to(device)
                labels = batch["label"].cpu()

                feats, _ = visual(imgs.type(model.dtype))
                feats = F.normalize(feats, dim=-1)  # [B, D]

                all_embeds.append(feats.cpu())
                all_labels.append(labels)

                if "impath" in batch:
                    all_paths.extend(batch["impath"])

            payload = {
                "embeddings": torch.cat(all_embeds, dim=0),  # [N, D]
                "labels": torch.cat(all_labels, dim=0),      # [N]
                "classnames": classnames,
                "client": client_id,
                "split": split,
                "round_raw": raw_epoch,   # 0-based
                "round": epoch,           # 1-based（你想要的 1/10/20...）
            }
            if len(all_paths) > 0:
                payload["impath"] = all_paths

            save_path = osp.join(save_root, f"client_{client_id}_{split}_image_embeddings.pt")
            torch.save(payload, save_path)
            print(f"✅ Client {client_id}: saved {payload['embeddings'].shape[0]} embeddings → {save_path}", flush=True)

        # -----------------------------
        # 5) Restore state
        # -----------------------------
        if visual_was_training:
            visual.train()

        print("🎉 Finished saving test image embeddings.", flush=True)
