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
                      "vision_depth": cfg.TRAINER.VPT.PROMPT_DEPTH_VISION,
                      "language_depth": 0, "vision_ctx": cfg.TRAINER.VPT.N_CTX_VISION,
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
        ctx_init = cfg.TRAINER.VPT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            prompt_prefix = ctx_init

        print(f'Initial context: "{prompt_prefix}"')

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.n_cls = n_cls
        self.embedding = nn.Parameter(embedding).requires_grad_(False)

        self.tokenized_prompts = tokenized_prompts  
        self.name_lens = name_lens
    def forward(self):
        prompts = self.embedding
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



class VPT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.VPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.VPT.PREC == "fp32" or cfg.TRAINER.VPT.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        
        
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "VPT" in name:
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

        self.scaler = GradScaler() if cfg.TRAINER.VPT.PREC == "amp" else None

     

    def forward_backward(self, idx,batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.VPT.PREC
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

    @torch.no_grad()
    def save_test_image_embeddings_after_test(
            self,
            round_idx: int,
            base_root: str = "./embeddings_test_img",
            split: str = "test",
    ):
        raw_epoch = int(round_idx)
        epoch = raw_epoch + 1

        should_save = (epoch == 1) or (epoch % 10 == 0)
        print(f"[SAVE-DBG] raw_epoch={raw_epoch}, epoch(+1)={epoch}, should_save={should_save}", flush=True)

        if not should_save:
            return

        device = self.device
        model = self.model
        visual = model.image_encoder
        visual_was_training = visual.training
        visual.eval()

        classnames = [c.replace("_", " ") for c in self.dm.dataset.classnames]

        dataset_name = getattr(self.cfg.DATASET, "NAME", "unknown_dataset")
        trainer_name = getattr(self.cfg.TRAINER, "NAME", "VPT")
        depth = getattr(self.cfg.TRAINER.VPT, "PROMPT_DEPTH_VISION", "d?")
        nctx = getattr(self.cfg.TRAINER.VPT, "N_CTX_VISION", "ctx?")

        save_root = osp.join(
            base_root,
            dataset_name,
            trainer_name,
            f"depth_{depth}_ctx_{nctx}",
            f"round_{epoch:03d}",
            split
        )
        os.makedirs(save_root, exist_ok=True)

        loaders_dict = getattr(self, "fed_test_loader_x_dict", None)
        if loaders_dict is None:
            loaders_dict = {"global": getattr(self, "test_loader", None)}
            if loaders_dict["global"] is None:
                print("❌ No fed_test_loader_x_dict and no test_loader found. Skip saving embeddings.", flush=True)
                return

        print(f"🧪 Saving {split} image embeddings (epoch={epoch}, raw_epoch={raw_epoch}) → {save_root}", flush=True)

        for client_id, dl in tqdm(loaders_dict.items(), desc=f"🔹 Clients ({split})"):
            if dl is None:
                continue

            all_embeds = []
            all_labels = []
            all_paths = []

            for batch in tqdm(dl, desc=f"  ↳ Client {client_id}", leave=False):
                imgs = batch["img"].to(device)
                labels = batch["label"].cpu()

                feats, _ = visual(imgs.type(model.dtype))
                feats = F.normalize(feats, dim=-1)

                all_embeds.append(feats.cpu())
                all_labels.append(labels)

                if "impath" in batch:
                    all_paths.extend(batch["impath"])

            payload = {
                "embeddings": torch.cat(all_embeds, dim=0),
                "labels": torch.cat(all_labels, dim=0),
                "classnames": classnames,
                "client": client_id,
                "split": split,
                "round_raw": raw_epoch,
                "round": epoch,
            }
            if len(all_paths) > 0:
                payload["impath"] = all_paths

            save_path = osp.join(save_root, f"client_{client_id}_{split}_image_embeddings.pt")
            torch.save(payload, save_path)
            print(f"✅ Client {client_id}: saved {payload['embeddings'].shape[0]} embeddings → {save_path}", flush=True)

        if visual_was_training:
            visual.train()

        print("🎉 Finished saving test image embeddings.", flush=True)

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

    def _get_trainer_depth_nctx(self):
        cfg = self.cfg
        trainer_name = cfg.TRAINER.NAME  # e.g. "VPT_Ma"

        # 1) 优先尝试 cfg.TRAINER.<trainer_name>
        trainer_cfg = getattr(cfg.TRAINER, trainer_name, None)

        # 3) 读取 depth / nctx（严格要求存在）
        if not hasattr(trainer_cfg, "PROMPT_DEPTH_VISION"):
            raise AttributeError("Missing PROMPT_DEPTH_VISION in trainer config")
        if not hasattr(trainer_cfg, "N_CTX_VISION"):
            raise AttributeError("Missing N_CTX_VISION in trainer config")

        depth = trainer_cfg.PROMPT_DEPTH_VISION
        nctx = trainer_cfg.N_CTX_VISION

        return str(trainer_name), str(depth), str(nctx)

    @torch.no_grad()
    def save_test_image_embeddings_by_client_if_needed(
            self,
            epoch: int,
            base_root: str = "./embeddings/tsne",
            save_rounds=(1, 10, 20, 30, 40, 50),
    ):
        """
        在每次 test 结束后调用：
        仅在指定轮次 save_rounds 保存 test image embeddings（按 client/domain 分组）。

        保存目录：
        {base_root}/{TRAINER}/{depth}/{nctx}/{round}/client_{id}_test_image_embeddings.pt
        """
        if int(epoch+1) not in set(int(x) for x in save_rounds):
            return  # 非指定轮次，直接跳过

        # ✅ 从 cfg 动态拿：trainer_name / depth / nctx
        trainer_name, depth, nctx = self._get_trainer_depth_nctx()

        # round 目录
        save_root = osp.join(base_root, str(trainer_name), str(depth), str(nctx), str(int(epoch)))
        os.makedirs(save_root, exist_ok=True)

        device = self.device

        # ✅ 用当前模型的视觉编码器（含 VPT prompt）
        if not (hasattr(self, "model") and hasattr(self.model, "image_encoder")):
            raise AttributeError("Expected self.model.image_encoder (CustomCLIP).")

        visual = self.model.image_encoder
        dtype = getattr(self.model, "dtype", torch.float32)

        visual.eval()
        visual.requires_grad_(False)

        # classnames
        classnames = [c.replace("_", " ") for c in self.dm.dataset.classnames]

        # per-client test loader
        if not hasattr(self, "fed_test_loader_x_dict"):
            raise AttributeError("self.fed_test_loader_x_dict not found (need per-client test loaders).")

        print(f"🧪 [Round {epoch}] Saving TEST image embeddings → {save_root}")

        for client_id, dl in tqdm(self.fed_test_loader_x_dict.items(), desc="🔹 Clients (test)"):
            samples = []

            for batch in tqdm(dl, desc=f"  ↳ Client {client_id}", leave=False):
                imgs = batch["img"].to(device)
                labels = batch["label"]

                feats = visual(imgs.type(dtype))
                if isinstance(feats, (tuple, list)):  # 兼容 (feats, aux)
                    feats = feats[0]
                feats = F.normalize(feats, dim=-1)

                impaths = batch.get("impath", None)

                for i in range(imgs.size(0)):
                    cls_idx = int(labels[i].item())
                    cls_name = classnames[cls_idx]

                    item = {
                        "embedding": feats[i].detach().float().cpu(),  # float32 更通用
                        "class": cls_name,
                        "label": cls_idx,
                        "client": int(client_id) if str(client_id).isdigit() else client_id,
                        "split": "test",
                        "round": int(epoch),
                        "trainer": str(trainer_name),
                        "depth": str(depth),
                        "nctx": str(nctx),
                    }
                    if impaths is not None:
                        item["impath"] = impaths[i]

                    samples.append(item)

            save_path = osp.join(save_root, f"client_{client_id}_test_image_embeddings.pt")
            torch.save(samples, save_path)
            print(f"✅ Round {epoch} | Client {client_id}: saved {len(samples)} → {save_path}")

        print(f"🎉 [Round {epoch}] Finished saving test embeddings.")
