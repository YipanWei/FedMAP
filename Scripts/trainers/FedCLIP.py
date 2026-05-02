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

from Dassl.dassl.utils import (
    count_num_param, load_checkpoint,load_pretrained_weights
)
from tqdm import tqdm

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
    design_details = {"trainer": 'CLIP',
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
        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # use given words to initialize context vectors
        ctx_init = "a photo of a"
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
        # self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.FedCLIP.CLASS_TOKEN_POSITION

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

        image_embedding = self.image_encoder.output_dim

        self.img_adap = nn.Sequential(nn.Linear(image_embedding, image_embedding), nn.Tanh(
        ), nn.Linear(image_embedding, image_embedding), nn.Softmax(dim=1)).type(self.dtype)


    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features,_ = self.image_encoder(image.type(self.dtype))

        image_features_att = self.img_adap(image_features)
        image_features = torch.mul(image_features_att, image_features)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        return logits



# @TRAINER_REGISTRY.register("CLIP")
class FedCLIP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.FedCLIP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.FedCLIP.PREC == "fp32" or cfg.TRAINER.FedCLIP.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")

        for name, param in self.model.named_parameters():
            if "img_adap" not in name:
                param.requires_grad_(False)
        print(f"# params: {count_num_param(self.model):,}")

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.FedCLIP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,3,2,1"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            # self.model = nn.DataParallel(self.model, device_ids=[1])

    def forward_backward(self,idx, batch_idx,batch, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.FedCLIP.PREC
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

        【重要修改】：已修正逻辑，现在会应用训练好的 'img_adap' 模块，
        确保保存的特征是经过 FedCLIP 调整后的特征。
        """

        # -----------------------------
        # 0) Save schedule: 1,10,20...
        # -----------------------------
        raw_epoch = int(round_idx)      # 0-based
        epoch = raw_epoch + 1           # 1-based

        should_save = (epoch == 1) or (epoch % 10 == 0)
        # print(f"[SAVE-DBG] raw_epoch={raw_epoch}, epoch(+1)={epoch}, should_save={should_save}", flush=True)

        if not should_save:
            return

        # -----------------------------
        # 1) Prepare model / encoder
        # -----------------------------
        device = self.device
        model = self.model

        # 获取视觉编码器 (冻结的 backbone)
        visual = model.image_encoder
        visual_was_training = visual.training
        visual.eval()

        # 确保 adapter 也是 eval 模式 (虽然它通常只包含 Linear/Tanh，但也可能有 Dropout 等)
        model.img_adap.eval()

        classnames = [c.replace("_", " ") for c in self.dm.dataset.classnames]

        dataset_name = getattr(self.cfg.DATASET, "NAME", "unknown_dataset")
        trainer_name = getattr(self.cfg.TRAINER, "NAME", "FedCLIP")
        depth = getattr(self.cfg.TRAINER.FedCLIP, "PROMPT_DEPTH_VISION", "12")
        nctx = getattr(self.cfg.TRAINER.FedCLIP, "N_CTX_VISION", "1")
        dataset_dir_name = dataset_name
        # -----------------------------
        # 2) Build save path
        # -----------------------------
        save_root = osp.join(
            base_root,
            dataset_dir_name,
            trainer_name,
            f"depth_{depth}_ctx_{nctx}",
            f"round_{epoch:03d}",   # 1-based epoch
            split
        )
        os.makedirs(save_root, exist_ok=True)

        # -----------------------------
        # 3) Pick loaders
        # -----------------------------
        loaders_dict = getattr(self, "fed_test_loader_x_dict", None)
        if loaders_dict is None:
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
            all_paths = []

            for batch in tqdm(dl, desc=f"  ↳ Client {client_id}", leave=False):
                imgs = batch["img"].to(device)
                labels = batch["label"].cpu()

                # ==========================
                # 【关键修改开始】
                # ==========================
                # 1. 获取原始 CLIP 特征 (frozen)
                image_features, _ = visual(imgs.type(model.dtype))

                # 2. 应用训练好的 Adapter (img_adap)
                #    这部分是 FedCLIP 真正训练的东西
                image_features_att = model.img_adap(image_features)
                image_features = torch.mul(image_features_att, image_features)

                # 3. 归一化 (这一步必须做，否则与 text feature 点积时尺度不对)
                feats = F.normalize(image_features, dim=-1)
                # ==========================
                # 【关键修改结束】
                # ==========================

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
                "round_raw": raw_epoch,
                "round": epoch,
            }
            if len(all_paths) > 0:
                payload["impath"] = all_paths

            save_path = osp.join(save_root, f"client_{client_id}_{split}_image_embeddings.pt")
            torch.save(payload, save_path)
            # print(f"✅ Client {client_id}: saved {payload['embeddings'].shape[0]} embeddings", flush=True)

        # -----------------------------
        # 5) Restore state
        # -----------------------------
        if visual_was_training:
            visual.train()

        # 记得把 img_adap 也切回之前的状态 (通常是 train)
        model.img_adap.train()

        print("🎉 Finished saving test image embeddings.", flush=True)
