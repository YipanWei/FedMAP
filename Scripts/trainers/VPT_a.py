import os.path as osp
import os
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

    design_details = {"trainer": 'VPT_a',
                      "vision_depth": cfg.TRAINER.VPT_a.PROMPT_DEPTH_VISION,
                      "language_depth": 0, "vision_ctx": cfg.TRAINER.VPT_a.N_CTX_VISION,
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


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # No need for PromptLearner anymore
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.class_embeddings = None

    def forward(self, image):
        logit_scale = self.logit_scale.exp()

        # Instead of using dynamic prompts, use precomputed class embeddings directly
        avg_class_embeddings = self.class_embeddings.mean(dim=1)  # Averaging the embeddings for each class
        image_features, _ = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        avg_class_embeddings = avg_class_embeddings / avg_class_embeddings.norm(dim=-1, keepdim=True)

        logits = logit_scale * image_features @ avg_class_embeddings.t()

        return logits,image_features,avg_class_embeddings


class VPT_a(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.VPT_a.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.VPT_a.PREC == "fp32" or cfg.TRAINER.VPT_a.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        FILE_PREFIX_MAP = {
            "fedisic": "FedISIC",
            "fedcamelyon17md": "FedCamelyon17MD",
            "covidflmd": "COVIDFLMD",  # 或 "CovidFLMD" 取决于你文件名
            "whu": "WHU",
            "pacs": "PACS"
        }
        print("VPT_A")

        dataset_name = self.cfg.DATASET.NAME.lower()
        if dataset_name in FILE_PREFIX_MAP:
            prefix = FILE_PREFIX_MAP[dataset_name]
        else:
            raise ValueError(f"Dataset name '{dataset_name}' is not in the FILE_PREFIX_MAP")

        # 构建属性embedding路径
        attribute_embeddings_path = f"./embeddings/{dataset_name}/text/{prefix}_template_attributes.pt"

        # 1. 加载保存的文件
        if osp.exists(attribute_embeddings_path):
            print(f"Loading embeddings from: {attribute_embeddings_path}")
            attribute_embeddings_data = torch.load(attribute_embeddings_path,weights_only=True)

            # 这里获取到的是一个字典: {'class_A': Tensor, 'class_B': Tensor ...}
            raw_embedding_dict = attribute_embeddings_data["embeddings"]

            # 2. 获取数据集定义的类别顺序 (非常重要！必须和 Label 0,1,2... 对应)
            # self.dm.dataset.classnames 是一个列表 ['Melanoma', 'Nevus', ...]
            target_classnames = self.dm.dataset.classnames

            embedding_list = []

            # 3. 按照正确的顺序，从字典里把 Tensor 拿出来
            print(f"Aligning embeddings for {len(target_classnames)} classes...")
            for cls_name in target_classnames:
                if cls_name in raw_embedding_dict:
                    # 获取该类的 Tensor [Num_Templates, Dim]
                    cls_tensor = raw_embedding_dict[cls_name]
                    embedding_list.append(cls_tensor)
                else:
                    # 如果字典里的 Key 和数据集的类名不匹配（比如大小写问题），这里会报错
                    raise ValueError(f"Error: Class '{cls_name}' defined in dataset not found in the loaded embedding dict keys: {list(raw_embedding_dict.keys())}")

            # 4. 堆叠成一个大 Tensor [Num_Classes, Num_Templates, Dim]
            # 这样就变成了 CustomCLIP 想要的格式，拥有 .mean() 方法了
            self.class_embeddings = torch.stack(embedding_list).to(self.device)

            print(f"✅ Class embeddings loaded and aligned. Shape: {self.class_embeddings.shape}")

            # [新增] 初始化理想语义拓扑矩阵 (机制 B)
            # 确保你的 CustomCLIP 类里有 init_text_topology 方法
            # self.model.init_text_topology(self.class_embeddings)

        else:
            raise FileNotFoundError(f"Attribute embeddings not found at {attribute_embeddings_path}")

        self.model.class_embeddings =  self.class_embeddings

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

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.VPT_a.PREC == "amp" else None

    def forward_backward(self, idx, batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.VPT_a.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)  # Pass the pre-loaded attribute embeddings
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)  # Pass the pre-loaded attribute embeddings
            loss = F.cross_entropy(output[0], label)
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
