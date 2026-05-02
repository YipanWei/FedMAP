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
from Dassl.dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)

_tokenizer = _Tokenizer()

# ... (load_clip_to_cpu 和 TextEncoder 保持不变，此处省略以节省篇幅) ...
# 请保留你原代码中的 load_clip_to_cpu 和 TextEncoder

def load_clip_to_cpu(cfg):
    # (保持你原代码不变)
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'VPT',
                      "vision_depth": cfg.TRAINER.VPT_M.PROMPT_DEPTH_VISION,
                      "language_depth": 0, "vision_ctx": cfg.TRAINER.VPT_M.N_CTX_VISION,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model

class TextEncoder(nn.Module):
    # (保持你原代码不变)
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
        ctx_init = cfg.TRAINER.VPT_M.CTX_INIT
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

# ============================================================
# 核心修改区域：CustomCLIP 实现机制 A 和 机制 B
# ============================================================
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # ... (基础组件保持不变) ...
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.num_classes = len(classnames)
        self.feat_dim = clip_model.visual.output_dim

        # [Buffer] 视觉原型库
        self.register_buffer("visual_prototypes", torch.zeros(self.num_classes, self.feat_dim))

        # [Buffer] 记录哪些类别已经被更新过 (用于 Mask)
        self.register_buffer("active_classes", torch.zeros(self.num_classes, dtype=torch.bool))

        # [Buffer] 理想语义拓扑矩阵 M_text (C x C)
        self.register_buffer("text_gram_matrix", torch.zeros(self.num_classes, self.num_classes))

        # [Buffer] 文本锚点 (用于计算 Batch-to-Global 的关系)
        self.register_buffer("fixed_text_anchors", torch.zeros(self.num_classes, self.feat_dim))

        self.proto_momentum = 0.9

    def init_text_topology(self, class_embeddings):
        with torch.no_grad():
            # 1. 计算平均语义锚点
            avg_anchors = class_embeddings.mean(dim=1)
            avg_anchors = avg_anchors / avg_anchors.norm(dim=-1, keepdim=True)

            # 2. 存下来备用 (Fix Mechanism A)
            self.fixed_text_anchors.copy_(avg_anchors)

            # 3. 计算理想 Gram 矩阵
            M_text = torch.matmul(avg_anchors, avg_anchors.t())
            self.text_gram_matrix.copy_(M_text)

            # [优化] 4. 用文本锚点初始化视觉原型 (Warm Start)
            # 这样一开始 M_vis 就等于 M_text，Loss 从 0 开始慢慢随着特征变化
            self.visual_prototypes.copy_(avg_anchors)
            # 初始化时，假设所有类都处于"潜在"激活状态，或者保持 false 等遇到数据再开
            # 这里保守策略：保持 false，只计算见过的

            return avg_anchors

    def update_visual_prototypes(self, features, labels):
        """
        仅更新 Buffer，不涉及梯度
        """
        with torch.no_grad():
            features = features / features.norm(dim=-1, keepdim=True)
            unique_labels = torch.unique(labels)

            for c in unique_labels:
                c_mask = (labels == c)
                c_mean = features[c_mask].mean(dim=0)
                c_mean = c_mean / c_mean.norm(dim=-1, keepdim=True)

                old_proto = self.visual_prototypes[c]

                # 如果是第一次见到这个类 (或者初始化后的第一次更新)
                if not self.active_classes[c]:
                    new_proto = c_mean
                    self.active_classes[c] = True # 标记为活跃
                else:
                    new_proto = self.proto_momentum * old_proto + (1 - self.proto_momentum) * c_mean

                self.visual_prototypes[c] = new_proto / new_proto.norm(dim=-1, keepdim=True)

    def forward(self, image,  label=None):
        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        # -----------------------------------------------------
        # Part 1: 获取视觉特征 (带梯度)
        # -----------------------------------------------------
        image_features, _ = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # -----------------------------------------------------
        # Part 2: 机制 A - 绝对对齐
        # -----------------------------------------------------
        # 直接使用预存的 fixed_text_anchors，省去重复计算
        logits = logit_scale * image_features @ text_features.t()

        # -----------------------------------------------------
        # Part 3: 机制 B - 结构蒸馏 (修复版)
        # -----------------------------------------------------
        loss_struct = torch.tensor(0.0).to(image.device)

        if self.training and label is not None:
            # 1. 更新原型 (detach防止梯度干扰)
            self.update_visual_prototypes(image_features.detach(), label)

            # 2. 构建视觉关系矩阵
            # image_features: [B, D] (FP16)
            # visual_prototypes: [C, D] (FP32 or FP16)

            # [关键修改] 统统转成 float32 再做矩阵乘法
            # 这样 current_vis_relations 就是 float32 类型
            current_vis_relations = image_features.float() @ self.visual_prototypes.float().t()

            # 3. 构建文本关系矩阵 (Target)
            batch_text_anchors = self.fixed_text_anchors[label]
            target_text_relations = batch_text_anchors.float() @ self.fixed_text_anchors.float().t()

            # 4. 计算 MSE Loss (输入都是 float32，安全！)
            distillation_loss = F.mse_loss(current_vis_relations, target_text_relations, reduction='none')

            # 5. 应用 Mask
            mask = self.active_classes.unsqueeze(0).expand(image_features.shape[0], -1)

            if mask.sum() > 0:
                loss_struct = (distillation_loss * mask).sum() / mask.sum()
            else:
                loss_struct = torch.tensor(0.0).to(image.device)

        # 这里的 loss_struct 已经是 float32 了
        return logits,image_features,self.fixed_text_anchors,loss_struct


class VPT_M(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.VPT_M.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.VPT_M.PREC == "fp32" or cfg.TRAINER.VPT_M.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # ... (文件前缀映射代码保持不变) ...
        FILE_PREFIX_MAP = {
            "fedisic": "FedISIC",
            "fedcamelyon17md": "FedCamelyon17MD",
            "covidflmd": "COVIDFLMD",
            "whu": "WHU",
            "pacs": "PACS"
        }
        dataset_name = self.cfg.DATASET.NAME.lower()
        if dataset_name in FILE_PREFIX_MAP:
            prefix = FILE_PREFIX_MAP[dataset_name]
        else:
            # 为了兼容性，这里可以写个 default 或者 raise error
            raise ValueError(f"Dataset name '{dataset_name}' is not in the FILE_PREFIX_MAP")

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
            self.model.init_text_topology(self.class_embeddings)

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

        self.scaler = GradScaler() if cfg.TRAINER.VPT_M.PREC == "amp" else None

        # 定义结构损失的权重 (超参数)
        self.lambda_struct = 10.0 # 可以根据实验调整，通常设为 1.0 到 10.0

    def forward_backward(self, idx, batch_idx, batch, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.VPT_M.PREC

        # 获取模型的数据类型 (通常是 float16)
        model_dtype = self.model.dtype

        if prec == "amp":
            with autocast():
                output, loss_struct = self.model(image, label=label)
                loss_cls = F.cross_entropy(output, label)
                loss = loss_cls + self.lambda_struct * loss_struct

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        else:
            # [非 AMP 模式]
            output,_,_, loss_struct = self.model(image, label=label)

            # F.cross_entropy 可能会返回 float32，即使输入是 float16
            loss_cls = F.cross_entropy(output, label)

            # 计算总 Loss
            loss = loss_cls + self.lambda_struct * loss_struct

            # [关键修复] 如果模型是 float16，必须把 Loss 也强转回 float16
            # 否则 backward 会报错 "Found dtype Float but expected Half"
            if loss.dtype != model_dtype:
                loss = loss.type(model_dtype)

            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_cls": loss_cls.item(),
            "loss_struct": loss_struct.item(),
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