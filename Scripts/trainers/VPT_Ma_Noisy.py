import os.path as osp
import torch
from torch.cuda.amp import GradScaler

from Dassl.dassl.optim import build_optimizer, build_lr_scheduler
from Dassl.dassl.utils import load_pretrained_weights

from trainers.VPT_Ma import (
    VPT_Ma,
    CustomCLIP,
    load_clip_to_cpu,
    _get_trainer_cfg,
)


class CustomCLIPNoisy(CustomCLIP):
    def __init__(self, cfg, classnames, clip_model, trainer_name="VPT_Ma_Noisy"):
        super().__init__(cfg, classnames, clip_model, trainer_name=trainer_name)
        trainer_cfg = _get_trainer_cfg(cfg, trainer_name)
        self.anchor_noise_std = float(getattr(trainer_cfg, "ANCHOR_NOISE_STD", 0.0))

    def init_text_topology(self, class_embeddings):
        with torch.no_grad():
            avg_anchors = class_embeddings.mean(dim=1)
            avg_anchors = avg_anchors / avg_anchors.norm(dim=-1, keepdim=True)

            if self.anchor_noise_std > 0:
                noise = torch.randn_like(avg_anchors) * self.anchor_noise_std
                avg_anchors = avg_anchors + noise
                avg_anchors = avg_anchors / avg_anchors.norm(dim=-1, keepdim=True)

            self.fixed_text_anchors.copy_(avg_anchors)
            M_text = torch.matmul(avg_anchors, avg_anchors.t())
            self.text_gram_matrix.copy_(M_text)
            self.visual_prototypes.copy_(avg_anchors)

            return avg_anchors


class VPT_Ma_Noisy(VPT_Ma):
    def build_model(self):
        cfg = self.cfg
        trainer_name = self._trainer_name()
        trainer_cfg = self._trainer_cfg()
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg, trainer_name=trainer_name)

        if trainer_cfg.PREC == "fp32" or trainer_cfg.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIPNoisy(cfg, classnames, clip_model, trainer_name=trainer_name)

        FILE_PREFIX_MAP = {
            "fedisic": "FedISIC",
            "fedcamelyon17md": "FedCamelyon17MD",
            "covidflmd": "COVIDFLMD",
            "whu": "WHU",
            "pacs": "PACS",
            "cifar100md": "CIFAR100MD",
            "tinyimagenetmd": "TinyImageNetMD",
        }
        dataset_name = self.cfg.DATASET.NAME.lower()
        if dataset_name in FILE_PREFIX_MAP:
            prefix = FILE_PREFIX_MAP[dataset_name]
        else:
            raise ValueError(f"Dataset name '{dataset_name}' is not in the FILE_PREFIX_MAP")

        attribute_embeddings_path = f"./embeddings/{dataset_name}/text/{prefix}_template_attributes.pt"

        if osp.exists(attribute_embeddings_path):
            print(f"Loading embeddings from: {attribute_embeddings_path}")
            attribute_embeddings_data = torch.load(attribute_embeddings_path, weights_only=True)
            raw_embedding_dict = attribute_embeddings_data["embeddings"]
            target_classnames = self.dm.dataset.classnames
            embedding_list = []

            print(f"Aligning embeddings for {len(target_classnames)} classes...")
            for cls_name in target_classnames:
                if cls_name in raw_embedding_dict:
                    embedding_list.append(raw_embedding_dict[cls_name])
                else:
                    raise ValueError(
                        f"Error: Class '{cls_name}' defined in dataset not found in the loaded embedding dict keys: {list(raw_embedding_dict.keys())}"
                    )

            self.class_embeddings = torch.stack(embedding_list).to(self.device)
            print(f"✅ Class embeddings loaded and aligned. Shape: {self.class_embeddings.shape}")
            print(f"Applying anchor noise with std={self.model.anchor_noise_std}")
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

        print("\n" + "=" * 24 + " Params & Comm (FP16) " + "=" * 24)
        all_params = 0
        trainable_params = 0
        ONLY_COUNT_VPT_TRAINABLE = True

        for name, p in self.model.named_parameters():
            n = p.numel()
            all_params += n
            if p.requires_grad:
                if (not ONLY_COUNT_VPT_TRAINABLE) or ("VPT" in name):
                    trainable_params += n

        print(f"Total Params:            {all_params:,}")
        print(f"Trainable Params:        {trainable_params:,}")
        print(f"Trainable Ratio:         {trainable_params / max(all_params,1):.4%}")
        print(f"Trainable Params (M):    {trainable_params / 1e6:.4f} M")

        extra_keys = [
            "fixed_text_anchors",
            "text_gram_matrix",
            "visual_prototypes",
            "active_classes",
        ]

        sd = self.model.state_dict()
        extra_elems = 0
        extra_bytes_one_way = 0

        print("\n[Extra communicated buffers]")
        for k in extra_keys:
            if k not in sd:
                print(f"- {k}: NOT FOUND in state_dict (skip)")
                continue
            t = sd[k]
            if not torch.is_tensor(t):
                print(f"- {k}: not a tensor (skip)")
                continue

            n = t.numel()
            if t.dtype == torch.bool:
                bpe = 1
            elif t.is_floating_point():
                bpe = 2
            else:
                bpe = t.element_size()

            extra_elems += n
            extra_bytes_one_way += n * bpe
            print(f"- {k:18s} shape={tuple(t.shape)} dtype={t.dtype} numel={n:,} bytes/elem={bpe}")

        print(f"Extra elems:             {extra_elems:,}")

        bytes_per_param_fp16 = 2
        trainable_bytes_one_way = trainable_params * bytes_per_param_fp16

        total_bytes_one_way = trainable_bytes_one_way + extra_bytes_one_way
        bytes_per_round_per_client = 2 * total_bytes_one_way
        mb_per_round_per_client = bytes_per_round_per_client / (1024 ** 2)

        print("\n[Communication Cost]")
        print("Assumption: FP16 transmission; per round includes download + upload.")
        print(f"Comm/round/client:       {mb_per_round_per_client:.4f} MB")

        num_clients_per_round = getattr(self.cfg, "NUM_CLIENTS_PER_ROUND", None)
        num_rounds = getattr(self.cfg, "MAX_EPOCH", None)

        if num_clients_per_round is not None:
            mb_per_round_total = mb_per_round_per_client * num_clients_per_round
            print(f"Comm/round total (N={num_clients_per_round}): {mb_per_round_total:.4f} MB")

            if num_rounds is not None:
                gb_total_training = (mb_per_round_total * num_rounds) / 1024.0
                print(f"Comm total training (R={num_rounds}):        {gb_total_training:.4f} GB")

        print("=" * 70 + "\n")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if trainer_cfg.PREC == "amp" else None

        self.lambda_struct = trainer_cfg.lambda_struct
        self.proto_momentum = float(getattr(trainer_cfg, "PROTO_MOMENTUM", 0.9))
        self.struct_loss_type = str(getattr(trainer_cfg, "STRUCT_LOSS", "mse")).lower()
