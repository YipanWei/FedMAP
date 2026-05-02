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
      
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CLIP_OB',
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
        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"


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

        self.tokenized_prompts = tokenized_prompts 
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.CLIP_OB.CLASS_TOKEN_POSITION

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

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2) 
        x = self.transformer(x)
        x = x.permute(1, 0, 2) 
        x = self.ln_final(x).type(self.dtype)

  
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def forward(self, image):
        image_features,_ = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits,image_features,text_features

class CLIP_OB(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.CLIP_OB.PREC in ["fp16", "fp32", "amp"]

    def load_class_exp(
            self,
            dataset_name: str,
            exp_root="./CLS_Exp/ClassExp",
    ):
        dataset_dir = osp.join(exp_root, dataset_name)
        assert osp.isdir(dataset_dir), f"Dataset not found: {dataset_dir}"

        class_exp_dict = {}
        for fname in os.listdir(dataset_dir):
            if not fname.endswith(".txt"):
                continue
            if fname.startswith("_"):
                continue

            class_name = fname.replace(".txt", "").replace("_", " ")
            fpath = osp.join(dataset_dir, fname)
            with open(fpath, "r") as f:
                text = f.read().strip()

            class_exp_dict[class_name] = text

        if len(class_exp_dict) == 0:
            raise RuntimeError(f"No class expansion found in {dataset_dir}")

        return class_exp_dict

    def load_class_exp_multi(
            self,
            dataset_name: str,
            exp_root="./CLS_Exp/",
    ):
        dataset_dir = osp.join(exp_root, dataset_name)
        assert osp.isdir(dataset_dir), f"Dataset not found: {dataset_dir}"

        class_exp_dict = {}
        for fname in os.listdir(dataset_dir):
            if not fname.endswith(".txt"):
                continue
            if fname.startswith("_"):
                continue

            class_name = fname.replace(".txt", "").replace("_", " ")
            fpath = osp.join(dataset_dir, fname)
            with open(fpath, "r") as f:
                lines = [line.strip() for line in f.readlines() if len(line.strip()) > 0]

            if len(lines) == 0:
                continue

            class_exp_dict[class_name] = lines

        if len(class_exp_dict) == 0:
            raise RuntimeError(f"No class expansion found in {dataset_dir}")

        return class_exp_dict

    def resolve_attr_root(self, dataset_name: str):
        """
        Prefer GPT_4o-curated attribute files if available, otherwise fall back
        to the original ClassAttr/<dataset> directory.
        """
        base_root = "./CLS_Exp/ClassAttr"
        preferred = osp.join(base_root, "GPT_4o", dataset_name)
        fallback = osp.join(base_root, dataset_name)

        if osp.isdir(preferred):
            print(f"Using GPT_4o attribute root: {preferred}")
            return osp.join(base_root, "GPT_4o")

        if osp.isdir(fallback):
            print(f"Using default attribute root: {fallback}")
            return base_root

        raise FileNotFoundError(
            f"No attribute directory found for dataset '{dataset_name}'. "
            f"Checked: {preferred} and {fallback}"
        )

    # ---------------------------------------------------------------------
    # NEW: save class-name semantic text embeddings (name-only / templated)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def extract_classname_semantic_embeddings(
            self,
            save_root: str,
            template: str,
            save_name: str,
    ):
        """
        Extract and save class-level text embeddings using ONLY class names.

        Args:
            save_root: directory to save
            template: e.g. "{}" or "a photo of a {}."
            save_name: file name, e.g. "PACS_class_semantic_nameonly.pt"

        Saved file format:
        {
          "dataset": str,
          "type": "class_semantic_embedding",
          "embedding_dim": int,
          "template": str,
          "embeddings": { class_name (str): Tensor[D] }
        }
        """
        clip_model = self.clip_model
        device = self.device
        dtype = clip_model.dtype

        # move text modules to device
        clip_model.token_embedding = clip_model.token_embedding.to(device)
        clip_model.positional_embedding = clip_model.positional_embedding.to(device)
        clip_model.ln_final = clip_model.ln_final.to(device)
        clip_model.text_projection = clip_model.text_projection.to(device)

        text_encoder = TextEncoder(clip_model).to(device)
        text_encoder.eval()

        classnames = [c.replace("_", " ") for c in self.dm.dataset.classnames]

        embeddings = {}
        for cname in classnames:
            text = template.format(cname)
            tokenized = clip.tokenize(text).to(device)               # [1, 77]
            prompts = clip_model.token_embedding(tokenized).type(dtype)

            feat = text_encoder(prompts, tokenized)                  # [1, D]
            feat = F.normalize(feat, dim=-1).squeeze(0).cpu()        # [D]
            embeddings[cname] = feat

        os.makedirs(save_root, exist_ok=True)
        save_path = osp.join(save_root, save_name)

        torch.save(
            {
                "dataset": self.cfg.DATASET.NAME,
                "type": "class_semantic_embedding",
                "embedding_dim": next(iter(embeddings.values())).shape[0],
                "template": template,
                "embeddings": embeddings,
            },
            save_path,
        )
        print(f"✅ Saved class-name semantic embeddings to {save_path}")

    # ---------------------------------------------------------------------
    # existing: save single text per class (from self.class_exp)
    # ---------------------------------------------------------------------
    def extract_embedding(self, save_root):
        clip_model = self.clip_model
        device = self.device
        dtype = clip_model.dtype

        clip_model.token_embedding = clip_model.token_embedding.to(device)
        clip_model.positional_embedding = clip_model.positional_embedding.to(device)
        clip_model.ln_final = clip_model.ln_final.to(device)
        clip_model.text_projection = clip_model.text_projection.to(device)

        text_encoder = TextEncoder(clip_model).to(device)
        text_encoder.eval()

        class_exp_dict = self.class_exp  # {class_name: semantic_text}
        class_exp_embedding = {}

        with torch.no_grad():
            for class_name, text in class_exp_dict.items():
                tokenized = clip.tokenize(text).to(device)
                prompts = clip_model.token_embedding(tokenized).type(dtype)

                feat = text_encoder(prompts, tokenized)
                feat = F.normalize(feat, dim=-1)  # [1, D]
                class_exp_embedding[class_name] = feat.squeeze(0).cpu()  # [D]

        os.makedirs(save_root, exist_ok=True)
        save_path = osp.join(save_root, f"{self.cfg.DATASET.NAME}_class_semantic_embeddings.pt")

        torch.save(
            {
                "dataset": self.cfg.DATASET.NAME,
                "type": "class_semantic_embedding",
                "embedding_dim": next(iter(class_exp_embedding.values())).shape[0],
                "embeddings": class_exp_embedding,
            },
            save_path,
        )
        print(f"✅ Saved class semantic embeddings to {save_path}")

    # ---------------------------------------------------------------------
    # existing: save multiple texts per class (descriptions / attributes)
    # ---------------------------------------------------------------------
    def extract_embedding_multi(self, save_root, save_name):
        clip_model = self.clip_model
        device = self.device
        dtype = clip_model.dtype

        clip_model.token_embedding = clip_model.token_embedding.to(device)
        clip_model.positional_embedding = clip_model.positional_embedding.to(device)
        clip_model.ln_final = clip_model.ln_final.to(device)
        clip_model.text_projection = clip_model.text_projection.to(device)

        text_encoder = TextEncoder(clip_model).to(device)
        text_encoder.eval()

        class_exp_dict = self.class_exp  # {class_name: List[str]}
        class_exp_embeddings = {}

        with torch.no_grad():
            for class_name, texts in class_exp_dict.items():
                feats_per_class = []
                for text in texts:
                    tokenized = clip.tokenize(text).to(device)
                    prompts = clip_model.token_embedding(tokenized).type(dtype)

                    feat = text_encoder(prompts, tokenized)
                    feat = F.normalize(feat, dim=-1)  # [1, D]
                    feats_per_class.append(feat.squeeze(0).cpu())

                class_exp_embeddings[class_name] = torch.stack(feats_per_class, dim=0)  # [N, D]

        os.makedirs(save_root, exist_ok=True)
        save_path = osp.join(save_root, save_name)

        torch.save(
            {
                "dataset": self.cfg.DATASET.NAME,
                "embeddings": class_exp_embeddings,
            },
            save_path,
        )
        print(f"✅ Saved class text embeddings to {save_path}")

    @torch.no_grad()
    def extract_train_image_embeddings(self, save_root):
        clip_model = self.clip_model
        device = self.device

        visual = clip_model.visual
        visual.eval()
        visual.requires_grad_(False)

        classnames = [c.replace("_", " ") for c in self.dm.dataset.classnames]
        os.makedirs(save_root, exist_ok=True)

        print("🧪 Extracting TRAIN image embeddings...")
        for client_id, dl in tqdm(self.fed_train_loader_x_dict.items(), desc="🔹 Clients (train)"):
            samples = []
            for batch in tqdm(dl, desc=f"  ↳ Client {client_id}", leave=False):
                imgs = batch["img"].to(device)
                labels = batch["label"]

                feats, _ = visual(imgs.type(clip_model.dtype))
                feats = F.normalize(feats, dim=-1)

                for i in range(imgs.size(0)):
                    cls_idx = labels[i].item()
                    cls_name = classnames[cls_idx]
                    samples.append(
                        {"embedding": feats[i].cpu(), "class": cls_name, "client": client_id, "split": "train"}
                    )

            save_path = osp.join(save_root, f"client_{client_id}_train_image_embeddings.pt")
            torch.save(samples, save_path)
            print(f"✅ Client {client_id}: saved {len(samples)} train embeddings → {save_path}")

        print("🎉 Finished extracting train image embeddings.")

    @torch.no_grad()
    def extract_test_image_embeddings(self, save_root):
        clip_model = self.clip_model
        device = self.device

        visual = clip_model.visual
        visual.eval()
        visual.requires_grad_(False)

        classnames = [c.replace("_", " ") for c in self.dm.dataset.classnames]
        os.makedirs(save_root, exist_ok=True)

        print("🧪 Extracting TEST image embeddings...")
        for client_id, dl in tqdm(self.fed_test_loader_x_dict.items(), desc="🔹 Clients (test)"):
            samples = []
            for batch in tqdm(dl, desc=f"  ↳ Client {client_id}", leave=False):
                imgs = batch["img"].to(device)
                labels = batch["label"]

                feats, _ = visual(imgs.type(clip_model.dtype))
                feats = F.normalize(feats, dim=-1)

                for i in range(imgs.size(0)):
                    cls_idx = labels[i].item()
                    cls_name = classnames[cls_idx]
                    samples.append(
                        {"embedding": feats[i].cpu(), "class": cls_name, "client": client_id, "split": "test"}
                    )

            save_path = osp.join(save_root, f"client_{client_id}_test_image_embeddings.pt")
            torch.save(samples, save_path)
            print(f"✅ Client {client_id}: saved {len(samples)} test embeddings → {save_path}")

        print("🎉 Finished extracting test image embeddings.")

    @torch.no_grad()
    def extract_template_attributes_embeddings(
            self,
            save_root: str,
            save_name: str,
    ):
        """
        Extract and save template-based attribute text embeddings for each class.

        In addition to the original "{class} : {attribute}" embeddings, this function also saves:
          1) "a photo of a {cls}."  (per-class, 1 embedding)
          2) "{cls}"               (per-class, 1 embedding)
          3) "{attribute}"         (per-class, K embeddings; attributes grouped by class)

        Args:
            save_root: directory to save the embeddings
            save_name: base file name for "{class} : {attribute}" embeddings (kept for backward compatibility)
        """
        clip_model = self.clip_model
        device = self.device
        dtype = clip_model.dtype

        # move text modules to device
        clip_model.token_embedding = clip_model.token_embedding.to(device)
        clip_model.positional_embedding = clip_model.positional_embedding.to(device)
        clip_model.ln_final = clip_model.ln_final.to(device)
        clip_model.text_projection = clip_model.text_projection.to(device)

        text_encoder = TextEncoder(clip_model).to(device)
        text_encoder.eval()

        # Load class attribute descriptions
        class_exp_dict = self.class_exp  # {class_name: List[attributes]}

        # ----------------------------
        # Helpers
        # ----------------------------
        def _encode_batch(text_list):
            """
            text_list: List[str]
            return: Tensor [N, D] on CPU
            """
            if len(text_list) == 0:
                return None
            tokenized = clip.tokenize(text_list).to(device)                 # [N, 77]
            prompts = clip_model.token_embedding(tokenized).type(dtype)     # [N, 77, dim]
            feats = text_encoder(prompts, tokenized)                        # [N, D]
            feats = F.normalize(feats, dim=-1)
            return feats.cpu()

        os.makedirs(save_root, exist_ok=True)

        # ============================================================
        # A) Original: "{class} : {attribute}"  (per-class K embeddings)
        # ============================================================
        class_attr_embeddings = {}
        for class_name, attributes in class_exp_dict.items():
            if attributes is None or len(attributes) == 0:
                continue
            texts = [f"{class_name} : {attr}" for attr in attributes]
            feats = _encode_batch(texts)  # [K, D]
            class_attr_embeddings[class_name] = feats  # cpu

        save_path = osp.join(save_root, save_name)
        torch.save(
            {
                "dataset": self.cfg.DATASET.NAME,
                "type": "class_attr",
                "template": "{} : {}",
                "embeddings": class_attr_embeddings,  # {class: [K,D]}
            },
            save_path,
        )
        print(f"✅ Saved template-based (class:attribute) text embeddings to {save_path}")

        # ============================================================
        # B) "a photo of a {cls}."  (per-class 1 embedding)
        # ============================================================
        classnames = list(class_exp_dict.keys())
        photo_texts = [f"a photo of a {c}." for c in classnames]
        photo_feats = _encode_batch(photo_texts)  # [C, D]
        cls_photo_embeddings = {c: photo_feats[i] for i, c in enumerate(classnames)} if photo_feats is not None else {}

        base, ext = osp.splitext(save_name)
        save_path_photo = osp.join(save_root, f"{base}_cls_photo{ext or '.pt'}")
        torch.save(
            {
                "dataset": self.cfg.DATASET.NAME,
                "type": "cls_photo",
                "template": "a photo of a {}.",
                "embeddings": cls_photo_embeddings,  # {class: [D]}
            },
            save_path_photo,
        )
        print(f"✅ Saved class-name (photo template) embeddings to {save_path_photo}")

        # ============================================================
        # C) "{cls}"  (per-class 1 embedding)
        # ============================================================
        cls_texts = [f"{c}" for c in classnames]
        cls_feats = _encode_batch(cls_texts)  # [C, D]
        cls_only_embeddings = {c: cls_feats[i] for i, c in enumerate(classnames)} if cls_feats is not None else {}

        save_path_cls = osp.join(save_root, f"{base}_cls_only{ext or '.pt'}")
        torch.save(
            {
                "dataset": self.cfg.DATASET.NAME,
                "type": "cls_only",
                "template": "{}",
                "embeddings": cls_only_embeddings,  # {class: [D]}
            },
            save_path_cls,
        )
        print(f"✅ Saved class-name (cls only) embeddings to {save_path_cls}")

        # ============================================================
        # D) "{attribute}"  (per-class K embeddings; grouped by class)
        # ============================================================
        attr_only_embeddings = {}
        for class_name, attributes in class_exp_dict.items():
            if attributes is None or len(attributes) == 0:
                continue
            feats = _encode_batch(attributes)  # [K, D]
            attr_only_embeddings[class_name] = feats  # cpu

        save_path_attr = osp.join(save_root, f"{base}_attr_only{ext or '.pt'}")
        torch.save(
            {
                "dataset": self.cfg.DATASET.NAME,
                "type": "attr_only",
                "template": "{attribute}",
                "embeddings": attr_only_embeddings,  # {class: [K,D]}
            },
            save_path_attr,
        )
        print(f"✅ Saved attribute-only embeddings to {save_path_attr}")


    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(self.dm.dataset)
        print(f"Loading CLIP_OB (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        self.clip_model = clip_model

        if cfg.TRAINER.CLIP_OB.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP_OB")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        print(f"# params: {count_num_param(self.model):,}")
        print(f"# prompt learner params: {count_num_param(self.model.prompt_learner):,}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.scaler = GradScaler() if cfg.TRAINER.CLIP_OB.PREC == "amp" else None

        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")

        dataset_name = self.cfg.DATASET.NAME.lower()
        dataset_dir_name = dataset_name
        beta = getattr(self.cfg.DATASET, "BETA", None)
        if beta is not None and "cifar100" in dataset_name:
            dataset_dir_name = f"{dataset_name}_beta_{float(beta):.1f}"
        ATTR_ROOT = self.resolve_attr_root(dataset_name)
        text_save_root = f"./embeddings/{dataset_dir_name}/text"
        train_save_root = f"./embeddings/{dataset_dir_name}/train"
        test_save_root = f"./embeddings/{dataset_dir_name}/test"
        save_template_attr = os.environ.get("CLIP_OB_SAVE_TEMPLATE_ATTR", "1") == "1"
        save_images = os.environ.get("CLIP_OB_SAVE_IMAGES", "0") == "1"

        if save_template_attr:
            self.class_exp = self.load_class_exp_multi(dataset_name=dataset_name, exp_root=ATTR_ROOT)
            self.extract_template_attributes_embeddings(
                save_root=text_save_root,
                save_name=f"{self.cfg.DATASET.NAME}_template_attributes.pt",
            )

        if save_images:
            self.extract_train_image_embeddings(save_root=train_save_root)
            self.extract_test_image_embeddings(save_root=test_save_root)


    # # -----------------------------------------------------------------
    #     # NEW: Save class-name semantic embeddings (name-only + templated)
    #     # -----------------------------------------------------------------
    #     self.extract_classname_semantic_embeddings(
    #         save_root=text_save_root,
    #         template="{}",
    #         save_name=f"{self.cfg.DATASET.NAME}_class_semantic_nameonly.pt",
    #     )
    #     self.extract_classname_semantic_embeddings(
    #         save_root=text_save_root,
    #         template="a photo of a {}.",
    #         save_name=f"{self.cfg.DATASET.NAME}_class_semantic_template.pt",
    #     )
    #
    #     # -----------------------------------------------------------------
    #     # Save descriptions (sentence-level)
    #     # -----------------------------------------------------------------
    #     self.class_exp = self.load_class_exp_multi(dataset_name=dataset_name, exp_root=DESC_ROOT)
    #     self.extract_embedding_multi(
    #         save_root=text_save_root,
    #         save_name=f"{self.cfg.DATASET.NAME}_class_descriptions.pt",
    #     )
    #
    #     # -----------------------------------------------------------------
    #     # Save attributes (phrase-level)
    #     # -----------------------------------------------------------------
    #     self.class_exp = self.load_class_exp_multi(dataset_name=dataset_name, exp_root=ATTR_ROOT)
    #     self.extract_embedding_multi(
    #         save_root=text_save_root,
    #         save_name=f"{self.cfg.DATASET.NAME}_class_attributes.pt",
    #     )
    #
    #     # -----------------------------------------------------------------
    #     # Save image embeddings
    #     # -----------------------------------------------------------------
    #     self.extract_train_image_embeddings(save_root=f"./embeddings/{dataset_name}/train")
    #     self.extract_test_image_embeddings(save_root=f"./embeddings/{dataset_name}/test")

    def forward_backward(self,idx, batch, global_weight=None, fedprox=False, mu=0.5):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.CLIP_OB.PREC
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
            if fedprox:
                model_weight = self.model.state_dict()
                fed_prox_reg = ((mu / 2) * torch.norm((model_weight['prompt_learner.ctx'] - global_weight['prompt_learner.ctx'])) ** 2)
                loss += fed_prox_reg
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }



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

