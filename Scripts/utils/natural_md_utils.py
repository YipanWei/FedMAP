import os
import random
from collections import defaultdict

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from utils.data_utils import Datum, Dataset_partition_domain, get_domain_client_num


class GenericMDDataset(Dataset):
    def __init__(self, base_path, dataset_dirname, center, classnames, class2idx, net_dataidx_map=None, train=True, transform=None):
        self.base_path = os.path.join(base_path, dataset_dirname)
        self.center = str(center)
        self.site = self.center
        self.train = train
        self.transform = transform
        self.classnames = list(classnames)
        self.lab2cname = {i: c for i, c in enumerate(self.classnames)}
        self.class2idx = dict(class2idx)

        split = "train" if train else "test"
        split_dir = os.path.join(self.base_path, self.center, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"找不到数据目录: {split_dir}")

        ds = ImageFolder(split_dir, transform=None)
        paths = [p for (p, _) in ds.samples]

        labels = []
        for p in paths:
            cls_name = os.path.basename(os.path.dirname(p))
            if cls_name not in self.class2idx:
                raise KeyError(
                    f"未知类别文件夹: {cls_name}，请确保类名属于 {self.classnames}"
                )
            labels.append(self.class2idx[cls_name])

        self.imgs = np.asarray(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.label = self.labels

        if net_dataidx_map is not None:
            idx = np.asarray(net_dataidx_map, dtype=np.int64)
            self.imgs = self.imgs[idx]
            self.labels = self.labels[idx]
            self.label = self.labels

        self.data_detailed = self._convert()

    def _convert(self):
        data = []
        for i in range(len(self.labels)):
            path = self.imgs[i]
            y = int(self.labels[i])
            cname = self.lab2cname[y]
            data.append(Datum(impath=path, label=y, domain=self.center, classname=cname))
        return data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        y = int(self.labels[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, y


def infer_md_classnames(base_path, dataset_dirname, source_domains):
    dataset_root = os.path.join(base_path, dataset_dirname)
    classes = set()
    for domain in source_domains:
        for split in ("train", "test"):
            split_dir = os.path.join(dataset_root, domain, split)
            if not os.path.isdir(split_dir):
                continue
            for name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, name)
                if os.path.isdir(class_dir):
                    classes.add(name)
    if not classes:
        raise FileNotFoundError(f"在 {dataset_root} 下未找到任何 train/test 类别目录")
    classnames = sorted(classes)
    class2idx = {c: i for i, c in enumerate(classnames)}
    return classnames, class2idx


def prepare_natural_md_dataset(cfg, data_base_path, dataset_dirname, partition_mode="dirichlet"):
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    total_client_num = cfg.DATASET.USERS
    domain_names = list(cfg.DATASET.SOURCE_DOMAINS)
    domain_num = len(domain_names)
    min_pic_require_size = 2
    domain_client_num = get_domain_client_num(domain_num, total_client_num)
    classnames, class2idx = infer_md_classnames(data_base_path, dataset_dirname, domain_names)
    use_percent_partition = bool(
        getattr(cfg.DATASET, "USE_PERCENT_PARTITION", hasattr(cfg.DATASET, "DOMAIN_P"))
    )
    percent = float(getattr(cfg.DATASET, "DOMAIN_P", 1.0))

    all_domain_trainset = []
    all_domain_testset = []
    global_test_set = []

    for domain_name_index in range(domain_num):
        current_domain_name = domain_names[domain_name_index]
        domain_n_clients = int(domain_client_num[domain_name_index])

        global_domain_trainset = GenericMDDataset(
            data_base_path,
            dataset_dirname,
            current_domain_name,
            classnames,
            class2idx,
            transform=transform_train,
            train=True,
        )
        global_domain_testset = GenericMDDataset(
            data_base_path,
            dataset_dirname,
            current_domain_name,
            classnames,
            class2idx,
            transform=transform_test,
            train=False,
        )

        if use_percent_partition:
            net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domain(
                global_domain_trainset,
                global_domain_testset,
                beta=cfg.DATASET.BETA,
                K=len(classnames),
                n_parties=domain_n_clients,
                min_require_size=min_pic_require_size,
                partition_mode="percent",
                percent=percent,
            )
        else:
            net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domain(
                global_domain_trainset,
                global_domain_testset,
                beta=cfg.DATASET.BETA,
                K=len(classnames),
                n_parties=domain_n_clients,
                min_require_size=min_pic_require_size,
                partition_mode=partition_mode,
            )

        global_domain_train_data = global_domain_trainset.data_detailed
        global_domain_test_data = global_domain_testset.data_detailed

        domain_trainset = [[] for _ in range(domain_n_clients)]
        domain_testset = [[] for _ in range(domain_n_clients)]

        for i in range(domain_n_clients):
            domain_trainset[i] = GenericMDDataset(
                data_base_path,
                dataset_dirname,
                current_domain_name,
                classnames,
                class2idx,
                net_dataidx_map_train[i],
                transform=transform_train,
                train=True,
            ).data_detailed
            domain_testset[i] = GenericMDDataset(
                data_base_path,
                dataset_dirname,
                current_domain_name,
                classnames,
                class2idx,
                net_dataidx_map_test[i],
                transform=transform_test,
                train=False,
            ).data_detailed

        all_domain_trainset.append(domain_trainset)
        all_domain_testset.append(domain_testset)
        global_test_set.append(global_domain_test_data)

    train_data_num_list = []
    test_data_num_list = []
    train_set = []
    test_set = []

    for dataset in all_domain_trainset:
        for subset in dataset:
            train_data_num_list.append(len(subset))
            train_set.append(subset)

    for dataset in all_domain_testset:
        for subset in dataset:
            test_data_num_list.append(len(subset))
            test_set.append(subset)

    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)

    lab2cname = {i: classnames[i] for i in range(len(classnames))}
    return train_set, test_set, global_test_set, classnames, lab2cname


def stratified_split(items_by_class, train_ratio=0.8, seed=42):
    rng = random.Random(seed)
    train_items = []
    test_items = []
    for cls_name, items in sorted(items_by_class.items()):
        items = list(items)
        rng.shuffle(items)
        if len(items) <= 1:
            train_cls = items
            test_cls = []
        else:
            split_idx = max(1, min(len(items) - 1, int(round(len(items) * train_ratio))))
            train_cls = items[:split_idx]
            test_cls = items[split_idx:]
        train_items.extend((cls_name, p) for p in train_cls)
        test_items.extend((cls_name, p) for p in test_cls)
    return train_items, test_items
