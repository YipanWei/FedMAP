
import re
import sys, os

from torchvision.datasets import ImageFolder

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from collections import Counter

class Datum:

    def __init__(self, impath, label=0, domain=0, classname=""):
        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname

def prepare_data_domain_partition_train(cfg, data_base_path):
    data_base_path = data_base_path
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
    domain_name = cfg.DATASET.SOURCE_DOMAINS
    domain_num = len(domain_name)
    min_pic_require_size = 2
    domain_client_num = get_domain_client_num(domain_num, total_client_num)

    all_domain_trainset = []
    all_domain_testset = []
    global_test_set = []

    for domain_name_index in range(domain_num):
        current_domain_name = domain_name[domain_name_index]
        domain_n_clients = int(domain_client_num[domain_name_index])
        if cfg.DATASET.NAME == 'FedISIC':
            global_domain_trainset = FedISICDataset(data_base_path, current_domain_name, transform=transform_train,
                                                    train=True)
            global_domain_testset = FedISICDataset(data_base_path, current_domain_name, transform=transform_test,
                                                   train=False)
            net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domain(
                global_domain_trainset, global_domain_testset,
                beta=cfg.DATASET.BETA,
                K=len(global_domain_trainset.classnames),
                n_parties=domain_n_clients,
                min_require_size=min_pic_require_size,
                partition_mode="dirichlet",  # 'dirichlet' or 'percent'
            )

        elif cfg.DATASET.NAME == 'FedCamelyon17MD':
            global_domain_trainset = FedCamelyon17MDDataset(data_base_path, current_domain_name, transform=transform_train,
                                                            train=True)
            global_domain_testset = FedCamelyon17MDDataset(data_base_path, current_domain_name, transform=transform_test,
                                                           train=False)
            net_dataidx_map_train, net_dataidx_map_test = Dataset_partition_domain(
                global_domain_trainset,
                global_domain_testset,
                beta=cfg.DATASET.BETA,
                K = len(global_domain_trainset.classnames),
                n_parties=domain_n_clients,
                min_require_size=min_pic_require_size,
                partition_mode="percent",  # 'dirichlet' or 'percent'
                percent=cfg.DATASET.DOMAIN_P,
            )
        else:
            raise ValueError(f"Unsupported DATASET.NAME: {cfg.DATASET.NAME}")

        # 统一获取类名映射（优先 trainset）
        if hasattr(global_domain_trainset, 'imagefolder_obj'):
            classnames = global_domain_trainset.imagefolder_obj.classes
            lab2cname = {i: classnames[i] for i in range(len(classnames))}
        elif hasattr(global_domain_trainset, 'classnames'):
            classnames = list(global_domain_trainset.classnames)
            lab2cname = {i: classnames[i] for i in range(len(classnames))}
        elif hasattr(global_domain_testset, 'imagefolder_obj'):
            classnames = global_domain_testset.imagefolder_obj.classes
            lab2cname = {i: classnames[i] for i in range(len(classnames))}
        elif hasattr(global_domain_testset, 'classnames'):
            classnames = list(global_domain_testset.classnames)
            lab2cname = {i: classnames[i] for i in range(len(classnames))}
        else:
            raise RuntimeError("Cannot infer classnames/lab2cname from dataset objects")


        global_domain_trainset = global_domain_trainset.data_detailed
        global_domain_testset = global_domain_testset.data_detailed

        domain_trainset = [[] for i in range(domain_n_clients)]
        domain_testset = [[] for i in range(domain_n_clients)]

        for i in range(domain_n_clients):
            if cfg.DATASET.NAME == 'FedISIC':
                domain_trainset[i] = FedISICDataset(
                    data_base_path,
                    current_domain_name,
                    net_dataidx_map_train[i],
                    transform=transform_train
                )
                domain_testset[i] = FedISICDataset(
                    data_base_path,
                    current_domain_name,
                    net_dataidx_map_test[i],
                    train=False,
                    transform=transform_test
                ).data_detailed

            elif cfg.DATASET.NAME == 'FedCamelyon17MD':
                domain_trainset[i] = FedCamelyon17MDDataset(
                    data_base_path,
                    current_domain_name,
                    net_dataidx_map_train[i],
                    transform=transform_train
                )
                domain_testset[i] = FedCamelyon17MDDataset(
                    data_base_path,
                    current_domain_name,
                    net_dataidx_map_test[i],
                    train=False,
                    transform=transform_test
                ).data_detailed

            domain_trainset[i] = domain_trainset[i].data_detailed

        all_domain_trainset.append(domain_trainset)
        all_domain_testset.append(domain_testset)
        global_test_set.append(global_domain_testset)

    train_data_num_list = []
    test_data_num_list = []
    train_set = []
    test_set = []
    for dataset in all_domain_trainset:
        for i in range(len(dataset)):
            train_data_num_list.append(len(dataset[i]))
            train_set.append(dataset[i])
    for dataset in all_domain_testset:
        for i in range(len(dataset)):
            test_data_num_list.append(len(dataset[i]))
            test_set.append(dataset[i])
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)

    return train_set, test_set, global_test_set, classnames, lab2cname



def get_domain_client_num(domain_num, total_client_num):

    n_clients = total_client_num // domain_num
    not_allocated_num = total_client_num % domain_num

    domain_client_num = np.ones(domain_num) * n_clients
    remain_num = np.random.randint(domain_num, size=not_allocated_num)
    for i in range(len(remain_num)):
        domain_client_num[remain_num[i]] += 1
   
    return domain_client_num

ISIC_CLASSNAMES = ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC"]
ISIC_CLASS2IDX = {c:i for i,c in enumerate(ISIC_CLASSNAMES)}

class FedISICDataset(Dataset):

    def __init__(self, base_path, center, net_dataidx_map=None, train=True, transform=None):
        self.base_path = os.path.join(base_path,"FedISIC-MD")
        self.center = str(center)
        self.site = self.center
        self.train = train
        self.transform = transform

        split = "train" if train else "test"
        split_dir = os.path.join(self.base_path, self.center, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"找不到数据目录: {split_dir}")

        ds = ImageFolder(split_dir, transform=None)
        paths = [p for (p, _) in ds.samples]

        labels = []
        for p in paths:
            cls_name = os.path.basename(os.path.dirname(p))
            if cls_name not in ISIC_CLASS2IDX:
                raise KeyError(f"未知类别文件夹：{cls_name}，请确保类名属于 {ISIC_CLASSNAMES}")
            labels.append(ISIC_CLASS2IDX[cls_name])

        self.imgs = np.asarray(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.label = self.labels
        self.classnames = ISIC_CLASSNAMES[:]
        self.lab2cname = {i:c for i,c in enumerate(self.classnames)}

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


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts


CAMELYON17_CLASSNAMES = ["normal tissue", "tumor tissue"]
CAMELYON17_CLASS2IDX = {c: i for i, c in enumerate(CAMELYON17_CLASSNAMES)}

class FedCamelyon17MDDataset(Dataset):
    def __init__(self, base_path, center, net_dataidx_map=None, train=True, transform=None):
        self.base_path = os.path.join(base_path,"FedCamelyon17-MD")
        self.center = str(center)
        self.site = self.center
        self.train = train
        self.transform = transform


        split = "train" if train else "test"
        split_dir = os.path.join(self.base_path, self.center, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"找不到数据目录: {split_dir}")


        ds = ImageFolder(split_dir, transform=None)
        paths = [p for (p, _) in ds.samples]


        labels = []
        for p in paths:
            cls_name = os.path.basename(os.path.dirname(p))
            if "." in cls_name:
                cls_name = cls_name.split(".")[0]
            if cls_name not in CAMELYON17_CLASS2IDX:
                raise KeyError(
                    f"未知类别文件夹: {cls_name}，请确保类名属于 {CAMELYON17_CLASSNAMES}"
                )
            labels.append(CAMELYON17_CLASS2IDX[cls_name])

        self.imgs = np.asarray(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.label = self.labels
        self.classnames = CAMELYON17_CLASSNAMES[:]
        self.lab2cname = {i: c for i, c in enumerate(self.classnames)}

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

def Dataset_partition_domain(
        global_domain_trainset, global_domain_testset,
        beta, K,
        n_parties=5,
        min_require_size=2,
        partition_mode="dirichlet",   # 'dirichlet' or 'percent'
        percent=0.01,                  # 用于 percent 模式
):
    """
    统一的域内划分函数:
    - partition_mode='dirichlet'：使用 Dirichlet 分布划分 (默认)
    - partition_mode='percent'：使用按百分比切块划分
    """
    # -------------------- 模式 1: 百分比划分 --------------------
    if partition_mode == "percent":
        print(f"Using percent-based domain partition (percent={percent})")
        train_labels = np.array(global_domain_trainset.label)
        test_labels = np.array(global_domain_testset.label)

        net_dataidx_map_train, train_cls_counts = partition_domain_skew_loaders(
            train_labels, n_participants=n_parties, percent=percent, seed=42
        )
        net_dataidx_map_test, test_cls_counts = partition_domain_skew_loaders(
            test_labels, n_participants=n_parties, percent=percent, seed=42
        )

        print(global_domain_trainset.site, "Training data split:", train_cls_counts)
        print(global_domain_trainset.site, "Testing data split:", test_cls_counts)
        return net_dataidx_map_train, net_dataidx_map_test

    # -------------------- 模式 2: Dirichlet / 均匀划分 --------------------
    min_size = 0


    train_labels = global_domain_trainset.label
    train_labels = np.array(train_labels)

    test_labels = global_domain_testset.label
    test_labels = np.array(test_labels)

    N_train = len(train_labels)
   
    net_dataidx_map_train = {}
    net_dataidx_map_test = {}

    while min_size < min_require_size:
        idx_batch_train = [[] for _ in range(n_parties)]
        idx_batch_test = [[] for _ in range(n_parties)]
        for k in range(K):
            train_idx_k = np.where(train_labels == k)[0]
            test_idx_k = np.where(test_labels == k)[0]
            train_idx_k = np.array(train_idx_k)
            test_idx_k = np.array(test_idx_k)
            np.random.seed(0)
            np.random.shuffle(train_idx_k)
            np.random.shuffle(test_idx_k)
            if beta == 0:
                idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.array_split(train_idx_k, n_parties))]
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.array_split(test_idx_k, n_parties))]
            else:
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])
                proportions = proportions / proportions.sum()
                proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
                proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]
                train_part_list = np.split(train_idx_k, proportions_train)
                test_part_list = np.split(test_idx_k, proportions_test)
                idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, train_part_list)]
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, test_part_list)]

            min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
            min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
            min_size = min(min_size_test, min_size_train)

    for j in range(n_parties):
        np.random.shuffle(idx_batch_train[j])
        np.random.shuffle(idx_batch_test[j])
        net_dataidx_map_train[j] = idx_batch_train[j]
        net_dataidx_map_test[j] = idx_batch_test[j]

    traindata_cls_counts = record_net_data_stats(train_labels, net_dataidx_map_train)
    print(global_domain_trainset.site, "Training data split: ", traindata_cls_counts)
    testdata_cls_counts = record_net_data_stats(test_labels, net_dataidx_map_test)
    print(global_domain_trainset.site, "Testing data split: ", testdata_cls_counts)
    return net_dataidx_map_train, net_dataidx_map_test


def partition_domain_skew_loaders(
        train_labels: np.ndarray,
        n_participants: int,
        percent: float,
        seed: int = 42,
):
    """
    域内“按百分比切块”的划分：
    - 有一个域的训练样本，共 N 条（train_labels 仅用于统计类分布）
    - 将该域分给 n_participants 个客户端
    - 每个客户端从“剩余索引池”中拿走 int(percent * N) 条（不重叠），每次拿之前都会对剩余池重新打乱
    - 如果样本不够，后面的客户端可能拿到更少甚至 0 条

    返回:
      net_dataidx_map: Dict[int, np.ndarray]  # 每个客户端对应的样本索引
      net_cls_counts : Dict[int, Dict[int,int]]  # 每个客户端的类别计数
    """
    assert 0 <= percent <= 1.0, f"percent should be in [0,1], got {percent}"
    N = len(train_labels)
    ini_len = N
    k = int(percent * ini_len)

    rng = np.random.default_rng(seed)
    not_used = np.arange(N, dtype=np.int64)

    net_dataidx_map = {}
    net_cls_counts = {}

    for i in range(n_participants):
        if not_used.size == 0 or k == 0:
            selected = np.array([], dtype=np.int64)
        else:
            idxs = rng.permutation(not_used)
            take = min(k, idxs.size)
            selected = idxs[:take]
            not_used = idxs[take:]

        net_dataidx_map[i] = selected

    num_classes = int(train_labels.max()) + 1 if N > 0 else 0
    for i in range(n_participants):
        lbls_i = train_labels[net_dataidx_map[i]] if net_dataidx_map[i].size > 0 else np.array([], dtype=int)
        binc = np.bincount(lbls_i, minlength=num_classes) if num_classes > 0 else np.array([])
        net_cls_counts[i] = {int(c): int(binc[c]) for c in range(num_classes)}

    return net_dataidx_map, net_cls_counts
