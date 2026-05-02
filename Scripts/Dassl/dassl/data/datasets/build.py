from Dassl.dassl.utils import Registry, check_availability

from datasets.fedisic import FedISIC
from datasets.fedcamelyon17 import FedCamelyon17MD

DATASET_REGISTRY = Registry("DATASET")

DATASET_REGISTRY.register(FedISIC)
DATASET_REGISTRY.register(FedCamelyon17MD)


def build_dataset(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)
