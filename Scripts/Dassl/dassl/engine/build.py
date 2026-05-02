from Dassl.dassl.utils import Registry, check_availability

from trainers.CLIP import CLIP
from trainers.VPT import VPT
from trainers.FedAPT import FedAPT
from trainers.FedCLIP import FedCLIP
from trainers.FedCoCoOP import FedCoCoOP
from trainers.FedKgCoOP import FedKgCoOP
from trainers.PromptFL import PROMPTFL
from trainers.FedProxLPT import FedProxLPT
from trainers.FOCoOP import FOCoOP
from trainers.FedMVP import FedMVP
from trainers.VPT_Ma import VPT_Ma

TRAINER_REGISTRY = Registry("TRAINER")

TRAINER_REGISTRY.register(CLIP)
TRAINER_REGISTRY.register(VPT)
TRAINER_REGISTRY.register(FedAPT)
TRAINER_REGISTRY.register(FedCLIP)
TRAINER_REGISTRY.register(FedCoCoOP)
TRAINER_REGISTRY.register(FedKgCoOP)
TRAINER_REGISTRY.register(PROMPTFL)
TRAINER_REGISTRY.register(FedProxLPT)
TRAINER_REGISTRY.register(FOCoOP)
TRAINER_REGISTRY.register(FedMVP)
TRAINER_REGISTRY.register(VPT_Ma)

def build_trainer(args,cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(args,cfg)
