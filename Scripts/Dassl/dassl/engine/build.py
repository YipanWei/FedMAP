from Dassl.dassl.utils import Registry, check_availability

from trainers.CLIP import CLIP

from trainers.LPT import LPT
from trainers.IVLP import IVLP
from trainers.VPT import VPT

from trainers.FedAPT import FedAPT
from trainers.FedCLIP import FedCLIP
from trainers.FedCoCoOP import FedCoCoOP
from trainers.FedKgCoOP import FedKgCoOP
from trainers.PromptFL import PROMPTFL
from trainers.FedProxLPT import FedProxLPT
from trainers.FedTPG import FedTPG

from trainers.PromptFL_OB import PROMPTFL_OB
from trainers.CLIP_OB import CLIP_OB
from trainers.PromptFL_Exp import PROMPTFL_Exp

from trainers.GL_SVDMSE_HE import GL_SVDMSE_HE
from trainers.FOCoOP import FOCoOP
from trainers.FedMVP import FedMVP

from trainers.PromptFL_KL_Global import PROMPTFL_KL_Global
from trainers.PromptFL_Anchor import PROMPTFL_Anchor
from trainers.PromptFL_KL import PROMPTFL_KL
from trainers.PromptFL_KL_Anchor import PROMPTFL_KL_Anchor
from trainers.PromptFL_Anchor2 import PROMPTFL_Anchor2
from trainers.PromptFL_Anchor3 import PROMPTFL_Anchor3
from trainers.PromptFL_KL_VPT import PROMPTFL_KL_VPT
from trainers.VPT_LPT import VPT_LPT
from trainers.PromptFL_KL_VPT_Inter import PROMPTFL_KL_VPT_Inter
from trainers.VPT_Ma import VPT_Ma
from trainers.VPT_M import VPT_M

from trainers.VPT_a import VPT_a

from trainers.VPT_Ma_KL import VPT_Ma_KL
from trainers.VPT_Ma_T import VPT_Ma_T

from trainers.VPT_Ma_One import VPT_Ma_One
from trainers.VPT_Ma_R import VPT_Ma_R

from trainers.VPT_Ma_A import VPT_Ma_A
from trainers.VPT_Ma_Cos import VPT_Ma_Cos
from trainers.VPT_Ma_Wass import VPT_Ma_Wass
from trainers.VPT_Ma_Noisy import VPT_Ma_Noisy

TRAINER_REGISTRY = Registry("TRAINER")

TRAINER_REGISTRY.register(CLIP)
TRAINER_REGISTRY.register(LPT)
TRAINER_REGISTRY.register(IVLP)
TRAINER_REGISTRY.register(VPT)
TRAINER_REGISTRY.register(FedAPT)
TRAINER_REGISTRY.register(FedCLIP)
TRAINER_REGISTRY.register(FedCoCoOP)
TRAINER_REGISTRY.register(FedKgCoOP)
TRAINER_REGISTRY.register(PROMPTFL)
TRAINER_REGISTRY.register(FedProxLPT)
TRAINER_REGISTRY.register(FedTPG)
TRAINER_REGISTRY.register(PROMPTFL_OB)
TRAINER_REGISTRY.register(CLIP_OB)
TRAINER_REGISTRY.register(PROMPTFL_Exp)
TRAINER_REGISTRY.register(GL_SVDMSE_HE)
TRAINER_REGISTRY.register(FOCoOP)
TRAINER_REGISTRY.register(FedMVP)
TRAINER_REGISTRY.register(PROMPTFL_KL_Global)
TRAINER_REGISTRY.register(PROMPTFL_Anchor)
TRAINER_REGISTRY.register(PROMPTFL_KL)
TRAINER_REGISTRY.register(PROMPTFL_KL_Anchor)
TRAINER_REGISTRY.register(PROMPTFL_Anchor2)
TRAINER_REGISTRY.register(PROMPTFL_Anchor3)
TRAINER_REGISTRY.register(PROMPTFL_KL_VPT)
TRAINER_REGISTRY.register(VPT_a)
TRAINER_REGISTRY.register(VPT_LPT)
TRAINER_REGISTRY.register(PROMPTFL_KL_VPT_Inter)
TRAINER_REGISTRY.register(VPT_Ma)
TRAINER_REGISTRY.register(VPT_M)
TRAINER_REGISTRY.register(VPT_Ma_KL)
TRAINER_REGISTRY.register(VPT_Ma_T)
TRAINER_REGISTRY.register(VPT_Ma_One)
TRAINER_REGISTRY.register(VPT_Ma_R)
TRAINER_REGISTRY.register(VPT_Ma_A)
TRAINER_REGISTRY.register(VPT_Ma_Cos)
TRAINER_REGISTRY.register(VPT_Ma_Wass)
TRAINER_REGISTRY.register(VPT_Ma_Noisy)

def build_trainer(args,cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(args,cfg)
