from Dassl.dassl.config import get_cfg_default

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)



def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.resume:
        cfg.RESUME = args.resume
    if args.seed:
        cfg.SEED = args.seed
    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    if hasattr(args, "num_users") and args.num_users is not None and args.num_users > 0:
        cfg.DATASET.USERS = args.num_users
    if hasattr(args, "partition") and args.partition:
        cfg.DATASET.PARTITION = args.partition
    if hasattr(args, "beta"):
        cfg.DATASET.BETA = args.beta
    if hasattr(args, "aggregation") and args.aggregation:
        cfg.DATASET.AGGREGATION = args.aggregation
    if hasattr(cfg.DATASET, "DOMAIN_P") and getattr(args, "domain_p", -1.0) >= 0:
        cfg.DATASET.DOMAIN_P = float(args.domain_p)
    if hasattr(args, "num_workers") and args.num_workers is not None:
        cfg.DATALOADER.NUM_WORKERS = int(args.num_workers)
    if hasattr(args, "gamma"):
        cfg.OPTIM.GAMMA = args.gamma
    cfg.DATASET.REPEATRATE = 0.0
    cfg.MODEL.BACKBONE.PRETRAINED = True


def extend_cfg(cfg, args):
    from yacs.config import CfgNode as CN

    cfg.TRAINER.CLIP = CN()
    cfg.TRAINER.CLIP.PREC = "fp16"
    cfg.TRAINER.CLIP.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.CLIP_OB = CN()
    cfg.TRAINER.CLIP_OB.PREC = "fp16"
    cfg.TRAINER.CLIP_OB.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION =args.nctxv
    cfg.TRAINER.IVLP.N_CTX_TEXT = 16
    cfg.TRAINER.IVLP.CTX_INIT = False
    cfg.TRAINER.IVLP.CSC = False
    cfg.TRAINER.IVLP.PREC = "fp16"
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = (
        1
    )
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = (
        1
    )

    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION =args.nctxv
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT.PREC = "fp16"
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = args.prompt_depth

    cfg.TRAINER.VPTPR = CN()
    cfg.TRAINER.VPTPR.PREC = "fp16"
    cfg.TRAINER.VPTPR.N_CTX_VISION =args.nctxv
    cfg.TRAINER.VPTPR.CTX_INIT = False
    cfg.TRAINER.VPTPR.PROMPT_DEPTH_VISION = args.prompt_depth
    cfg.TRAINER.VPTPR.RATIO = 0.8

    cfg.TRAINER.LPT = CN()
    cfg.TRAINER.LPT.N_CTX_TEXT = args.nctx
    cfg.TRAINER.LPT.CTX_INIT = False
    cfg.TRAINER.LPT.CSC = False
    cfg.TRAINER.LPT.PREC = "fp16"
    cfg.TRAINER.LPT.PROMPT_DEPTH_TEXT = args.prompt_depth

    cfg.TRAINER.FedAPT = CN()
    cfg.TRAINER.FedAPT.N_CTX_TEXT = args.nctx
    cfg.TRAINER.FedAPT.CTX_INIT = False
    cfg.TRAINER.FedAPT.CSC = False
    cfg.TRAINER.FedAPT.PREC = "fp16"
    cfg.TRAINER.FedAPT.BETA = 0.5

    cfg.TRAINER.FedCLIP = CN()
    cfg.TRAINER.FedCLIP.PREC = "fp16"
    cfg.TRAINER.FedCLIP.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.FedCoCoOP = CN()
    cfg.TRAINER.FedCoCoOP.N_CTX_TEXT = args.nctx
    cfg.TRAINER.FedCoCoOP.CTX_INIT = False
    cfg.TRAINER.FedCoCoOP.CSC = False
    cfg.TRAINER.FedCoCoOP.PREC = "fp16"

    cfg.TRAINER.FedKgCoOP = CN()
    cfg.TRAINER.FedKgCoOP.N_CTX_TEXT = args.nctx
    cfg.TRAINER.FedKgCoOP.CTX_INIT = False
    cfg.TRAINER.FedKgCoOP.CSC = False
    cfg.TRAINER.FedKgCoOP.PREC = "fp16"
    cfg.TRAINER.FedKgCoOP.KG_LAMBDA = 8.0
    cfg.TRAINER.FedKgCoOP.KG_TEMPLATE = "a photo of a {}."

    cfg.TRAINER.FedProxLPT = CN()
    cfg.TRAINER.FedProxLPT.N_CTX_TEXT = args.nctx
    cfg.TRAINER.FedProxLPT.CTX_INIT = False
    cfg.TRAINER.FedProxLPT.CSC = False
    cfg.TRAINER.FedProxLPT.PREC = "fp16"
    cfg.TRAINER.FedProxLPT.mu = 0.5

    cfg.TRAINER.PROMPTFL = CN()
    cfg.TRAINER.PROMPTFL.N_CTX = args.nctx
    cfg.TRAINER.PROMPTFL.CSC = False
    cfg.TRAINER.PROMPTFL.CTX_INIT = False
    cfg.TRAINER.PROMPTFL.PREC = "fp16"
    cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.PROMPTFL_OB = CN()
    cfg.TRAINER.PROMPTFL_OB.N_CTX = args.nctx
    cfg.TRAINER.PROMPTFL_OB.CSC = False
    cfg.TRAINER.PROMPTFL_OB.CTX_INIT = False
    cfg.TRAINER.PROMPTFL_OB.PREC = "fp16"
    cfg.TRAINER.PROMPTFL_OB.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.PROMPTFL_Exp = CN()
    cfg.TRAINER.PROMPTFL_Exp.N_CTX = args.nctx
    cfg.TRAINER.PROMPTFL_Exp.CSC = False
    cfg.TRAINER.PROMPTFL_Exp.CTX_INIT = False
    cfg.TRAINER.PROMPTFL_Exp.PREC = "fp16"
    cfg.TRAINER.PROMPTFL_Exp.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.FedTPG = CN()
    cfg.TRAINER.FedTPG.N_CTX_TEXT = 16
    cfg.TRAINER.FedTPG.CTX_INIT = False
    cfg.TRAINER.FedTPG.PREC = "fp16"
    cfg.TRAINER.FedTPG.PROMPT_DEPTH_TEXT = 1
    cfg.TRAINER.FedTPG.D_CTX = 1

    cfg.TRAINER.GL_SVDMSE_HE = CN()
    cfg.TRAINER.GL_SVDMSE_HE.N_CTX_GLOBAL = 16  # number of context vectors
    cfg.TRAINER.GL_SVDMSE_HE.CSC = False  # class-specific context
    cfg.TRAINER.GL_SVDMSE_HE.CTX_INIT = False  # initialization words
    cfg.TRAINER.GL_SVDMSE_HE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.GL_SVDMSE_HE.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.GL_SVDMSE_HE.N = 1  # number of prompts
    cfg.TRAINER.GL_SVDMSE_HE.lambda_orthogonal = 1
    cfg.TRAINER.GL_SVDMSE_HE.alpha = 1.0
    cfg.TRAINER.GL_SVDMSE_HE.ratio = 0.8

    cfg.TRAINER.FOCoOP = CN()
    cfg.TRAINER.FOCoOP.N_CTX = args.nctx
    cfg.TRAINER.FOCoOP.CSC = False
    cfg.TRAINER.FOCoOP.CTX_INIT = False
    cfg.TRAINER.FOCoOP.PREC = "fp16"
    cfg.TRAINER.FOCoOP.CLASS_TOKEN_POSITION = "end"

    # FOCoOP (simplified) extras
    cfg.TRAINER.FOCoOP.RHO = 0.5
    cfg.TRAINER.FOCoOP.U = 16
    cfg.TRAINER.FOCoOP.LAMBDA_OOD = 0.01
    cfg.TRAINER.FOCoOP.OOD_WORD = " "

    cfg.TRAINER.FedMVP = CN()
    cfg.TRAINER.FedMVP.PREC = "fp16"   # "fp16" / "fp32" / "amp"
    cfg.TRAINER.FedMVP.M = 8
    cfg.TRAINER.FedMVP.HEADS = 4
    cfg.TRAINER.FedMVP.DROPOUT = 0.0
    cfg.TRAINER.FedMVP.TEXT_TEMPLATE = "a photo of a {}."

    cfg.TRAINER.PROMPTFL_KL_Global = CN()
    cfg.TRAINER.PROMPTFL_KL_Global.N_CTX = args.nctx
    cfg.TRAINER.PROMPTFL_KL_Global.CSC = False
    cfg.TRAINER.PROMPTFL_KL_Global.CTX_INIT = False
    cfg.TRAINER.PROMPTFL_KL_Global.PREC = "fp16"
    cfg.TRAINER.PROMPTFL_KL_Global.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.PROMPTFL_KL_Global.LAM_KL =0.5
    cfg.TRAINER.PROMPTFL_KL_Global.KD_T =2.0

    cfg.TRAINER.PROMPTFL_Anchor = CN()
    cfg.TRAINER.PROMPTFL_Anchor.N_CTX = 20
    cfg.TRAINER.PROMPTFL_Anchor.CSC = False
    cfg.TRAINER.PROMPTFL_Anchor.CTX_INIT = False
    cfg.TRAINER.PROMPTFL_Anchor.PREC = "fp16"
    cfg.TRAINER.PROMPTFL_Anchor.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.PROMPTFL_KL = CN()
    cfg.TRAINER.PROMPTFL_KL.N_CTX = args.nctx
    cfg.TRAINER.PROMPTFL_KL.CSC = False
    cfg.TRAINER.PROMPTFL_KL.CTX_INIT = False
    cfg.TRAINER.PROMPTFL_KL.PREC = "fp16"
    cfg.TRAINER.PROMPTFL_KL.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.PROMPTFL_KL.LAM_KL =args.lambda_kl
    cfg.TRAINER.PROMPTFL_KL.KD_T =args.kd_t

    cfg.TRAINER.PROMPTFL_KL_Anchor = CN()
    cfg.TRAINER.PROMPTFL_KL_Anchor.N_CTX = args.nctx
    cfg.TRAINER.PROMPTFL_KL_Anchor.CSC = False
    cfg.TRAINER.PROMPTFL_KL_Anchor.CTX_INIT = False
    cfg.TRAINER.PROMPTFL_KL_Anchor.PREC = "fp16"
    cfg.TRAINER.PROMPTFL_KL_Anchor.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.PROMPTFL_KL_Anchor.LAM_KL =0.5
    cfg.TRAINER.PROMPTFL_KL_Anchor.KD_T =2.0
    cfg.TRAINER.PROMPTFL_KL_Anchor.LAMBDA_DIVERSE = args.diverse
    cfg.TRAINER.PROMPTFL_KL_Anchor.PROMPT_DEPTH_VISION = args.prompt_depth

    cfg.TRAINER.PROMPTFL_Anchor2 = CN()
    cfg.TRAINER.PROMPTFL_Anchor2.N_CTX = args.nctx
    cfg.TRAINER.PROMPTFL_Anchor2.CSC = False
    cfg.TRAINER.PROMPTFL_Anchor2.CTX_INIT = False
    cfg.TRAINER.PROMPTFL_Anchor2.PREC = "fp16"
    cfg.TRAINER.PROMPTFL_Anchor2.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.PROMPTFL_Anchor2.LAMBDA_DIVERSE = args.diverse

    cfg.TRAINER.PROMPTFL_Anchor3 = CN()
    cfg.TRAINER.PROMPTFL_Anchor3.N_CTX = args.nctx
    cfg.TRAINER.PROMPTFL_Anchor3.CSC = False
    cfg.TRAINER.PROMPTFL_Anchor3.CTX_INIT = False
    cfg.TRAINER.PROMPTFL_Anchor3.PREC = "fp16"
    cfg.TRAINER.PROMPTFL_Anchor3.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.PROMPTFL_KL_VPT = CN()
    cfg.TRAINER.PROMPTFL_KL_VPT.N_CTX = args.nctx
    cfg.TRAINER.PROMPTFL_KL_VPT.N_CTX_VISION =args.nctxv
    cfg.TRAINER.PROMPTFL_KL_VPT.CSC = False
    cfg.TRAINER.PROMPTFL_KL_VPT.CTX_INIT = False
    cfg.TRAINER.PROMPTFL_KL_VPT.PREC = "fp16"
    cfg.TRAINER.PROMPTFL_KL_VPT.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.PROMPTFL_KL_VPT.LAM_KL =0.5
    cfg.TRAINER.PROMPTFL_KL_VPT.KD_T =2.0
    cfg.TRAINER.PROMPTFL_KL_VPT.PROMPT_DEPTH_VISION = args.prompt_depth

    cfg.TRAINER.VPT_a = CN()
    cfg.TRAINER.VPT_a.N_CTX_VISION =args.nctxv
    cfg.TRAINER.VPT_a.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT_a.PREC = "fp16"
    cfg.TRAINER.VPT_a.PROMPT_DEPTH_VISION = args.prompt_depth

    cfg.TRAINER.VPT_LPT = CN()
    cfg.TRAINER.VPT_LPT.N_CTX_VISION =args.nctxv
    cfg.TRAINER.VPT_LPT.N_CTX_TEXT = args.nctx
    cfg.TRAINER.VPT_LPT.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT_LPT.PREC = "fp16"
    cfg.TRAINER.VPT_LPT.PROMPT_DEPTH_VISION = args.prompt_depth
    cfg.TRAINER.VPT_LPT.CSC = False

    cfg.TRAINER.PROMPTFL_KL_VPT_Inter = CN()
    cfg.TRAINER.PROMPTFL_KL_VPT_Inter.N_CTX = args.nctx
    cfg.TRAINER.PROMPTFL_KL_VPT_Inter.N_CTX_VISION =args.nctxv
    cfg.TRAINER.PROMPTFL_KL_VPT_Inter.CSC = False
    cfg.TRAINER.PROMPTFL_KL_VPT_Inter.CTX_INIT = False
    cfg.TRAINER.PROMPTFL_KL_VPT_Inter.PREC = "fp16"
    cfg.TRAINER.PROMPTFL_KL_VPT_Inter.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.PROMPTFL_KL_VPT_Inter.LAM_KL =0.5
    cfg.TRAINER.PROMPTFL_KL_VPT_Inter.KD_T =2.0
    cfg.TRAINER.PROMPTFL_KL_VPT_Inter.PROMPT_DEPTH_VISION = args.prompt_depth

    cfg.TRAINER.VPT_Ma = CN()
    cfg.TRAINER.VPT_Ma.N_CTX_VISION =args.nctxv
    cfg.TRAINER.VPT_Ma.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT_Ma.PREC = "fp16"
    cfg.TRAINER.VPT_Ma.PROMPT_DEPTH_VISION = args.prompt_depth
    cfg.TRAINER.VPT_Ma.lambda_struct = args.lambda_struct
    cfg.TRAINER.VPT_Ma.PROTO_MOMENTUM = args.proto_momentum
    cfg.TRAINER.VPT_Ma.STRUCT_LOSS = "mse"

    cfg.TRAINER.VPT_M = CN()
    cfg.TRAINER.VPT_M.N_CTX_VISION =args.nctxv
    cfg.TRAINER.VPT_M.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT_M.PREC = "fp16"
    cfg.TRAINER.VPT_M.PROMPT_DEPTH_VISION = args.prompt_depth

    cfg.TRAINER.VPT_Ma_KL = CN()
    cfg.TRAINER.VPT_Ma_KL.N_CTX_VISION =args.nctxv
    cfg.TRAINER.VPT_Ma_KL.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT_Ma_KL.PREC = "fp16"
    cfg.TRAINER.VPT_Ma_KL.PROMPT_DEPTH_VISION = args.prompt_depth
    cfg.TRAINER.VPT_Ma_KL.lambda_struct = args.lambda_struct
    cfg.TRAINER.VPT_Ma_KL.PROTO_MOMENTUM = args.proto_momentum
    cfg.TRAINER.VPT_Ma_KL.STRUCT_LOSS = "kl"

    cfg.TRAINER.VPT_Ma_T = CN()
    cfg.TRAINER.VPT_Ma_T.N_CTX_VISION =args.nctxv
    cfg.TRAINER.VPT_Ma_T.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT_Ma_T.PREC = "fp16"
    cfg.TRAINER.VPT_Ma_T.PROMPT_DEPTH_VISION = args.prompt_depth
    cfg.TRAINER.VPT_Ma_T.lambda_struct = args.lambda_struct
    cfg.TRAINER.VPT_Ma_T.PROTO_MOMENTUM = args.proto_momentum
    cfg.TRAINER.VPT_Ma_T.STRUCT_LOSS = "mse"

    cfg.TRAINER.VPT_Ma_One = CN()
    cfg.TRAINER.VPT_Ma_One.N_CTX_VISION =args.nctxv
    cfg.TRAINER.VPT_Ma_One.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT_Ma_One.PREC = "fp16"
    cfg.TRAINER.VPT_Ma_One.PROMPT_DEPTH_VISION = args.prompt_depth
    cfg.TRAINER.VPT_Ma_One.lambda_struct = args.lambda_struct
    cfg.TRAINER.VPT_Ma_One.PROTO_MOMENTUM = args.proto_momentum
    cfg.TRAINER.VPT_Ma_One.STRUCT_LOSS = "mse"

    cfg.TRAINER.VPT_Ma_R = CN()
    cfg.TRAINER.VPT_Ma_R.N_CTX_VISION =args.nctxv
    cfg.TRAINER.VPT_Ma_R.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT_Ma_R.PREC = "fp16"
    cfg.TRAINER.VPT_Ma_R.PROMPT_DEPTH_VISION = args.prompt_depth
    cfg.TRAINER.VPT_Ma_R.lambda_struct = args.lambda_struct
    cfg.TRAINER.VPT_Ma_R.PROTO_MOMENTUM = args.proto_momentum
    cfg.TRAINER.VPT_Ma_R.STRUCT_LOSS = "mse"

    cfg.TRAINER.VPT_Ma_A = CN()
    cfg.TRAINER.VPT_Ma_A.N_CTX_VISION =args.nctxv
    cfg.TRAINER.VPT_Ma_A.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT_Ma_A.PREC = "fp16"
    cfg.TRAINER.VPT_Ma_A.PROMPT_DEPTH_VISION = args.prompt_depth
    cfg.TRAINER.VPT_Ma_A.lambda_struct = args.lambda_struct
    cfg.TRAINER.VPT_Ma_A.PROTO_MOMENTUM = args.proto_momentum
    cfg.TRAINER.VPT_Ma_A.STRUCT_LOSS = "mse"
    cfg.TRAINER.VPT_Ma_A.TEXT_EMB_SOURCE = args.source

    cfg.TRAINER.VPT_Ma_Cos = CN()
    cfg.TRAINER.VPT_Ma_Cos.N_CTX_VISION = args.nctxv
    cfg.TRAINER.VPT_Ma_Cos.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT_Ma_Cos.PREC = "fp16"
    cfg.TRAINER.VPT_Ma_Cos.PROMPT_DEPTH_VISION = args.prompt_depth
    cfg.TRAINER.VPT_Ma_Cos.lambda_struct = args.lambda_struct
    cfg.TRAINER.VPT_Ma_Cos.PROTO_MOMENTUM = args.proto_momentum
    cfg.TRAINER.VPT_Ma_Cos.STRUCT_LOSS = "cosine"

    cfg.TRAINER.VPT_Ma_Wass = CN()
    cfg.TRAINER.VPT_Ma_Wass.N_CTX_VISION = args.nctxv
    cfg.TRAINER.VPT_Ma_Wass.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT_Ma_Wass.PREC = "fp16"
    cfg.TRAINER.VPT_Ma_Wass.PROMPT_DEPTH_VISION = args.prompt_depth
    cfg.TRAINER.VPT_Ma_Wass.lambda_struct = args.lambda_struct
    cfg.TRAINER.VPT_Ma_Wass.PROTO_MOMENTUM = args.proto_momentum
    cfg.TRAINER.VPT_Ma_Wass.STRUCT_LOSS = "wasserstein"

    cfg.TRAINER.VPT_Ma_Noisy = CN()
    cfg.TRAINER.VPT_Ma_Noisy.N_CTX_VISION = args.nctxv
    cfg.TRAINER.VPT_Ma_Noisy.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT_Ma_Noisy.PREC = "fp16"
    cfg.TRAINER.VPT_Ma_Noisy.PROMPT_DEPTH_VISION = args.prompt_depth
    cfg.TRAINER.VPT_Ma_Noisy.lambda_struct = args.lambda_struct
    cfg.TRAINER.VPT_Ma_Noisy.PROTO_MOMENTUM = args.proto_momentum
    cfg.TRAINER.VPT_Ma_Noisy.STRUCT_LOSS = "mse"
    cfg.TRAINER.VPT_Ma_Noisy.ANCHOR_NOISE_STD = 0.0

    if not hasattr(cfg.DATASET, "USERS"):
        cfg.DATASET.USERS = args.num_users if args.num_users is not None and args.num_users > 0 else 0
    if not hasattr(cfg.DATASET, "PARTITION"):
        cfg.DATASET.PARTITION = args.partition
    if not hasattr(cfg.DATASET, "BETA"):
        cfg.DATASET.BETA = args.beta
    if not hasattr(cfg.DATASET, "AGGREGATION"):
        cfg.DATASET.AGGREGATION = args.aggregation
    if not hasattr(cfg.DATASET, "REPEATRATE"):
        cfg.DATASET.REPEATRATE = 0.0
    if not hasattr(cfg.DATASET, "DOMAIN_P"):
        cfg.DATASET.DOMAIN_P = 0.01
    if not hasattr(cfg.DATASET, "USE_PERCENT_PARTITION"):
        cfg.DATASET.USE_PERCENT_PARTITION = True
    if not hasattr(cfg.OPTIM, "ROUND"):
        cfg.OPTIM.ROUND = 50

    current_trainer_cfg = cfg["TRAINER"][args.trainer]
    cfg["TRAINER"] = CN()
    cfg["TRAINER"][args.trainer] = current_trainer_cfg
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.DATASET.Max_Class = 0


    # =========================================================
    # [NEW] LLM semantic expansion + injection config
    # =========================================================
    if not hasattr(cfg, "SEMANTIC"):
        cfg.SEMANTIC = CN()
    if not hasattr(cfg, "INJECT"):
        cfg.INJECT = CN()

    # ---- SEMANTIC (reading/selection + multi aggregation) ----
    cfg.SEMANTIC.ROOT = args.semantic_root          # "."
    cfg.SEMANTIC.FORM = args.semantic_form          # "attr" | "desc"
    cfg.SEMANTIC.USE_ALL = args.semantic_use_all    # bool
    cfg.SEMANTIC.IDX = args.semantic_idx            # int (supports -1)
    cfg.SEMANTIC.MULTI_REDUCE = args.multi_reduce   # "calc_then_mean" | "mean_then_calc"

    # ---- INJECT (paradigm + hyper-params) ----
    cfg.INJECT.MODE = args.inject_mode              # "concat" | "kl" | "mse"
    cfg.INJECT.LAMBDA_KL = float(args.lambda_kl)
    cfg.INJECT.LAMBDA_MSE = float(args.lambda_mse)
    cfg.INJECT.T = float(args.temp)

    # concat-specific formatting (optional, but good to have)
    if hasattr(args, "concat_sep"):
        cfg.INJECT.CONCAT_SEP = args.concat_sep
    else:
        cfg.INJECT.CONCAT_SEP = "; "

    # (optional) print a short fingerprint in log
    print(
        f"[CFG] INJECT.MODE={cfg.INJECT.MODE} | FORM={cfg.SEMANTIC.FORM} | "
        f"USE_ALL={cfg.SEMANTIC.USE_ALL} | IDX={cfg.SEMANTIC.IDX} | "
        f"MULTI_REDUCE={cfg.SEMANTIC.MULTI_REDUCE} | "
        f"lam_kl={cfg.INJECT.LAMBDA_KL} | lam_mse={cfg.INJECT.LAMBDA_MSE} | T={cfg.INJECT.T}"
    )


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)
    if args.dataset:
        cfg.merge_from_file(f"configs/datasets/{args.dataset}.yaml")
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    reset_cfg(cfg, args)
    cfg.OUTPUT_DIR = f"output/{args.dataset}/beta:{args.beta}/{args.trainer}"
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg
