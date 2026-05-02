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

    supported_trainers = {
        "CLIP",
        "VPT",
        "FedAPT",
        "FedCLIP",
        "FedCoCoOP",
        "FedKgCoOP",
        "FedProxLPT",
        "PROMPTFL",
        "FOCoOP",
        "FedMVP",
        "VPT_Ma",
    }
    if args.trainer not in supported_trainers:
        raise ValueError(
            f"Unsupported trainer '{args.trainer}'. "
            f"Choose from: {', '.join(sorted(supported_trainers))}"
        )

    cfg.TRAINER.CLIP = CN()
    cfg.TRAINER.CLIP.PREC = "fp16"
    cfg.TRAINER.CLIP.CLASS_TOKEN_POSITION = "end"

    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION =args.nctxv
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT.PREC = "fp16"
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = args.prompt_depth

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

    cfg.TRAINER.VPT_Ma = CN()
    cfg.TRAINER.VPT_Ma.N_CTX_VISION =args.nctxv
    cfg.TRAINER.VPT_Ma.CTX_INIT = "a photo of a"
    cfg.TRAINER.VPT_Ma.PREC = "fp16"
    cfg.TRAINER.VPT_Ma.PROMPT_DEPTH_VISION = args.prompt_depth
    cfg.TRAINER.VPT_Ma.lambda_struct = args.lambda_struct
    cfg.TRAINER.VPT_Ma.PROTO_MOMENTUM = args.proto_momentum
    cfg.TRAINER.VPT_Ma.STRUCT_LOSS = "mse"

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
