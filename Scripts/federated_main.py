

from utils.fed_utils import (
    count_parameters,
    save_acc_csv,
)
from Dassl.dassl.utils import setup_logger, set_random_seed
from Dassl.dassl.engine import build_trainer
import setproctitle
import numpy as np
import argparse
import torch
import time
import copy

from utils.train_utils import TRAINER_METHODS

from utils.utils import print_args,setup_cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    args.para_dir = setup_logger(cfg)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    print_args(args, cfg)

    ckpt_path = args.para_dir + "/model.pth"
    local_weights = [[] for i in range(cfg.DATASET.USERS)]

    local_trainer = build_trainer(args, cfg)
    local_trainer.fed_before_train()
    # count_parameters(local_trainer.model, "prompt_learner")
    # count_parameters(local_trainer.model, "image_encoder")
    # count_parameters(local_trainer.model, "text_encoder")
    datanumber_client = []

    if args.trainer == "CLIP":
        global_weights = copy.deepcopy(local_trainer.model.state_dict())
    else:
        if args.aggregation == "Weight":
            for net_i in range(cfg.DATASET.USERS):
                datanumber_client.append(
                    len(local_trainer.fed_train_loader_x_dict[net_i].dataset)
                )
        elif args.aggregation == "Equal":
            for net_i in range(cfg.DATASET.USERS):
                datanumber_client.append(1 / cfg.DATASET.USERS)
        global_weights = copy.deepcopy(local_trainer.model.state_dict())

    start_epoch = 0
    end_epoch = cfg.OPTIM.ROUND
    global_test_acc_dict = {}
    global_time_list = []
    global_similarity_list = []
    local_similarity_list = []
    start = time.time()

    trainer_fn = TRAINER_METHODS[args.trainer]

    if args.trainer == "CLIP":
        global_weights, global_test_acc_dict = trainer_fn(
            args, cfg, 0, local_trainer, global_weights, local_weights,
            global_test_acc_dict, global_time_list, start,datanumber_client
        )

    elif args.trainer == "FOCoOP":
        # --------- FOCoOP needs persistent local prompts ctx_l for each client ----------
        local_prompt_states = {}        # dict[int] -> Tensor (prompt_learner.ctx_l)
        local_full_weights = {}         # dict[int] -> full model state_dict after local training

        for epoch in range(start_epoch, end_epoch):
            global_weights, global_test_acc_dict, local_prompt_states = trainer_fn(
                args, cfg, epoch, local_trainer,
                global_weights,
                local_full_weights,          # 注意：这里传 local_full_weights，而不是 local_weights
                global_test_acc_dict, global_time_list, start,
                datanumber_client,
                local_prompt_states=local_prompt_states
            )

            if args.resume:
                torch.save(local_trainer.model.state_dict(), ckpt_path)
    else:
        for epoch in range(start_epoch, end_epoch):
            global_weights, global_test_acc_dict = trainer_fn(
                args, cfg, epoch, local_trainer, global_weights, local_weights,
                global_test_acc_dict, global_time_list, start,datanumber_client
            )

            if args.resume:
                torch.save(local_trainer.model.state_dict(), ckpt_path)

    m = max(int(args.frac * cfg.DATASET.USERS), 1)
    idxs_users = list(range(m))
    for idx in idxs_users:
        local_trainer.fed_after_train()

    for key, global_test_acc_list in global_test_acc_dict.items():
        print(key, "global_test_acc_list:", global_test_acc_list)
        print(key, "maximum test acc:", max(global_test_acc_list))
        print(key, "mean of acc:", np.mean(global_test_acc_list[-5:]))
        print(key, "std of acc:", np.std(global_test_acc_list[-5:]))

    save_acc_csv(local_trainer.args.para_dir, global_test_acc_dict, cfg,start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer",
        type=str,
        default="FedMAP",
        help="name of trainer, choose from: "
        "CLIP, FedAPT, FedCLIP, FedCoCoOP, FedKgCoOP, FedProxLPT, FedMVP, FOCoOP, PROMPTFL, VPT, FedMAP",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fedisic",
        help="name of dataset, choose from: fedisic, fedcamelyon",
    )
    parser.add_argument(
        "--backbone", type=str, default="ViT-B/16", help="name of CNN backbone"
    )
    parser.add_argument(
        "--aggregation", type=str, default="Weight", help="name of Aggregation strategy"
    )
    parser.add_argument(
        "--device_id", type=int, default=0, help="The Device Id for Experiment"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="The parameter for the dirichlet distribution",
    )
    parser.add_argument("--num_users", type=int, default=-1, help="number of users: K; negative keeps dataset config")
    parser.add_argument(
        "--frac", type=float, default=1, help="the fraction of clients: C"
    )
    parser.add_argument("--gamma", type=float, default=1, help="gamma of single_step")
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="number of trainer batch size"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=128, help="number of test batch size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers. Use 0 in restricted environments.",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="only positive value enables a fixed seed"
    )


    parser.add_argument(
        "--partition",
        type=str,
        default="noniid-labeldir",
        help="the data partitioning strategy,"
        ' select from "noniid-labeluni, noniid-labeldir,noniid-labeldir100"',
    )
  
    parser.add_argument(
        "--logdir",
        type=str,
        required=False,
        default="./logs/",
        help="Log directory path",
    )
    parser.add_argument(
        "--root", type=str, default="./data/", help="path to dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/..", help="output directory"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument(
        "--eval-only", action="store_true", default=False, help="evaluation only"
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )

    parser.add_argument("--nctx",type=int,help = "Number of Learnable CTX",default=16)

    # ======= LLM semantic expansion / injection switches =======
    parser.add_argument(
        "--inject_mode",
        type=str,
        default="concat",
        choices=["concat", "kl", "mse"],
        help="Injection paradigm: concat|kl|mse",
    )

    parser.add_argument(
        "--semantic_form",
        type=str,
        default="attr",
        choices=["attr", "desc"],
        help="LLM semantic form: attr|desc",
    )

    # reading strategy: SINGLE (IDX) vs ALL
    parser.add_argument(
        "--semantic_use_all",
        action="store_true",
        default=False,
        help="If set, use ALL expansions per class (multi). If not set, use a single expansion by --semantic_idx.",
    )

    parser.add_argument(
        "--semantic_idx",
        type=int,
        default=0,
        help="When --semantic_use_all is False, pick one expansion at this index (supports -1 for last).",
    )

    # multi aggregation strategy (applies when semantic_use_all=True)
    parser.add_argument(
        "--multi_reduce",
        type=str,
        default="calc_then_mean",
        choices=["calc_then_mean", "mean_then_calc"],
        help="Multi-expansion aggregation: calc_then_mean (loss mean) | mean_then_calc (prob/mean then loss).",
    )

    # hyper-params
    parser.add_argument(
        "--lambda_kl",
        type=float,
        default=0.5,
        help="Weight for KL alignment loss (only for inject_mode=kl).",
    )

    parser.add_argument(
        "--lambda_mse",
        type=float,
        default=0.1,
        help="Weight for MSE alignment loss (only for inject_mode=mse).",
    )

    parser.add_argument(
        "--temp",
        type=float,
        default=2.0,
        help="Temperature T for KL (only for inject_mode=kl).",
    )

    # root path for CLS_Exp and embeddings
    parser.add_argument(
        "--semantic_root",
        type=str,
        default=".",
        help="Project root that contains CLS_Exp/ and embeddings/.",
    )

    parser.add_argument(
        "--diverse",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--prompt_depth",
        type=float,
        default=12,
    )


    parser.add_argument(
        "--kd_t",
        type=float,
        default=2,
    )

    parser.add_argument(
        "--nctxv",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lambda_struct",
        type=float,
        default=10,
    )

    parser.add_argument(
        "--proto_momentum",
        type=float,
        default=0.9,
        help="EMA momentum for prototype updates in FedMAP-style trainers.",
    )

    parser.add_argument(
        "--source",
        type=str,
        default="class_attr",
    )

    parser.add_argument(
        "--domain_p",
        type=float,
        default=-1.0,
        help="Override DOMAIN_P for percent-based intra-domain splitting; negative keeps dataset config.",
    )

    args = parser.parse_args()
    setproctitle.setproctitle(
        "{}_{}_{}".format(args.trainer, args.backbone, args.dataset)
    )
    main(args)
