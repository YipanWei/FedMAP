# train_utils.py
import copy
import time
from collections import defaultdict
import torch


from utils.fed_utils import (
    average_weights,
    show_results,
    FOCOOP_KEYS_GLOBAL,
    FOCOOP_KEY_LOCAL,
    average_focoop_global_parts,
    overwrite_keys_,
    build_global_eval_state
)


def cpu_state_dict(state_dict):
    """Clone a model state_dict onto CPU to avoid accumulating GPU copies across clients."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}

# ------------------ 1) CLIP ------------------ #
def trainer_clip(args, cfg, epoch, local_trainer, global_weights,
                 local_weights, global_test_acc_dict, global_time_list, start,datanumber_client):
    print("------------Global test start -------------")

    m = max(int(args.frac * cfg.DATASET.USERS), 1)
    idxs_users = list(range(m))
    for idx in idxs_users:
        local_trainer.model.load_state_dict(global_weights, strict=False)

    results = local_trainer.test()
    global_test_acc, global_test_acc_dict = show_results(
        cfg, results, epoch, global_test_acc_dict
    )
    global_time_list.append(time.time() - start)

    print("------------Global test finish-------------")
    return global_weights, global_test_acc_dict

def trainer_clip_ob(args, cfg, epoch, local_trainer, global_weights,
                 local_weights, global_test_acc_dict, global_time_list, start,datanumber_client):

    m = max(int(args.frac * cfg.DATASET.USERS), 1)
    idxs_users = list(range(m))
    for idx in idxs_users:
        local_trainer.model.load_state_dict(global_weights, strict=False)

    results = local_trainer.test()
    global_test_acc, global_test_acc_dict = show_results(
        cfg, results, epoch, global_test_acc_dict
    )
    global_time_list.append(time.time() - start)

    print("------------Global test finish-------------")
    return global_weights, global_test_acc_dict


# ------------------ 2) VPTPR ------------------ #
def trainer_vptpr(args, cfg, epoch, local_trainer, global_weights,
                  local_weights, global_test_acc_dict, global_time_list, start,datanumber_client):

    m = max(int(args.frac * cfg.DATASET.USERS), 1)
    idxs_users = list(range(m))
    print("idxs_users", idxs_users)

    print("------------local train start epoch:", epoch, "-------------")
    for idx in idxs_users:
        local_trainer.model.load_state_dict(global_weights, strict=False)
        local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
        local_weights[idx] = cpu_state_dict(local_trainer.model.state_dict())
    print("------------local train finish epoch:", epoch, "-------------")

    global_weights = average_weights(local_weights, idxs_users, datanumber_client)

    print(f"------------{args.trainer}:Global test start-------------")
    for idx in range(cfg.DATASET.USERS):
        local_trainer.model.load_state_dict(global_weights, strict=False)

    results = local_trainer.test()
    _, global_test_acc_dict = show_results(cfg, results, epoch, global_test_acc_dict)
    global_time_list.append(time.time() - start)

    print("------------Global test finish-------------")

    fea_in = defaultdict(dict)
    fea_in[0] = torch.mm(local_trainer.model.image_encoder.VPT.T,
                         local_trainer.model.image_encoder.VPT)
    local_trainer.fea_in = fea_in

    return global_weights, global_test_acc_dict



# ------------------ 3) VPT / LPT / IVLP / PROMPTFL 共用 ------------------ #
def trainer_default_fed(args, cfg, epoch, local_trainer, global_weights,
                        local_weights, global_test_acc_dict, global_time_list, start,datanumber_client):

    m = max(int(args.frac * cfg.DATASET.USERS), 1)
    idxs_users = list(range(m))

    print("------------local train start epoch:", epoch, "-------------")
    for idx in idxs_users:
        local_trainer.model.load_state_dict(global_weights, strict=False)
        local_trainer.train(
            idx=idx, global_epoch=epoch, is_fed=True, global_weights=global_weights
        )
        local_weights[idx] = cpu_state_dict(local_trainer.model.state_dict())
    print("------------local train finish epoch:", epoch, "-------------")

    global_weights = average_weights(local_weights, idxs_users, datanumber_client)

    print(f"------------{args.trainer}:Global test start-------------")
    local_trainer.model.load_state_dict(global_weights, strict=False)
    results = local_trainer.test()

    _, global_test_acc_dict = show_results(cfg, results, epoch, global_test_acc_dict)
    global_time_list.append(time.time() - start)

    print("------------Global test finish-------------")
    return global_weights, global_test_acc_dict


# ------------------ 3) VPT / LPT / IVLP / PROMPTFL 共用 ------------------ #
def trainer_tsne_fed(args, cfg, epoch, local_trainer, global_weights,
                        local_weights, global_test_acc_dict, global_time_list, start,datanumber_client):

    m = max(int(args.frac * cfg.DATASET.USERS), 1)
    idxs_users = list(range(m))

    print("------------local train start epoch:", epoch, "-------------")
    for idx in idxs_users:
        local_trainer.model.load_state_dict(global_weights, strict=False)
        local_trainer.train(
            idx=idx, global_epoch=epoch, is_fed=True, global_weights=global_weights
        )
        local_weights[idx] = cpu_state_dict(local_trainer.model.state_dict())
    print("------------local train finish epoch:", epoch, "-------------")

    global_weights = average_weights(local_weights, idxs_users, datanumber_client)

    print(f"------------{args.trainer}:Global test start-------------")
    local_trainer.model.load_state_dict(global_weights, strict=False)
    results = local_trainer.test()

    _, global_test_acc_dict = show_results(cfg, results, epoch, global_test_acc_dict)
    global_time_list.append(time.time() - start)
    local_trainer.save_test_image_embeddings_after_test(round_idx = epoch)

    print("------------Global test finish-------------")
    return global_weights, global_test_acc_dict

def trainer_default_fed_focoop(
        args, cfg, epoch, local_trainer,
        global_weights,
        local_full_weights,              # dict[idx] -> full state_dict after local train
        global_test_acc_dict, global_time_list, start,
        datanumber_client,               # 你说先不考虑权重，这里保留接口
        local_prompt_states=None         # dict[idx] -> ctx_l
):
    """
    Simplified FOCoOp federated loop:
      - Downlink: ctx_g/ctx_o from global_weights + client-specific ctx_l
      - Client train: updates ctx_l/ctx_g/ctx_o locally
      - Uplink: save ctx_l to local_prompt_states, save full state to local_full_weights
      - Server agg: average ctx_g/ctx_o (equal weights) -> update global_weights
      - Global test: use ctx_l = ctx_g (simplified)
    """

    if local_prompt_states is None:
        local_prompt_states = {}

    m = max(int(args.frac * cfg.DATASET.USERS), 1)
    idxs_users = list(range(m))  # 你现在是顺序取前 m 个

    # --------- init local ctx_l if not exists ----------
    # 如果某个客户端还没保存过 ctx_l，就用 global_weights 里的 ctx_l 初始化
    if FOCOOP_KEY_LOCAL not in global_weights:
        raise KeyError(f"global_weights missing key: {FOCOOP_KEY_LOCAL}. "
                       f"Make sure PromptLearner defines ctx_l in state_dict.")

    for idx in idxs_users:
        if idx not in local_prompt_states:
            local_prompt_states[idx] = global_weights[FOCOOP_KEY_LOCAL].detach().clone()

    print("------------local train start epoch:", epoch, "-------------")

    for idx in idxs_users:
        # 1) 下发：先加载 global_weights（含 ctx_g/ctx_o + 旧的 ctx_l）
        local_trainer.model.load_state_dict(global_weights, strict=False)

        # 2) 覆盖为该客户端的 ctx_l（个性化 local）
        #    注意：这里直接 load_state_dict 局部覆盖也可以，但更简单是直接写 model.state_dict()
        sd = local_trainer.model.state_dict()
        sd[FOCOOP_KEY_LOCAL] = local_prompt_states[idx].to(sd[FOCOOP_KEY_LOCAL].device)
        local_trainer.model.load_state_dict(sd, strict=False)

        # 3) 本地训练
        local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True, global_weights=global_weights)

        # 4) 回传：保存该客户端最新的 ctx_l，保存完整 state_dict（用于提取 ctx_g/ctx_o）
        new_sd = copy.deepcopy(local_trainer.model.state_dict())
        local_full_weights[idx] = new_sd
        local_prompt_states[idx] = new_sd[FOCOOP_KEY_LOCAL].detach().cpu().clone()

    print("------------local train finish epoch:", epoch, "-------------")

    # --------- server aggregation: only ctx_g/ctx_o ----------
    focoop_global_avg = average_focoop_global_parts(local_full_weights, idxs_users)

    # 用聚合结果更新 global_weights（其他参数保持不变）
    global_weights = copy.deepcopy(global_weights)
    overwrite_keys_(global_weights, focoop_global_avg, FOCOOP_KEYS_GLOBAL)

    # --------- global test (simplified) ----------
    print(f"------------{args.trainer}:Global test start-------------")

    # 测试时没有某个 client 的 ctx_l：简化令 ctx_l = ctx_g
    eval_weights = build_global_eval_state(global_weights)
    local_trainer.model.load_state_dict(eval_weights, strict=False)

    results = local_trainer.test()
    _, global_test_acc_dict = show_results(cfg, results, epoch, global_test_acc_dict)
    global_time_list.append(time.time() - start)

    print("------------Global test finish-------------")

    # 返回时把 local_prompt_states 也带出去（跨轮保存）
    return global_weights, global_test_acc_dict, local_prompt_states

def trainer_promptfl_observe(args, cfg, epoch, local_trainer, global_weights,
                        local_weights, global_test_acc_dict, global_time_list, start,datanumber_client):

    m = max(int(args.frac * cfg.DATASET.USERS), 1)
    idxs_users = list(range(m))

    print("------------local train start epoch:", epoch, "-------------")
    for idx in idxs_users:
        local_trainer.model.load_state_dict(global_weights, strict=False)
        local_trainer.train(
            idx=idx, global_epoch=epoch, is_fed=True, global_weights=global_weights
        )

        local_weights[idx] = cpu_state_dict(local_trainer.model.state_dict())
    print("------------local train finish epoch:", epoch, "-------------")

    global_weights = average_weights(local_weights, idxs_users, datanumber_client)

    print(f"------------{args.trainer}:Global test start-------------")
    local_trainer.model.load_state_dict(global_weights, strict=False)
    local_trainer.save_global_text_embedding(round_id=epoch)

    results = local_trainer.test()

    _, global_test_acc_dict = show_results(cfg, results, epoch, global_test_acc_dict)
    global_time_list.append(time.time() - start)

    print("------------Global test finish-------------")
    return global_weights, global_test_acc_dict



# ------------------ 4) 注册表 ------------------ #
TRAINER_METHODS = {
    "CLIP": trainer_clip,
    "CLIP_OB":trainer_clip_ob,
    "VPTPR": trainer_vptpr,
    "VPT": trainer_tsne_fed,
    "LPT": trainer_default_fed,
    "IVLP": trainer_default_fed,
    "FedAPT":trainer_default_fed,
    "FedCLIP":trainer_tsne_fed,
    "FedCoCoOP":trainer_default_fed,
    "FedKgCoOP":trainer_default_fed,
    "FedProxLPT":trainer_default_fed,
    "FedTPG":trainer_default_fed,
    "PROMPTFL": trainer_default_fed,
    "PROMPTFL_OB":trainer_promptfl_observe,
    "PROMPTFL_Exp":trainer_promptfl_observe,
    "GL_SVDMSE_HE":trainer_default_fed,
    "FOCoOP":trainer_default_fed_focoop,
    "FedMVP":trainer_default_fed,
    "PROMPTFL_KL_Global":trainer_promptfl_observe,
    "PROMPTFL_Anchor":trainer_default_fed,
    "PROMPTFL_KL":trainer_promptfl_observe,
    "PROMPTFL_KL_Anchor":trainer_default_fed,
    "PROMPTFL_Anchor2":trainer_default_fed,
    "PROMPTFL_Anchor3":trainer_default_fed,
    "PROMPTFL_KL_VPT":trainer_default_fed,
    "VPT_a":trainer_default_fed,
    "VPT_LPT":trainer_tsne_fed,
    "PROMPTFL_KL_VPT_Inter":trainer_default_fed,
    "VPT_Ma":trainer_default_fed,
    "VPT_M":trainer_default_fed,
    "VPT_Ma_KL":trainer_default_fed,
    "VPT_Ma_T":trainer_tsne_fed,
    "VPT_Ma_One":trainer_default_fed,
    "VPT_Ma_R":trainer_default_fed,
    "VPT_Ma_A":trainer_default_fed,
    "VPT_Ma_Cos":trainer_default_fed,
    "VPT_Ma_Wass":trainer_default_fed,
    "VPT_Ma_Noisy":trainer_default_fed
}
