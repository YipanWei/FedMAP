import torch


def fedprox_penalty(model_weight, global_weight, mu):
    reference = global_weight.detach().to(
        device=model_weight.device,
        dtype=model_weight.dtype,
    )
    return (mu / 2) * torch.norm(model_weight - reference).pow(2)
