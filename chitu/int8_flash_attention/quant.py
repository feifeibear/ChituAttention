import torch


def quant_pertoken(X):
    X_max, _ = torch.abs(X).max(dim=-1)
    X_scale = X_max / 127
    ret = torch.round(X / X_scale[:, :, :, None]).to(torch.int8)
    return ret, X_scale


def quant_pertensor(X):
    X_max, _ = torch.abs(X).max(dim=-1)
    X_max, _ = torch.max(X_max, dim=-1)
    X_scale = X_max / 127
    ret = torch.round(X / X_scale[:, :, None, None]).to(torch.int8)
    return ret, X_scale
