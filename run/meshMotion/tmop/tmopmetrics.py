import torch
from .registries import register_metric

@register_metric
def tmop_metric(T):
    arg = T - torch.eye(2).unsqueeze(0)
    return torch.linalg.matrix_norm(arg)**2 #/ torch.linalg.det(T).unsqueeze(0)

@register_metric
def mu_2(T):
    """
    Barrier shape
    """
    return 0.5 * torch.linalg.matrix_norm(T)**2 / torch.linalg.det(T) - 1

@register_metric
def mu_7(T):
    Tt = T.transpose(-2,-1)
    arg = T - torch.linalg.inv(Tt)
    return torch.linalg.matrix_norm(arg)**2

@register_metric
def mu_9(T):
    Tt = T.transpose(-2,-1)
    arg = T - torch.linalg.inv(Tt)
    return torch.linalg.matrix_norm(arg)**2 * torch.linalg.det(T).unsqueeze(0)

@register_metric
def mu_4(T):
    """
    Non-barrier shape
    """
    return torch.linalg.matrix_norm(T)**2 - 2 * torch.linalg.det(T)

@register_metric
def mu_55(T):
    """
    Non-barrier size
    """
    return (torch.linalg.det(T) - 1)**2

@register_metric
def mu_66(T, gamma=0.5):
    """
    Non-barrier shape + size
    """
    return (1-gamma) * mu_4(T) + gamma * mu_55(T)

@register_metric
def mu_77(T):
    """
    Barrier size
    """
    tau = torch.linalg.det(T)
    return 0.5 * (tau - 1/tau)**2

@register_metric
def mu_80(T, gamma = 0.5):
    """
    Barrier shape+size
    """
    return (1-gamma) * mu_2(T) + gamma * mu_77(T)
