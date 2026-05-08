import torch
from ..registries import METRIC_REGISTRY

def det2x2(T):
    row0, row1 = T.unbind(-2)   # each (..., 2)
    return row0[..., 0] * row1[..., 1] - row0[..., 1] * row1[..., 0]
    # return T[..., 0, 0]*T[..., 1, 1] - T[..., 0, 1]*T[..., 1, 0]

@METRIC_REGISTRY.register()
def tmop_metric(T):
    arg = T - torch.eye(2).unsqueeze(0)
    return torch.linalg.matrix_norm(arg)**2 #/ det2x2(T).unsqueeze(0)

@METRIC_REGISTRY.register()
def mu_2(T):
    """
    Barrier shape
    """
    return 0.5 * torch.linalg.matrix_norm(T)**2 / det2x2(T) - 1

@METRIC_REGISTRY.register()
def mu_7(T):
    Tt = T.transpose(-2,-1)
    arg = T - torch.linalg.inv(Tt)
    return torch.linalg.matrix_norm(arg)**2

@METRIC_REGISTRY.register()
def mu_9(T):
    Tt = T.transpose(-2,-1)
    arg = T - torch.linalg.inv(Tt)
    return torch.linalg.matrix_norm(arg)**2 * det2x2(T).unsqueeze(0)

@METRIC_REGISTRY.register()
def mu_4(T):
    """
    Non-barrier shape
    """
    return (T*T).sum(dim=(-2,-1)) - 2 * det2x2(T)

@METRIC_REGISTRY.register()
def mu_55(T):
    """
    Non-barrier size
    """
    return (det2x2(T) - 1)**2

@METRIC_REGISTRY.register()
def mu_66(T, gamma=0.5):
    """
    Non-barrier shape + size
    """
    return (1-gamma) * mu_4(T) + gamma * mu_55(T)

@METRIC_REGISTRY.register()
def mu_77(T):
    """
    Barrier size
    """
    tau = det2x2(T)
    return 0.5 * (tau - 1/tau)**2

@METRIC_REGISTRY.register()
def mu_80(T, gamma = 0.5):
    """
    Barrier shape+size
    """
    return (1-gamma) * mu_2(T) + gamma * mu_77(T)
