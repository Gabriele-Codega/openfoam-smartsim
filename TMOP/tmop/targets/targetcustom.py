import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..reference import Reference
from ..registries import TARGET_REGISTRY
from .targetbase import TargetBase

class TargetCustom(TargetBase):
    def __init__(self, config):
        self.config = config
        raise NotImplemented

    def _make_target(self, ref: "Reference"):
        pts_all = ref.mesh.pts[ref.mesh.elements]
        W0 = torch.einsum("epsi,esj->epij", ref.shape_grad_all, pts_all).detach() # (n_elements, n_samples, n_sides, spacedim), (n_elements, n_sides, spacedim) -> (n_elements, n_samples, spacedim, spacedim)
        _ps = self.config.preserve_size
        _po = self.config.preserve_orientation
        _pq = self.config.preserve_skewness
        _pa = self.config.preserve_aspect

        if (_ps and _po and _pq and _pa):
            return W0

        w1 = W0[...,0]
        w2 = W0[...,1]
        n1 = torch.linalg.norm(w1,dim=-1)
        n2 = torch.linalg.norm(w2,dim=-1)
        w1_orth = torch.stack((-w1[...,1], w1[...,0]),dim=-1)

        zeta = n1*n2 if _ps else torch.ones_like(n1)

        if _po :
            cos_theta = W0[...,0,0]/n1
            sin_theta = W0[...,1,0]/n1
            R = torch.stack([
                torch.stack([cos_theta, -sin_theta], dim = -1),
                torch.stack([sin_theta,  cos_theta], dim = -1)],
                dim = -2
            )
        else:
            R = torch.eye(ref.mesh.spacedim)

        if _pq :
            cos_phi = torch.linalg.vecdot(w1,w2)/(n1*n2)
            sin_phi = torch.linalg.vecdot(w1_orth,w2)/(n1*n2)
            Q = torch.stack([
                torch.stack([torch.ones_like(cos_phi) , cos_phi], dim = -1),
                torch.stack([torch.zeros_like(cos_phi), sin_phi], dim = -1)],
                dim = -2
            )
        else:
            Q = torch.eye(ref.mesh.spacedim)

        if _pa :
            rho = torch.sqrt(n2/n1)
            delta = torch.stack([
                torch.stack([1./rho, torch.zeros_like(rho)], dim = -1),
                torch.stack([torch.zeros_like(rho), rho], dim = -1)],
                dim = -2
            )
        else:
            delta = torch.eye(ref.mesh.spacedim)

        W = torch.sqrt(zeta)[...,None,None]*torch.matmul(R, torch.matmul(Q,delta))
        return W
