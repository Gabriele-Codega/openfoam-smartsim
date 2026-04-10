import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..reference import Reference
from ..registries import TARGET_REGISTRY
from .targetbase import TargetBase

@TARGET_REGISTRY.register()
class TargetInitial(TargetBase):
    def make_target(self, ref: "Reference"):
        pts_all = ref.mesh.pts[ref.mesh.elements]
        W0 = torch.einsum("...psi,...sj->...pij", ref.shape_grad_all, pts_all).detach() # (n_elements, n_samples, n_sides, spacedim), (n_elements, n_sides, spacedim) -> (n_elements, n_samples, spacedim, spacedim)
        return W0
