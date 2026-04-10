import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..reference import Reference
from ..registries import TARGET_REGISTRY
from .targetbase import TargetBase

@TARGET_REGISTRY.register()
class TargetIdentity(TargetBase):
    def make_target(self, ref: "Reference"):
        return torch.eye(ref.mesh.spacedim).detach()
