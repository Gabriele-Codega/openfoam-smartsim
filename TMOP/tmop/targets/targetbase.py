import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..reference import Reference

class TargetBase(ABC):
    @abstractmethod
    def make_target(self, ref: "Reference")->torch.Tensor:
        pass
