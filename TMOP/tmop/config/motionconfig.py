from dataclasses import dataclass, field
from typing import Optional, Type

import torch

from ..optimisers import TMOPOptimiser, WCUOptimiser
from .tmopconfig import TMOPConfig

@dataclass
class MotionConfig:
    """Main configuration."""
    mpi_ranks: int = 1
    """Number of MPI ranks in OpenFOAM."""
    device: Optional[str] = None
    """Which accelerator should be used in PyTorch. Defaults to GPU ('cuda') if available, otherwise falls back to CPU ('cpu')."""
    dtype: torch.dtype = torch.float64
    """Default data type to use in PyTorch. Defaults to 'torch.float64'."""
    mode: str = "incremental"
    """Either 'incremental' or 'points0'. If 'incremental', the motion is computed with respect to the mesh at the previous time step. If 'points0', the motion is computed with respect to the initial mesh."""
    tmop_optimiser: Type[TMOPOptimiser] = WCUOptimiser
    """Which TMOP optimiser should be used. Either `TMOPOptimiser` for the standard, non-untangling version, or `WCUOptimiser` for the simultaneous Worst-Case Untangle Optimiser."""

    tmop: TMOPConfig = field(default_factory=TMOPConfig)
    """Dataclass to handle TMOP configuration."""
