from dataclasses import dataclass

@dataclass
class ReferenceConfig:
    """
    Configure the reference element.
    """
    shape_fn: str = "stable_mean_value_coordinates"
    """Name of the shape function. Should be defined in `shape_functions.py`."""
    reference_type: str = "regular"
    """What reference element should be used. Must be one of ['regular', 'initial', 'svd']"""
    n_samples: int = 10
    """Number of sample points per element."""

