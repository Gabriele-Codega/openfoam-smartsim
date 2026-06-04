import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

# torch.set_default_dtype(torch.float64)

class Mesh(nn.Module):
    def __init__(self, 
                 boundary_points, 
                 interior_points, 
                 boundary_ids,
                 interior_ids,
                 elements,
                 elements_area,
                ):
        super().__init__()

        self.n_bd_pts: int = boundary_points.shape[0]
        self.n_int_pts: int = interior_points.shape[0]
        self.n_pts: int = self.n_bd_pts + self.n_int_pts
        self.spacedim: int = boundary_points.shape[-1]
        if self.spacedim != 2:
            raise RuntimeError("Only 2D cases are supported. Got {self.spacedim:%d}D problem instead.")

        self.register_buffer( "bd_pts", boundary_points.detach()) # boundary points are buffers, so they are fixed
        self.int_pts = nn.Parameter(interior_points.detach()) # interior points are parameters, so they can be optimised
        self.boundary_ids = boundary_ids
        self.interior_ids = interior_ids
        # The property `pts` returns boundary and interior stacked.
        # Use this `inverse_perm` to shuffle the points so that they are
        # in the same order as their global id.
        storage_order = torch.cat([self.boundary_ids, self.interior_ids])
        self.register_buffer(
            "inverse_perm",
            torch.argsort(storage_order)
        )

        self.register_buffer("elements", elements) # assume elements are a padded tensor of node ids
        self.n_elements = elements.shape[0]
        self.n_sides = torch.sum(elements >= 0, dim=1, dtype=torch.int)
        self.n_sides_unique, self.n_sides_count = torch.unique(self.n_sides, return_counts=True)
        self.register_buffer("elements_area", elements_area.detach())
        self.register_buffer("elements_area_inv", 1./elements_area.detach())
        self.register_buffer("elements_area_inv_sum", self.elements_area_inv.sum().detach())

    @property
    def pts(self):
        return torch.cat([self.bd_pts, self.int_pts], dim=0)[self.inverse_perm]

    @property
    def pts_all(self):
        return self.pts[self.elements]
