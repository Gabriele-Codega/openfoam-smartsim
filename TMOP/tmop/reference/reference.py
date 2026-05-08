import torch
import torch.nn as nn
from torch.func import jacrev, vmap

from math import sin, cos

from ..mesh import Mesh
from ..registries import SHAPE_REGISTRY

class Reference(nn.Module):
    def __init__(self,
                 mesh: Mesh,
                 config):
        super().__init__()
        self.mesh = mesh

        self.shape = SHAPE_REGISTRY[config.shape_fn]
        self.ref_type = config.reference_type
        valid_refs = ["regular", "initial", "svd"]
        if self.ref_type not in valid_refs:
            raise ValueError(f"Invalid reference type {self.ref_type}. Choose one of {valid_refs}.")
        self._make_ref_elements()

        self.n_sample_pts = config.n_samples
        # self._sample_ref_element()
        self._sample_ref_sides()

        # Register buffer for shape gradients, so that 
        # they can be automatically moved to GPU if needed.
        self._evaluate_shape_grad()

        # Gather all shape gradients per physical element
        # and per sample point. Pad rows with zero. Needed
        # for jacobian computation without loops.
        # TODO: consider differentiating by element type.

        #NOTE: probably do not need to gather all if we differentiate by 
        # element type. 
        shape_grad_all = torch.zeros((self.mesh.n_elements, self.n_sample_pts, self.mesh.elements.shape[1], self.mesh.spacedim))
        for ns in self.mesh.n_sides_unique:
            mask = self.mesh.n_sides == ns
            g = getattr(self, self.shape_grad[ns.item()])
            shape_grad_all[mask,:,:ns] = g
        self.register_buffer("shape_grad_all", shape_grad_all.detach())

    def _make_ref_elements(self):
        mesh = self.mesh
        self.ref_elements = {}
        match self.ref_type :
            case "regular":
                r = 1.
                for ns in mesh.n_sides_unique:
                    self.ref_elements[ns.item()] = torch.tensor([(r*cos(2*torch.pi*i/ns),r*sin(2*torch.pi*i/ns)) for i in range(1,ns+1)])
            case "initial":
                for ns in mesh.n_sides_unique:
                    mask = (mesh.n_sides == ns)
                    phys = mesh.pts[mesh.elements[mask]][:,:ns]
                    self.ref_elements[ns.item()] = phys
            case "svd":
                for ns in mesh.n_sides_unique:
                    mask = (mesh.n_sides == ns)
                    phys = mesh.pts[mesh.elements[mask]][:,:ns]
                    U,S,V = torch.linalg.svd(phys - phys.mean(dim=1).unsqueeze(1), full_matrices = False)
                    ref = (U@V)
                    self.ref_elements[ns.item()] = ref

    def _sample_ref_sides(self):
        self.sample_points = {}
        for n_sides, ref_el in self.ref_elements.items():
            # we want n_sample_points points in total. So we evenly distribute those in the n_sides triangles
            # in a round-robin fashion (i.e. optional small imbalance)
            n_pts_base = self.n_sample_pts//n_sides 
            rem = self.n_sample_pts % n_sides
            n_pts_per_side = torch.tensor([n_pts_base + (i < rem) for i in range(n_sides)],dtype=torch.int) #(n_sides,)

            batch_shape = ref_el.shape[:-2]
            idx = torch.arange(n_sides, dtype=torch.int)
            sides = torch.stack((idx,torch.roll(idx,1)),dim=-1) # (n_sides, 2)

            pts = ref_el[...,sides,:] 
            samples_list = []

            for i in range(n_sides):
                k = n_pts_per_side[i].item()

                # evenly spaced in (0,1)
                t = torch.arange(1, k + 1)
                t = (t / (k + 1)).view(*([1] * len(batch_shape)), k, 1)
                # shape: (..., k, 1) via broadcasting

                p0 = pts[..., i, 0, :]  # (..., dim)
                p1 = pts[..., i, 1, :]

                # expand to (..., k, dim)
                p0 = p0.unsqueeze(-2)
                p1 = p1.unsqueeze(-2)

                s = (1 - t) * p0 + t * p1  # (..., k, dim)

                samples_list.append(s)

            samples = torch.cat(samples_list, dim=-2)  # (..., n_sample_pts, dim)

            name = f"samples_{n_sides}"
            self.register_buffer(name, samples.detach())
            self.sample_points[n_sides] = name

    def _sample_ref_element(self):
        # Sample uniformly from a unit trianlge
        u1,u2 = torch.rand(size=(2,self.n_sample_pts))
        x = 1-torch.sqrt(u1)
        y = (1-x)*u2
        tri_samples = torch.stack([x, y], dim = -1)
        # Barycentric coordinates of samples
        tri_basis = torch.stack([1-tri_samples[...,0]-tri_samples[...,1], tri_samples[...,0], tri_samples[...,1]],dim=-1) #(n_sample_pts, 3)

        self.sample_points = {}
        for n_sides, ref_el in self.ref_elements.items():
            # we want n_sample_points points in total. So we evenly distribute those in the n_sides triangles
            # in a round-robin fashion (i.e. optional small imbalance)
            n_pts_base = self.n_sample_pts//n_sides 
            rem = self.n_sample_pts % n_sides
            n_pts_per_tri = torch.tensor([n_pts_base + (i < rem) for i in range(n_sides)],dtype=torch.int) #(n_sides,)

            triangles = torch.stack((torch.full((n_sides,), n_sides, dtype=torch.int),
                                     torch.arange(n_sides, dtype=torch.int),
                                     torch.roll(torch.arange(n_sides, dtype=torch.int),1)
                                     ),dim=-1) # (n_sides, 3)

            centre = torch.mean(ref_el, dim=-2, keepdim=True) # (..., 1, spacedim)
            pts = torch.cat((ref_el,centre),dim=-2) # (..., n_sides+1, spacedim)
            samples = torch.einsum("si,...tid -> ...tsd",tri_basis,pts[...,triangles,:]) # (..., n_sides, n_samples, spacedim)
            samples = torch.cat([samples[...,i,:n_pts_per_tri[i],:] for i in range(n_sides)], dim=-2) # (..., n_samples, spacedim)
            name = f"samples_{n_sides}"
            self.register_buffer(name, samples.detach())
            self.sample_points[n_sides] = name

    def _evaluate_shape_grad(self):
        _grad = jacrev(self.shape, argnums=0)
        _vgrad = vmap(vmap(_grad, in_dims=(0,None)),in_dims=(0,0)) if self.ref_type != "regular" else vmap(_grad, in_dims=(0,None))# inner vmap maps over samples, outer vmap maps over el
        self.shape_grad = {}
        for n_sides, ref_el in self.ref_elements.items():
            samples = getattr(self, self.sample_points[n_sides])
            name = f"_shape_grad_{n_sides}"
            self.register_buffer(name, _vgrad(samples, ref_el)) 
            self.shape_grad[n_sides] = name

