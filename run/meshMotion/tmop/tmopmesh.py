import torch
import torch.nn as nn
from torch.func import jacrev, vmap
import numpy as np

import sys
from math import sin, cos
from functools import partial
import inspect

from .configutils import TMOPConfig
from .shape_functions import SHAPE_REGISTRY
from .tmopmetrics import METRIC_REGISTRY

torch.set_default_dtype(torch.float64)

class TMOPMesh(nn.Module):
    def __init__(self, 
                 boundary_points, 
                 interior_points, 
                 boundary_ids,
                 interior_ids,
                 elements,
                 config: TMOPConfig = TMOPConfig(),
                 log_client = None 
                ):
        super().__init__()

        self.n_bd_pts = boundary_points.shape[0]
        self.n_int_pts = interior_points.shape[0]
        self.n_pts = self.n_bd_pts + self.n_int_pts
        self.spacedim = boundary_points.shape[-1]
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

        self.elements = elements # assume elements are a padded tensor of node ids
        self.n_elements = elements.shape[0]
        self.n_sides = torch.sum(elements >= 0, dim=1, dtype=torch.int)
        self.n_sides_unique, self.n_sides_count = torch.unique(self.n_sides, return_counts=True)


        shape_config = config.shape
        self.shape = SHAPE_REGISTRY[shape_config.shape_fn]
        self.regular_reference = shape_config.regular_ref
        if not self.regular_reference:
            raise ValueError("No support for non-regular reference elements yet. Set `regular_ref` to `True`.")
        self._make_ref_elements()

        self.n_sample_pts = shape_config.n_samples
        self._sample_ref_element()

        # These two register buffers for shape values and gradients
        # so that they can be automatically moved to GPU if needed.
        self._evaluate_shape()
        self._evaluate_shape_grad()

        # This a non-elegant solution to make things efficient on GPU.
        # Basically build a big tensor for shape gradients, where
        # all the gradients for all sample points, all elements of all 
        # shapes are stored, and pad that with zeros.
        # Essentially retrieve the buffers set in the previous call and
        # fill the big tensor.
        # TODO: consider differentiating by element type, and merge this 
        # and the call to _evaluate_shape_grad(). Also, not sure if 
        # shape_vals are actually needed.
        shape_grad_all = torch.zeros((self.n_elements, self.n_sample_pts, self.elements.shape[1], self.spacedim))
        for i in range(self.n_elements):
            n_sides = self.n_sides[i].item()
            g = getattr(self, self.shape_grad[n_sides])
            shape_grad_all[i,:,:n_sides] = g
        self.register_buffer("shape_grad_all", shape_grad_all.detach())

        metric_config = config.metric

        metric_fn = METRIC_REGISTRY[metric_config.metric_fn]
        gamma = metric_config.gamma
        sig = inspect.signature(metric_fn)
        metric_args = sig.parameters
        if ("gamma" in metric_args) and (gamma is not None):
            metric_fn = partial(metric_fn, gamma=gamma)
        elif ("gamma" in metric_args) and (gamma is None):
            print(f"Metric {metric_config.metric_fn} requires parameter `gamma` but got {gamma}. Defaulting to `gamma = 0.5`.")
            metric_fn = partial(metric_fn, gamma=0.5)
        elif ("gamma" not in metric_args) and (gamma is not None):
            print(f"Metric {metric_config.metric_fn} does not require parameter `gamma`. Ignoring supplied value.")

        self.mu = metric_fn
        self.untangle = metric_config.untangle
        # parameters for untangling metric
        self.c = metric_config.c
        self.d = metric_config.d

        # Set the target W.
        # NOTE: might want to implement a TargetFactory class
        # for finer control over the target (preserve original,
        # regular ngon, user defined...)
        target_config = config.target
        self._make_target(target_config)
        self.register_buffer("W_inv", torch.linalg.inv(self.W))

        # Allows smartsim driver to log progress of optimisation to stdout
        if log_client:
            self.log_client = log_client

    @property
    def pts(self):
        return torch.cat([self.bd_pts, self.int_pts], dim=0)[self.inverse_perm]

    def map_element(self, ref_coords, ref_element, physical_node_ids):
        return self.shape(ref_coords, ref_element) @ self.pts[physical_node_ids]

    def _make_ref_elements(self):
        if self.regular_reference:
            # ref is a regula r n-gon. Build a dict where key is number of sides
            # and val is the tensor of vertices.
            r = 1.
            self.ref_elements = {}
            for ns in self.n_sides.unique():
                self.ref_elements[ns.item()] = torch.tensor([(r*cos(2*torch.pi*i/ns),r*sin(2*torch.pi*i/ns)) for i in range(1,ns+1)])
        else:
            self.ref_elements = []
            for el in self.elements:
                phys = self.pts[el]
                U,S,V = torch.linalg.svd(phys - phys.mean(dim=0).unsqueeze(0), full_matrices = False)
                ref = (U[...,:2]@V)
                self.ref_elements.append(ref)

    def _sample_ref_element(self):
        # Sample uniformly from a unit trianlge
        u1,u2 = torch.rand(size=(2,self.n_sample_pts), dtype=torch.float64)
        x = 1-torch.sqrt(u1)
        y = (1-x)*u2
        tri_samples = torch.stack([x, y], dim = -1)
        # Barycentric coordinates of samples
        tri_basis = torch.stack([1-tri_samples[...,0]-tri_samples[...,1], tri_samples[...,0], tri_samples[...,1]],dim=-1)

        if self.regular_reference:
            # For each reference element, build a triangle between
            # pairs of consecutive vertices and barycenter. Map 
            # random samples to these triangles to cover the whole
            # element.
            self.sample_points = {}
            for n_sides, ref_el in self.ref_elements.items():
                # we want n_sample_points points in total. So we evenly distribute those in the n_sides triangles
                # in a round-robin fashion (i.e. optional small imbalance)
                n_pts_base = self.n_sample_pts//n_sides 
                rem = self.n_sample_pts % n_sides

                samples = []
                triangles = torch.tensor([[n_sides, i, (i+1)%n_sides] for i in range(n_sides)])
                for i, tri in enumerate(triangles):
                    n_pts_per_tri = n_pts_base + (i < rem)
                    pts = torch.vstack((ref_el,torch.tensor([0.,0.])))[tri]
                    samples.append(tri_basis[:n_pts_per_tri]@pts)

                name = f"samples_{n_sides}"
                self.register_buffer(name, torch.vstack(samples).detach())
                self.sample_points[n_sides] = name
        else:
            self.sample_points = []
            for i in range(self.n_elements):
                n_sides = self.n_sides[i]
                ref_el = self.ref_elements[i]
                # we want n_sample_points points in total. So we evenly distribute those in the n_sides triangles
                # in a round-robin fashion (i.e. optional small imbalance)
                n_pts_base = self.n_sample_pts//n_sides 
                rem = self.n_sample_pts % n_sides
                samples = []
                triangles = torch.tensor([[n_sides, i, (i+1)%n_sides] for i in range(n_sides)])
                for i, tri in enumerate(triangles):
                    n_pts_per_tri = n_pts_base + (i < rem)
                    pts = torch.vstack((ref_el,torch.tensor([0.,0.])))[tri] ## WARNING: assumes that the origin is inside the polygon, which is not always true. In general should be the centroid of the kernel.
                    samples.append(tri_basis[:n_pts_per_tri]@pts)

                name = f"samples_{i}"
                self.register_buffer(name, torch.vstack(samples).detach())
                self.sample_points.append(name)

    def _evaluate_shape(self):
        if self.regular_reference:
            self.shape_vals = {}
            for n_sides, ref_el in self.ref_elements.items():
                samples = getattr(self, self.sample_points[n_sides])
                name = f"_shape_vals_{n_sides}"
                self.register_buffer(name, vmap(self.shape, in_dims=(0,None))(samples, ref_el))
                self.shape_vals[n_sides] = name
        else:
            self.shape_vals = []
            for i in range(self.n_elements):
                ref_el = self.ref_elements[i]
                samples = getattr(self, self.sample_points[i])
                name = f"_shape_vals_{i}"
                self.register_buffer(name, vmap(self.shape, in_dims=(0,None))(samples, ref_el))
                self.shape_vals.append(name)
        
    def _evaluate_shape_grad(self):
        _grad = jacrev(self.shape, argnums=0) # NOTE: this is not stable wit jacfwd
        if self.regular_reference:
            self.shape_grad = {}
            for n_sides, ref_el in self.ref_elements.items():
                samples = getattr(self, self.sample_points[n_sides])
                name = f"_shape_grad_{n_sides}"
                self.register_buffer(name, vmap(_grad, in_dims=(0,None))(samples, ref_el))
                self.shape_grad[n_sides] = name
        else:
            self.shape_grad = []
            for i in range(self.n_elements):
                ref_el = self.ref_elements[i]
                samples = getattr(self, self.sample_points[i])
                name = f"_shape_grad_{i}"
                self.register_buffer(name, vmap(_grad, in_dims=(0,None))(samples, ref_el))
                self.shape_grad.append(name)

    def _make_target(self, config):
        pts_all = self.pts[self.elements]
        W0 = torch.einsum("epsi,esj->epij", self.shape_grad_all, pts_all).detach() # (n_elements, n_samples, n_sides, spacedim), (n_elements, n_sides, spacedim) -> (n_elements, n_samples, spacedim, spacedim)
        _ps = config.preserve_size
        _po = config.preserve_orientation
        _pq = config.preserve_skewness
        _pa = config.preserve_aspect

        if (_ps and _po and _pq and _pa):
            self.register_buffer("W",W0)
            return

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
            R = torch.eye(self.spacedim)

        if _pq :
            cos_phi = torch.linalg.vecdot(w1,w2)/(n1*n2)
            sin_phi = torch.linalg.vecdot(w1_orth,w2)/(n1*n2)
            Q = torch.stack([
                torch.stack([torch.ones_like(cos_phi) , cos_phi], dim = -1),
                torch.stack([torch.zeros_like(cos_phi), sin_phi], dim = -1)],
                dim = -2
            )
        else:
            Q = torch.eye(self.spacedim)

        if _pa :
            rho = torch.sqrt(n2/n1)
            delta = torch.stack([
                torch.stack([1./rho, torch.zeros_like(rho)], dim = -1),
                torch.stack([torch.zeros_like(rho), rho], dim = -1)],
                dim = -2
            )
        else:
            delta = torch.eye(self.spacedim)

        W = torch.sqrt(zeta)[...,None,None]*torch.matmul(R, torch.matmul(Q,delta))
        self.register_buffer("W", W.detach())

    def _mapping_jacobian_regular(self, element):
        n_sides = len(element)
        return torch.einsum("psd,se->pde", getattr(self,self.shape_grad[n_sides]), self.pts[element]) # (n_samples, n_sides, spacedim), (n_sides, spacedim) -> (n_samples, spacedim, spacedim)

    def _mapping_jacobian_svd(self, idx):
        shape = self.shape_grad[idx]
        x = self.pts[self.elements[idx]]
        return torch.einsum("psd,se->pde", shape, x) # (n_samples, n_sides, spacedim), (n_sides, spacedim) -> (n_samples, spacedim, spacedim)

    def optimise(self, optimiser, max_epochs, patience=50, rtol=1e-2):

        t = 0 # tracks minimum tau
        beta = 0 # tracks maximum mu_hat
        min_beta = torch.inf
        epochs_since_improvement = 0
        for epoch in range(max_epochs):
            if epochs_since_improvement > patience:
                break
            loss = 0
            # A = shape_grad @ pts[elements], with the correct grad selected depending on the type of element
            if self.regular_reference :
                pts_all = self.pts[self.elements]
                # GPU-friendly version
                A = torch.einsum("epsi,esj->epij", self.shape_grad_all, pts_all) # (n_elements, n_samples, n_sides, spacedim), (n_elements, n_sides, spacedim) -> (n_elements, n_samples, spacedim, spacedim)
            else:
                #WARNING: NOT WORKING
                As =  list(map(partial(torch.einsum,"psd,se->pde"), self.shape_grad , self.pts[self.elements])) 
                A = torch.vstack(As)

            T = (A @ self.W_inv).reshape(-1,2,2)

            if self.untangle:
                tau = torch.linalg.det(T)
                min_tau = tau.min().item()
                t = 0.0 if min_tau > 0.0 else min_tau * (1 + self.c)

                mu_hat = 0.5*self.mu(T)/(tau - t)
                max_mu_hat = mu_hat.max().item()
                #beta = max(beta, max_mu_hat * (1 + d))
                beta =  max_mu_hat * (1 + self.d)
                assert torch.all(tau > t), f"Invalid t: {t} > {tau.min().item()}"
                assert torch.all(beta > mu_hat), f"Invalid beta: {beta} < {mu_hat.max().item()}"

                # Absolute priority to untangling.
                # If untangled, then t = 0, and we can start
                # checking if we made improvements on the worst
                # element.
                if t >= 0:
                    if beta < min_beta and abs((beta-min_beta)/beta) > rtol:
                        min_beta = beta
                        epochs_since_improvement = 0
                    else:
                        epochs_since_improvement += 1

                # wcuo as in Worst Case Untangle Optimise (or something like that)
                wcuo = mu_hat/(beta - mu_hat)
                loss = wcuo.mean()
            else:
                loss = self.mu(T).mean()

            if not loss < torch.inf:
                print("Loss blew up. Stopping.")
                break

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()


            # log stuff
            plen = int((epoch/max_epochs)*40 + 0.5)
            barstr = "[" + u"\u2501" * plen + " " * (40-plen) + "]"
            if self.log_client:
                self.log_client.put_tensor("tmop_epoch", np.array([epoch]))
                self.log_client.put_tensor("tmop_loss", np.array([loss.detach().item()]))
                self.log_client.put_tensor("tmop_t", np.array([t]))
                self.log_client.put_tensor("tmop_beta", np.array([beta]))
                self.log_client.put_tensor("tmop_progress", np.array([plen]))
            if sys.stdout.isatty():
                logstr =  f"\x1b[38;2;61;69;106m{barstr} epoch {epoch} \x1b[0m" + f"\x1b[38;2;172;82;37m-- loss = {loss.item():0.3e}; t = {t:0.3e}; beta = {beta:0.3e}\x1b[0m"
                # logstr = logstr.ljust(200)
                print('\x1b[k'+logstr+'\r',end='')
            else:
                logstr = f"epoch = {epoch}, loss = {loss.item():0.3e}, t = {t:0.3e}, beta = {beta:0.3e}"
                print(logstr)
        print()
