import sys
import torch
import torch.nn as nn
import numpy as np
from torch.func import jacrev, vmap
from math import sin, cos
from functools import partial

torch.set_default_dtype(torch.float64)

class TMOPMesh(nn.Module):
    def __init__(self, 
                 boundary_points, 
                 interior_points, 
                 boundary_ids,
                 interior_ids,
                 elements,
                 basis_functions = None,
                 regular_reference = True,
                 n_sample_points = 5,
                 tmop_metric = None,
                 untangle = False,
                 c = 1e-3,
                 d = 1e-3,
                 log_client = None 
                ):
        super().__init__()

        self.n_bd_pts = boundary_points.shape[0]
        self.n_int_pts = interior_points.shape[0]
        self.n_pts = self.n_bd_pts + self.n_int_pts

        self.spacedim = boundary_points.shape[-1]
        self.regular_reference = regular_reference
        if regular_reference == False:
            raise ValueError("Non-regular reference not quite supported yet.")
        if self.spacedim == 3 and regular_reference == True:
            print("Regular n-gon as reference is not supported in 3D. Falling back to SVD.")
            self.regular_reference = False


        # self.bd_pts = boundary_points.detach()
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
        # self.n_elements = len(elements)
        # self.n_sides = torch.tensor([len(el) for el in elements], dtype=torch.int64)
        self.n_elements = elements.shape[0]
        self.n_sides = torch.sum(elements >= 0, dim=1, dtype=torch.int)
        self.n_sides_unique, self.n_sides_count = torch.unique(self.n_sides, return_counts=True)

        self._make_ref_elements()

        self.basis = basis_functions
        self.n_sample_pts = n_sample_points
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

        self.mu = tmop_metric
        self.untangle = untangle
        # parameters for untangling metric
        self.c = c
        self.d = d

        # Set the target W somehow.
        # Either initialise to None, and set later in the main code,
        # or set W to match the initial mesh.
        # TODO:: add a flag or some runtime arg to control W creation.

        # self._W = None
        # set W as the jacobian of map from reference to original mesh
        pts_all = self.pts[self.elements]
        W = torch.einsum("epsi,esj->epij", self.shape_grad_all, pts_all) # (n_elements, n_samples, n_sides, spacedim), (n_elements, n_sides, spacedim) -> (n_elements, n_samples, spacedim, spacedim)
        self.register_buffer("W", W.detach())
        # self.register_buffer("W", torch.eye(self.spacedim))

        # Allows smartsim driver to log progress of optimisation to stdout
        if log_client:
            self.log_client = log_client

    @property
    def pts(self):
        return torch.cat([self.bd_pts, self.int_pts], dim=0)[self.inverse_perm]

    def map_element(self, ref_coords, ref_element, physical_node_ids):
        return self.basis(ref_coords, ref_element) @ self.pts[physical_node_ids]

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
                U,S,V = torch.linalg.svd(phys - phys.mean(dim=0).unsqueeze(0))
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
                # self.sample_points[n_sides] = getattr(self, name)
                # self.sample_points[n_sides] = torch.vstack(samples).detach()#.requires_grad_(True)
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
                # self.sample_points.append(getattr(self, name))
                # self.sample_points.append(torch.vstack(samples).detach())#.requires_grad_(True))

    def _evaluate_shape(self):
        if self.regular_reference:
            self.shape_vals = {}
            for n_sides, ref_el in self.ref_elements.items():
                # samples = self.sample_points[n_sides]
                samples = getattr(self, self.sample_points[n_sides])
                name = f"_shape_vals_{n_sides}"
                self.register_buffer(name, vmap(self.basis, in_dims=(0,None))(samples, ref_el))
                self.shape_vals[n_sides] = name
                # self.shape_vals[n_sides] = getattr(self, name)
                # self.shape_vals[n_sides] = vmap(self.basis, in_dims=(0,None))(samples, ref_el)
        else:
            self.shape_vals = []
            for i in range(self.n_elements):
                ref_el = self.ref_elements[i]
                # samples = self.sample_points[i]
                samples = getattr(self, self.sample_points[i])
                name = f"_shape_vals_{i}"
                self.register_buffer(name, vmap(self.basis, in_dims=(0,None))(samples, ref_el))
                self.shape_vals.append(name)
                # self.shape_vals.append(getattr(self, name))
                # self.shape_vals.append(vmap(self.basis, in_dims=(0,None))(samples, ref_el))
        
    def _evaluate_shape_grad(self):
        _grad = jacrev(self.basis, argnums=0) # NOTE: this is not stable wit jacfwd
        if self.regular_reference:
            self.shape_grad = {}
            for n_sides, ref_el in self.ref_elements.items():
                # samples = self.sample_points[n_sides]
                samples = getattr(self, self.sample_points[n_sides])
                name = f"_shape_grad_{n_sides}"
                self.register_buffer(name, vmap(_grad, in_dims=(0,None))(samples, ref_el))
                self.shape_grad[n_sides] = name
                # self.shape_grad[n_sides] = getattr(self, name)
                # self.shape_grad[n_sides] = vmap(_grad, in_dims=(0,None))(samples, ref_el)
        else:
            self.shape_grad = []
            for i in range(self.n_elements):
                ref_el = self.ref_elements[i]
                # samples = self.sample_points[i]
                samples = getattr(self, self.sample_points[i])
                name = f"_shape_grad_{i}"
                self.register_buffer(name, vmap(_grad, in_dims=(0,None))(samples, ref_el))
                self.shape_grad.append(name)
                # self.shape_grad.append(getattr(self, name))
                # self.shape_grad.append(vmap(_grad, in_dims=(0,None))(samples, ref_el))

    def _mapping_jacobian_regular(self, element):
        n_sides = len(element)
        # print(self.shape_grad[n_sides].device)
        return torch.einsum("psd,se->pde", getattr(self,self.shape_grad[n_sides]), self.pts[element]) # (n_samples, n_sides, spacedim), (n_sides, spacedim) -> (n_samples, spacedim, spacedim)

    def _mapping_jacobian_svd(self, idx):
        shape = self.shape_grad[idx]
        x = self.pts[self.elements[idx]]
        return torch.einsum("psd,se->pde", shape, x) # (n_samples, n_sides, spacedim), (n_sides, spacedim) -> (n_samples, spacedim, spacedim)

    # @property
    # def W(self):
    #     return self._W
    # @W.setter
    # def W(self,new_W):
    #     self._W = new_W

    def optimise(self, optimiser, max_epochs, patience=50, rtol=1e-2):
        W_inv = torch.linalg.inv(self.W) ## TODO: W should be a function of space as well (or something to allow for different W for different physical elements)
        
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

                # Slow python loop
                # A = torch.vstack([
                #         self._mapping_jacobian_regular(el)
                #         for el in self.elements
                #     ])
            else:
                #WARNING: NOT WORKING
                As =  list(map(partial(torch.einsum,"psd,se->pde"), self.shape_grad , self.pts[self.elements])) 
                A = torch.vstack(As)

            T = (A @ W_inv).reshape(-1,2,2)

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

                # no idea why it's called wcuo
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
