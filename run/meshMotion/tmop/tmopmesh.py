import torch
import torch.nn as nn
from torch.func import jacrev, vmap
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import sys
import shutil
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
        # self.el_dataset = TensorDataset(torch.range(0,self.n_elements-1,dtype = torch.int))
        self.el_dataset = TensorDataset(self.interior_ids)


        shape_config = config.shape
        self.shape = SHAPE_REGISTRY[shape_config.shape_fn]
        self.ref_type = shape_config.reference_type
        valid_refs = ["regular", "initial", "svd"]
        if self.ref_type not in valid_refs:
            raise ValueError(f"Invalid reference type {self.ref_type}. Choose one of {valid_refs}.")
        self._make_ref_elements()

        self.n_sample_pts = shape_config.n_samples
        self._sample_ref_element()

        # Register buffer for shape gradients, so that 
        # they can be automatically moved to GPU if needed.
        self._evaluate_shape_grad()

        # Gather all shape gradients per physical element
        # and per sample point. Pad rows with zero. Needed
        # for jacobian computation without loops.
        # TODO: consider differentiating by element type.

        #NOTE: probably do not need to gather all if we differentiate by 
        # element type. 
        shape_grad_all = torch.zeros((self.n_elements, self.n_sample_pts, self.elements.shape[1], self.spacedim))
        for ns in self.n_sides_unique:
            mask = self.n_sides == ns
            g = getattr(self, self.shape_grad[ns.item()])
            shape_grad_all[mask,:,:ns] = g
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
        self.compute_metric = self._compute_metric_untangle if self.untangle else self._compute_metric

        # Set the target W.
        target_config = config.target
        self._make_target(target_config)
        self.register_buffer("W_inv", torch.linalg.inv(self.W))

        # Allows smartsim driver to log progress of optimisation to stdout
        if log_client:
            self.log_client = log_client

    @property
    def pts(self):
        return torch.cat([self.bd_pts, self.int_pts], dim=0)[self.inverse_perm]

    def _make_ref_elements(self):
        self.ref_elements = {}
        match self.ref_type :
            case "regular":
                r = 1.
                for ns in self.n_sides_unique:
                    self.ref_elements[ns.item()] = torch.tensor([(r*cos(2*torch.pi*i/ns),r*sin(2*torch.pi*i/ns)) for i in range(1,ns+1)])
            case "initial":
                for ns in self.n_sides_unique:
                    mask = (self.n_sides == ns)
                    phys = self.pts[self.elements[mask]][:,:ns]
                    self.ref_elements[ns.item()] = phys
            case "svd":
                for ns in self.n_sides_unique:
                    mask = (self.n_sides == ns)
                    phys = self.pts[self.elements[mask]][:,:ns]
                    U,S,V = torch.linalg.svd(phys - phys.mean(dim=1).unsqueeze(1), full_matrices = False)
                    ref = (U@V)
                    self.ref_elements[ns.item()] = ref

    def _sample_ref_element(self):
        # Sample uniformly from a unit trianlge
        u1,u2 = torch.rand(size=(2,self.n_sample_pts), dtype=torch.float64)
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

    def _compute_metric_untangle(self, T):
        tau = torch.linalg.det(T)
        min_tau = tau.min().item()
        self.t = 0.0 if min_tau > 0.0 else min_tau * ( 1 + self.c)

        mu_hat = 0.5*self.mu(T)/(tau - self.t)
        max_mu_hat = mu_hat.max().item()
        # self.beta = min(self.beta, max_mu_hat + self.d)
        self.beta =  max_mu_hat * (1 + self.d)
        assert torch.all(tau > self.t), f"Invalid t: {self.t} > {tau.min().item()}"
        assert torch.all(self.beta > mu_hat), f"Invalid beta: {self.beta} < {mu_hat.max().item()}"

        # Absolute priority to untangling.
        # If untangled, then t = 0, and we can start
        # checking if we made improvements on the worst
        # element.
        if self.t >= 0:
            if self.beta < self.min_beta and abs((self.beta-self.min_beta)/self.beta) > self.rtol:
                self.min_beta = self.beta
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1

        # wcuo as in Worst Case Untangle Optimise (or something like that)
        wcuo = mu_hat/(self.beta - mu_hat)
        return wcuo.mean()

    def _compute_metric(self, T):
        return self.mu(T).mean()

    def optimise(self, optimiser, scheduler, config):
        max_epochs = config.max_steps
        patience = config.patience
        self.rtol = config.rtol
        batch_size = self.n_int_pts if config.batch_size == -1 else config.batch_size
        assert batch_size > 0, f"Invalid batch size {batch_size}. Values need to be either -1 (no batches) or positive."
        el_dataloader = DataLoader(self.el_dataset, batch_size=batch_size, shuffle=True)
        n_batches = len(el_dataloader)

        self.t = 0 # tracks minimum tau
        self.beta = torch.inf # tracks maximum mu_hat
        self.min_beta = torch.inf
        self.epochs_since_improvement = 0
        for epoch in range(max_epochs):
            if self.epochs_since_improvement > patience:
                break
            for batch, indices in enumerate(el_dataloader):
                loss = 0
                indices = indices[0]
                el_mask = torch.isin(self.elements, indices)
                el = self.elements[el_mask.any(dim=1)]
                grads = self.shape_grad_all[el_mask.any(dim=1)]

                # A = shape_grad @ pts[elements], with the correct grad selected depending on the type of element
                pts_all = self.pts[el]
                A = torch.einsum("epsi,esj->epij", grads, pts_all) # (n_elements, n_samples, n_sides, spacedim), (n_elements, n_sides, spacedim) -> (n_elements, n_samples, spacedim, spacedim)

                T = (A @ self.W_inv[el_mask.any(dim=1)]).reshape(-1,2,2)

                loss = self.compute_metric(T)

                if not loss < torch.inf:
                    print("Loss blew up. Stopping.")
                    break

                optimiser.zero_grad()
                loss.backward()
                mask = torch.isin(self.interior_ids,indices)
                self.int_pts.grad[~mask].zero_()
                optimiser.step()
                if self.t >= 0:
                    try:
                        scheduler.step()
                    except:
                        scheduler.step(metrics=self.beta)

                # log stuff
                self._log(epoch, max_epochs, batch, n_batches, loss, self.t, self.beta, scheduler.get_last_lr()[0])
        print()

    def _log(self, epoch, max_epochs, batch, n_batches, loss, t, beta, lr):
        ecol = "\033[38;2;61;69;106m"
        bcol = "\033[48;2;0;147;155m"
        dfg = "\033[97m"
        dbg = "\033[107m"
        rcol = "\033[0m"

        blen = int(shutil.get_terminal_size().columns/5)
        eplen = int((epoch+1)/max_epochs*blen + 0.5)
        bplen = int((batch+1)/n_batches*blen + 0.5)

        if eplen >= bplen:
            fill = (ecol + bcol + "\u2580") * bplen + rcol + (dbg+ecol+"\u2580") * (eplen-bplen) + rcol + (dfg+dbg+"\033[7m\u2588") * (blen - eplen) + rcol
        elif eplen < bplen:
            fill = (ecol + bcol + "\u2580") * eplen + rcol + (dfg+bcol+"\u2580") * (bplen-eplen) + rcol + (dfg+dbg+"\033[7m\u2588") * (blen - bplen) + rcol

        barstr = "\u2595" + fill + "\u258F"
        logstr =  barstr + f"\033[38;2;61;69;106m epoch {epoch+1}\033[0m,\033[38;2;0;147;155m batch {batch+1} \033[0m" + f"-- loss = {loss.item():0.3e};\x1b[38;2;172;82;37m t = {t:0.3e}; beta = {beta:0.3e}; lr = {lr:0.3e}\x1b[0m"
        if self.log_client:
            self.log_client.put_tensor("tmop_string", np.frombuffer(logstr.encode('utf-8'),dtype=np.uint8))
            self.log_client.put_tensor("tmop_epoch", np.array([epoch]))
        if sys.stdout.isatty():
            logstr =  barstr + f"\033[38;2;61;69;106m epoch {epoch+1}\033[0m,\033[38;2;0;147;155m batch {batch+1} \033[0m" + f"-- loss = {loss.item():0.3e};\x1b[38;2;172;82;37m t = {t:0.3e}; beta = {beta:0.3e}; lr = {lr:0.3e}\x1b[0m"
            print('\x1b[k'+logstr+'\r',end='')
        else:
            logstr = f"epoch = {epoch}, batch = {batch}, loss = {loss.item():0.3e}, t = {t:0.3e}, beta = {beta:0.3e}, lr = {lr:0.3e}"
            print(logstr)
