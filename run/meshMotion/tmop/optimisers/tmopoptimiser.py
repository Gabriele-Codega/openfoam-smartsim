import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import sys
import shutil
from functools import partial
import inspect
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..config import TMOPConfig

from ..mesh import Mesh
from ..reference import Reference
from ..registries import METRIC_REGISTRY,TARGET_REGISTRY

class TMOPOptimiser(nn.Module):
    def __init__(self,
                 mesh: Mesh,
                 config: "TMOPConfig",
                 log_client = None 
                 ):
        super().__init__()
        self.mesh = mesh
        self.reference = Reference(mesh, config.reference)

        metric_config = config.metric
        self._setup_metric(metric_config)

        # Set the target W.
        target_config = config.target
        self._setup_target(target_config)

        optim_config = config.optim
        self._setup_optimisation(optim_config)

        self._log_dict = {"epoch": 0,
                          "max_epochs": self.max_epochs,
                          "batch": 0,
                          "n_batches": self.n_batches,
                          "loss": 0
                          }
        # Allows smartsim driver to log progress of optimisation to stdout
        if log_client:
            self.log_client = log_client

    def optimise(self):
        self._reset_optimisation()
        self.best = torch.inf
        self.n_bad_epochs = 0
        for epoch in range(self.max_epochs):
            self.batch_mean_loss = 0
            for batch, indices in enumerate(self.el_dataloader):
                loss = 0
                indices = indices[0]
                el_mask = torch.isin(self.mesh.elements, indices)
                el = self.mesh.elements[el_mask.any(dim=1)]
                grads = self.reference.shape_grad_all[el_mask.any(dim=1)]

                # A = shape_grad @ pts[elements], with the correct grad selected depending on the type of element
                pts_all = self.mesh.pts[el]
                A = torch.einsum("epsi,esj->epij", grads, pts_all) # (n_elements, n_samples, n_sides, spacedim), (n_elements, n_sides, spacedim) -> (n_elements, n_samples, spacedim, spacedim)
                try:
                    T = (A @ self.W_inv[el_mask.any(dim=1)]).reshape(-1,2,2)
                except IndexError: # workaround to allow target to be the same for all elements (i.e. just one matrix)
                    T = (A @ self.W_inv).reshape(-1,2,2)

                loss = self._compute_metric(T)
                self.batch_mean_loss += loss

                self.optimiser.zero_grad()
                loss.backward()
                mask = torch.isin(self.mesh.interior_ids,indices)
                self.mesh.int_pts.grad[~mask].zero_()
                self.optimiser.step()

                # log stuff
                self._log_dict["epoch"] = epoch
                self._log_dict["batch"] = batch
                self._log_dict["loss"] = loss
                self._log_dict["t"] = self.t
                self._log_dict["beta"] = self.beta
                self._log_dict["lr"] = self.scheduler.get_last_lr()[0]
                self._log()

            self.batch_mean_loss /= self.n_batches
            try:
                self.scheduler.step()
            except:
                self.scheduler.step(metrics=self.batch_mean_loss)
            if self._should_stop():
                break
        print()

    def _compute_metric(self,T):
        return self.mu(T).mean()

    def _setup_metric(self, metric_config):
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

    def _setup_target(self, target_config):
        # try:
        self.target_factory = TARGET_REGISTRY[target_config.target_factory]()
        # except KeyError as e:
        #     e.add_note(f"Valid values for `target_factory` are {list(TARGET_FACTORY_REGISTRY.keys())}. Got '{target_config.target_factory}' instead.")
            # raise e
        W = self.target_factory.make_target(self.reference)
        self.register_buffer("W", W.detach())
        self.register_buffer("W_inv", torch.linalg.inv(self.W))

    def _setup_optimisation(self, optim_config):
        self.optimiser = optim_config.optimiser(self.parameters(), **optim_config.optimiser_kwargs)
        self.scheduler = optim_config.scheduler(self.optimiser, **optim_config.scheduler_kwargs)
        self._optim_state0 = self.optimiser.state_dict()
        self._sched_state0 = self.scheduler.state_dict()
        self.max_epochs = optim_config.max_steps
        self.patience = optim_config.patience
        self.rtol = optim_config.rtol
        self.batch_size = self.mesh.n_int_pts if optim_config.batch_size == -1 else optim_config.batch_size
        assert self.batch_size > 0, f"Invalid batch size {self.batch_size}. Values need to be either -1 (no batches) or positive."
        self.el_dataloader = DataLoader(self.mesh.el_dataset, batch_size=self.batch_size, shuffle=True)
        self.n_batches = len(self.el_dataloader)

    def _reset_optimisation(self):
        self.scheduler.load_state_dict(self._sched_state0)
        self.optimiser.load_state_dict(self._optim_state0)

    def _should_stop(self):
        assert self.batch_mean_loss < torch.inf, "Loss blew up, stopping."

        if (self.batch_mean_loss < self.best) and abs((self.batch_mean_loss - self.best)/self.best) > self.rtol:
            self.n_bad_epochs = 0
            self.best = self.batch_mean_loss
        else:
            self.n_bad_epochs += 1

        return self.n_bad_epochs > self.patience

    def _log(self):
        ecol = "\033[38;2;61;69;106m"
        bcol = "\033[48;2;172;82;37m"
        dfg = "\033[97m"
        dbg = "\033[107m"
        rcol = "\033[0m"

        epoch = self._log_dict["epoch"]
        max_epochs = self._log_dict["max_epochs"]
        batch = self._log_dict["batch"]
        n_batches = self._log_dict["n_batches"]
        loss = self._log_dict["loss"]

        blen = int(shutil.get_terminal_size().columns/5)
        eplen = int((epoch+1)/max_epochs*blen + 0.5)
        bplen = int((batch+1)/n_batches*blen + 0.5)

        if eplen >= bplen:
            fill = (ecol + bcol + "\u2580") * bplen + rcol + (dbg+ecol+"\u2580") * (eplen-bplen) + rcol + (dfg+dbg+"\033[7m\u2588") * (blen - eplen) + rcol
        elif eplen < bplen:
            fill = (ecol + bcol + "\u2580") * eplen + rcol + (dfg+bcol+"\u2580") * (bplen-eplen) + rcol + (dfg+dbg+"\033[7m\u2588") * (blen - bplen) + rcol

        barstr = "\u2595" + fill + "\u258F"
        other = ""
        for k,v in self._log_dict.items():
            if k not in ["epoch", "max_epochs", "batch", "n_batches", "loss"]:
                other += f"; {k} = {v:0.3e}"

        logstr =  barstr + f"\033[38;2;61;69;106m epoch {epoch+1}\033[0m,\033[38;2;172;82;37m batch {batch+1} \033[0m" + f"-- loss = {float(loss):0.3e}"+ other 
        if self.log_client:
            self.log_client.put_tensor("tmop_string", np.frombuffer(logstr.encode('utf-8'),dtype=np.uint8))
            self.log_client.put_tensor("tmop_epoch", np.array([epoch]))
        if sys.stdout.isatty():
            logstr =  barstr + f"\033[38;2;61;69;106m epoch {epoch+1}\033[0m,\033[38;2;172;82;37m batch {batch+1} \033[0m" + f"-- loss = {float(loss):0.3e}"+ other 
            print('\x1b[k'+logstr+'\r',end='')
        else:
            logstr = f"epoch = {epoch}; batch = {batch}; loss = {float(loss):0.3e}" + other
            print(logstr)
