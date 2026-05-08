import torch
import torch.nn as nn
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
                          "loss": 0
                          }
        # Allows smartsim driver to log progress of optimisation to stdout
        if log_client:
            self.log_client = log_client

    def optimise(self):
        self._reset_optimisation()
        while not self._update_state():
            self.loss = self._step()
            self._adjust_lr()

            # log stuff
            self._log_dict["epoch"] = self.epoch
            self._log_dict["loss"]  = self.loss
            self._log()

    def _step(self):
        loss = 0
        T = self._compute_weighted_jacobian()
        _loss = self._compute_loss(T)
        loss = (_loss.mean(dim=-1)/self.mesh.elements_area).sum()
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        self.epoch += 1
        return loss

    def _compute_weighted_jacobian(self):
        A = torch.einsum("epsi,esj->epij", 
                         self.reference.shape_grad_all, 
                         self.mesh.pts_all) # (n_elements, n_samples, n_sides, spacedim), (n_elements, n_sides, spacedim) -> (n_elements, n_samples, spacedim, spacedim)
        return A @ self.W_inv

    def _compute_loss(self,T):
        return self.mu(T)

    def _adjust_lr(self):
        try:
            self.scheduler.step()
        except:
            self.scheduler.step(metrics=self.loss)

    def _update_state(self):
        if torch.any((self.best - self.loss)/self.loss> self.rtol):
            self.n_bad_epochs = 0
            self.best = self.beta
        else:
            self.n_bad_epochs += 1
            should_stop = (self.n_bad_epochs > self.patience) \
                        | (self.loss < self.stopping_threshold) \
                        | (self.epoch > self.max_epochs)

        return should_stop

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
        self.target_factory = TARGET_REGISTRY[target_config.target_factory]()
        W = self.target_factory.make_target(self.reference)
        self.register_buffer("W", W.detach())
        self.register_buffer("W_inv", torch.linalg.inv(self.W))

    def _setup_optimisation(self, optim_config):
        self.optimiser = optim_config.optimiser(self.parameters(), **optim_config.optimiser_kwargs)
        self.scheduler = optim_config.scheduler(self.optimiser, **optim_config.scheduler_kwargs)
        self._optim_state0 = self.optimiser.state_dict()
        self._sched_state0 = self.scheduler.state_dict()
        self.max_epochs = optim_config.max_steps
        self.stopping_threshold = optim_config.stopping_threshold
        self.patience = optim_config.patience
        self.rtol = optim_config.rtol

    def _reset_optimisation(self):
        self.scheduler.load_state_dict(self._sched_state0)
        self.optimiser.load_state_dict(self._optim_state0)
        device=self.mesh.pts.device
        self.loss             = torch.tensor(float('inf'),device=device)
        self.best             = torch.tensor(float('inf'),device=device, requires_grad=False)
        self.epoch            = torch.tensor(0,     device=device, dtype=torch.long, requires_grad=False)
        self.n_bad_epochs     = torch.tensor(0,     device=device, dtype=torch.long, requires_grad=False)

    def _log(self):
        with torch.no_grad():
            # COL = "\033[38;2;251;179;23m"
            # RES = "\033[0m"
            #
            epoch = self._log_dict.pop("epoch")
            # max_epochs = self.max_epochs
            loss = self._log_dict.pop("loss")
            #
            # blen = int(shutil.get_terminal_size().columns/5)
            # plen = int((epoch)/max_epochs*blen + 0.5)
            #
            # fill = (COL + "\u2588") * plen + RES + "\u2591" * (blen - plen) + RES
            #
            # barstr = "\u2595" + fill + "\u258F"
            other = ""
            for k,v in self._log_dict.items():
                # if k not in ["epoch", "max_epochs", "loss"]:
                other += f"; {k} = {v:0.3e}"

            logstr = f"epoch = {epoch}; loss = {loss:0.3e}" + other
            print(logstr)
            # logstr =  barstr + COL + f" epoch {epoch}" + RES + f" -- loss = {float(loss):0.3e}"+ other 
            # if self.log_client:
            #     self.log_client.put_tensor("tmop_string", np.frombuffer(logstr.encode('utf-8'),dtype=np.uint8))
            #     self.log_client.put_tensor("tmop_epoch", np.array([epoch]))
            # if sys.stdout.isatty():
            #     print('\x1b[k'+logstr+'\r',end='')
            # else:
            #     logstr = f"epoch = {epoch}; loss = {float(loss):0.3e}" + other
            #     print(logstr)
