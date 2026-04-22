import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..config import TMOPConfig
from .tmopoptimiser import TMOPOptimiser
from ..mesh import Mesh

class WCUOptimiser(TMOPOptimiser):
    def __init__(self,
                 mesh: Mesh,
                 config: "TMOPConfig",
                 log_client = None 
                 ):
        super().__init__(mesh, config, log_client)
        # parameters for untangling metric
        metric_config = config.metric
        assert (metric_config.c is not None) and (metric_config.d is not None), "WCU optimiser requires parameters `c` and `d`."
        self.register_buffer("c", torch.tensor(metric_config.c))
        self.register_buffer("d", torch.tensor(metric_config.d))

    def optimise(self):
        self._reset_optimisation()
        while not self._should_stop():
            self.loss = self._step()
            self._adjust_lr()

            if self.untangle:
                self.untangle = (self.t < 0).item()

            # log stuff
            self._log_dict["epoch"] = self.epoch
            self._log_dict["loss"]  = self.loss
            self._log_dict["t"]     = self.t
            self._log_dict["beta"]  = self.beta
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

    def _compute_loss(self, T):
        if self.untangle :
            return self._untangle_loss(T)
        else:
            return self._worst_case_loss(T)

    def _untangle_loss(self, T):
        tau = torch.linalg.det(T)
        min_tau = tau.min().detach()
        self.t = torch.minimum(torch.zeros_like(min_tau), min_tau * ( 1. + self.c))
        _loss = (0.5*self.mu(T)/(tau - self.t))
        return _loss

    def _worst_case_loss(self, T):
        tau = torch.linalg.det(T)
        min_tau = tau.min().detach()
        self.t = torch.minimum(torch.zeros_like(min_tau), min_tau * ( 1. + self.c))
        mu_hat = 0.5*self.mu(T)/(tau - self.t)
        max_mu_hat = mu_hat.max().detach()
        self.beta =  max_mu_hat * (1. + self.d)
        _loss = (mu_hat/(self.beta - mu_hat))
        return _loss

    def _adjust_lr(self):
        _sched  = self.t_scheduler if self.untangle else self.scheduler
        _met    = self.t if self.untangle else self.beta
        try:
            _sched.step()
        except:
            _sched.step(metrics=_met)

    def _should_stop(self):
        if not self.untangle:
            if torch.any((self.best - self.beta)/self.beta > self.rtol):
                self.n_bad_epochs = 0
                self.best = self.beta
            else:
                self.n_bad_epochs += 1
            should_stop = (self.n_bad_epochs > self.patience) \
                        | (self.beta < self.stopping_threshold) \
                        | (self.epoch > self.max_epochs)
        else:
            should_stop = False
        return should_stop 

    def _setup_optimisation(self, optim_config):
        super()._setup_optimisation(optim_config)
        self.t_scheduler = optim_config.t_scheduler(self.optimiser, **optim_config.t_scheduler_kwargs)
        self._t_sched_state0 = self.t_scheduler.state_dict()

    def _reset_optimisation(self):
        super()._reset_optimisation()
        self.t_scheduler.load_state_dict(self._t_sched_state0)
        self.untangle = True
        self.t = torch.tensor(-1.,device=self.mesh.pts.device) # tracks minimum tau
        self.beta = torch.full_like(self.t,float("inf")) # tracks maximum mu_hat
