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
        # self.untangle = metric_config.untangle
        # parameters for untangling metric
        metric_config = config.metric
        assert (metric_config.c is not None) and (metric_config.d is not None), "WCU optimiser requires parameters `c` and `d`."
        self.c: float = metric_config.c
        self.d: float = metric_config.d

    def optimise(self):
        self._reset_optimisation()
        self.t = 0 # tracks minimum tau
        self.beta = torch.inf # tracks maximum mu_hat
        self.best = torch.inf
        self.n_bad_epochs = 0
        for epoch in range(self.max_epochs):
            self.batch_mean_loss = 0
            self.batch_mean_beta = 0
            self.batch_worst_t = 0
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
                self.batch_mean_beta += self.beta
                self.batch_worst_t = min(self.t, self.batch_worst_t)

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
            self.batch_mean_beta /= self.n_batches
            _sched = self.t_scheduler if self.batch_worst_t<0 else self.scheduler
            _met = self.batch_worst_t if self.batch_worst_t<0 else self.batch_mean_beta
            try:
                _sched.step()
            except:
                _sched.step(metrics=_met)
            if self._should_stop():
                break
        print()

    def _compute_metric(self, T):
        tau = torch.linalg.det(T)
        min_tau = tau.min().item()
        self.t = 0.0 if min_tau > 0.0 else min_tau * ( 1 + self.c)

        mu_hat = 0.5*self.mu(T)/(tau - self.t)
        max_mu_hat = mu_hat.max().item()
        # self.beta = min(self.beta, max_mu_hat + self.d)
        self.beta =  max_mu_hat * (1 + self.d)
        assert torch.all(tau > self.t), f"Invalid t: {self.t} > {tau.min().item()}"
        assert torch.all(self.beta > mu_hat), f"Invalid beta: {self.beta} < {mu_hat.max().item()}"

        # wcuo as in Worst Case Untangle Optimise (or something like that)
        wcuo = mu_hat/(self.beta - mu_hat)
        return wcuo.mean()

    def _should_stop(self):
        assert self.batch_mean_loss < torch.inf, "Loss blew up, stopping."

        self.n_bad_epochs *= (self.batch_worst_t >= 0)
        if (self.batch_mean_beta < self.best) and abs((self.batch_mean_beta - self.best)/self.best) > self.rtol:
            self.n_bad_epochs = 0
            self.best = self.batch_mean_beta
        else:
            self.n_bad_epochs += 1

        return self.n_bad_epochs > self.patience

    def _setup_optimisation(self, optim_config):
        super()._setup_optimisation(optim_config)
        self.t_scheduler = optim_config.t_scheduler(self.optimiser, **optim_config.t_scheduler_kwargs)
        self._t_sched_state0 = self.t_scheduler.state_dict()

    def _reset_optimisation(self):
        super()._reset_optimisation()
        self.t_scheduler.load_state_dict(self._t_sched_state0)

