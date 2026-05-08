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
        # self._step()
        # self._update_state()
        # self._reset_optimisation()
        # with torch.profiler.profile() as p:
        CHECK_INTERVAL = 20
        while True :
            for _ in range(CHECK_INTERVAL):
                self.loss = self._step()
                self.epoch.add_(1)
                self._update_state()
            self._adjust_lr()

            if self._should_stop.item():
            # if self.epoch >= self.max_epochs:
                # log stuff
                self._log_dict["epoch"] = self.epoch.item()
                self._log_dict["loss"]  = self.loss.item()
                self._log_dict["t"]     = self.t.item()
                self._log_dict["beta"]  = self.beta.item()
                self._log()
                break
        # p.export_chrome_trace("./profiler_trace.json")

    @torch.compile(fullgraph=False, mode='max-autotune')
    def _step(self):
        loss = 0
        T = self._compute_weighted_jacobian()
        _loss, t, beta = self._compute_loss(T)
        self.t = t.detach()
        self.beta = beta.detach()
        loss = (_loss.mean(dim=-1)*self.mesh.elements_area_inv).sum()
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        return loss

    def _compute_weighted_jacobian(self):
        sg : torch.Tensor = self.reference.shape_grad_all  # (n_el, n_pts, n_sides, sdim)
        pts = self.mesh.pts_all             # (n_el, n_sides, sdim)
        # n_el, n_pts, n_sides, sdim = sg.shape

        A00 = (sg[...,0] * pts[:,None,:,0]).sum(dim=-1)
        A01 = (sg[...,0] * pts[:,None,:,1]).sum(dim=-1)
        A10 = (sg[...,1] * pts[:,None,:,0]).sum(dim=-1)
        A11 = (sg[...,1] * pts[:,None,:,1]).sum(dim=-1)
        A = torch.stack((torch.stack((A00,A01), dim=-1),
                         torch.stack((A10,A11), dim=-1)),dim=-2)
        # Reshape sg: (n_el, n_pts, n_sides, sdim) -> (n_el, n_pts*sdim, n_sides)
        # pts:        (n_el, n_sides, sdim)
        # bmm gives: (n_el, n_pts*sdim, sdim)
        # then reshape to (n_el, n_pts, sdim, sdim)
        # TODO : DIRECTLY CHANGE STORAGE ORDER OF SHAPE_GRAD_ALL TO BE CONTIGUOUS
        # sg_r = sg.permute(0, 1, 3, 2).reshape(n_el, n_pts * sdim, n_sides)  # contiguous after reshape
        # TODO : CONSIDER INLINING THIS AS WELL
        # A = torch.bmm(sg_r, pts).reshape(n_el, n_pts, sdim, sdim)
        # A = torch.einsum("epsi,esj->epij", 
        #                  self.reference.shape_grad_all, 
        #                  self.mesh.pts_all) # (n_elements, n_samples, n_sides, spacedim), (n_elements, n_sides, spacedim) -> (n_elements, n_samples, spacedim, spacedim)
        # TODO : INLINE THIS
        T00 = (A[...,0,:] * self.W_inv[...,:,0]).sum(dim=-1)
        T01 = (A[...,0,:] * self.W_inv[...,:,1]).sum(dim=-1)
        T10 = (A[...,1,:] * self.W_inv[...,:,0]).sum(dim=-1)
        T11 = (A[...,1,:] * self.W_inv[...,:,1]).sum(dim=-1)
        return torch.stack((torch.stack((T00,T01), dim=-1),
                            torch.stack((T10,T11), dim=-1)),dim=-2)
        # return A @ self.W_inv

    def _compute_loss(self, T):
        return torch.cond(self.untangle,
                          self._untangle_loss,
                          self._worst_case_loss,
                          (T,)
                          )
        # return self._worst_case_loss(T)

    def _untangle_loss(self, T):
        tau = T[..., 0, 0]*T[..., 1, 1] - T[..., 0, 1]*T[..., 1, 0]
        min_tau = tau.min().detach()
        t = torch.clamp(min_tau * ( 1. + self.c),max=0)
        _loss = (0.5*self.mu(T)/(tau - t))
        return _loss, t, self.beta.clone()

    def _worst_case_loss(self, T):
        tau = T[..., 0, 0]*T[..., 1, 1] - T[..., 0, 1]*T[..., 1, 0]
        min_tau = tau.min().detach()
        t = torch.clamp(min_tau * ( 1. + self.c),max=0)
        mu_hat = 0.5*self.mu(T)/(tau - t)
        max_mu_hat = mu_hat.max().detach()
        beta =  max_mu_hat * (1. + self.d)
        _loss = (mu_hat/(beta - mu_hat))
        return _loss, t, beta

    def _adjust_lr(self):
        _sched  = self.t_scheduler if self.untangle else self.scheduler
        _met    = self.t if self.untangle else self.beta
        try:
            _sched.step()
        except:
            _sched.step(metrics=_met)

    @torch.compile
    def _update_state(self):
        self.untangle = self.t < 0.

        improved = (self.best - self.beta) / self.beta > self.rtol
        self.best = torch.where(improved, self.beta, self.best)
        torch.where(
            improved | self.untangle,
            torch.zeros_like(self.n_bad_epochs),
            self.n_bad_epochs + 1,
            out = self.n_bad_epochs
        )
        patience_exceeded = self.n_bad_epochs > self.patience
        step_too_small    = self._get_max_step_size() < self.stopping_threshold
        epoch_exceeded    = self.epoch >= self.max_epochs
        self._should_stop = torch.where(
            self.untangle,
            epoch_exceeded,
            patience_exceeded | step_too_small | epoch_exceeded
        )

    def _get_max_step_size(self):
        sd = self.optimiser.state_dict()
        lr = sd['param_groups'][0]['lr']
        beta1, beta2 = sd['param_groups'][0]['betas']
        eps = sd['param_groups'][0]['eps']
        step = sd['state'][0]['step']
        exp_avg = sd['state'][0]['exp_avg']
        exp_avg_sq = sd['state'][0]['exp_avg_sq']

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = bias_correction2**0.5

        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

        return torch.linalg.norm(exp_avg/denom * step_size,dim=-1).max()

    def _setup_optimisation(self, optim_config):
        super()._setup_optimisation(optim_config)
        self.t_scheduler = optim_config.t_scheduler(self.optimiser, **optim_config.t_scheduler_kwargs)
        self._t_sched_state0 = self.t_scheduler.state_dict()

    def _reset_optimisation(self):
        super()._reset_optimisation()
        self.t_scheduler.load_state_dict(self._t_sched_state0)
        device=self.mesh.pts.device
        self.untangle         = torch.tensor(True,  device=device, dtype=torch.bool)
        self.t                = torch.tensor(-1.,   device=device, requires_grad=False)
        self.beta             = torch.tensor(float('inf'), device=device, requires_grad=False)
        self._should_stop     = torch.tensor(False, device=device, dtype=torch.bool)
