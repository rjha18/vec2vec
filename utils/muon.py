import math

import torch
from torch.optim import Optimizer


def _newton_schulz_orthogonalize(x: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate orthogonalization used by Muon-style updates.

    Input is treated as a matrix update direction. For non-square matrices, we
    orthogonalize in the smaller dimension and project back.
    """
    orig_shape = x.shape
    if x.ndim != 2:
        return x

    transposed = False
    if x.shape[0] < x.shape[1]:
        x = x.t()
        transposed = True

    # Normalize for numerical stability.
    x = x / (x.norm() + eps)
    eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)

    for _ in range(steps):
        xtx = x.t() @ x
        x = 0.5 * x @ (3.0 * eye - xtx)

    if transposed:
        x = x.t()
    return x.view(orig_shape)


class MuonW(Optimizer):
    """Hybrid MuonW optimizer.

    Best-practice style behavior:
    - Matrix parameters (ndim >= 2): Muon momentum + orthogonalized update.
    - Vector/scalar parameters (ndim < 2): AdamW-style update.
    """

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        muon_beta: float = 0.95,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            muon_beta=muon_beta,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            muon_beta = group["muon_beta"]
            beta1, beta2 = group["adamw_betas"]
            adamw_eps = group["adamw_eps"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("MuonW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if p.ndim >= 2:
                        state["momentum"] = torch.zeros_like(p)
                    else:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1

                # Decoupled weight decay for both branches.
                if wd > 0:
                    p.mul_(1.0 - lr * wd)

                if p.ndim >= 2:
                    m = state["momentum"]
                    m.mul_(muon_beta).add_(g, alpha=1.0 - muon_beta)
                    m_hat = m / (1.0 - muon_beta ** state["step"])
                    update = _newton_schulz_orthogonalize(m_hat, steps=ns_steps)
                    p.add_(update, alpha=-lr)
                else:
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(adamw_eps)
                    step_size = lr / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
