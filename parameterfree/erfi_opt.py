import torch
import numpy as np
from scipy.special import erfi
from torch.optim.optimizer import Optimizer


class ERFI(Optimizer):
    def __init__(self, params, eps: float = 1, weight_decay: float = 0):
        if not 0.0 <= eps:
            raise ValueError("Invalid eps value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(weight_decay=weight_decay)
        self._eps = eps
        self._iter = 1
        self._firstep = True

        super(ERFI, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):

        self._iter += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            if self._firstep:
                group['x0'] = [torch.clone(p).detach() for p in group['params']]
                group['theta'] = [torch.zeros_like(p).detach() for p in group['params']]
                self._firstep = False

            if weight_decay > 0:
                for p in group['params']:
                    p.grad.add_(p, alpha=weight_decay)

            # update the sum of the negative gradients and the weights
            for p, t, x in zip(group['params'], group['theta'], group['x0']):
                if p.grad is None:
                    continue
                else:
                    t.add_(p.grad, alpha=-1)
                t_norm = t.norm()
                device = t_norm.device
                z = t_norm.to('cpu') / np.sqrt(2 * self._iter)
                magnitude = self._eps * erfi(z)
                p.copy_(t.mul(magnitude.to(device) / t_norm).add(x))
