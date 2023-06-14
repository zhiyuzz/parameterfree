import torch
import numpy as np
from scipy.special import erfi
from torch.optim.optimizer import Optimizer


class ERFI(Optimizer):
    r"""Implements an algorithm based on the imaginary error function.
    The original version is from the paper `PDE-Based Optimal Strategy for Unconstrained Online Learning`_.
    The implemented version is a variant without performance guarantees.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        eps (float, optional): scaling parameter (default: 1)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. _PDE-Based Optimal Strategy for Unconstrained Online Learning:
        https://arxiv.org/abs/2201.07877
    """

    def __init__(self, params, eps: float = 1.0, weight_decay: float = 0):
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
        """Performs a single optimization step.
        """

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            if self._firstep:
                group['x0'] = [torch.clone(p).detach() for p in group['params']]
                group['theta'] = [torch.zeros_like(p).detach() for p in group['params']]
                self._firstep = False

            if weight_decay > 0:
                for p in group['params']:
                    p.grad.add_(p, alpha=weight_decay)

            self._iter += 1

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
