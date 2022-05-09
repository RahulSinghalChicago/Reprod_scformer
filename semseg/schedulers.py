import torch
import math
from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1, power=0.9, last_epoch=-1) -> None:
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
            return self.base_lrs
        else:
            factor = (1 - self.last_epoch / float(self.max_iter)) ** self.power
            return [factor*lr for lr in self.base_lrs]


class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_iter=500, warmup_ratio=5e-4, warmup='exp', last_epoch=-1) -> None:
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        return [ratio * lr for lr in self.base_lrs]

    def get_lr_ratio(self):
        return self.get_warmup_ratio() if self.last_epoch < self.warmup_iter else self.get_main_ratio()

    def get_main_ratio(self):
        raise NotImplementedError

    def get_warmup_ratio(self):
        assert self.warmup in ['linear', 'exp']
        alpha = self.last_epoch / self.warmup_iter

        return self.warmup_ratio + (1. - self.warmup_ratio) * alpha if self.warmup == 'linear' else self.warmup_ratio ** (1. - alpha)


class WarmupPolyLR(WarmupLR):
    def __init__(self, optimizer, power, max_iter, warmup_iter=500, warmup_ratio=5e-4, warmup='exp', last_epoch=-1) -> None:
        self.power = power
        self.max_iter = max_iter
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        alpha = real_iter / real_max_iter

        return (1 - alpha) ** self.power


class WarmupExpLR(WarmupLR):
    def __init__(self, optimizer, gamma, interval=1, warmup_iter=500, warmup_ratio=5e-4, warmup='exp', last_epoch=-1) -> None:
        self.gamma = gamma
        self.interval = interval
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        return self.gamma ** (real_iter // self.interval)


class WarmupCosineLR(WarmupLR):
    def __init__(self, optimizer, max_iter, eta_ratio=0, warmup_iter=500, warmup_ratio=5e-4, warmup='exp', last_epoch=-1) -> None:
        self.eta_ratio = eta_ratio
        self.max_iter = max_iter
        super().__init__(optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        
        return self.eta_ratio + (1 - self.eta_ratio) * (1 + math.cos(math.pi * self.last_epoch / real_max_iter)) / 2

class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    
    def __init__(self, optimizer, max_decay_steps=40, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr

__all__ = ['polylr', 'warmuppolylr', 'warmupcosinelr', 'warmupsteplr', 'plrd']


def get_scheduler(scheduler_name: str, optimizer, max_iter: int, power: int, warmup_iter: int, warmup_ratio: float):
    assert scheduler_name in __all__, f"Unavailable scheduler name >> {scheduler_name}.\nAvailable schedulers: {__all__}"
    if scheduler_name == 'warmuppolylr':
        return WarmupPolyLR(optimizer, power, max_iter, warmup_iter, warmup_ratio, warmup='linear')
    elif scheduler_name == 'warmupcosinelr':
        return WarmupCosineLR(optimizer, max_iter, warmup_iter=warmup_iter, warmup_ratio=warmup_ratio)
    elif scheduler_name == 'plrd':
        return PolynomialLRDecay(optimizer) 
    return PolyLR(optimizer, max_iter)


if __name__ == '__main__':
    model = torch.nn.Conv2d(3, 16, 3, 1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    max_iter = 20000
    sched = WarmupPolyLR(optim, power=0.9, max_iter=max_iter, warmup_iter=200, warmup_ratio=0.1, warmup='exp', last_epoch=-1)

    lrs = []

    for _ in range(max_iter):
        lr = sched.get_lr()[0]
        lrs.append(lr)
        optim.step()
        sched.step()

    import matplotlib.pyplot as plt
    import numpy as np 

    plt.plot(np.arange(len(lrs)), np.array(lrs))
    plt.grid()
    plt.show()