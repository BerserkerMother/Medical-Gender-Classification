import math


class CosineSchedularLinearWarmup:
    def __init__(self, optimizer, num_training_steps, warmup_steps, lr):
        self.opt = optimizer
        self.num_training_steps = num_training_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = self.num_training_steps - self.warmup_steps
        self.step = 0
        self.lr = lr

    def get_scale(self):
        if self.warmup_steps > self.step:
            return self.step / self.warmup_steps
        else:
            ratio = (self.step - self.warmup_steps) / self.decay_steps
            return 0.5 * (1 + math.cos(ratio * math.pi))

    def update(self):
        scale = self.get_scale()
        for param_group in self.opt.param_groups:
            param_group["lr"] = self.lr * scale
            self.step += 1
            return param_group["lr"]
