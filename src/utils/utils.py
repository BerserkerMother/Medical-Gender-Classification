import random
import logging

import torch
import numpy as np

import wandb


class AverageMeter:
    def __init__(self):
        self.sum = 0.
        self.num = 0

    def reset(self):
        self.sum = 0.
        self.num = 0

    def update(self, value, num=1):
        self.sum += value
        self.num += num

    def avg(self):
        return self.sum / self.num


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log_and_display(train_acc, val_acc, train_loss, val_loss):
    wandb.log({
        "accuracy": {
            "train": train_acc,
            "val": val_acc
        },
        "loss": {
            "train": train_loss,
            "val": val_loss
        }
    }
    )
    logging.info('train accuracy: %.2f%%, val accuracy: %.2f%%' %
                 (train_acc, val_acc))
