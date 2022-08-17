import random
import logging
import os

import torch

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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


def plot_hist(scores, experiment_name):
    sns.set_theme()
    names = ["train", "validation", "test1", "test2", "test3"]

    for name, data in zip(names, scores):
        sns.kdeplot(data, label=name)
    plt.legend()

    save_path = os.path.join("plots", experiment_name)
    if not os.path.exists("plots"):
        os.mkdir("plots")

    plt.savefig(save_path)
