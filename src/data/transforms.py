"""implements data augmentation on 3D images"""

import random
import numpy as np

import torch
from torch import Tensor


class Normalize:
    def __init__(self, mean: np.array, std: np.array):
        """

        :param mean: data mean
        :param std: data standard deviation
        :return:
        """

        self.mean = torch.tensor(mean, dtype=torch.float).squeeze(0) + 1e-9
        self.std = torch.tensor(std, dtype=torch.float).squeeze(0) + 1e-9

    def __call__(self, x: Tensor):
        """

        :param x: 3D image tensor (L, W, H)
        :return: normalized 3D image
        """

        return (x - self.mean) / self.std


# DON'T USE, the integrity of this class hasn't been verified yet
class RandomResizeCrop:
    def __init__(self, size: tuple):
        """

        :param size: desired size of 3D image
        """
        self.size = size

    def __call__(self, x):
        """

        :param x: 3D image tensor (L, W, H)
        :return: cropped and resized 3D image
        """
        L, W, H = x.size()
        l_range = L - self.size[0]
        w_range = W - self.size[1]
        h_range = H - self.size[2]

        l_start = random.randint(0, l_range)
        w_start = random.randint(0, w_range)
        h_start = random.randint(0, h_range)

        x = x[
            l_start: l_start + self.size[0],
            w_start:w_start + self.size[1],
            h_start: h_range + self.size[2]
            ]
        return x


# DON'T USE, the integrity of this class hasn't been verified yet
class RandomPermute:
    def __init__(self, p: float = 0.5):
        """

        :param p: probability that image will be rotated
        """

        self.p = p
        self.permute_states = [
            (2, 1, 0),
            (2, 0, 1),
            (1, 2, 0),
            (1, 0, 2),
            (0, 2, 1)
        ]

    def __call__(self, x: Tensor):
        """

        :param x: 3D image tensor (L, W, H)
        :return: permuted 3D image tensor
        """

        change = random.random()
        if change < self.p:
            random_index = random.randint(0, len(self.permute_states) - 1)
            x = x.permute(self.permute_states[random_index])
        return x
