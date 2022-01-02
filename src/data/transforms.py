"""implements data augmentation on 3D images"""

import random
import numpy as np

from torch import Tensor


class Normalize:
    def __int__(self, mean: np.array, std: np.array):
        """

        :param mean: data mean
        :param std: data standard deviation
        :return:
        """

        self.mean = mean
        self.std = std

    def __call__(self, x: Tensor):
        """

        :param x: 3D image tensor (L, W, H)
        :return: normalized 3D image
        """

        return (x - self.mean) / self.std


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
