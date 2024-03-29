"""Resnet 3D network for MRI Images implementation https://arxiv.org/pdf/1701.06643.pdf """

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Resnet(nn.Module):
    def __init__(self, age_group_size=15, age_feature_dim=32):
        """

        :param age_group_size: number of aging groups
        :param age_feature_dim: age embedding dimension
        """
        super(Resnet, self).__init__()

        self.num_age_groups = age_group_size
        self.age_feature_dim = age_feature_dim

        # define the model
        self.age_embedding = nn.Embedding(self.num_age_groups,
                                          self.age_feature_dim)
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(7, 7, 7),
                      stride=(2, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3),
                      padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3))
        )

        self.layer2 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3),
                      padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3),
                      padding=1)
        )

        self.layer3 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3),
                      padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )

        # stride layer
        self.layer4 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3),
                      stride=(2, 2, 2))
        )

        self.layer5 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3),
                      padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3),
                      padding=1)
        )

        self.layer6 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3),
                      padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3),
                      padding=1)
        )

        # stride layer
        self.layer7 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3),
                      stride=(2, 2, 2))
        )

        self.layer8 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3),
                      padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3),
                      padding=1)
        )

        self.layer9 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3),
                      padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3),
                      padding=1)
        )

        self.maxpool3d = nn.AdaptiveMaxPool3d(output_size=(2, 2, 2))

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x: Tensor, age: int) -> Tensor:
        """

        :param x: brain image 3d features
        :param age: age group number
        :return: logits for each batch image
        """
        x = self.layer1(x)
        x = x + self.layer2(x)
        x = x + self.layer3(x)
        x = self.layer4(x)
        x = x + self.layer5(x)
        x = x + self.layer6(x)
        x = self.layer7(x)
        x = x + self.layer8(x)
        x = x + self.layer9(x)
        x = self.maxpool3d(x)

        x = x.view(x.size()[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc2(x)
        return logits
