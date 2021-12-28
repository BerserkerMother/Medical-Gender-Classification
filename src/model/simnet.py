import torch
import torch.nn as nn
import torch.nn.functional as F


class SimNet(nn.Module):
    def __init__(self, age_group_size=15, age_feature_dim=32):
        super(SimNet, self).__init__()

        # age embeddings
        self.age_encoder = nn.Embedding(age_group_size, age_feature_dim)

        self.conv1 = nn.Conv3d(1, 32, kernel_size=(7, 7, 7), stride=(2, 2, 2))
        self.layer_norm1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()

        self.block1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(32, 48, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(48),
            nn.ReLU(),
            nn.Conv3d(48, 48, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(48),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        self.block3 = nn.Sequential(
            nn.Conv3d(48, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        self.block4 = nn.Sequential(
            nn.Conv3d(64, 80, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(80),
            nn.ReLU(),
            nn.Conv3d(80, 80, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(80),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        self.fc = nn.Linear(2880, 1024)
        self.decoder = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(.2)

    # TODO: fix high over-fitting problem
    def forward(self, x, age):
        batch_size = x.size()[0]

        x = self.relu1(self.layer_norm1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x.view(batch_size, -1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)

        # age_ft = self.age_encoder(age)
        # x = x + age_ft
        x = self.decoder(x)

        return x
