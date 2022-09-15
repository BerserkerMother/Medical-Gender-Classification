"""Simple 3D network based on presented powerpoint"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video.resnet
from torch import Tensor


class SimNet(nn.Module):
    def __init__(self, dropout: float = 0.2, num_classes: int = 1):
        """

        :param num_classes: output vector dimension
        :param dropout: dropout rate
        """
        super(SimNet, self).__init__()
        # model info
        self.name = "Model#1"

        self.conv1 = nn.Conv3d(1, 32, kernel_size=(7, 7, 7), stride=(2, 2, 2), bias=False)
        self.layer_norm1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()

        self.block1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(32, 48, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(),
            nn.Conv3d(48, 48, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        self.block3 = nn.Sequential(
            nn.Conv3d(48, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        self.block4 = nn.Sequential(
            nn.Conv3d(64, 80, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(80),
            nn.ReLU(),
            nn.Conv3d(80, 80, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(80),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        self.fc = nn.Linear(2880, 1024)
        self.decoder = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.image_encoder = nn.Sequential(
            self.conv1,
            self.layer_norm1,
            nn.ReLU(),
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            nn.Flatten(),
            self.fc,
            self.dropout,
            self.decoder
        )

    def init_weights(self):
        for module in self.image_encoder.children():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    # TODO: fix high over-fitting problem
    def forward(self, x: Tensor) -> Tensor:
        """
        currently not using age argument!

        :param x: brain image 3d features
        :return: logits for each batch image
        """
        batch_size = x.size()[0]

        x = self.relu1(self.layer_norm1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x.view(batch_size, -1)
        # include age
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.decoder(x)

        return x

    def forward_seq(self, x: Tensor) -> Tensor:
        """
        currently not using age argument!

        :param x: brain image 3d features
        :return: logits for each batch image
        """
        return self.image_encoder(x)


# extra information dim for each variant of model
MODEL_DIM = {
    1: 1,
    2: 1,
    3: 2,
    4: 3,
    5: 2,
    6: 3,
    7: 4,
}


class R3D18(nn.Module):
    def __init__(self, num_classes=64):
        super(R3D18, self).__init__()
        # model info
        self.name = "3D Resnet"

        # original R3D18
        res = torchvision.models.video.r3d_18(pretrained=True)
        res = list(res.children())[1:-1]
        basic_stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), bias=False),
            nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.image_encoder = nn.Sequential(
            basic_stem,
            *res
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.image_encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def forward_seq(self, x):
        return self.forward(x)


class SimNetExtra(R3D18):
    def __init__(self, model_num: int = 2, **kwargs):
        super(SimNetExtra, self).__init__(num_classes=64)
        # model info
        self.name = ("Model#%d" % model_num)
        self.model_num = model_num
        self.extra_dim = MODEL_DIM[model_num]

        # encoder age into 64 vector
        self.extra_info_encoder = nn.Sequential(
            nn.Linear(self.extra_dim, 64),
            nn.ReLU(),
            nn.Dropout(kwargs["dropout"]),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(kwargs["dropout"]),
            nn.Linear(64, 64)
        )
        # decoder after fusing multi modal features
        self.decoder = nn.Linear(64, 1)

    def forward_with_extra(self, x: Tensor,
                           age: Tensor,
                           GMv: Tensor,
                           TIV: Tensor,
                           GMn: Tensor,
                           WMn: Tensor,
                           CSFn: Tensor,
                           fusion: str = "sum",
                           ):
        """
        currently not using age argument!

        :param x: brain image 3d features
        :param age: the age of patient
        :param fusion: fusion method (sum, hadamard)
        :param GMv: the GMv number
        :param TIV: the TIV number
        :param GMn: the GMn number
        :param WMn: the WMn number
        :param CSFn: the CSFn number
        :return: logits for each batch image
        """

        image_ft = self.forward_seq(x)  # gets image features
        # cat together extra info
        if self.model_num == 1:
            return self.decoder(image_ft), image_ft
        elif self.model_num == 2:
            extra = age
        elif self.model_num == 3:
            extra = torch.cat([age, TIV], dim=1)  # since each will be (bs, 1)
        elif self.model_num == 4:
            # since each will be (bs, 1)
            extra = torch.cat([age, TIV, GMv], dim=1)
        elif self.model_num == 5:
            extra = torch.cat([age, GMn], dim=1)  # since each will be (bs, 1)
        elif self.model_num == 6:
            extra = torch.cat([age, GMn, WMn], dim=1)
        elif self.model_num == 7:
            extra = torch.cat([age, GMn, WMn, CSFn], dim=1)
        else:
            raise Exception("Invalid Model number!!")

        extra_ft = self.extra_info_encoder(extra)

        # fusion
        if fusion == "sum":
            fused_ft = image_ft + extra_ft
        elif fusion == "hadamard":
            fused_ft = image_ft * extra_ft
        else:
            raise Exception("Not Implemented fusion method!")
        logits = self.decoder(fused_ft)
        return logits, fused_ft
