"""Simple 3D network based on presented powerpoint"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttentionNet(nn.Module):
    def __init__(self, dropout: float = 0.5, age_group_size: int = 20,
                 age_feature_dim: int = 32, att_dim: int = 128,
                 sim_dim: int = 128, att_kernel: tuple = (6, 6, 6),
                 att_cat: bool = True):
        """

        :param age_group_size: number of aging groups
        :param age_feature_dim: age embedding dimension
        :param att_dim: attended feature dimension
        :param sim_dim: similarity space dim
        :param att_kernel: cube size to group voxels
        """
        super(AttentionNet, self).__init__()

        # model info
        self.age_group_size = age_group_size
        self.age_feature_dim = age_feature_dim
        self.att_dim = att_dim
        self.sim_dim = sim_dim
        self.att_kernel = att_kernel
        self.att_cat = att_cat

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

        # attention modules, number next to name indicates corresponding block
        self.attention2 = Attention(
            in_channels=48, output_channels=att_dim,
            attention_kernel=(3, 3, 3), g_dim=1024, similarity_dim=sim_dim)
        self.attention3 = Attention(
            in_channels=64, output_channels=att_dim,
            attention_kernel=(2, 2, 2), g_dim=1024, similarity_dim=sim_dim)
        self.attention4 = Attention(
            in_channels=80, output_channels=att_dim,
            attention_kernel=(1, 1, 1), g_dim=1024, similarity_dim=sim_dim)

        self.fc1 = nn.Linear(2880, 992)

        if att_cat:
            self.att_cat_layer = nn.Linear(3 * att_dim, att_dim)

        self.decoder = nn.Linear(att_dim, 1)
        self.dropout = nn.Dropout(dropout)

    # TODO: fix high over-fitting problem
    def forward(self, x: Tensor, age: int) -> tuple:
        """
        :param x: brain image 3d features
        :param age: age group number
        :return: logits for each batch image
        """
        batch_size = x.size()[0]

        x = self.relu1(self.layer_norm1(self.conv1(x)))
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)

        x = block4.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        age_ft = self.age_encoder(age)
        x = torch.cat([x, age_ft], dim=1)
        # attention forward
        att2, w2 = self.attention2(block2, x)
        att3, w3 = self.attention3(block3, x)
        att4, w4 = self.attention4(block4, x)
        if self.att_cat:
            x = torch.cat([att2, att3, att4], dim=1)
        else:
            x = att4 + att3 + att2

        x = self.decoder(x)

        return x, [w2, w3, w4]


class Attention(nn.Module):
    def __init__(self, in_channels: int, output_channels: int = 128,
                 attention_kernel: tuple = (6, 6, 6), g_dim: int = 128,
                 similarity_dim: int = 128):
        """
        :param in_channels: Conv3d input channels
        :param output_channels: Conv3d output channels
        :param attention_kernel: size of kernel which groups voxels together
        :param g_dim: dimension of global features
        :param similarity_dim: dimension of space which we take similarities in
        """
        super(Attention, self).__init__()
        # model info
        self.in_channels = in_channels
        self.out_channels = output_channels
        self.attention_kernel = attention_kernel
        self.similarity_space = similarity_dim

        # model layers
        self.conv_layer = nn.Conv3d(
            in_channels=in_channels, out_channels=output_channels,
            kernel_size=attention_kernel, stride=attention_kernel)
        self.conv_projection = nn.Linear(output_channels, similarity_dim)
        self.g_projection = nn.Linear(g_dim, similarity_dim)

    def forward(self, x, g):
        """

        :param x: input tensor image, shape(batch_size, C, H, W, L)
        :param g: global features, shape(batch_size, g_dim)
        :return: attended global features, shape(batch_size, dimension)
        """
        batch_size, channels, H, W, L = x.size()
        conv_features = self.conv_layer(x)
        _, _, H_, W_, L_ = conv_features.size()
        # flatten them for dot product
        conv_features = conv_features.view(batch_size, self.out_channels, -1)
        # permute C dimension because channels are used as features
        # for each cube
        conv_features = conv_features.permute(0, 2, 1)
        # project conv features to other space to measure similarity
        conv_hat = self.conv_projection(conv_features)

        # project g into other space, unsqueeze for dot product
        g_hat = self.g_projection(g).unsqueeze(1)

        # take dot product to measure similarity score
        scores = torch.matmul(g_hat, conv_hat.transpose(1, 2))
        weights = F.softmax(scores, dim=2)
        # compute weighted average
        attended_features = torch.matmul(weights, conv_features)

        return attended_features.squeeze(1), weights.detach().cpu() \
            .view(batch_size, H_, W_, L_)
