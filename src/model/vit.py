import torch
import torch.nn as nn
import torch.nn.functional as F


class VIT(nn.Module):
    def __init__(self, d_model=512, num_layers=6, num_heads=8, dropout_p=0.3):
        super(VIT, self).__init__()
        # embedding
        self.embedding = Embedding(d_model)

        # layers
        encoder_layers = []
        for i in range(num_layers):
            layer = EncoderLayer(d_model, num_heads, dropout_p)
            encoder_layers.append(layer)
        self.encoder_layer = nn.ModuleList(encoder_layers)

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, vis_attn=False):
        # visual attention
        attn_weights = []
        # embedding
        x = self.embedding(x)

        for layer in self.encoder_layer:
            x = layer(x, attn_weights)

        x = self.norm(x[:, 0, :])
        x = self.fc(x)
        if vis_attn:
            return x, attn_weights
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout_p):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout_p)
        self.feed_forward = FeedForward(d_model, scale=2)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, x, att_weights):
        x1 = x
        x = self.norm1(x)
        x = self.attention(x, att_weights)
        x = self.dropout1(x)
        x = x + x1

        x1 = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = x + x1
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_p):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_attention = d_model // num_heads
        assert (d_model % num_heads) == 0

        self.q_transformation = nn.Linear(d_model, d_model, bias=False)
        self.k_transformation = nn.Linear(d_model, d_model, bias=False)
        self.v_transformation = nn.Linear(d_model, d_model, bias=False)

        self.projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, attn_weights):
        batch_size, num_token, d_model = x.size()

        # size x (b, n, d)
        q = self.q_transformation(x). \
            view(batch_size, num_token, self.num_heads, self.d_attention). \
            permute(0, 2, 1, 3)
        k = self.k_transformation(x). \
            view(batch_size, num_token, self.num_heads, self.d_attention). \
            permute(0, 2, 1, 3)
        v = self.v_transformation(x). \
            view(batch_size, num_token, self.num_heads, self.d_attention). \
            permute(0, 2, 1, 3)
        # size q (b, n, d_)

        scores = torch.matmul(q, k.transpose(2, 3))
        # size scores (b, n, n)
        weights = F.softmax(scores, dim=3)
        # size weights (b, n, n)
        weights = self.dropout(weights)

        final_result = torch.matmul(weights, v).permute(0, 2, 1, 3). \
            contiguous().view(batch_size, num_token, d_model)
        final_result = self.projection(final_result)

        # save attention weights
        # attn_weights.append(weights.detach().cpu())
        return final_result


class FeedForward(nn.Module):
    def __init__(self, d_model, scale=4, dropout_p=0.2):
        super(FeedForward, self).__init__()

        self.d_model = d_model
        self.scale = scale

        self.fc1 = nn.Linear(d_model, d_model * scale)
        self.fc2 = nn.Linear(d_model * scale, d_model)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Embedding(nn.Module):
    def __init__(self, d_model, max_size=2000):
        super(Embedding, self).__init__()
        self.d_model = d_model

        self.patcher = nn.Conv3d(1, d_model, kernel_size=(12, 12, 12),
                                 stride=(12, 12, 12))

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.positions = nn.Parameter(torch.randn(1, max_size, d_model))

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.patcher(x[:, :, :120, :144, :120]) \
            .view(batch_size, self.d_model, -1).permute(0, 2, 1)

        # x (b, n, d_model) -> (b, n+1, d_model)
        _, num_token, d_model = x.size()
        cls_token = self.cls_token.expand(batch_size, 1, d_model)
        x = torch.cat([cls_token, x], dim=1)

        x = x + self.positions[:, :num_token + 1, :]
        return x
