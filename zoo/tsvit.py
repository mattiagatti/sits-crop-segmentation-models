"""
Adapted from the TSViT implementation in DeepSatModels:
https://github.com/michaeltrs/DeepSatModels

Original project:
Michail Tarasiou et al., "ViTs for SITS: Vision Transformers for Satellite Image Time Series", CVPR 2023.

DeepSatModels is released under the Apache License 2.0.
Modifications in this file include changes to temporal embeddings and DOY handling.
"""

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
            qkv,
        )
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class TSViT(nn.Module):
    def __init__(
        self,
        img_res=48,
        patch_size=3,
        num_classes=18,
        max_seq_len=32,
        dim=128,
        heads=4,
        dim_head=64,
        num_channels=12,
        dropout=0.0,
        emb_dropout=0.0,
        scale_dim=4,
        temporal_depth=8,
        spatial_depth=4,
    ):
        super().__init__()
        self.image_size = img_res
        self.patch_size = patch_size
        self.num_patches_1d = self.image_size // self.patch_size
        self.num_classes = num_classes
        self.num_frames = max_seq_len
        self.dim = dim
        self.temporal_depth = temporal_depth
        self.spatial_depth = spatial_depth
        self.heads = heads
        self.dim_head = dim_head
        self.dropout_p = dropout
        self.emb_dropout = emb_dropout
        self.scale_dim = scale_dim

        num_patches = self.num_patches_1d**2
        patch_dim = num_channels * self.patch_size**2

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)",
                p1=self.patch_size,
                p2=self.patch_size,
            ),
            nn.Linear(patch_dim, self.dim),
        )

        self.to_temporal_embedding_input = nn.Linear(366, self.dim)

        self.temporal_token = nn.Parameter(
            torch.randn(1, self.num_classes, self.dim)
        )
        self.temporal_transformer = Transformer(
            self.dim,
            self.temporal_depth,
            self.heads,
            self.dim_head,
            self.dim * self.scale_dim,
            self.dropout_p,
        )

        self.space_pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, self.dim)
        )
        self.space_transformer = Transformer(
            self.dim,
            self.spatial_depth,
            self.heads,
            self.dim_head,
            self.dim * self.scale_dim,
            self.dropout_p,
        )

        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2),
        )

    def forward(self, x, doys):
        """
        x:    (B, C, T, H, W)
        doys: (B, T) with integer values in [1, 366]
        """
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        B, T, C, H, W = x.shape

        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"H and W must be divisible by patch_size={self.patch_size}, got H={H}, W={W}"
            )

        if doys.shape != (B, T):
            raise ValueError(
                f"doys must have shape {(B, T)}, got {tuple(doys.shape)}"
            )

        doys = doys.long().clamp(min=1, max=366) - 1
        xt = F.one_hot(doys, num_classes=366).to(device=x.device, dtype=torch.float32)

        xt = xt.reshape(-1, 366)
        temporal_pos_embedding = self.to_temporal_embedding_input(xt).reshape(
            B, T, self.dim
        )

        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x = x + temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)

        cls_temporal_tokens = repeat(
            self.temporal_token,
            "() N d -> b N d",
            b=B * self.num_patches_1d**2,
        )
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, : self.num_classes]

        x = (
            x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim)
            .permute(0, 2, 1, 3)
            .reshape(B * self.num_classes, self.num_patches_1d**2, self.dim)
        )

        x = x + self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))

        x = (
            x.reshape(
                B,
                self.num_classes,
                self.num_patches_1d**2,
                self.patch_size**2,
            )
            .permute(0, 2, 3, 1)
        )
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x


class TSViT_lookup(nn.Module):
    """
    TSViT with lookup-based temporal embeddings.

    During training:
        DOYs are mapped to embeddings using the set of train_dates.

    During inference:
        embeddings for all days 1..366 are built by linear interpolation
        between the embeddings of train_dates.
    """

    def __init__(
        self,
        train_dates,
        img_res=48,
        patch_size=3,
        num_classes=18,
        max_seq_len=32,
        dim=128,
        heads=4,
        dim_head=64,
        num_channels=12,
        dropout=0.0,
        emb_dropout=0.0,
        scale_dim=4,
        temporal_depth=8,
        spatial_depth=4,
    ):
        super().__init__()

        train_dates = sorted(set(int(d) for d in train_dates))
        if len(train_dates) == 0:
            raise ValueError("train_dates must not be empty.")

        self.train_dates = nn.Parameter(
            torch.tensor(train_dates, dtype=torch.long),
            requires_grad=False,
        )
        self.eval_dates = nn.Parameter(
            torch.arange(1, 367, dtype=torch.long),  # 1..366
            requires_grad=False,
        )

        self.image_size = img_res
        self.patch_size = patch_size
        self.num_patches_1d = self.image_size // self.patch_size
        self.num_classes = num_classes
        self.num_frames = max_seq_len
        self.dim = dim
        self.temporal_depth = temporal_depth
        self.spatial_depth = spatial_depth
        self.heads = heads
        self.dim_head = dim_head
        self.dropout_p = dropout
        self.emb_dropout = emb_dropout
        self.scale_dim = scale_dim

        num_patches = self.num_patches_1d**2
        patch_dim = num_channels * self.patch_size**2

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)",
                p1=self.patch_size,
                p2=self.patch_size,
            ),
            nn.Linear(patch_dim, self.dim),
        )

        self.temporal_pos_embedding = nn.Parameter(
            torch.randn(len(train_dates), self.dim),
            requires_grad=True,
        )
        self.update_inference_temporal_position_embeddings()

        self.temporal_token = nn.Parameter(
            torch.randn(1, self.num_classes, self.dim)
        )
        self.temporal_transformer = Transformer(
            self.dim,
            self.temporal_depth,
            self.heads,
            self.dim_head,
            self.dim * self.scale_dim,
            self.dropout_p,
        )

        self.space_pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, self.dim)
        )
        self.space_transformer = Transformer(
            self.dim,
            self.spatial_depth,
            self.heads,
            self.dim_head,
            self.dim * self.scale_dim,
            self.dropout_p,
        )

        self.dropout = nn.Dropout(self.emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2),
        )

    def forward(self, x, doys, inference=False):
        """
        x:    (B, C, T, H, W)
        doys: (B, T) with integer values in [1, 366]
        """
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        B, T, C, H, W = x.shape

        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"H and W must be divisible by patch_size={self.patch_size}, got H={H}, W={W}"
            )

        if doys.shape != (B, T):
            raise ValueError(
                f"doys must have shape {(B, T)}, got {tuple(doys.shape)}"
            )

        doys = doys.long().clamp(min=1, max=366)

        if inference:
            self.update_inference_temporal_position_embeddings()
            temporal_pos_embedding = self.get_inference_temporal_position_embeddings(
                doys
            ).to(x.device)
        else:
            temporal_pos_embedding = self.get_temporal_position_embeddings(doys).to(
                x.device
            )

        x = self.to_patch_embedding(x)
        x = x.reshape(B, -1, T, self.dim)
        x = x + temporal_pos_embedding.unsqueeze(1)
        x = x.reshape(-1, T, self.dim)

        cls_temporal_tokens = repeat(
            self.temporal_token,
            "() N d -> b N d",
            b=B * self.num_patches_1d**2,
        )
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x[:, : self.num_classes]

        x = (
            x.reshape(B, self.num_patches_1d**2, self.num_classes, self.dim)
            .permute(0, 2, 1, 3)
            .reshape(B * self.num_classes, self.num_patches_1d**2, self.dim)
        )

        x = x + self.space_pos_embedding
        x = self.dropout(x)
        x = self.space_transformer(x)
        x = self.mlp_head(x.reshape(-1, self.dim))

        x = (
            x.reshape(
                B,
                self.num_classes,
                self.num_patches_1d**2,
                self.patch_size**2,
            )
            .permute(0, 2, 3, 1)
        )
        x = x.reshape(B, H, W, self.num_classes)
        x = x.permute(0, 3, 1, 2)
        return x

    def update_inference_temporal_position_embeddings(self):
        device = self.temporal_pos_embedding.device
        train_dates = self.train_dates.to(device)

        min_val = int(train_dates.min().item())
        max_val = int(train_dates.max().item())

        pos_eval = torch.zeros(len(self.eval_dates), self.dim, device=device)

        for i, evdate in enumerate(self.eval_dates.to(device)):
            ev = int(evdate.item())

            if ev <= min_val:
                pos_eval[i] = self.temporal_pos_embedding[0]
                continue

            if ev >= max_val:
                pos_eval[i] = self.temporal_pos_embedding[-1]
                continue

            exact_match = (train_dates == ev)
            if exact_match.any():
                idx = exact_match.nonzero(as_tuple=True)[0][0]
                pos_eval[i] = self.temporal_pos_embedding[idx]
                continue

            lower_idx = (train_dates < ev).nonzero(as_tuple=True)[0][-1]
            upper_idx = (train_dates > ev).nonzero(as_tuple=True)[0][0]

            lower_date = int(train_dates[lower_idx].item())
            upper_date = int(train_dates[upper_idx].item())

            alpha = (ev - lower_date) / (upper_date - lower_date)
            pos_eval[i] = (
                (1.0 - alpha) * self.temporal_pos_embedding[lower_idx]
                + alpha * self.temporal_pos_embedding[upper_idx]
            )

        self.inference_temporal_pos_embedding = nn.Parameter(
            pos_eval,
            requires_grad=False,
        )

    def get_temporal_position_embeddings(self, doys):
        """
        Exact lookup during training.

        Requires each doy to be present in train_dates.
        """
        B, T = doys.shape
        doys = doys.to(self.train_dates.device)

        idx = torch.searchsorted(self.train_dates, doys.reshape(-1))
        valid = idx < len(self.train_dates)
        exact = valid & (self.train_dates[idx.clamp(max=len(self.train_dates) - 1)] == doys.reshape(-1))

        if not exact.all():
            bad_vals = doys.reshape(-1)[~exact].unique().tolist()
            raise ValueError(
                f"Some DOY values are not present in train_dates: {bad_vals}"
            )

        return self.temporal_pos_embedding[idx].reshape(B, T, self.dim)

    def get_inference_temporal_position_embeddings(self, doys):
        """
        Interpolated lookup during inference.
        doys: (B, T), values in [1, 366]
        """
        B, T = doys.shape
        doys = doys.to(self.eval_dates.device).clamp(min=1, max=366)

        idx = doys.reshape(-1) - 1  # 1..366 -> 0..365
        return self.inference_temporal_pos_embedding[idx].reshape(B, T, self.dim)


if __name__ == "__main__":
    res = 48
    batch_size = 8
    channels = 12
    seq_len = 32

    x = torch.rand((batch_size, channels, seq_len, res, res))
    doys = torch.randint(1, 367, (batch_size, seq_len))

    model = TSViT(
        img_res=res,
        patch_size=3,
        num_classes=18,
        max_seq_len=seq_len,
        dim=128,
        heads=4,
        dim_head=64,
        num_channels=channels,
        dropout=0.0,
        emb_dropout=0.0,
        pool="cls",
        scale_dim=4,
        depth=4,
        temporal_depth=8,
        spatial_depth=2,
    )

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum(np.prod(p.size()) for p in parameters) / 1_000_000
    print("Trainable Parameters TSViT: %.3fM" % parameters)

    out = model(x, doys)
    print("TSViT output shape:", out.shape)