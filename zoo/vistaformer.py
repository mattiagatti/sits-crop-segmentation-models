"""
Adapted from the VistaFormer implementation:
https://github.com/macdonaldezra/VistaFormer

Original project:
Ezra MacDonald, Derek Jacoby, and Yvonne Coady,
"VistaFormer: Scalable Vision Transformers for Satellite Image Time Series Segmentation", 2024.

Modifications in this file include local refactoring and implementation changes.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_


# ============================================================
# Utilities
# ============================================================


def get_activation_layer(activation: str) -> nn.Module:
    activation = activation.lower()
    if activation == "gelu":
        return nn.GELU()
    if activation == "swish":
        return nn.SiLU()
    if activation == "mish":
        return nn.Mish()
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    raise ValueError(
        "Invalid activation function. Choose from "
        "'gelu', 'swish', 'mish', 'relu', or 'leakyrelu'."
    )


def _init_transformer_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, nn.Conv3d):
        fan_out = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * module.kernel_size[2]
            * module.out_channels
        )
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()


# ============================================================
# Core Layers
# ============================================================


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x + self.fn(x, **kwargs)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.norm(self.fn(x, **kwargs))


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.to_out = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        h = self.num_heads

        qkv = self.to_qkv(x).reshape(b, n, 3, h, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.to_out(out)
        out = self.proj_dropout(out)
        return out


class SEBlock3D(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class GatedConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.gate = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) * torch.sigmoid(self.gate(x))


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
    ):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class AdaptiveTemporalFeaturePooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_temporal_size: int = 1,
        pool_type: str = "avg",
    ):
        super().__init__()
        if pool_type not in {"avg", "max"}:
            raise ValueError("pool_type must be 'avg' or 'max'")

        if pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool3d((output_temporal_size, None, None))
        else:
            self.pool = nn.AdaptiveMaxPool3d((output_temporal_size, None, None))

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class PatchEmbed3D(nn.Module):
    """
    Input:  (B, C, T, H, W)
    Output: (B, D, T', H', W')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int],
        use_squeeze: bool,
        gate: bool,
        norm_type: str = "batch2d",
    ):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=patch_size,
            stride=stride,
        )

        self.use_gate = gate
        if self.use_gate:
            self.gate_conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=stride,
            )

        self.use_squeeze = use_squeeze
        if self.use_squeeze:
            self.se_block = SEBlock3D(out_channels)

        self.norm_type = norm_type
        if norm_type == "batch2d":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == "batch3d":
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm_type in {"", "none", None}:
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(
        self,
        x: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.use_gate:
            x_conv = self.proj(x)
            gate = torch.sigmoid(self.gate_conv(x))
            x = x_conv * gate
        else:
            x = self.proj(x)
            gate = None

        if self.use_squeeze:
            x = self.se_block(x)

        if self.norm_type == "batch2d":
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).contiguous()
            x = self.norm(x)
            x = x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        else:
            x = self.norm(x)

        if return_gate:
            if gate is None:
                raise ValueError("return_gate=True requested, but gating is disabled.")
            return x, gate

        return x


class DWConv3d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv3d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=dim,
        )

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        bt, hw, c = x.shape
        b = bt // T

        x = x.reshape(b, T, H, W, c).permute(0, 4, 1, 2, 3).contiguous()
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1).reshape(b * T, H * W, c).contiguous()
        return x


class PosFeedForward3d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        hidden_dim = embed_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dwconv = DWConv3d(hidden_dim)
        self.act = get_activation_layer(activation)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.drop = nn.Dropout(dropout)
        self.apply(_init_transformer_weights)

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, T, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoder(nn.Module):
    """
    Input shape: (B*T, H*W, C)
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        feed_forward: type[nn.Module],
        activation: str,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            SelfAttention(dim, num_heads=num_heads, dropout=dropout),
                        ),
                        PreNorm(
                            dim,
                            feed_forward(
                                dim,
                                mlp_dim,
                                dropout=dropout,
                                activation=activation,
                            ),
                        ),
                        DropPath(drop_path) if drop_path > 0.0 else nn.Identity(),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)
        self.apply(_init_transformer_weights)

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        for attn, ff, drop_path in self.layers:
            x = x + drop_path(attn(x))
            x = x + drop_path(ff(x, T=T, H=H, W=W))
        return self.norm(x)


# ============================================================
# Backbone
# ============================================================


class VistaFormerBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dims: list[int],
        patch_sizes: list[int],
        strides: list[int],
        depths: list[int],
        num_heads: list[int],
        mlp_dims: list[int],
        dropout: float,
        drop_path: float,
        gate: bool,
        use_squeeze: bool,
        activation: str,
    ):
        super().__init__()

        n = len(embed_dims)
        if not (
            n
            == len(patch_sizes)
            == len(strides)
            == len(depths)
            == len(num_heads)
            == len(mlp_dims)
        ):
            raise ValueError("All stage configuration lists must have the same length")

        self.embeddings = nn.ModuleList()
        self.transformers = nn.ModuleList()

        for i in range(n):
            stage_in_channels = embed_dims[i - 1] if i > 0 else in_channels

            self.embeddings.append(
                PatchEmbed3D(
                    in_channels=stage_in_channels,
                    out_channels=embed_dims[i],
                    patch_size=patch_sizes[i],
                    stride=strides[i],
                    use_squeeze=use_squeeze,
                    norm_type="batch2d" if i == 0 else "none",
                    gate=gate,
                )
            )

            self.transformers.append(
                TransformerEncoder(
                    dim=embed_dims[i],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    mlp_dim=mlp_dims[i],
                    feed_forward=PosFeedForward3d,
                    activation=activation,
                    dropout=dropout,
                    drop_path=drop_path,
                )
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Input:
            x: (B, T, C, H, W)

        Output:
            list of stage outputs, each shaped (B, C_i, T_i, H_i, W_i)
        """
        # x = x.permute(0, 2, 1, 3, 4).contiguous()  # -> (B, C, T, H, W)
        outputs: list[torch.Tensor] = []

        for embedding, transformer in zip(self.embeddings, self.transformers):
            x = embedding(x)  # (B, C, T, H, W)
            b, c, t, h, w = x.shape

            x_tokens = x.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c).contiguous()
            x_tokens = transformer(x_tokens, T=t, H=h, W=w)

            x = x_tokens.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
            outputs.append(x)

        return outputs


# ============================================================
# Head
# ============================================================


def get_temporal_agg_layer(
    layer_type: str,
    in_channels: int,
    out_channels: int,
    T: int,
) -> nn.Module:
    layer_type = layer_type.lower()

    if layer_type == "conv":
        return nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(T, 1, 1),
            stride=(T, 1, 1),
        )
    if layer_type == "gatedconv":
        return GatedConv3d(
            in_channels,
            out_channels,
            kernel_size=(T, 1, 1),
            stride=(T, 1, 1),
            padding=0,
        )
    if layer_type == "depthwise":
        return DepthwiseSeparableConv3d(
            in_channels,
            out_channels,
            kernel_size=(T, 1, 1),
            stride=(T, 1, 1),
            padding=0,
        )
    if layer_type == "adaptive_avg_pool":
        return AdaptiveTemporalFeaturePooling(
            in_channels,
            out_channels,
            output_temporal_size=1,
            pool_type="avg",
        )
    if layer_type == "adaptive_max_pool":
        return AdaptiveTemporalFeaturePooling(
            in_channels,
            out_channels,
            output_temporal_size=1,
            pool_type="max",
        )

    raise ValueError(
        "Invalid temporal aggregation type. Choose from "
        "'conv', 'gatedconv', 'depthwise', 'adaptive_avg_pool', or 'adaptive_max_pool'."
    )


class VistaFormerHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dims: list[int],
        seq_lens: list[int],
        num_classes: int,
        dropout: float,
        temporal_agg_type: str,
        conv_embed_dim: int = 64,
        upsample_type: str = "trilinear",
        norm_type: str = "batch",
        activation: str = "mish",
    ):
        super().__init__()

        if len(embed_dims) != len(seq_lens):
            raise ValueError("embed_dims and seq_lens must have the same length")

        if upsample_type not in {"bilinear", "trilinear", "conv"}:
            raise ValueError("upsample_type must be 'bilinear', 'trilinear', or 'conv'")

        self.num_classes = num_classes
        self.output_dim = input_dim
        self.upsample_type = upsample_type

        if upsample_type == "bilinear":
            self.upsample = nn.Upsample(
                size=(input_dim, input_dim),
                mode="bilinear",
                align_corners=False,
            )
        elif upsample_type == "conv":
            self.upsample = nn.ModuleList(
                [
                    nn.ConvTranspose3d(
                        in_channels=embed_dims[i],
                        out_channels=embed_dims[i],
                        kernel_size=(1, 2 ** (i + 1), 2 ** (i + 1)),
                        stride=(1, 2 ** (i + 1), 2 ** (i + 1)),
                    )
                    for i in range(len(embed_dims))
                ]
            )
        else:
            self.upsample = nn.ModuleList(
                [
                    nn.Upsample(
                        size=(seq_lens[i], input_dim, input_dim),
                        mode="trilinear",
                        align_corners=False,
                    )
                    for i in range(len(seq_lens))
                ]
            )

        self.temp_downsample = nn.ModuleList(
            [
                get_temporal_agg_layer(
                    temporal_agg_type,
                    embed_dims[i],
                    conv_embed_dim,
                    seq_lens[i],
                )
                for i in range(len(embed_dims))
            ]
        )

        self.fuse = nn.Conv2d(
            conv_embed_dim * len(embed_dims),
            conv_embed_dim,
            kernel_size=1,
        )

        norm_type = norm_type.lower()
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(conv_embed_dim)
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm2d(conv_embed_dim)
        elif norm_type == "group":
            groups = min(8, conv_embed_dim)
            while conv_embed_dim % groups != 0 and groups > 1:
                groups -= 1
            self.norm = nn.GroupNorm(groups, conv_embed_dim)
        elif norm_type == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(
                "norm_type must be one of: 'batch', 'instance', 'group', 'none'"
            )

        self.act = get_activation_layer(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Conv2d(conv_embed_dim, num_classes, kernel_size=1)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        x: list of tensors shaped (B, C, T, H, W)
        """
        if self.upsample_type == "bilinear":
            upsampled = []
            for x_i in x:
                b, c, t, h, w = x_i.shape
                x_i = x_i.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).contiguous()
                x_i = self.upsample(x_i)
                x_i = (
                    x_i.reshape(b, t, c, self.output_dim, self.output_dim)
                    .permute(0, 2, 1, 3, 4)
                    .contiguous()
                )
                upsampled.append(x_i)
            x = upsampled
        else:
            x = [self.upsample[i](x[i]) for i in range(len(x))]

        x = [self.temp_downsample[i](x[i]).squeeze(2) for i in range(len(x))]
        x = torch.cat(x, dim=1)
        x = self.fuse(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.out(x)
        return x


# ============================================================
# Single-Input Model
# ============================================================


class VistaFormer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_dim: int = 48,
        num_classes: int = 1,
        depths: list[int] = [1, 1, 1],
        embed_dims: list[int] = [32, 64, 128],
        seq_lens: list[int] = None,
        patch_sizes: list[int] = [2, 2, 2],
        strides: list[int] = [2, 2, 2],
        num_heads: list[int] = [4, 8, 8],
        mlp_mult: int = 4,
        gate: bool = True,
        activation: str = "mish",
        use_squeeze: bool = False,
        head_conv_dim: int = 64,
        head_upsample_type: str = "trilinear",
        head_temporal_agg_type: str = "conv",
        head_norm_type: str = "batch",
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        # handle dynamic default for seq_lens
        if seq_lens is None:
            seq_lens = [input_dim // 2, input_dim // 4, input_dim // 8]

        self.seq_lens = seq_lens

        mlp_dims = [dim * mlp_mult for dim in embed_dims]

        self.backbone = VistaFormerBackbone(
            in_channels=in_channels,
            embed_dims=embed_dims,
            patch_sizes=patch_sizes,
            strides=strides,
            depths=depths,
            num_heads=num_heads,
            mlp_dims=mlp_dims,
            dropout=dropout,
            drop_path=drop_path,
            gate=gate,
            use_squeeze=use_squeeze,
            activation=activation,
        )

        self.head = VistaFormerHead(
            input_dim=input_dim,
            embed_dims=embed_dims,
            seq_lens=seq_lens,
            num_classes=num_classes,
            dropout=dropout,
            conv_embed_dim=head_conv_dim,
            upsample_type=head_upsample_type,
            temporal_agg_type=head_temporal_agg_type,
            norm_type=head_norm_type,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ============================================================
# Fusion Modules
# ============================================================


class CrossAttentionTransformerLayer(nn.Module):
    """
    Cross-attention over flattened spatiotemporal tokens.

    Inputs:
        x1, x2: (B, N, C)
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        num_heads: int,
        dropout: float,
        drop_path: float,
        activation: str = "gelu",
    ):
        super().__init__()

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.ffn = PreNorm(
            embed_dim,
            PosFeedForward3d(
                input_dim=embed_dim,
                embed_dim=mlp_dim,
                dropout=dropout,
                activation=activation,
            ),
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        T: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        q = self.norm_q(x1)
        kv = self.norm_kv(x2)

        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = x1 + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.ffn(x, T=T, H=H, W=W))
        return x


class FeatureFusionConcat(nn.Module):
    def __init__(self, in_channels1: int, in_channels2: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels1 + in_channels2, out_channels, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat((x1, x2), dim=1))


class FeatureFusionAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.attention = nn.Sigmoid()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1_proj = self.conv1(x1)
        x2_proj = self.conv2(x2)
        weights = self.attention(x1_proj + x2_proj)
        return weights * x1_proj + (1.0 - weights) * x2_proj


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        in_embed_dim: int,
        out_embed_dim: int,
        fusion_type: str,
        dropout: float,
        drop_path: float,
        mlp_mult: int = 4,
        attn_heads: Optional[int] = None,
        use_depthwise: bool = False,
    ):
        super().__init__()
        self.fusion_type = fusion_type.lower()
        self.use_depthwise = use_depthwise

        if self.fusion_type == "concat":
            self.fusion = FeatureFusionConcat(
                in_channels1=in_embed_dim,
                in_channels2=in_embed_dim,
                out_channels=out_embed_dim,
            )
        elif self.fusion_type == "attention":
            self.fusion = FeatureFusionAttention(
                in_channels=in_embed_dim,
                out_channels=out_embed_dim,
            )
        elif self.fusion_type == "crossattn":
            if attn_heads is None:
                raise ValueError("attn_heads must be provided for fusion_type='crossattn'")
            self.fusion = CrossAttentionTransformerLayer(
                embed_dim=in_embed_dim,
                mlp_dim=in_embed_dim * mlp_mult,
                num_heads=attn_heads,
                dropout=dropout,
                drop_path=drop_path,
            )
        else:
            raise ValueError(
                "fusion_type must be one of: 'concat', 'attention', 'crossattn'"
            )

        if use_depthwise:
            self.dw_conv1 = nn.Conv3d(
                in_embed_dim,
                in_embed_dim,
                kernel_size=3,
                padding=1,
                groups=in_embed_dim,
            )
            self.dw_conv2 = nn.Conv3d(
                in_embed_dim,
                in_embed_dim,
                kernel_size=3,
                padding=1,
                groups=in_embed_dim,
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1, x2: (B, C, T, H, W)
        """
        b, c, t, h, w = x1.shape

        if self.use_depthwise:
            x1 = self.dw_conv1(x1)
            x2 = self.dw_conv2(x2)

        if self.fusion_type == "crossattn":
            x1_tokens = x1.view(b, c, -1).transpose(1, 2).contiguous()  # (B, N, C)
            x2_tokens = x2.view(b, c, -1).transpose(1, 2).contiguous()  # (B, N, C)

            x = self.fusion(x1_tokens, x2_tokens, T=t, H=h, W=w)
            x = x.transpose(1, 2).reshape(b, c, t, h, w).contiguous()
        else:
            x = self.fusion(x1, x2)

        return self.dropout(x)


# ============================================================
# Auxiliary Head
# ============================================================


class AuxLayer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        scale_factor: int,
        in_channels: int,
        num_classes: int,
    ):
        super().__init__()
        self.temporal_pool = nn.AvgPool3d(
            kernel_size=(seq_len, 1, 1),
            stride=(seq_len, 1, 1),
        )
        self.upsample = nn.Upsample(
            scale_factor=(scale_factor, scale_factor),
            mode="bilinear",
            align_corners=False,
        )
        self.class_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_pool(x).squeeze(2)
        x = self.upsample(x)
        x = self.class_conv(x)
        return x


# ============================================================
# Multi-Input Model
# ============================================================


class VistaFormerMulti(nn.Module):
    def __init__(
        self,
        first_in_channels: int,
        second_in_channels: int,
        input_dim: int,
        num_classes: int,
        depths: list[int],
        embed_dims: list[int],
        seq_lens: list[int],
        patch_sizes: list[int],
        strides: list[int],
        num_heads: list[int],
        mlp_mult: int,
        gate: bool,
        padding: Optional[list[int]] = None,
        fusion_type: str = "concat",
        activation: str = "gelu",
        use_squeeze: bool = False,
        head_conv_dim: int = 64,
        head_upsample_type: str = "trilinear",
        head_temporal_agg_type: str = "depthwise",
        head_norm_type: str = "batch",
        dropout: float = 0.0,
        drop_path: float = 0.0,
        aux_loss_weight: float = 0.0,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        _ = padding  # currently unused, preserved for API compatibility

        mlp_dims = [dim * mlp_mult for dim in embed_dims]

        self.backbone1 = VistaFormerBackbone(
            in_channels=first_in_channels,
            embed_dims=embed_dims,
            patch_sizes=patch_sizes,
            strides=strides,
            depths=depths,
            num_heads=num_heads,
            mlp_dims=mlp_dims,
            dropout=dropout,
            drop_path=drop_path,
            gate=gate,
            use_squeeze=use_squeeze,
            activation=activation,
        )

        self.backbone2 = VistaFormerBackbone(
            in_channels=second_in_channels,
            embed_dims=embed_dims,
            patch_sizes=patch_sizes,
            strides=strides,
            depths=depths,
            num_heads=num_heads,
            mlp_dims=mlp_dims,
            dropout=dropout,
            drop_path=drop_path,
            gate=gate,
            use_squeeze=use_squeeze,
            activation=activation,
        )

        self.fusion_blocks = nn.ModuleList(
            [
                FeatureFusionBlock(
                    in_embed_dim=embed_dims[i],
                    out_embed_dim=head_conv_dim,
                    fusion_type=fusion_type,
                    attn_heads=num_heads[i],
                    dropout=dropout,
                    drop_path=drop_path,
                )
                for i in range(len(embed_dims))
            ]
        )

        head_embed_dims = (
            embed_dims if fusion_type.lower() == "crossattn" else [head_conv_dim] * len(embed_dims)
        )

        self.head = VistaFormerHead(
            input_dim=input_dim,
            embed_dims=head_embed_dims,
            seq_lens=seq_lens,
            num_classes=num_classes,
            dropout=dropout,
            conv_embed_dim=head_conv_dim,
            upsample_type=head_upsample_type,
            temporal_agg_type=head_temporal_agg_type,
            norm_type=head_norm_type,
            activation=activation,
        )

        self.ignore_index = ignore_index
        self.aux_loss_weight = aux_loss_weight

        if aux_loss_weight > 0.0:
            aux_in_channels = (
                embed_dims if fusion_type.lower() == "crossattn" else [head_conv_dim] * len(embed_dims)
            )
            self.auxiliary_heads = nn.ModuleList(
                [
                    AuxLayer(
                        seq_len=seq_lens[i],
                        scale_factor=2 ** (i + 1),
                        in_channels=aux_in_channels[i],
                        num_classes=num_classes,
                    )
                    for i in range(len(embed_dims))
                ]
            )
        else:
            self.auxiliary_heads = None

    def compute_aux_loss(
        self,
        aux_outputs: list[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if not aux_outputs:
            return target.new_tensor(0.0, dtype=torch.float32)

        aux_loss = 0.0
        for aux_output in aux_outputs:
            aux_loss = aux_loss + F.cross_entropy(
                aux_output,
                target,
                ignore_index=self.ignore_index if self.ignore_index is not None else -100,
            )

        aux_loss = aux_loss / len(aux_outputs)
        return aux_loss * self.aux_loss_weight

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        x_outs = self.backbone1(x)
        y_outs = self.backbone2(y)

        fused_outputs: list[torch.Tensor] = []
        aux_outputs: list[torch.Tensor] = []

        for i, fusion_block in enumerate(self.fusion_blocks):
            out = fusion_block(x_outs[i], y_outs[i])
            fused_outputs.append(out)

            if return_aux and self.auxiliary_heads is not None:
                aux_outputs.append(self.auxiliary_heads[i](out))

        logits = self.head(fused_outputs)

        if return_aux:
            return logits, aux_outputs

        return logits


# ============================================================
# Quick sanity test
# ============================================================

if __name__ == "__main__":
    model = VistaFormer(
        in_channels=3,
        input_dim=64,
        num_classes=5,
        depths=[1, 1, 1],
        embed_dims=[32, 64, 128],
        seq_lens=[4, 2, 1],
        patch_sizes=[2, 2, 2],
        strides=[2, 2, 2],
        num_heads=[4, 8, 8],
        mlp_mult=4,
        gate=True,
        activation="gelu",
        use_squeeze=False,
        head_conv_dim=64,
        head_upsample_type="trilinear",
        head_temporal_agg_type="conv",
        head_norm_type="batch",
        dropout=0.1,
        drop_path=0.1,
    )

    x = torch.randn(2, 3, 8, 64, 64)  # (B, T, C, H, W)
    y = model(x)
    print("VistaFormer output:", y.shape)

    multi = VistaFormerMulti(
        first_in_channels=3,
        second_in_channels=2,
        input_dim=64,
        num_classes=5,
        depths=[1, 1, 1],
        embed_dims=[32, 64, 128],
        seq_lens=[4, 2, 1],
        patch_sizes=[2, 2, 2],
        strides=[2, 2, 2],
        num_heads=[4, 8, 8],
        mlp_mult=4,
        gate=True,
        fusion_type="concat",
        activation="gelu",
        head_conv_dim=64,
        head_upsample_type="trilinear",
        head_temporal_agg_type="conv",
        head_norm_type="batch",
    )

    x1 = torch.randn(2, 3, 8, 64, 64)
    x2 = torch.randn(2, 2, 8, 64, 64)
    y2 = multi(x1, x2)
    print("VistaFormerMulti output:", y2.shape)