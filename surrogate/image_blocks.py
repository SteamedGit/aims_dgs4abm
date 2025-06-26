# https://github.com/eloialonso/diamond/blob/main/src/models/blocks.py
from flax import nnx
from flax.nnx import initializers
import jax
import jax.numpy as jnp
from functools import partial
from jax.typing import ArrayLike
from typing import List, Optional
import math
from surrogate.torch_to_flax_helpers import torch_conv_init


# Settings for GroupNorm and Attention

GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8

# default_kernel_init = initializers.lecun_normal()
# default_bias_init = initializers.zeros_init()


# Depthwise Separable Convolutions
# https://arxiv.org/pdf/1704.04861
class DepthWiseSeparableConv(nnx.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size,
        strides,
        padding,
        kernel_init=initializers.lecun_normal(),
        bias_init=initializers.zeros_init(),
        *,
        use_bias=True,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.layers = nnx.Sequential(
            # Depthwise Convolution
            nnx.Conv(
                in_features=in_features,
                out_features=in_features,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                feature_group_count=in_features,  # This makes it a depthwise convolution
                use_bias=use_bias,
                kernel_init=kernel_init,
                bias_init=bias_init,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            ),
            # Pointwise Convolution
            nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(1, 1),
                strides=1,
                padding=0,
                use_bias=use_bias,
                kernel_init=kernel_init,
                bias_init=bias_init,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            ),
        )

    def __call__(self, x):
        return self.layers(x)


# Convs
Conv1x1 = (
    lambda dwsc: partial(
        DepthWiseSeparableConv, kernel_size=(1, 1), strides=1, padding=0
    )
    if dwsc
    else partial(nnx.Conv, kernel_size=(1, 1), strides=1, padding=0)
)

Conv3x3 = (
    lambda dwsc: partial(
        DepthWiseSeparableConv, kernel_size=(3, 3), strides=1, padding=1
    )
    if dwsc
    else partial(nnx.Conv, kernel_size=(3, 3), strides=1, padding=1)
)


# Identity Layer
class IdentityLayer(nnx.Module):
    def __call__(self, x):
        return x


# GroupNorm and conditional GroupNorm
class GroupNorm(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.norm = nnx.GroupNorm(
            num_features=in_channels,
            num_groups=num_groups,
            epsilon=GN_EPS,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        return self.norm(x)


class AdaGroupNorm(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.linear = nnx.Linear(
            cond_channels,
            in_channels * 2,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.no_params_gn = nnx.GroupNorm(
            num_features=in_channels,
            num_groups=self.num_groups,
            epsilon=GN_EPS,
            use_bias=False,
            use_scale=False,
            dtype=dtype,
            param_dtype=param_dtype,  # For consistency
            rngs=rngs,
        )

    def __call__(self, x, cond):
        assert x.shape[-1] == self.in_channels
        x_normed = self.no_params_gn(x)

        cond_transformed = self.linear(cond)
        scale, shift = jnp.split(
            cond_transformed.reshape(-1, 1, 1, 2 * self.in_channels), 2, axis=3
        )
        return x_normed * (1 + scale) + shift


# Self Attention for images
# dtype: jnp.dtype = jnp.float32, param_dtype: jnp.dtype = jnp.float32,
# dtype=dtype, param_dtype=param_dtype,


class SelfAttention2d(nnx.Module):
    def __init__(
        self,
        in_channels,
        head_dim: int = ATTN_HEAD_DIM,
        depthwise_sep_conv: bool = False,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.norm = GroupNorm(
            in_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.qkv_proj = Conv1x1(depthwise_sep_conv)(
            in_features=in_channels,
            out_features=in_channels * 3,
            kernel_init=torch_conv_init(in_channels, jnp.ones((2,))),
            bias_init=torch_conv_init(in_channels, jnp.ones((2,))),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.out_proj = Conv1x1(depthwise_sep_conv)(
            in_features=in_channels,
            out_features=in_channels,
            kernel_init=initializers.zeros,
            bias_init=initializers.zeros,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike):
        n, h, w, c = x.shape
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(n, self.n_head * 3, c // self.n_head, h * w)
        qkv = jnp.transpose(qkv, (0, 1, 3, 2))
        q, k, v = jnp.array_split(qkv, 3, axis=1)

        # NOTE: Currently infeerring from inputs
        y = nnx.dot_product_attention(q, k, v)
        y = y.transpose(0, 1, 3, 2).reshape(n, h, w, c)
        return x + self.out_proj(y)


# Embedding of the noise level


class FourierFeatures(nnx.Module):
    def __init__(
        self,
        cond_channels: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        assert cond_channels % 2 == 0
        keys_for_weights = rngs.params()
        self.dtype = dtype

        self.weight = nnx.Variable(
            jax.random.normal(
                keys_for_weights, (1, cond_channels // 2), dtype=param_dtype
            ),
            collection="buffers",
        )

    def __call__(self, x: ArrayLike):
        assert x.ndim == 1
        f = 2 * jnp.pi * x[:, jnp.newaxis] @ self.weight

        return jnp.concat([jnp.cos(f), jnp.sin(f)], axis=1).astype(self.dtype)


# [Down|Up]sampling


class DownSample(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        depthwise_sep_conv: bool = False,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if depthwise_sep_conv:
            self.conv = DepthWiseSeparableConv(
                in_features=in_channels,
                out_features=in_channels,
                kernel_size=(3, 3),
                strides=2,
                padding=1,
                kernel_init=initializers.orthogonal(),  # NOTE: Need to call orthogonal to get initialiser whereas for zeros you pass in the func
                bias_init=torch_conv_init(in_channels, jnp.array([3, 3])),
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        else:
            self.conv = nnx.Conv(
                in_features=in_channels,
                out_features=in_channels,
                kernel_size=(3, 3),
                strides=2,
                padding=1,
                kernel_init=initializers.orthogonal(),  # NOTE: Need to call orthogonal to get initialiser whereas for zeros you pass in the func
                bias_init=torch_conv_init(in_channels, jnp.array([3, 3])),
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def __call__(self, x):
        return self.conv(x)


class UpSample(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        depthwise_sep_conv: bool = False,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs=nnx.Rngs,
    ):
        self.conv = Conv3x3(depthwise_sep_conv)(
            in_channels,
            in_channels,
            kernel_init=torch_conv_init(in_channels, jnp.array([3, 3])),
            bias_init=torch_conv_init(in_channels, jnp.array([3, 3])),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        n, h, w, c = x.shape
        x = jax.image.resize(image=x, shape=(n, 2 * h, 2 * w, c), method="nearest")
        return self.conv(x)


# Residual block (conditioning with AdaGroupNorm, no [down|up]sampling, optional self-attention)


class ResBlock(nnx.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_channels,
        attn: bool,
        depthwise_sep_conv: bool = False,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        if in_channels == out_channels:
            self.proj = IdentityLayer()  # lambda x: x
        else:
            self.proj = Conv1x1(depthwise_sep_conv)(
                in_features=in_channels,
                out_features=out_channels,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        self.norm1 = AdaGroupNorm(
            in_channels, cond_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.conv1 = Conv3x3(depthwise_sep_conv)(
            in_features=in_channels,
            out_features=out_channels,
            kernel_init=torch_conv_init(in_channels, jnp.array([3, 3])),
            bias_init=torch_conv_init(in_channels, jnp.array([3, 3])),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.norm2 = AdaGroupNorm(
            out_channels, cond_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.conv2 = Conv3x3(depthwise_sep_conv)(
            in_features=out_channels,
            out_features=out_channels,
            kernel_init=initializers.zeros,
            bias_init=torch_conv_init(out_channels, jnp.array([3, 3])),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        if attn:
            self.attn = SelfAttention2d(
                out_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )
        else:
            self.attn = IdentityLayer()  # lambda x: x

    def __call__(self, x, cond):
        residual = self.proj(x)
        x = self.conv1(nnx.silu(self.norm1(x, cond)))
        x = self.conv2(nnx.silu(self.norm2(x, cond)))
        x = x + residual
        x = self.attn(x)
        return x


# Sequence of residual blocks (in_channels -> mid_channels -> ... -> mid_channels -> out_channels)
class ResBlocks(nnx.Module):
    def __init__(
        self,
        list_in_channels: List[int],
        list_out_channels: List[int],
        cond_channels: int,
        attn: bool,
        depthwise_sep_conv: bool = False,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        assert len(list_in_channels) == len(list_out_channels)
        self.in_channels = list_in_channels[0]
        self.resblocks = [
            ResBlock(
                in_ch,
                out_ch,
                cond_channels,
                attn,
                depthwise_sep_conv=depthwise_sep_conv,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            for (in_ch, out_ch) in zip(list_in_channels, list_out_channels)
        ]

    def __call__(self, x, cond, to_cat: Optional[List[ArrayLike]] = None):
        outputs = []
        for i, resblock in enumerate(self.resblocks):
            x = x if to_cat is None else jnp.concat([x, to_cat[i]], axis=-1)
            x = resblock(x, cond)
            outputs.append(x)
        return x, outputs


def pad_img(h, w, n, x):
    # NOTE: Had to use math.ceil insteead of jnp.ceil
    # Despite everythoing actually being concrete when partial is applied
    # using jnp.ceil and jnp.int32 leads to the padding values being
    # treated as tracers
    padding_h = math.ceil(h / 2**n) * 2**n - h
    padding_w = math.ceil(w / 2**n) * 2**n - w

    x = jnp.pad(
        x,
        ((0, 0), (0, padding_h), (0, padding_w), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return x


class UNet(nnx.Module):
    def __init__(
        self,
        cond_channels: int,
        depths: List[int],
        channels: List[int],
        attn_depths: List[int],
        static_h: int,
        static_w: int,
        depthwise_sep_conv: bool = False,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        assert len(depths) == len(channels) == len(attn_depths)
        self._num_down = len(channels) - 1
        # So that we can pad inside jit
        self.pad_img = partial(pad_img, static_h, static_w, self._num_down)

        d_blocks, u_blocks = [], []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            d_blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                    depthwise_sep_conv=depthwise_sep_conv,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                )
            )
            u_blocks.append(
                ResBlocks(
                    list_in_channels=[2 * c2] * n + [c1 + c2],
                    list_out_channels=[c2] * n + [c1],
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                    depthwise_sep_conv=depthwise_sep_conv,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                )
            )
        self.d_blocks = d_blocks
        self.u_blocks = list(reversed(u_blocks))
        self.mid_blocks = ResBlocks(
            list_in_channels=[channels[-1]] * 2,
            list_out_channels=[channels[-1]] * 2,
            cond_channels=cond_channels,
            attn=True,
            depthwise_sep_conv=depthwise_sep_conv,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.downsamples = [IdentityLayer()] + [
            DownSample(
                c,
                depthwise_sep_conv=depthwise_sep_conv,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            for c in channels[:-1]
        ]
        self.upsamples = [IdentityLayer()] + [
            UpSample(
                c,
                depthwise_sep_conv=depthwise_sep_conv,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            for c in reversed(channels[:-1])
        ]

    def __call__(self, x: ArrayLike, cond: ArrayLike) -> jax.Array:
        _, h, w, c = x.shape
        x = self.pad_img(x)

        d_outputs = []
        for block, down in zip(self.d_blocks, self.downsamples):
            x_down = down(x)
            x, block_outputs = block(x_down, cond)
            d_outputs.append((x_down, *block_outputs))

        x, _ = self.mid_blocks(x, cond)

        u_outputs = []
        for block, up, skip in zip(self.u_blocks, self.upsamples, reversed(d_outputs)):
            x_up = up(x)
            x, block_outputs = block(x_up, cond, skip[::-1])
            u_outputs.append((x_up, *block_outputs))
        x = x[:, :h, :w, :]
        return x, d_outputs, u_outputs
