# https://github.com/eloialonso/diamond/blob/main/src/models/diffusion/inner_model.py
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers
from jax.typing import ArrayLike
from surrogate.image_blocks import Conv3x3, FourierFeatures, GroupNorm, UNet
from surrogate.torch_to_flax_helpers import torch_conv_init


class InnerModel(nnx.Module):
    def __init__(
        self,
        img_channels,
        num_steps_conditioning,
        cond_channels,
        depths,
        channels,
        attn_depths,
        num_abm_params,
        static_h,
        static_w,
        depthwise_sep_conv_unet: bool = False,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs=nnx.Rngs,
    ):
        self.noise_emb = FourierFeatures(
            cond_channels, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        # NOTE: Here we deviate from DIAMOND.
        # DIAMOND conditions on the actions associated with
        # all the previous frames
        # In our case we will just run the same
        # parameters the whole way through
        # Also our params are continous, so we don't embed
        self.abm_params_proj = nnx.Linear(
            num_abm_params,
            cond_channels,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.cond_proj = nnx.Sequential(
            nnx.Linear(
                cond_channels,
                cond_channels,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            ),
            nnx.silu,
            nnx.Linear(
                cond_channels,
                cond_channels,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            ),
        )

        self.conv_in = Conv3x3(dwsc=False)(
            (num_steps_conditioning + 1) * img_channels,
            channels[0],
            kernel_init=torch_conv_init(
                (num_steps_conditioning + 1) * img_channels, jnp.array([3, 3])
            ),
            bias_init=torch_conv_init(
                (num_steps_conditioning + 1) * img_channels, jnp.array([3, 3])
            ),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.unet = UNet(
            cond_channels,
            depths,
            channels,
            attn_depths,
            static_h,
            static_w,
            depthwise_sep_conv=depthwise_sep_conv_unet,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.norm_out = GroupNorm(
            channels[0], dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.conv_out = Conv3x3(dwsc=False)(
            channels[0],
            img_channels,
            kernel_init=initializers.zeros,
            bias_init=torch_conv_init(channels[0], jnp.array([3, 3])),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        noisy_next_obs: ArrayLike,
        c_noise: ArrayLike,
        obs: ArrayLike,
        abm_param: ArrayLike,
    ):
        cond = self.cond_proj(self.noise_emb(c_noise) + self.abm_params_proj(abm_param))
        x = self.conv_in(jnp.concat((obs, noisy_next_obs), axis=-1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(nnx.silu(self.norm_out(x)))
        return x
