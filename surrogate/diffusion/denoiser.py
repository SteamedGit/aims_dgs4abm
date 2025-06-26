# https://github.com/eloialonso/diamond/blob/main/src/models/diffusion/denoiser.py
# EDM Denoiser from DIAMOND
from flax import nnx
import jax
import jax.numpy as jnp
from surrogate.diffusion.inner_model import InnerModel
from dataclasses import dataclass
from typing import Optional, Dict
from jax.typing import ArrayLike
import optax


# NOTE: Experimental snapping
def find_closest_palette_color(pixel_vector, palette_colors):
    """
    Finds the color in palette_colors closest to pixel_vector.

    Args:
        pixel_vector (jax.Array): A 1D array of shape (C,) representing a single pixel's color values.
        palette_colors (jax.Array): A 2D array of shape (N, C) representing the palette
                                    of N colors, each with C channels.

    Returns:
        jax.Array: The color vector from palette_colors (shape (C,)) that is closest
                   to pixel_vector.
    """
    # Calculate squared Euclidean distances.
    # (palette_colors - pixel_vector) uses broadcasting:
    #   palette_colors: (N, C)
    #   pixel_vector:   (C,)  -> effectively (1, C) for subtraction
    #   Resulting diff: (N, C)
    differences = palette_colors - pixel_vector
    # Sum over the C channels, result shape (N,)
    distances_sq = jnp.sum(differences**2, axis=1)

    # Find the index of the palette color with the minimum distance.
    closest_index = jnp.argmin(distances_sq)

    # Return the closest color from the palette.
    return palette_colors[closest_index]


def map_image_to_palette(image_data, palette):
    """
    Maps each pixel in an image (or a batch of images) to its closest color from a given palette.
    If image_data is 3D (H, W, C), it processes a single image.
    If image_data is 4D (B, H, W, C), it processes a batch of images.

    Args:
        image_data (jax.Array): A 3D array of shape (H, W, C) for a single image,
                                 or a 4D array of shape (B, H, W, C) for a batch of images.
                                 B is batch size, H is height, W is width, C is number of channels.
        palette (jax.Array): A 2D array of shape (N, C), where N is the number of
                             colors in the palette.

    Returns:
        jax.Array: An array with the same shape as image_data, where each pixel's color
                   is replaced by the closest color from the palette.
    """

    # 1. Vectorize `find_closest_palette_color` to work on a row of pixels.
    # `in_axes=(0, None)` means:
    #   - Map over the 0-th axis of the first argument (`pixel_vector`s in a row).
    #   - The second argument (`palette_colors`) is fixed (not mapped over).
    # This function will take a row of pixels (W, C) and the palette (N, C),
    # and return a mapped row of pixels (W, C).
    map_row_to_palette = jax.vmap(find_closest_palette_color, in_axes=(0, None))

    # 2. Vectorize `map_row_to_palette` to work on an entire single image (all rows).
    # `in_axes=(0, None)` means:
    #   - Map over the 0-th axis of the first argument (rows of an `image_data`).
    #   - The second argument (`palette`) is fixed.
    # This function will take a single image_data (H, W, C) and the palette (N, C),
    # and return the fully mapped image (H, W, C).
    map_single_image_pixels = jax.vmap(map_row_to_palette, in_axes=(0, None))

    # Check if the input is a batch of images or a single image based on ndim
    # Batched input (B, H, W, C)
    # 3. Vectorize `map_single_image_pixels` to work on a batch of images.
    # `in_axes=(0, None)` means:
    #   - Map over the 0-th axis of the first argument (images in the `image_data`).
    #   - The second argument (`palette`) is fixed.
    # This function will take the batch of images (B, H, W, C) and the palette (N, C),
    # and return the batch of mapped images (B, H, W, C).
    map_batch_pixels = jax.vmap(map_single_image_pixels, in_axes=(0, None))
    mapped_output = map_batch_pixels(image_data, palette)

    return mapped_output


def add_dims(x: ArrayLike, n: int) -> jax.Array:
    return x.reshape(x.shape + (1,) * (n - x.ndim))


@dataclass
class Conditioners:
    c_in: jax.Array
    c_out: jax.Array
    c_skip: jax.Array
    c_noise: jax.Array


@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float


@dataclass
class DenoiserConfig:
    inner_model: Dict  # InnerModelConfig
    sigma_data: float
    # NOTE: Not from EDM. Trick introduced in https://www.crosslabs.org/blog/diffusion-with-offset-noise
    # Intuition: Noise does not completely destroy low frequencies in the image
    # so the model can pick up on low frequencies from the uncorrupted image
    # during training and never has to create them. However, at inference
    # it starts from truly random noise and may struggle to produce the low
    # frequencies. Adding offset noise during training makes the model learn
    # to modify the low frequencies  (See also: https://github.com/eloialonso/diamond/issues/36)
    sigma_offset_noise: float


class Denoiser(nnx.Module):
    def __init__(
        self,
        cfg: DenoiserConfig,
        rgb_palette=None,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs=nnx.Rngs(0),
    ):
        self.cfg = cfg
        self.inner_model = InnerModel(
            **cfg.inner_model, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.sample_sigma_training = None

        self.rngs = rngs  # nnx.Variable(rngs)
        self.rgb_palette = (
            nnx.Variable(rgb_palette) if rgb_palette is not None else None
        )

    def setup_training(self, cfg: SigmaDistributionConfig):
        assert self.sample_sigma_training is None, (
            "Sample sigma training already setup!"
        )

        # Sample noise level/time from a lognormal distribution
        def sample_sigma(n: int):
            s = jax.random.normal(self.rngs.default()) * cfg.scale + cfg.loc
            return jnp.exp(s).clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma

    def apply_noise(
        self, x: ArrayLike, sigma: ArrayLike, sigma_offset_noise: float
    ) -> jax.Array:
        b, _, _, c = x.shape
        # NOTE: See comment in Denoiser Conf above
        offset_noise = sigma_offset_noise * jax.random.normal(
            self.rngs.default(),
            (b, 1, 1, c),
        )
        return (
            x
            + offset_noise
            + jax.random.normal(self.rngs.default(), x.shape) * add_dims(sigma, x.ndim)
        )

    def compute_conditioners(self, sigma: ArrayLike) -> Conditioners:
        sigma = jnp.sqrt(sigma**2 + self.cfg.sigma_offset_noise**2)
        c_in = 1 / jnp.sqrt((sigma**2 + self.cfg.sigma_data**2))
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * jnp.sqrt(c_skip)
        c_noise = jnp.log(sigma) / 4
        return Conditioners(
            *(
                add_dims(c, n)
                for c, n in zip((c_in, c_out, c_skip, c_noise), (4, 4, 4, 1, 1))
            )
        )

    def compute_model_output(
        self,
        noisy_next_obs: ArrayLike,
        obs: ArrayLike,
        abm_params: ArrayLike,
        cs: Conditioners,
    ) -> jax.Array:
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * cs.c_in
        return self.inner_model(rescaled_noise, cs.c_noise, rescaled_obs, abm_params)

    def wrap_model_output(
        self, noisy_next_obs: ArrayLike, model_output: ArrayLike, cs: Conditioners
    ) -> jax.Array:
        # Actual denoised prediction
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        # Quantize to {0, ..., 255}, then into [0,1]
        d = (((jnp.clip(d, -1.0, 1.0) + 1) / 2.0) * 255).astype(jnp.uint8).astype(
            jnp.float32
        ) / 255  # * 2.0 - 1

        #################################################################
        # Optionally: Snap x_0 prediction to be within our rgb palette. #
        # This is can help with predicting discrete outputs.            #
        #################################################################
        if self.rgb_palette is not None:
            # Map into the rgb palette by finding each pixel's closest
            # colour in the palette using MSE
            d = map_image_to_palette(d, self.rgb_palette)

        # Then back to [-1, 1]
        d = 2 * d - 1
        return d

    def denoise(
        self,
        noisy_next_obs: ArrayLike,
        sigma: ArrayLike,
        obs: ArrayLike,
        abm_params: ArrayLike,
    ) -> jax.Array:
        cs = self.compute_conditioners(sigma)
        model_output = self.compute_model_output(noisy_next_obs, obs, abm_params, cs)
        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return denoised

    @nnx.jit
    def __call__(self, batch):
        n = self.cfg.inner_model["num_steps_conditioning"]
        seq_length = batch["obs"].shape[1] - n
        # Shape (B,Seq Length, H,W,C)
        all_obs = jnp.copy(batch["obs"])
        loss = 0

        for i in range(seq_length):
            obs = all_obs[:, i : n + i]
            next_obs = all_obs[:, n + i]
            abm_params = batch["abm_params"]  # [:, i : n + i]

            b, t, h, w, c = obs.shape
            # Frame Stack
            obs = obs.reshape(b, h, w, t * c)
            sigma = self.sample_sigma_training(b)
            noisy_next_obs = self.apply_noise(
                next_obs, sigma, self.cfg.sigma_offset_noise
            )
            cs = self.compute_conditioners(sigma)

            model_output = self.compute_model_output(
                noisy_next_obs, obs, abm_params, cs
            )

            target = (next_obs - cs.c_skip * noisy_next_obs) / cs.c_out
            loss += optax.l2_loss(model_output, target).mean()

            denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
            # Feed our predictions back in
            # We are conditioning our model on its own (single step) predictions during training
            all_obs = all_obs.at[:, n + i].set(denoised)
        loss /= seq_length  # NOTE: They have this in DIAMOND because not all sequences have the same length but in our case they always are so its probably uncessecary
        return loss  # , {"loss_denoising": jax.lax.stop_gradient(loss.mean())}
