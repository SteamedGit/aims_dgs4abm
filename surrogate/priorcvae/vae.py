"""
File contains the variational autoencoder (VAE) class.

The class is baed on the flax VAE example: https://github.com/google/flax/blob/main/examples/vae/train.py.
"""

from flax import nnx
from functools import partial
from typing import Tuple, Union
import jax
from jax import random
import jax.numpy as jnp
from flax import linen as nn
from omegaconf import DictConfig
import omegaconf
import hydra
from jax.typing import ArrayLike
from typing import Union, Dict, Optional


class VAE(nnx.Module):
    def __init__(
        self,
        encoder_conf: DictConfig,
        decoder_conf: DictConfig,
        path_length: int,
        params_dim: int,
        latent_dim: int,
        vae_var: int,
        rngs: nnx.Rngs,
    ):
        self.encoder = hydra.utils.get_class(encoder_conf.enc_class)(
            **omegaconf.OmegaConf.to_object(encoder_conf.arch),
            latent_dim=latent_dim,
            input_dim=path_length + params_dim,
            rngs=nnx.Rngs(0),
        )
        assert self.encoder.hidden_layers[0].in_features == (
            path_length + params_dim
        ), (
            f"Input dimension of encoder must match path length + ABM parameter dimension, "
            f"{self.encoder.hidden_layers[0].in_features} != {path_length + params_dim} "
        )

        self.decoder = hydra.utils.get_class(decoder_conf.dec_class)(
            **omegaconf.OmegaConf.to_object(decoder_conf.arch),
            out_dim=path_length,
            input_dim=latent_dim + params_dim,
            rngs=nnx.Rngs(0),
        )
        assert self.decoder.hidden_layers[0].in_features == (latent_dim + params_dim), (
            f"Input dimension of decoder must match latent dimension + ABM parameter dimension, "
            f"{self.decoder.hidden_layers[0].in_features} != {latent_dim + params_dim}"
        )
        self.latent_dim = latent_dim
        self.vae_var = vae_var

    def __call__(
        self,
        y: ArrayLike,
        z_key: jax.random.PRNGKey,
        c: Optional[ArrayLike] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        def reparameterize(
            z_rng: jax.random.PRNGKey, mean: ArrayLike, logvar: ArrayLike
        ) -> jax.Array:
            """Sampling using the reparameterization trick."""
            std = jnp.exp(0.5 * logvar)
            eps = random.normal(z_rng, logvar.shape)
            return mean + eps * std

        if c is not None:
            y = jnp.concatenate([y, c], axis=-1)

        z_mu, z_logvar = self.encoder(y)
        z = reparameterize(z_key, z_mu, z_logvar)

        if c is not None:
            z = jnp.concatenate([z, c], axis=-1)

        y_hat = self.decoder(z)

        return y_hat, z_mu, z_logvar

    def generate_decoder_samples(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        c: Optional[ArrayLike] = None,
    ) -> jax.Array:
        z = jax.random.normal(key, (num_samples, self.latent_dim))
        if c is not None:
            z = jnp.concatenate([z, c], axis=-1)
        x = self.decoder(z)
        return x


# Losses from https://github.com/elizavetasemenova/PriorCVAE_JAX/tree/2db074ea6e297d521d90fb70952318a75994e14d/priorCVAE/losses


@jax.jit
def kl_divergence(mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """
    Kullback-Leibler divergence between the normal distribution given by the mean and logvar and the unit Gaussian
    distribution.
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions

        KL[N(m, S) || N(0, I)] = -0.5 * (1 + log(diag(S)) - diag(S) - m^2)

    Detailed derivation can be found here: https://learnopencv.com/variational-autoencoder-in-tensorflow/

    :param mean: the mean of the Gaussian distribution with shape (N,).
    :param logvar: the log-variance of the Gaussian distribution with shape (N,) i.e. only diagonal values considered.

    :return: the KL divergence value.
    """
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.jit
def scaled_sum_squared_loss(
    y: jnp.ndarray, reconstructed_y: jnp.ndarray, vae_var: float = 1.0
) -> jnp.ndarray:
    """
    Scaled sum squared loss, i.e.

    L(y, y') = 0.5 * sum(((y - y')^2) / vae_var)

    Note: This loss can be considered as negative log-likelihood as:

    -1 * log N (y | y', sigma) \approx -0.5 ((y - y'/sigma)^2)

    :param y: the ground-truth value of y with shape (N, D).
    :param reconstructed_y: the reconstructed value of y with shape (N, D).
    :param vae_var: a float value representing the varianc of the VAE.

    :returns: the loss value
    """
    assert y.shape == reconstructed_y.shape
    return 0.5 * jnp.sum((reconstructed_y - y) ** 2 / vae_var)


@jax.jit
def mean_squared_loss(y: jnp.ndarray, reconstructed_y: jnp.ndarray) -> jnp.ndarray:
    """
    Mean squared loss, MSE i.e.

    L(y, y') = mean(((y - y')^2))

    :param y: the ground-truth value of y with shape (N, D).
    :param reconstructed_y: the reconstructed value of y with shape (N, D).

    :returns: the loss value
    """
    assert y.shape == reconstructed_y.shape
    return jnp.mean((reconstructed_y - y) ** 2)


def squared_sum_and_kl_loss(
    model: nnx.Module, batch: Dict[str, ArrayLike], key: jax.random.PRNGKey
) -> jax.Array:
    pred, z_mu, z_logvar = model(y=batch["realisation"], z_key=key, c=batch["param"])
    rcl_loss = scaled_sum_squared_loss(
        batch["realisation"], pred, vae_var=model.vae_var
    )
    kld_loss = kl_divergence(z_mu, z_logvar)
    loss = rcl_loss + kld_loss
    return loss
