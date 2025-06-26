"""
File contains the Encoder models.
"""

from abc import ABC
from typing import Tuple, Union

from flax import linen as nn
from flax import nnx
import jax.numpy as jnp
from jaxlib.xla_extension import PjitFunction
import hydra
from jax.typing import ArrayLike
from typing import Tuple
import jax


####NNX version of the PriorCVAE Encoder####


class Encoder(ABC, nnx.Module):
    """Parent class for encoder model."""

    def __init__(self):
        super().__init__()


class MLPEncoder(Encoder):
    def __init__(
        self,
        hidden_dim: Union[Tuple[Tuple[int, int]], Tuple[int, int]],
        latent_dim: int,
        input_dim: int,
        rngs: nnx.Rngs,
        hidden_activations: Union[Tuple, PjitFunction] = nnx.sigmoid,
    ):
        hidden_dims = (
            [hidden_dim]
            if isinstance(hidden_dim, list) and isinstance(hidden_dim[0], int)
            else hidden_dim
        )
        hidden_dims = [[input_dim, hidden_dims[0][0]]] + (hidden_dims)

        self.latent_dim = latent_dim
        self.hidden_activations = (
            [hidden_activations] * len(hidden_dims)
            if not isinstance(hidden_activations, list)
            else hidden_activations
        )
        self.hidden_activations = (
            list(map(hydra.utils.get_method, self.hidden_activations))
            if isinstance(self.hidden_activations[0], str)
            else self.hidden_activations
        )
        self.hidden_layers = []
        for hidden_dim_pair in hidden_dims:
            self.hidden_layers.append(
                nnx.Linear(hidden_dim_pair[0], hidden_dim_pair[1], rngs=rngs)
            )
        self.mu_layer = nnx.Linear(hidden_dims[-1][1], self.latent_dim, rngs=rngs)
        self.logvar_layer = nnx.Linear(hidden_dims[-1][1], self.latent_dim, rngs=rngs)

    def __call__(self, y: ArrayLike) -> Tuple[jax.Array, jax.Array]:
        for hidden_layer, activation_fn in zip(
            self.hidden_layers, self.hidden_activations
        ):
            y = activation_fn(hidden_layer(y))
        z_mu = self.mu_layer(y)
        z_logvar = self.logvar_layer(y)
        return z_mu, z_logvar
