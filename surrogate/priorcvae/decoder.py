"""
File contains the Decoder models.
"""

from abc import ABC
from typing import Tuple, Union, Optional

from flax import linen as nn
from flax import nnx
import jax.numpy as jnp
import hydra
from jaxlib.xla_extension import PjitFunction
import jax
from jax.typing import ArrayLike


#### NNX version of PriorCVAE Decoder ####
class Decoder(ABC, nnx.Module):
    """Parent class for decoder model."""

    def __init__(self):
        super().__init__()


class MLPDecoder(Decoder):
    def __init__(
        self,
        hidden_dim: Union[Tuple[Tuple[int, int]], Tuple[int, int]],
        out_dim: int,
        input_dim: int,
        rngs: nnx.Rngs,
        hidden_activations: Union[Tuple, PjitFunction] = nnx.sigmoid,
        output_activation: Optional[PjitFunction] = None,
    ):
        hidden_dims = (
            [hidden_dim]
            if isinstance(hidden_dim, list) and isinstance(hidden_dim[0], int)
            else hidden_dim
        )
        hidden_dims = [[input_dim, hidden_dims[0][0]]] + (hidden_dims)
        self.out_dim = out_dim
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
        self.output_layer = nnx.Linear(hidden_dims[-1][1], self.out_dim, rngs=rngs)
        self.output_activation = (
            hydra.utils.get_method(output_activation)
            if output_activation is not None
            else None
        )

    def __call__(self, y: ArrayLike) -> jax.Array:
        for hidden_layer, activation_fn in zip(
            self.hidden_layers, self.hidden_activations
        ):
            y = activation_fn(hidden_layer(y))
        z = self.output_layer(y)
        if self.output_activation is not None:
            z = self.output_activation(z)
        return z
