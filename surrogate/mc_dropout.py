from flax import nnx
import optax
from typing import Dict, Optional
from jaxlib.xla_extension import PjitFunction
import jax
import hydra
from jax.typing import ArrayLike


class LinearDropLayer(nnx.Module):
    def __init__(self, input_dim: int, output_dim: int, p_drop: float, rngs: nnx.Rngs):
        self.linear = nnx.Linear(input_dim, output_dim, rngs=rngs)
        self.dropout = nnx.Dropout(p_drop, rngs=rngs)

    def __call__(self, x: ArrayLike, rngs) -> jax.Array:
        return self.dropout(
            nnx.relu(self.linear(x)),
            rngs=rngs,
        )


class MCMLP(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        path_length: int,
        num_layers: int,
        hidden_dim: int,
        p_drop: float,
        rngs: nnx.Rngs,
        output_activation: Optional[PjitFunction] = None,
    ):
        assert num_layers >= 3, "MCMLP must have at least 3 layers"
        layers = [LinearDropLayer(input_dim, hidden_dim, p_drop, rngs)]

        for idx in range(num_layers - 2):
            layers.append(LinearDropLayer(hidden_dim, hidden_dim, p_drop, rngs=rngs))
        layers.append(nnx.Linear(hidden_dim, path_length, rngs=rngs))
        self.sequential = nnx.Sequential(*layers)
        self.output_activation = (
            hydra.utils.get_method(output_activation)
            if output_activation is not None
            else None
        )

    def __call__(self, x: ArrayLike, key: jax.random.PRNGKey) -> jax.Array:
        x = self.sequential(x, rngs=nnx.Rngs(key))
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


# TODO: Rename to L2 Loss
def loss_fn(
    model: MCMLP, batch: Dict[str, ArrayLike], key: jax.random.PRNGKey
) -> jax.Array:
    preds = model(batch["param"], key)
    loss = optax.l2_loss(preds, batch["realisation"]).mean()
    return loss


# TODO: Aleatoric Loss
