from jax.nn.initializers import Initializer
from jax._src import core
from typing import Any
import jax
import jax.numpy as jnp

DTypeLikeFloat = Any
DTypeLikeComplex = Any
DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex
RealNumeric = Any  # Scalar jnp array or float
from jax._src import dtypes

# 2D CONVS:
# NOTE: Flax by default initialises biases to zeroes and kernels to yacun_normal
# whilst torch initialises biases and kernels to uniform(-scale,scale) where scale
# depends on the kernel size and number of input channels.
# Hence, in the torch version of the code when they initialised the kernel to zero
# and kept the biases as default, the model trains, whereas in the flax version the model always outputs zeros
# and can't learn
# https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# https://flax.readthedocs.io/en/latest/_modules/flax/nnx/nn/linear.html#Conv


def calculate_torch_init_scale(c_in, kernel_size):
    k = 1 / c_in * (jnp.prod(kernel_size))
    return jnp.sqrt(k)


def symmetric_uniform(
    scale: RealNumeric = 1e-2, dtype: DTypeLikeInexact = jnp.float_
) -> Initializer:
    """Builds an initializer that returns real uniformly-distributed random arrays.

    Args:
      scale: optional; the upper bound of the random distribution.
      dtype: optional; the initializer's default dtype.

    Returns:
      An initializer that returns arrays whose values are uniformly distributed in
      the range ``[-scale, scale)``.

    >>> import jax, jax.numpy as jnp
    >>> initializer = jax.nn.initializers.uniform(10.0)
    >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  # doctest: +SKIP
    Array([[7.298188 , 8.691938 , 8.7230015],
           [2.0818567, 1.8662417, 5.5022564]], dtype=float32)
    """

    def init(
        key: jax.Array, shape: core.Shape, dtype: DTypeLikeInexact = dtype
    ) -> jax.Array:
        dtype = dtypes.canonicalize_dtype(dtype)
        return jax.random.uniform(key, shape, dtype, minval=-1) * jnp.array(
            scale, dtype
        )

    return init


def torch_conv_init(
    c_in, kernel_size, dtype: DTypeLikeInexact = jnp.float_
) -> Initializer:
    """Builds an initializer that can initialise convolutional layer
    kernel and biases the way tha torch does.


    """

    def init(
        key: jax.Array, shape: core.Shape, dtype: DTypeLikeInexact = dtype
    ) -> jax.Array:
        dtype = dtypes.canonicalize_dtype(dtype)
        return jax.random.uniform(key, shape, dtype, minval=-1) * jnp.array(
            calculate_torch_init_scale(c_in, kernel_size), dtype
        )

    return init
