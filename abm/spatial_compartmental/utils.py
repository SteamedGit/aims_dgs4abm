from enum import Enum
import jax.numpy as jnp


def calc_start_n_initial(proportion, grid_size):
    return jnp.round(proportion * grid_size * grid_size, 0).astype(jnp.int32)


class Neighbourhood(Enum):
    DIAGONAL = "diagonal"
    MOORE = "moore"
    VONNEUMANN = "von neumann"


_NEIGHBOURHOOD_DICT = {
    Neighbourhood.DIAGONAL: jnp.array(
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ],
        dtype=jnp.int32,
    ),
    Neighbourhood.MOORE: jnp.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=jnp.int32,
    ),
    Neighbourhood.VONNEUMANN: jnp.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=jnp.int32,
    ),
}


def sir_abm_state_counts(abm_grid):
    counts = {}

    # Susceptible: (layer0==0) & (layer1==1) & (layer2==0)
    counts["Susceptible"] = jnp.sum(
        (abm_grid[0, :, :] == 0) & (abm_grid[1, :, :] == 1) & (abm_grid[2, :, :] == 0),
        axis=(0, 1),
    ).astype(jnp.int32)

    # Infected: (layer0==1) & (layer1==0) & (layer2==0)
    counts["Infected"] = jnp.sum(
        (abm_grid[0, :, :] == 1) & (abm_grid[1, :, :] == 0) & (abm_grid[2, :, :] == 0),
        axis=(0, 1),
    ).astype(jnp.int32)

    # Recovered: (layer0==0) & (layer1==0) & (layer2==1)
    counts["Recovered"] = jnp.sum(
        (abm_grid[0, :, :] == 0) & (abm_grid[1, :, :] == 0) & (abm_grid[2, :, :] == 1),
        axis=(0, 1),
    ).astype(jnp.int32)
    return counts
