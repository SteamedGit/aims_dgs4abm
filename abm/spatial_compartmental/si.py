import jax
import jax.numpy as jnp
from abm.spatial_compartmental.utils import Neighbourhood, _NEIGHBOURHOOD_DICT
from jax.typing import ArrayLike
from functools import wraps
from typing import Tuple, Callable


def get_abm(
    neighbourhood: Neighbourhood,
    vmap: bool = False,
) -> Callable[
    [jax.random.PRNGKey, int, int, float, int], Tuple[jax.Array, jax.Array, jax.Array]
]:
    convolution_kernel = _NEIGHBOURHOOD_DICT[neighbourhood]

    @wraps(_scan_convolution_SI)
    def abm(
        key: jax.random.PRNGKey,
        grid_size: int,
        num_steps: int,
        p_infect: float,
        n_initial_infected: int,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Simulates an SI (Susceptible-Infected) epidemic on a grid using convolution.

        This function models the spread of an infectious disease on a square grid,
        where individuals can be in either Susceptible (S) or Infected (I) states.
        The pattern of disease spread is specified through the convolution_kernel
        argument.

        Parameters
        ----------
        key : jax.random.PRNGKey
            A JAX PRNGKey used to generate random numbers for the simulation,
            ensuring reproducible results.
        grid_size : int
            The size of the square grid (e.g., grid_size=10 represents a 10x10 grid).
        num_steps : int
            The number of steps of disease spread.
        p_infect : float
            The probability of infection transmission from an infected individual
            to a susceptible neighbor.
        n_initial_infected : int
            The number of individuals initially infected at the start of the simulation.

        Returns
        -------
        tuple of (DeviceArray, DeviceArray, DeviceArray)
            A tuple containing the following JAX arrays:

            - states : jax.Array, shape (grid_size, grid_size)
                The final grid state after the simulation, where 0 represents
                Susceptible (S) and 1 represents Infected (I).
            - S_list : jax.Array, shape (num_steps + 1,)
                A 1D array representing the number of susceptible individuals at
                each time step, including the initial state.
            - I_list : jax.Array, shape (num_steps + 1,)
                A 1D array representing the number of infected individuals at
                each time step, including the initial state.
        """
        return _scan_convolution_SI(
            convolution_kernel,
            key,
            grid_size,
            num_steps,
            p_infect,
            n_initial_infected,
        )

    if vmap:
        return jax.vmap(abm, in_axes=[0, None, None, None, None])
    return abm


def _scan_convolution_SI(
    convolution_kernel: ArrayLike,
    key: jax.random.PRNGKey,
    grid_size: int,
    num_steps: int,
    p_infect: float,
    n_initial_infected: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Simulates an SI (Susceptible-Infected) epidemic on a grid using convolution.

    This function models the spread of an infectious disease on a square grid,
    where individuals can be in either Susceptible (S) or Infected (I) states.
    The pattern of disease spread is specified through the convolution_kernel
    argument.

    Parameters
    ----------
    convolution_kernel : ArrayLike
        A 2D array representing the convolution kernel.  This kernel
        determines the neighborhood structure
    key : jax.random.PRNGKey
        A JAX PRNGKey used to generate random numbers for the simulation,
        ensuring reproducible results.
    grid_size : int
        The size of the square grid (e.g., grid_size=10 represents a 10x10 grid).
    num_steps : int
        The number of steps of disease spread.
    p_infect : float
        The probability of infection transmission from an infected individual
        to a susceptible neighbor.
    n_initial_infected : int
        The number of individuals initially infected at the start of the simulation.

    Returns
    -------
    tuple of (DeviceArray, DeviceArray, DeviceArray)
        A tuple containing the following JAX arrays:

        - states : jax.Array, shape (grid_size, grid_size)
            The final grid state after the simulation, where 0 represents
            Susceptible (S) and 1 represents Infected (I).
        - S_list : jax.Array, shape (num_steps + 1,)
            A 1D array representing the number of susceptible individuals at
            each time step, including the initial state.
        - I_list : jax.Array, shape (num_steps + 1,)
            A 1D array representing the number of infected individuals at
            each time step, including the initial state.

    """
    thresh_map = jnp.array(
        [
            -1,
            p_infect,
            1 - (1 - p_infect) ** 2,
            1 - (1 - p_infect) ** 3,
            1 - (1 - p_infect) ** 4,
            1 - (1 - p_infect) ** 5,
            1 - (1 - p_infect) ** 6,
            1 - (1 - p_infect) ** 7,
            1 - (1 - p_infect) ** 8,
        ]
    )

    key, subkey = jax.random.split(key)
    states = jax.random.permutation(
        subkey,
        jnp.concatenate(
            [
                jnp.ones((n_initial_infected,), dtype=jnp.int32),
                jnp.zeros(
                    (grid_size * grid_size - n_initial_infected,), dtype=jnp.int32
                ),
            ]
        ),
    ).reshape((grid_size, grid_size))

    S_0 = jnp.sum(1 - states)
    I_0 = jnp.sum(states)

    def update_grid(states, step_key):
        prob_mat = jax.random.uniform(step_key, (grid_size, grid_size))
        convd = jax.scipy.signal.convolve2d(
            states,
            convolution_kernel,
            mode="same",
        ).astype(jnp.int32)

        thresholds = thresh_map[convd]
        new_infections = jnp.where(prob_mat < thresholds, 1, 0)
        states = jnp.clip(states + new_infections, max=1)
        return states

    def step_fn(states, step_key):
        next_states = update_grid(states, step_key)
        S_t = jnp.sum(1 - next_states)
        I_t = jnp.sum(next_states)
        return next_states, (S_t, I_t)

    keys = jax.random.split(key, num_steps)
    states, (S_list, I_list) = jax.lax.scan(step_fn, states, keys, unroll=False)

    return (
        states,
        jnp.concatenate((jnp.array([S_0]), S_list)),
        jnp.concatenate((jnp.array([I_0]), I_list)),
    )
