import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Tuple, Callable
from abm.spatial_compartmental.utils import Neighbourhood, _NEIGHBOURHOOD_DICT
from functools import wraps


# TODO: Docs
def get_valid_offsets_from_kernel(kernel):
    """
    Extracts valid (dr, dc) relative offsets from a 3x3 neighborhood kernel.

    Args:
        kernel: A 3x3 array-like object (list of lists, NumPy array, JAX array)
                with 1s for valid neighbor positions and 0s otherwise.
                kernel[1, 1] corresponds to the center (dr=0, dc=0).

    Returns:
        A JAX array of shape (N, 2) containing N valid (dr, dc) offset pairs,
        where N is the number of 1s in the kernel. Dtype is int32.
        Returns shape (0, 2) if the kernel contains no 1s.
    """
    kernel = jnp.asarray(kernel, dtype=jnp.int32)
    if kernel.shape != (3, 3):
        raise ValueError("Kernel must be a 3x3 array.")

    valid_offsets = []
    for dr in range(-1, 2):  # Row offset (-1, 0, 1)
        for dc in range(-1, 2):  # Column offset (-1, 0, 1)
            # Map offset (dr, dc) to kernel index (r_idx, c_idx)
            r_idx, c_idx = dr + 1, dc + 1
            if kernel[r_idx, c_idx] == 1:
                valid_offsets.append((dr, dc))

    if not valid_offsets:
        print("Warning: Kernel contains no valid offsets (all zeros?).")
        # Return an empty array with the correct shape
        return jnp.array([], dtype=jnp.int32).reshape(0, 2)

    return jnp.array(valid_offsets, dtype=jnp.int32)


def get_abm(
    neighbourhood: Neighbourhood,
    vmap: bool = False,
) -> Callable[
    [jax.random.PRNGKey, int, int, float, float, float, float, int, int, int],
    jax.Array,
]:
    convolution_kernel = _NEIGHBOURHOOD_DICT[neighbourhood]

    # Movement pattern defined by neighbourhood kernel
    valid_offsets_array = get_valid_offsets_from_kernel(convolution_kernel)

    @wraps(_scan_convolution_SIR)
    def abm(
        key: jax.random.PRNGKey,
        grid_size: int,
        num_steps: int,
        p_infect: float,
        p_recover: float,
        p_wane: float,
        p_move: float,
        total_population: int,
        n_initial_infected: int,
        n_initial_recovered: int,
    ) -> jax.Array:
        return _scan_convolution_SIR(
            convolution_kernel,
            valid_offsets_array,
            key,
            grid_size,
            num_steps,
            p_infect,
            p_recover,
            p_wane,
            p_move,
            total_population,
            n_initial_infected,
            n_initial_recovered,
        )

    if vmap:
        return jax.vmap(
            abm, in_axes=[0, None, None, None, None, None, None, None, None, None]
        )
    return abm


def _scan_convolution_SIR(
    convolution_kernel: ArrayLike,
    valid_offsets_array: ArrayLike,
    key: jax.random.PRNGKey,
    grid_size: int,
    num_steps: int,
    p_infect: float,
    p_recover: float,
    p_wane: float,
    p_move: float,
    total_population: int,
    n_initial_infected: int,
    n_initial_recovered: int,
) -> jax.Array:
    ###############################################################################
    # Initialise the Grid                                                         #
    ###############################################################################

    inf_thresh_map = jnp.array(
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

    # Grid Coordinates for movement updates
    row_coords, col_coords = jnp.meshgrid(
        jnp.arange(grid_size), jnp.arange(grid_size), indexing="ij"
    )

    flat_cell_indices = jnp.arange(grid_size**2)[:, None]

    # One-hot encoding of different agent states
    infected_state = jnp.array([1, 0, 0], dtype=jnp.int32)
    susceptible_state = jnp.array([0, 1, 0], dtype=jnp.int32)
    recovered_state = jnp.array([0, 0, 1], dtype=jnp.int32)
    empty_state = jnp.array([0, 0, 0], dtype=jnp.int32)
    mask_state = jnp.array([-1, -1, -1], dtype=jnp.int32)

    # First we fill in the infected agents
    abm_grid_flat = jnp.where(
        flat_cell_indices < n_initial_infected,
        infected_state[None, :],
        mask_state[None, :],
    )
    # Next we fill in the recovered agents
    abm_grid_flat = jnp.where(
        (flat_cell_indices >= n_initial_infected)
        & (flat_cell_indices < n_initial_infected + n_initial_recovered),
        recovered_state[None, :],
        abm_grid_flat,
    )

    # Next we fill with susceptible
    abm_grid_flat = jnp.where(
        (flat_cell_indices >= n_initial_infected + n_initial_recovered)
        & (flat_cell_indices < total_population),
        susceptible_state[None, :],
        abm_grid_flat,
    )

    # Finally, fill in empty
    abm_grid_flat = jnp.where(
        abm_grid_flat == mask_state, empty_state[None, :], abm_grid_flat
    )

    key, subkey = jax.random.split(key)
    shuffled_indices = jax.random.permutation(subkey, grid_size * grid_size)
    abm_grid_flat = abm_grid_flat[shuffled_indices]
    abm_grid = abm_grid_flat.reshape((grid_size, grid_size, 3)).transpose((2, 0, 1))

    ###############################################################################
    # Grid state update definition                                                #
    ###############################################################################
    def update_grid_states(abm_grid, step_key):
        #############################################
        # Prepare RNG                               #
        #############################################
        infect_key, recover_key, wane_key = jax.random.split(step_key, 3)

        #############################################
        # Infection Spread                          #
        #############################################

        convd = jax.scipy.signal.convolve2d(
            abm_grid[0, :, :],
            convolution_kernel,
            mode="same",
        ).astype(jnp.int32)

        thresholds = inf_thresh_map[convd]

        susceptible_indices = abm_grid[1, :, :] == 1

        inf_prob_mat = jax.random.uniform(infect_key, (grid_size, grid_size))

        new_infections = jnp.where(susceptible_indices, inf_prob_mat, 2) < thresholds

        #############################################
        # Recovery                                  #
        #############################################
        infected_indices = abm_grid[0, :, :] == 1

        rec_prob_mat = jax.random.uniform(recover_key, (grid_size, grid_size))
        new_recoveries = jnp.where(infected_indices, rec_prob_mat, 2) < p_recover

        #############################################
        # Waning Immunity                           #
        #############################################
        recovered_indices = abm_grid[2, :, :] == 1

        wane_mat = jax.random.uniform(wane_key, (grid_size, grid_size))
        waned_immunities = jnp.where(recovered_indices, wane_mat, 2) < p_wane

        waned_channel_0 = jnp.where(
            waned_immunities,
            jnp.array(0, dtype=abm_grid.dtype),  # Change to 0 for Susceptible state
            abm_grid[0, :, :],  # Otherwise, keep original Channel 0 value
        )

        waned_channel_2 = waned_immunities.astype(jnp.int32)
        #############################################
        # Update Grids                              #
        #############################################
        abm_grid = jnp.stack(
            [
                waned_channel_0 + new_infections - new_recoveries,
                abm_grid[1, :, :] - new_infections + waned_channel_2,
                abm_grid[2, :, :] - waned_channel_2 + new_recoveries,
            ],
            axis=0,
        )

        return abm_grid

    ###############################################################################
    # Agents movement definition                                                  #
    ###############################################################################
    def move_agents(
        abm_grid,
        step_key,
        valid_offsets_array: jax.Array,  # Shape (N, 2)
        row_coords: jax.Array,  # Shape (H, W)
        col_coords: jax.Array,  # Shape (H, W)
    ):
        """
        Moves agents on the grid using segment_max for conflict resolution.

        Args:
            abm_grid: Current state grid (2, H, W).
            step_key: JAX PRNG key for this step.
            p_move: Probability an agent attempts to move.
            valid_offsets_array: Array of valid (dr, dc) moves from the kernel.
            row_coords: Grid row coordinates.
            col_coords: Grid column coordinates.

        Returns:
            Updated abm_grid after movement.
        """
        key_attempt, key_offset_choice, key_priority = jax.random.split(step_key, 3)
        grid_shape_2d = abm_grid.shape[1:]
        H, W = grid_shape_2d
        num_cells = H * W

        # 1. Identify Agents and Movement Attempts
        is_agent = (abm_grid[0] != 0) | (abm_grid[1] != 0) | (abm_grid[2] != 0)
        attempts_move = is_agent & (
            jax.random.uniform(key_attempt, grid_shape_2d) < p_move
        )

        # 2. Propose Target Locations (for all cells initially)
        # Choose a random valid offset for each cell
        chosen_offset_index = jax.random.randint(
            key_offset_choice,
            grid_shape_2d,
            0,
            valid_offsets_array.shape[0],  # Max offset index (exclusive)
        )
        # Get the corresponding (dr, dc) for each cell
        dr = valid_offsets_array[chosen_offset_index, 0]
        dc = valid_offsets_array[chosen_offset_index, 1]

        # Calculate potential target coordinates (clipping boundary condition)
        target_r = jnp.clip(row_coords + dr, 0, H - 1)
        target_c = jnp.clip(col_coords + dc, 0, W - 1)

        # 3. Check if Proposed Target is Empty
        # Gather state of target cells using calculated coordinates
        target_ch0 = abm_grid[0, target_r, target_c]
        target_ch1 = abm_grid[1, target_r, target_c]
        target_ch2 = abm_grid[2, target_r, target_c]
        target_is_empty = (target_ch0 == 0) & (target_ch1 == 0) & (target_ch2 == 0)

        # 4. Identify Valid Proposals (Agent attempts move AND target is empty)
        is_valid_proposal = attempts_move & target_is_empty

        # 5. Assign Priorities and Prepare for Conflict Resolution
        # Assign priorities only to valid proposals, others get -1
        priority = jax.random.uniform(key_priority, grid_shape_2d)
        flat_priority = priority.ravel()
        flat_valid_proposal = is_valid_proposal.ravel()

        # Calculate flat source and target indices
        flat_source_idx = (row_coords * W + col_coords).ravel()
        flat_target_idx = (target_r * W + target_c).ravel()

        # 6. Conflict Resolution using segment_max

        # Identify the winning agents: A proposal wins if it's valid AND
        # its priority matches the max priority for its target.
        # Due to float precision, == comparison should be okay here.
        # This inherently handles ties: if multiple proposals have the exact same max
        # priority for a target, segment_max's behavior is undefined on which *original*
        # element yielded the max, but this check identifies *all* that match.
        # However, only one agent should move. Let's refine using the source index tie-breaker.

        # Refined conflict resolution with tie-breaking (lower source index wins)
        # Pack value: Higher priority wins. If priorities are equal, lower source index wins.
        # Since priority is [0,1), scale it up past num_cells. Add (num_cells - source_idx)
        # so lower source_idx gives a higher packed value in case of ties.

        packed_value = flat_priority * (num_cells + 1.0) + (
            num_cells - flat_source_idx.astype(jnp.float32)
        )
        segment_data_packed = jnp.where(flat_valid_proposal, packed_value, -1.0)

        max_packed_value_per_target = jax.ops.segment_max(
            data=segment_data_packed,
            segment_ids=flat_target_idx,
            num_segments=num_cells,
            indices_are_sorted=False,
            unique_indices=False,
        )

        # An agent wins if its proposal is valid AND its packed value matches the max for its target
        is_winner = flat_valid_proposal & (
            packed_value == max_packed_value_per_target[flat_target_idx]
        )
        # This ensures only one winner per target cell, even with priority ties.

        # 7. Update Grid
        # Get current state in flat arrays
        source_ch0_flat = abm_grid[0].ravel()
        source_ch1_flat = abm_grid[1].ravel()
        source_ch2_flat = abm_grid[2].ravel()

        # Create new grid, initialized to empty
        # Use a padded version for safe scatter updates if indices might be invalid (-1)
        # We use num_cells as the safe padding index.
        next_ch0_flat_padded = jnp.zeros(num_cells + 1, dtype=abm_grid.dtype)
        next_ch1_flat_padded = jnp.zeros(num_cells + 1, dtype=abm_grid.dtype)
        next_ch2_flat_padded = jnp.zeros(num_cells + 1, dtype=abm_grid.dtype)

        # Identify stayers: agents that are not winners
        stayers_mask = is_agent.ravel() & ~is_winner

        # Get indices and states for stayers
        stayer_source_indices = flat_source_idx
        stayer_ch0_vals = jnp.where(stayers_mask, source_ch0_flat, 0)
        stayer_ch1_vals = jnp.where(stayers_mask, source_ch1_flat, 0)
        stayer_ch2_vals = jnp.where(stayers_mask, source_ch2_flat, 0)
        stayer_write_indices = jnp.where(
            stayers_mask, stayer_source_indices, num_cells
        )  # Write stayers to original pos

        # Scatter stayers into the padded grid
        next_ch0_flat_padded = next_ch0_flat_padded.at[stayer_write_indices].set(
            stayer_ch0_vals
        )
        next_ch1_flat_padded = next_ch1_flat_padded.at[stayer_write_indices].set(
            stayer_ch1_vals
        )
        next_ch2_flat_padded = next_ch2_flat_padded.at[stayer_write_indices].set(
            stayer_ch2_vals
        )

        # Get indices and states for winners
        winner_target_indices = flat_target_idx  # Target locations of potential winners
        winner_ch0_vals = jnp.where(is_winner, source_ch0_flat, 0)  # State of winners
        winner_ch1_vals = jnp.where(is_winner, source_ch1_flat, 0)
        winner_ch2_vals = jnp.where(is_winner, source_ch2_flat, 0)
        winner_write_indices = jnp.where(
            is_winner, winner_target_indices, num_cells
        )  # Write winners to target pos

        # Scatter winners into the padded grid (overwriting whatever was there)
        # Using .set ensures we place the winner, even if the target was non-empty
        # (which it shouldn't be due to 'target_is_empty' check earlier, but safer)
        # or if a stayer was mistakenly written there by padding.
        final_ch0_padded = next_ch0_flat_padded.at[winner_write_indices].set(
            winner_ch0_vals
        )
        final_ch1_padded = next_ch1_flat_padded.at[winner_write_indices].set(
            winner_ch1_vals
        )
        final_ch2_padded = next_ch2_flat_padded.at[winner_write_indices].set(
            winner_ch2_vals
        )

        # Remove padding and reshape
        final_ch0_flat = final_ch0_padded[:-1]
        final_ch1_flat = final_ch1_padded[:-1]
        final_ch2_flat = final_ch2_padded[:-1]

        new_grid = jnp.stack(
            [
                final_ch0_flat.reshape(grid_shape_2d),
                final_ch1_flat.reshape(grid_shape_2d),
                final_ch2_flat.reshape(grid_shape_2d),
            ],
            axis=0,
        )

        return new_grid

    ###############################################################################
    # Update grid definition                                                      #
    ###############################################################################
    def update_grid(abm_grid, step_key):
        grid_states_key, move_agents_key = jax.random.split(step_key)
        abm_grid = update_grid_states(abm_grid, grid_states_key)
        abm_grid = move_agents(
            abm_grid, move_agents_key, valid_offsets_array, row_coords, col_coords
        )
        return abm_grid, abm_grid

    initial_abm_grid = abm_grid

    keys = jax.random.split(key, num_steps)
    abm_grid, abm_grid_timeseries = jax.lax.scan(
        update_grid,
        abm_grid,
        keys,
        unroll=False,  # TODO: Check the speed implications
    )

    return jnp.concatenate(
        (initial_abm_grid.reshape(1, 3, grid_size, grid_size), abm_grid_timeseries),
        axis=0,
    )
