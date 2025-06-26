import jax
import jax.numpy as jnp
from tqdm import tqdm
from abm.spatial_compartmental.sir import get_abm
from abm.spatial_compartmental.utils import (
    Neighbourhood,
    calc_start_n_initial,
    sir_abm_state_counts,
)
import scipy
import hydra
from omegaconf import DictConfig
import os


def map_batch_grid_series_to_rgb(x):
    """Map a batch of SIR grid timeseries into our RGB colour scheme:
    * RED is infected
    * BLUE is Susceptible
    * GREEN is recovered
    * Black is empty."""
    rgb_colours = jnp.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(
        jnp.float32
    )
    indexed_x = 2 * x[:, :, 0, :, :] + x[:, :, 1, :, :] + 3 * x[:, :, 2, :, :]
    return rgb_colours[indexed_x]


@hydra.main(version_base=None, config_path="../../configs/datasets")
def main(cfg: DictConfig):
    print(cfg)
    # We sample uniformly from a hypercube in [0,1]. Special cases to handle are the initial populations of infected and recovered individuals since
    # infected + recovered <= total.

    lhs_sampler = scipy.stats.qmc.LatinHypercube(d=cfg.num_params, rng=cfg.lhs_seed)
    sample = lhs_sampler.random(n=cfg.number_of_samples)

    non_masked_cond = (jnp.array(cfg.lower_bounds) > -1) & (
        jnp.array(cfg.upper_bounds) > -1
    )

    # If we want to scale into specific ranges
    scaled_sample = scipy.stats.qmc.scale(
        sample,
        jnp.array(cfg.lower_bounds)[non_masked_cond],
        jnp.array(cfg.upper_bounds)[non_masked_cond],
    )

    # First scale initial infected to be a fraction of total population
    # Next scale initial recovered to be a fraction of the remainder
    scaled_sample = jnp.array(scaled_sample)

    if not cfg.no_sparsity:
        scaled_sample = scaled_sample.at[:, cfg.initial_inf_idx].set(
            scaled_sample[:, cfg.initial_inf_idx] * scaled_sample[:, cfg.total_pop_idx]
        )
        scaled_sample = scaled_sample.at[:, cfg.initial_rec_idx].set(
            scaled_sample[:, cfg.initial_rec_idx]
            * (
                scaled_sample[:, cfg.total_pop_idx]
                - scaled_sample[:, cfg.initial_inf_idx]
            )
        )
    else:  # in this case tot_pop always equals 1
        scaled_sample = scaled_sample.at[:, cfg.initial_rec_idx].set(
            scaled_sample[:, cfg.initial_rec_idx]
            * (1 - scaled_sample[:, cfg.initial_inf_idx])
        )

    abm = get_abm(neighbourhood=Neighbourhood(cfg.neighbourhood), vmap=False)
    # VMAP across every parameter except grid size and num steps, not just the key

    vmap_axes = [0, None, None, 0, 0, 0, 0, 0, 0, 0]
    for idx, val in enumerate(non_masked_cond):
        if not val:
            vmap_axes[idx + 3] = None

    vabm = jax.vmap(
        abm,
        in_axes=vmap_axes,
    )
    gen_key = jax.random.PRNGKey(cfg.generation_seed)
    gen_key, subkey = jax.random.split(gen_key)
    train_proportion = cfg.train_prop
    indexes = jax.random.permutation(subkey, jnp.arange(0, cfg.number_of_samples))
    train_indices = indexes[: int(train_proportion * cfg.number_of_samples)]
    test_indices = indexes[int(train_proportion * cfg.number_of_samples) :]

    all_data = []
    grid_size = cfg.grid_size
    num_steps = cfg.num_steps
    os.makedirs(cfg.dataset_folder, exist_ok=True)

    is_image_dataset = bool(cfg.is_image_dataset)
    print(scaled_sample)
    for idx, params in tqdm(
        enumerate(
            scaled_sample.reshape(-1, cfg.data_generation_batch_size, cfg.num_params)
        ),
        total=cfg.number_of_samples // cfg.data_generation_batch_size,
    ):
        data = vabm(
            jax.random.split(
                jax.random.fold_in(gen_key, idx), cfg.data_generation_batch_size
            ),
            grid_size,
            num_steps,
            params[:, 0],  # p_inf
            params[:, 1],  # p_rec
            params[:, 2],  # p_wane
            params[:, 3] if not cfg.no_move else 0.0,  # p_move
            calc_start_n_initial(params[:, cfg.total_pop_idx], grid_size)
            if not cfg.no_sparsity
            else (grid_size * grid_size),  # total pop
            calc_start_n_initial(
                params[:, cfg.initial_inf_idx], grid_size
            ),  # iniital infected
            calc_start_n_initial(
                params[:, cfg.initial_rec_idx], grid_size
            ),  # initial recovered
        )
        if is_image_dataset:
            data = map_batch_grid_series_to_rgb(data).astype(jnp.uint8)
            # data = jax.device_get(data)
            # all_data.append(data)
            os.makedirs(f"{cfg.dataset_folder}/pieces", exist_ok=True)
            with open(f"{cfg.dataset_folder}/pieces/realisation_{idx}.npy", "wb") as f:
                jnp.save(f, data)
            with open(f"{cfg.dataset_folder}/pieces/params_{idx}.npy", "wb") as f:
                jnp.save(f, params)
        else:
            # First vmap over sequence, then over batch
            data = jax.vmap(jax.vmap(sir_abm_state_counts, in_axes=[0]), in_axes=[0])(
                data
            )
            if cfg.data_format == "channel_stack":
                data = jnp.stack(list(data.values()), axis=2)
            elif cfg.data_format == "flat":
                data = jnp.concatenate(list(data.values()), axis=1)
            else:
                raise ValueError("Invalid data format for count-based dataset")
            # Normalise into [0,1]
            all_data.append(data / (grid_size * grid_size))
    if not is_image_dataset:
        realisations = jnp.concatenate(
            all_data, axis=0, dtype=jnp.uint8 if is_image_dataset else jnp.float32
        )

        with open(cfg.train.realisation, "wb") as f:
            jnp.save(f, realisations[train_indices])

        with open(cfg.test.realisation, "wb") as f:
            jnp.save(f, realisations[test_indices])

        with open(cfg.train.param, "wb") as f:
            jnp.save(f, scaled_sample[train_indices])

        with open(cfg.test.param, "wb") as f:
            jnp.save(f, scaled_sample[test_indices])

        print(
            f"Created a dataset at {cfg.dataset_folder} with a train set "
            f"of length {len(train_indices)} and a test set of length "
            f"{len(test_indices)} "
        )
    else:
        print(
            f"Saved pieces of image dataset in {cfg.dataset_folder}. Use fuse_all.py with the same config to combine into contiguous .npy files."
        )


if __name__ == "__main__":
    main()
