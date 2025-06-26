import hydra
from omegaconf import DictConfig
import jax
import jax.numpy as jnp
from abm.spatial_compartmental.utils import calc_start_n_initial, Neighbourhood
import itertools
import os
from tqdm import tqdm
import numpy as np


@hydra.main(
    version_base=None,
    config_path="../../configs/datasets",
)
def main(cfg: DictConfig):
    print(cfg)
    param_samples = {}
    n_initial_samples = {}
    for param, param_settings in cfg.parameters.items():
        param_samples[param] = jnp.arange(
            param_settings.low, param_settings.high, param_settings.step_size
        )
    for n_initial_name, n_initial_settings in cfg.n_initials.items():
        n_initial_samples[n_initial_name] = jnp.arange(
            calc_start_n_initial(n_initial_settings.low, cfg.grid_size),
            calc_start_n_initial(n_initial_settings.high, cfg.grid_size),
            calc_start_n_initial(n_initial_settings.step_size, cfg.grid_size),
        )
    combined_list = list(param_samples.values()) + list(n_initial_samples.values())
    cartesian_product = np.array(list(itertools.product(*combined_list)), dtype=object)
    key = jax.random.PRNGKey(cfg.generation_seed)
    key, subkey = jax.random.split(key)
    train_proportion = cfg.train_prop
    indexes = jax.random.permutation(subkey, jnp.arange(0, len(cartesian_product)))
    train_indices = indexes[: int(train_proportion * len(cartesian_product))]
    test_indices = indexes[int(train_proportion * len(cartesian_product)) :]

    abm = hydra.utils.get_method(cfg.get_abm_fn)(
        Neighbourhood(cfg.neighbourhood), vmap=True
    )

    num_sims_per_param = cfg.num_sims_per_param
    grid_size = cfg.grid_size
    num_steps = cfg.num_steps
    abm_summary_start = cfg.abm_summary_stats.start
    abm_summary_end = cfg.abm_summary_stats.end

    os.makedirs(cfg.dataset_folder, exist_ok=True)

    for realisations_path, param_labels_path, indices in [
        (cfg.train.realisation, cfg.train.param, train_indices),
        (cfg.test.realisation, cfg.test.param, test_indices),
    ]:
        param_labels = []
        realisations = []
        for param in tqdm(
            cartesian_product[indices], "Sampling realisations from the ABM"
        ):
            key, subkey = jax.random.split(key)
            abm_keys = jax.random.split(subkey, num_sims_per_param)
            abm_summaries = abm(
                abm_keys,
                grid_size,
                num_steps,
                *(map(lambda x: x.tolist(), param)),
            )[1:]
            abm_summaries = list(jnp.stack(abm_summaries, axis=1))

            param_labels.extend(param for x in range(num_sims_per_param))
            realisations.extend(
                [x[abm_summary_start:abm_summary_end].flatten() for x in abm_summaries]
            )

        realisations = jnp.stack(realisations, axis=0) / (grid_size * grid_size)
        param_labels = jnp.stack(
            [
                jnp.array(  # Convert the n_intials to proportions for our NN
                    [
                        y / (grid_size * grid_size)
                        if jnp.isdtype(y.dtype, jnp.int32)
                        else y
                        for y in x
                    ]
                )
                for x in param_labels
            ],
            axis=0,
        )

        with open(realisations_path, "wb") as f:
            jnp.save(f, realisations)

        with open(param_labels_path, "wb") as f:
            jnp.save(f, param_labels)

    print(
        f"Created a dataset at {cfg.dataset_folder} with a train set "
        f"of length {len(train_indices) * num_sims_per_param} and a test set of length "
        f"{len(test_indices) * num_sims_per_param} "
    )


if __name__ == "__main__":
    main()
