import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
import jax.numpy as jnp
import hydra

jax.config.update("jax_platform_name", "cpu")


@hydra.main(version_base=None, config_path="../../configs/datasets")
def main(cfg):
    print(cfg)
    num_pieces = cfg.number_of_samples // cfg.data_generation_batch_size
    real_pieces = []
    param_pieces = []

    for i in range(num_pieces):
        with open(f"{cfg.dataset_folder}/pieces/realisation_{i}.npy", "rb") as f:
            real_arr = jnp.load(f)
            real_pieces.append(real_arr)

        with open(f"{cfg.dataset_folder}/pieces/params_{i}.npy", "rb") as f:
            param_arr = jnp.load(f)
            param_pieces.append(param_arr)

    full_realisations = jnp.concat(real_pieces, axis=0).astype(jnp.uint8)
    full_parameters = jnp.concat(param_pieces, axis=0)

    # So that same key is used as in create_sir
    gen_key = jax.random.PRNGKey(cfg.generation_seed)
    _, subkey = jax.random.split(gen_key)
    train_proportion = cfg.train_prop
    indexes = jax.random.permutation(subkey, jnp.arange(0, cfg.number_of_samples))
    train_indices = indexes[: int(train_proportion * cfg.number_of_samples)]
    test_indices = indexes[int(train_proportion * cfg.number_of_samples) :]

    # Train
    with open(f"{cfg.dataset_folder}/train_realisations.npy", "wb") as f:
        jnp.save(f, full_realisations[train_indices])
    with open(f"{cfg.dataset_folder}/train_param_labels.npy", "wb") as f:
        jnp.save(f, full_parameters[train_indices])

    # Test
    with open(f"{cfg.dataset_folder}/test_realisations.npy", "wb") as f:
        jnp.save(f, full_realisations[test_indices])
    with open(f"{cfg.dataset_folder}/test_param_labels.npy", "wb") as f:
        jnp.save(f, full_parameters[test_indices])


if __name__ == "__main__":
    main()
