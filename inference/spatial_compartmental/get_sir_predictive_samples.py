import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, AIES
from abm.spatial_compartmental.sir import get_abm
from abm.spatial_compartmental.utils import calc_start_n_initial, Neighbourhood
from inference.spatial_compartmental.si import mcmc_summary
import jax.numpy as jnp
import jax
import hydra
from omegaconf import DictConfig
import itertools
from tqdm import tqdm
from checkpointing import CheckPointer
import json
import os
from cloud_storage_utils import upload_many_blobs_with_transfer_manager
import time
from flax import nnx
from functools import partial
from surrogate.diffusion.denoiser import (
    Denoiser,
    DenoiserConfig,
    SigmaDistributionConfig,
    map_image_to_palette,
)
from surrogate.diffusion.diffusion_sampler import (
    DiffusionSampler,
    DiffusionSamplerConfig,
)


@hydra.main(
    version_base=None, config_path="../../configs/infer/spatial_compartmental/SIR"
)
def main(cfg: DictConfig):
    print(cfg)
    numpyro_model = hydra.utils.get_method(cfg.numpyro_model)

    all_abm_parameters = cfg.abm_params
    grid_size = cfg.abm.grid_size
    num_steps = cfg.abm.num_steps
    abm = get_abm(Neighbourhood(cfg.abm.neighbourhood), vmap=True)

    with open(cfg.posterior_samples_path, "rb") as f:
        posterior_samples = jnp.load(f, allow_pickle=True).item()

    post_pred_dict = {}
    for abm_param in tqdm(all_abm_parameters):
        inf_kwargs = {
            "rng_key": jax.random.key(cfg.inference_seed),
            "grid_size": grid_size,
            "n_initial_infected": calc_start_n_initial(
                abm_param["initial_infected"], grid_size
            ),
            "n_initial_recovered": calc_start_n_initial(
                abm_param["initial_recovered"], grid_size
            ),
            "n_sims": cfg.n_sims,
        }
        if cfg.model_name == "abm":
            inf_kwargs["abm"] = abm
            inf_kwargs["num_steps"] = num_steps
            inf_kwargs["total_population"] = calc_start_n_initial(
                abm_param["total_population"], grid_size
            )
        elif cfg.model_name == "mcmlp":
            mcmlp = CheckPointer.load(cfg.checkpoint)[0]
            inf_kwargs["mcmlp_fn"] = partial(
                nnx.vmap(
                    lambda module, x, key: module(x, key),
                    in_axes=(None, None, 0),
                    out_axes=0,
                ),
                mcmlp,
            )
            if "vanilla" not in cfg.numpyro_model:
                inf_kwargs["total_population"] = calc_start_n_initial(
                    abm_param["total_population"], grid_size
                )
        elif cfg.model_name == "priorcvae":
            priorcvae = CheckPointer.load(cfg.checkpoint)[0]
            inf_kwargs["priorcvae_fn"] = priorcvae.generate_decoder_samples
            if "vanilla" not in cfg.numpyro_model:
                inf_kwargs["total_population"] = (
                    calc_start_n_initial(abm_param["total_population"], grid_size),
                )
        subsample_keys = jax.random.split(jax.random.key(cfg.subsample_seed), 3)
        subsampled_posterior = {
            k: jax.random.choice(sub_key, v, shape=(cfg.subsample_size,), replace=False)
            for (k, v), sub_key in zip(
                posterior_samples[tuple(abm_param.values())][
                    "posterior_samples"
                ].items(),
                subsample_keys,
            )
        }
        posterior_predictive = numpyro.infer.Predictive(
            numpyro_model, subsampled_posterior
        )
        posterior_predictions = posterior_predictive(**inf_kwargs)
        post_pred_dict[tuple(abm_param.values())] = posterior_predictions

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    with open(f"{output_dir}/post_predictions.npy", "wb") as f:
        jnp.save(f, post_pred_dict)

    if cfg.cloud_save:
        upload_many_blobs_with_transfer_manager(
            "dgs4abm",
            os.path.relpath(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            ),
        )


if __name__ == "__main__":
    main()
