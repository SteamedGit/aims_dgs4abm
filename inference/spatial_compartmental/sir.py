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


def vectorized_abm_state_counts(abm_grids_batch):
    """
    Calculates state counts for a batch of ABM grids using JAX.

    Args:
        abm_grids_batch (jnp.ndarray): An array of shape (batch_size, steps, 3, H, W)
                                       representing all grids at a single time step.

    Returns:
        dict: A dictionary where keys are state names (str) and values are
              JAX arrays (shape (batch_size,)) containing the counts of that
              state for each grid in the batch.
    """
    counts = {}
    # Susceptible: (layer0==0) & (layer1==1) & (layer2==0)
    counts["Susceptible"] = jnp.sum(
        (abm_grids_batch[:, :, 0, :, :] == 0)
        & (abm_grids_batch[:, :, 1, :, :] == 1)
        & (abm_grids_batch[:, :, 2, :, :] == 0),
        axis=(2, 3),
    ).astype(jnp.int32)

    # Infected: (layer0==1) & (layer1==0) & (layer2==0)
    counts["Infected"] = jnp.sum(
        (abm_grids_batch[:, :, 0, :, :] == 1)
        & (abm_grids_batch[:, :, 1, :, :] == 0)
        & (abm_grids_batch[:, :, 2, :, :] == 0),
        axis=(2, 3),
    ).astype(jnp.int32)

    # Recovered: (layer0==0) & (layer1==0) & (layer2==1)
    counts["Recovered"] = jnp.sum(
        (abm_grids_batch[:, :, 0, :, :] == 0)
        & (abm_grids_batch[:, :, 1, :, :] == 0)
        & (abm_grids_batch[:, :, 2, :, :] == 1),
        axis=(2, 3),
    ).astype(jnp.int32)
    return counts


def rgb_vectorized_abm_state_counts_jax(abm_grids_batch):
    """
    Calculates state counts for a batch of ABM grids using JAX.

    Args:
        abm_grids_batch (jnp.ndarray): An array of shape (batch_size, seq, 3, H,W)
                                       representing all grids at a single time step.

    Returns:
        dict: A dictionary where keys are state names (str) and values are
              JAX arrays (shape (batch_size,)) containing the counts of that
              state for each grid in the batch.
    """
    counts = {}

    # Susceptible: (layer0==0) & (layer1==1) & (layer2==0)
    counts["Susceptible"] = jnp.sum(
        (abm_grids_batch[:, :, 0, :, :] == 0)
        & (abm_grids_batch[:, :, 1, :, :] == 0)
        & (abm_grids_batch[:, :, 2, :, :] == 1),
        axis=(2, 3),
    ).astype(jnp.int32)

    # Infected: (layer0==1) & (layer1==0) & (layer2==0)
    counts["Infected"] = jnp.sum(
        (abm_grids_batch[:, :, 0, :, :] == 1)
        & (abm_grids_batch[:, :, 1, :, :] == 0)
        & (abm_grids_batch[:, :, 2, :, :] == 0),
        axis=(2, 3),
    ).astype(jnp.int32)

    # Recovered: (layer0==0) & (layer1==0) & (layer2==1)
    counts["Recovered"] = jnp.sum(
        (abm_grids_batch[:, :, 0, :, :] == 0)
        & (abm_grids_batch[:, :, 1, :, :] == 1)
        & (abm_grids_batch[:, :, 2, :, :] == 0),
        axis=(2, 3),
    ).astype(jnp.int32)
    return counts


def abm_grid_initialise(
    key, grid_size, n_initial_infected, n_initial_recovered, total_population
):
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

    # key, subkey = jax.random.split(key)
    shuffled_indices = jax.random.permutation(key, grid_size * grid_size)
    abm_grid_flat = abm_grid_flat[shuffled_indices]
    abm_grid = abm_grid_flat.reshape((grid_size, grid_size, 3)).transpose((2, 0, 1))
    return abm_grid[None, :]


vabm_init = jax.vmap(abm_grid_initialise, in_axes=[0, None, None, None, None])


def no_move_prior():
    p_infect = numpyro.sample("p_infect", dist.Beta(2, 5))
    p_recover = numpyro.sample("p_recover", dist.Beta(2, 5))
    p_wane = numpyro.sample("p_wane", dist.Beta(2, 5))
    return p_infect, p_recover, p_wane


#########################################################################
# SIR ABM                                                               #
#########################################################################
def full_cov_no_move_sir_ABM_model(
    abm,
    n_sims,
    grid_size,
    num_steps,
    total_population,
    n_initial_infected,
    n_initial_recovered,
    data=None,
):
    p_infect, p_recover, p_wane = no_move_prior()

    multi_grid_timeseries = abm(
        jax.random.split(numpyro.prng_key(), n_sims),
        grid_size,
        num_steps,
        p_infect,
        p_recover,
        p_wane,
        0.0,  # No Movement
        total_population,
        n_initial_infected,
        n_initial_recovered,
    )

    state_count_timeseries = vectorized_abm_state_counts(multi_grid_timeseries)

    means = {k: jnp.mean(x[:, 1:], axis=0) for k, x in state_count_timeseries.items()}

    stacked = jnp.concatenate(
        [x[:, 1:] for x in list(state_count_timeseries.values())[1:]], axis=1
    )
    mean = jnp.concatenate((means["Infected"], means["Recovered"]))
    covariance = jnp.cov(stacked, rowvar=False) + 1e-7 * jnp.eye(mean.shape[0])
    numpyro.sample("obs", dist.MultivariateNormal(mean, covariance), obs=data)


def diag_cov_no_move_sir_ABM_model(
    abm,
    n_sims,
    grid_size,
    num_steps,
    total_population,
    n_initial_infected,
    n_initial_recovered,
    data=None,
):
    p_infect, p_recover, p_wane = no_move_prior()

    multi_grid_timeseries = abm(
        jax.random.split(numpyro.prng_key(), n_sims),
        grid_size,
        num_steps,
        p_infect,
        p_recover,
        p_wane,
        0.0,  # No Movement
        total_population,
        n_initial_infected,
        n_initial_recovered,
    )

    state_count_timeseries = vectorized_abm_state_counts(multi_grid_timeseries)

    means = {k: jnp.mean(x[:, 1:], axis=0) for k, x in state_count_timeseries.items()}
    stds = {
        k: jnp.std(x[:, 1:], axis=0, ddof=1) + 1e-7
        for k, x in state_count_timeseries.items()
    }

    means = jnp.concatenate(
        (means["Susceptible"], means["Infected"], means["Recovered"])
    )
    stds = jnp.concatenate((stds["Susceptible"], stds["Infected"], stds["Recovered"]))
    numpyro.sample("obs", dist.Normal(means, stds), obs=data)


def summarised_no_move_sir_ABM_model(
    abm,
    n_sims,
    grid_size,
    num_steps,
    total_population,
    n_initial_infected,
    n_initial_recovered,
    data=None,
):
    p_infect, p_recover, p_wane = no_move_prior()
    multi_grid_timeseries = abm(
        jax.random.split(numpyro.prng_key(), n_sims),
        grid_size,
        num_steps,
        p_infect,
        p_recover,
        p_wane,
        0.0,  # No Movement
        total_population,
        n_initial_infected,
        n_initial_recovered,
    )

    state_count_timeseries = vectorized_abm_state_counts(multi_grid_timeseries)

    # Cumulative sum of all new infections
    cumulative_incidence = jnp.sum(state_count_timeseries["Infected"][:, 1:], axis=1)

    # Number of timesteps to reach peak infection
    time_to_peak_infected = jnp.argmax(state_count_timeseries["Infected"], axis=1)

    time_to_min_infected = jnp.argmin(state_count_timeseries["Infected"], axis=1)

    # Largest increase in infection in a single timestep
    peak_incidence = jnp.max(
        jnp.diff(state_count_timeseries["Infected"], axis=1), axis=1
    )

    minimum_susceptible = jnp.min(state_count_timeseries["Susceptible"], axis=1)

    time_to_peak_recovered = jnp.argmax(state_count_timeseries["Recovered"], axis=1)
    time_to_min_recovered = jnp.argmin(state_count_timeseries["Recovered"], axis=1)

    summary_variables = jnp.stack(
        (
            cumulative_incidence,
            time_to_peak_infected,
            time_to_min_infected,
            peak_incidence,
            minimum_susceptible,
            time_to_peak_recovered,
            time_to_min_recovered,
        ),
        axis=1,
    )
    mean = jnp.mean(summary_variables, axis=0)
    covariance = jnp.cov(summary_variables, rowvar=False) + 1e7 * jnp.eye(7)
    numpyro.sample("obs", dist.MultivariateNormal(mean, covariance), obs=data)


#####################################################################################
# MCMLP Model                                                                       #
#####################################################################################


def full_cov_vanilla_sir_MCMLP_model(
    mcmlp_fn, n_sims, grid_size, n_initial_infected, n_initial_recovered, data=None
):
    p_infect, p_recover, p_wane = no_move_prior()

    initial_infected = n_initial_infected / grid_size**2
    initial_recovered = n_initial_recovered / grid_size**2

    mcmlp_timeseries = mcmlp_fn(
        jnp.array([p_infect, p_recover, p_wane, initial_infected, initial_recovered]),
        jax.random.split(numpyro.prng_key(), n_sims),
    )
    mcmlp_infected, mcmlp_recovered, mcmlp_susceptible = jnp.split(
        mcmlp_timeseries, 3, axis=-1
    )

    mcmlp_infected = mcmlp_infected[:, 1:]
    mcmlp_recovered = mcmlp_recovered[:, 1:]
    mean = jnp.concatenate(
        (
            jnp.mean(mcmlp_infected, axis=0),
            jnp.mean(mcmlp_recovered, axis=0),
        )
    )
    covariance = jnp.cov(
        jnp.concatenate((mcmlp_infected, mcmlp_recovered), axis=1), rowvar=False
    ) + 1e7 * jnp.eye(mean.shape[0])
    numpyro.sample("obs", dist.MultivariateNormal(mean, covariance), obs=data)


def fixed_uncertainty_vanilla_sir_MCMLP_model(
    mcmlp_fn, n_sims, grid_size, n_initial_infected, n_initial_recovered, data=None
):
    p_infect, p_recover, p_wane = no_move_prior()

    initial_infected = n_initial_infected / grid_size**2
    initial_recovered = n_initial_recovered / grid_size**2

    mcmlp_timeseries = mcmlp_fn(
        jnp.array([p_infect, p_recover, p_wane, initial_infected, initial_recovered]),
        jax.random.split(numpyro.prng_key(), n_sims),
    )
    mcmlp_infected, mcmlp_recovered, mcmlp_susceptible = jnp.split(
        mcmlp_timeseries, 3, axis=-1
    )

    mcmlp_infected = mcmlp_infected[:, 1:]
    mcmlp_recovered = mcmlp_recovered[:, 1:]
    mcmlp_susceptible = mcmlp_susceptible[:, 1:]
    mean = jnp.concatenate(
        (
            jnp.mean(mcmlp_susceptible, axis=0),
            jnp.mean(mcmlp_infected, axis=0),
            jnp.mean(mcmlp_recovered, axis=0),
        )
    )
    numpyro.sample("obs", dist.Normal(mean, 5), obs=data)


#####################################################################################
# PriorCVAE Model                                                                   #
#####################################################################################
def full_cov_vanilla_sir_PriorCVAE_model(
    priorcvae_fn, n_sims, grid_size, n_initial_infected, n_initial_recovered, data=None
):
    p_infect, p_recover, p_wane = no_move_prior()

    initial_infected = n_initial_infected / grid_size**2
    initial_recovered = n_initial_recovered / grid_size**2

    priorcvae_timeseries = priorcvae_fn(
        key=numpyro.prng_key(),
        num_samples=n_sims,
        c=jnp.broadcast_to(
            jnp.array(
                [p_infect, p_recover, p_wane, initial_infected, initial_recovered]
            ),
            (n_sims, 5),
        ),
    )
    priorcvae_infected, priorcvae_recovered, priorcvae_susceptible = jnp.split(
        priorcvae_timeseries, 3, axis=-1
    )

    priorcvae_infected = priorcvae_infected[:, 1:]
    priorcvae_recovered = priorcvae_recovered[:, 1:]
    mean = jnp.concatenate(
        (
            jnp.mean(priorcvae_infected, axis=0),
            jnp.mean(priorcvae_recovered, axis=0),
        )
    )
    covariance = jnp.cov(
        jnp.concatenate((priorcvae_infected, priorcvae_recovered), axis=1), rowvar=False
    ) + 1e7 * jnp.eye(mean.shape[0])
    numpyro.sample("obs", dist.MultivariateNormal(mean, covariance), obs=data)


#####################################################################################
# Diffusion Model                                                                   #
#####################################################################################
def full_cov_no_move_sir_Diff_model(
    diff_rollout_fn,
    abm_init,
    rgb_colours,
    n_sims,
    grid_size,
    total_population,
    n_initial_infected,
    n_initial_recovered,
    data=None,
):
    p_infect, p_recover, p_wane = no_move_prior()
    ###Initialise grid###

    multi_grid_timeseries = abm_init(
        jax.random.split(numpyro.prng_key(), n_sims),
        grid_size,
        n_initial_infected,
        n_initial_recovered,
        total_population,
    )

    # Map into RGB
    multi_grid_timeseries = map_batch_grid_series_to_rgb(
        multi_grid_timeseries, rgb_colours
    )

    # Rollout diffusion model conditioned on ABM
    diff_grid_series = diff_rollout_fn(
        numpyro.prng_key(),
        2 * multi_grid_timeseries - 1,
        jnp.tile(
            jnp.array([p_infect, p_recover, p_wane, total_population / grid_size**2]),
            (1, 1),
        ),
    )
    # Map back into RGB
    diff_grid_series = (diff_grid_series + 1) / 2

    # Same likelihood
    state_count_timeseries = rgb_vectorized_abm_state_counts_jax(diff_grid_series)

    means = {k: jnp.mean(x[:, 1:], axis=0) for k, x in state_count_timeseries.items()}

    stacked = jnp.concatenate(
        [x[:, 1:] for x in list(state_count_timeseries.values())[1:]], axis=1
    )
    mean = jnp.concatenate((means["Infected"], means["Recovered"]))
    covariance = jnp.cov(stacked, rowvar=False) + 1e-7 * jnp.eye(mean.shape[0])
    numpyro.sample("obs", dist.MultivariateNormal(mean, covariance), obs=data)


def map_batch_grid_series_to_rgb(x, rgb_colours):
    # B, T, C, H, W
    indexed_x = 2 * x[:, :, 0, :, :] + x[:, :, 1, :, :] + 3 * x[:, :, 2, :, :]
    return rgb_colours[indexed_x]


@hydra.main(
    version_base=None, config_path="../../configs/infer/spatial_compartmental/SIR"
)
def main(cfg: DictConfig):
    print(cfg)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    abm = get_abm(Neighbourhood(cfg.abm.neighbourhood), vmap=True)
    grid_size = cfg.abm.grid_size
    num_steps = cfg.abm.num_steps

    numpyro_model = hydra.utils.get_method(cfg.numpyro_model)
    mcmc_alg = hydra.utils.get_method(cfg.mcmc_algorithm)

    prior_func = hydra.utils.get_method(cfg.prior_func)
    prior_pred = numpyro.infer.Predictive(prior_func, num_samples=cfg.prior_num_samples)
    prior_predictions = prior_pred(
        rng_key=jax.random.PRNGKey(cfg.prior_predictive_seed)
    )
    data_gen_key = jax.random.PRNGKey(cfg.data_generation_seed)
    all_abm_parameters = cfg.abm_params

    post_dict = {}
    data_dict = {}
    for abm_param in tqdm(all_abm_parameters):
        data_gen_key, subkey = jax.random.split(data_gen_key)
        grid_timeseries = abm(
            subkey[None, :],
            grid_size,
            num_steps,
            abm_param["p_infect"],
            abm_param["p_recover"],
            abm_param["p_wane"],
            abm_param["p_move"],
            calc_start_n_initial(abm_param["total_population"], grid_size),
            calc_start_n_initial(abm_param["initial_infected"], grid_size),
            calc_start_n_initial(abm_param["initial_recovered"], grid_size),
        )
        state_count_timeseries = vectorized_abm_state_counts(grid_timeseries)

        if "summarised" in cfg.numpyro_model:
            cumulative_incidence = jnp.sum(state_count_timeseries["Infected"][:, 1:])

            # Number of timesteps to reach peak infection
            time_to_peak_infected = jnp.argmax(state_count_timeseries["Infected"])

            time_to_min_infected = jnp.argmin(state_count_timeseries["Infected"])

            # Largest increase in infection in a single timestep
            peak_incidence = jnp.max(jnp.diff(state_count_timeseries["Infected"]))

            minimum_susceptible = jnp.min(state_count_timeseries["Susceptible"])

            time_to_peak_recovered = jnp.argmax(state_count_timeseries["Recovered"])
            time_to_min_recovered = jnp.argmin(state_count_timeseries["Recovered"])

            data = jnp.array(
                [
                    cumulative_incidence,
                    time_to_peak_infected,
                    time_to_min_infected,
                    peak_incidence,
                    minimum_susceptible,
                    time_to_peak_recovered,
                    time_to_min_recovered,
                ]
            )
        else:
            # We exclude the first entry from the data since our likelihood also excludes it

            if "full_cov" in cfg.numpyro_model:
                data = jnp.concatenate(
                    (
                        state_count_timeseries["Infected"].squeeze()[1:],
                        state_count_timeseries["Recovered"].squeeze()[1:],
                    )
                )
            else:
                data = jnp.concatenate(
                    (
                        state_count_timeseries["Susceptible"].squeeze()[1:],
                        state_count_timeseries["Infected"].squeeze()[1:],
                        state_count_timeseries["Recovered"].squeeze()[1:],
                    )
                )
        data_dict[tuple(abm_param.values())] = data

        kernel = mcmc_alg(numpyro_model)
        mcmc = MCMC(
            kernel,
            num_warmup=cfg.num_warmup,
            num_samples=cfg.num_samples,
            num_chains=cfg.num_chains,
            chain_method=cfg.chain_method,
            progress_bar=True,
        )
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
            inf_kwargs["total_population"] = (
                calc_start_n_initial(abm_param["total_population"], grid_size),
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
                inf_kwargs["total_population"] = (
                    calc_start_n_initial(abm_param["total_population"], grid_size),
                )
        elif cfg.model_name == "priorcvae":
            priorcvae = CheckPointer.load(cfg.checkpoint)[0]
            inf_kwargs["priorcvae_fn"] = priorcvae.generate_decoder_samples
            if "vanilla" not in cfg.numpyro_model:
                inf_kwargs["total_population"] = (
                    calc_start_n_initial(abm_param["total_population"], grid_size),
                )
        elif cfg.model_name == "grid_diffuser":
            NUM_STEPS_CONDITIONING = 1
            NUM_STEPS_DENOISING = 3
            SNAP_AFTER_DIFFUSION = True
            SNAP_DURING_DIFFUSION = True
            rgb_colours = jnp.array(
                [[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
            ).astype(jnp.float32)
            den_cfg = DenoiserConfig(
                {
                    "img_channels": 3,
                    "num_steps_conditioning": NUM_STEPS_CONDITIONING,
                    "cond_channels": 256,
                    "depths": [2, 2, 2, 2],
                    "channels": [64, 64, 64, 64],
                    "attn_depths": [0, 0, 0, 0],
                    "num_abm_params": 4,
                    "static_h": 10,
                    "static_w": 10,
                },
                0.7,
                0.3,
            )
            # Trained on 20
            denoiser = Denoiser(
                den_cfg,
                rgb_palette=rgb_colours if SNAP_DURING_DIFFUSION else None,
                rngs=nnx.Rngs(0),
            )
            sigma_dist_conf = SigmaDistributionConfig(-0.4, 1.2, 2e-3, 20)
            denoiser.setup_training(sigma_dist_conf)

            _, abstract_model = nnx.state(denoiser, nnx.RngState, ...)
            checkpointer = CheckPointer(cfg.checkpoint)
            restored_data = checkpointer.restore(
                checkpointer.get_latest(), abstract_state=abstract_model
            )
            nnx.update(denoiser, restored_data["state"])

            diff_sampler_conf = DiffusionSamplerConfig(
                num_steps_denoising=NUM_STEPS_DENOISING,
                realisation_length=(num_steps + 1),
                snap_after_diffusion_palette=rgb_colours
                if SNAP_AFTER_DIFFUSION
                else None,
            )
            diff_sampler = DiffusionSampler(
                denoiser,
                diff_sampler_conf,
            )
            inf_kwargs["diff_rollout_fn"] = diff_sampler.sample_rollout
            inf_kwargs["abm_init"] = vabm_init
            inf_kwargs["rgb_colours"] = rgb_colours
            inf_kwargs["total_population"] = grid_size * grid_size

        else:
            raise ValueError("Invalid model name")

        start_time = time.perf_counter()
        jax.block_until_ready(
            mcmc.run(
                **inf_kwargs,
                data=data,
            )
        )
        inference_time = time.perf_counter() - start_time
        mcmc.print_summary()
        posterior_samples = mcmc.get_samples()
        if cfg.skip_prediction:
            posterior_predictions = {}
        else:
            posterior_predictive = numpyro.infer.Predictive(
                numpyro_model, posterior_samples
            )
            posterior_predictions = posterior_predictive(**inf_kwargs)
        posterior_predictions["posterior_samples"] = {}
        posterior_predictions["mcmc_summary"] = {}

        for param_name in cfg.params_to_infer:
            posterior_predictions["posterior_samples"][param_name] = posterior_samples[
                param_name
            ]
            posterior_predictions["mcmc_summary"][param_name] = mcmc_summary(
                {param_name: posterior_samples[param_name]}, prob=0.95
            )
        posterior_predictions["mcmc_time"] = inference_time
        post_dict[tuple(abm_param.values())] = posterior_predictions

        with open(f"{output_dir}/prior_predictions.npy", "wb") as f:
            jnp.save(f, prior_predictions)

    with open(f"{output_dir}/posterior.npy", "wb") as f:
        jnp.save(f, post_dict)
    with open(f"{output_dir}/data.npy", "wb") as f:
        jnp.save(f, data_dict)

    if cfg.cloud_save:
        upload_many_blobs_with_transfer_manager(
            "dgs4abm",
            os.path.relpath(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            ),
        )


if __name__ == "__main__":
    main()
