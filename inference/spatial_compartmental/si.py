import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, AIES
from abm.spatial_compartmental.si import get_abm, _scan_convolution_SI
from abm.spatial_compartmental.utils import calc_start_n_initial, Neighbourhood
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


# Modified from  numpyro.diagnostics
def mcmc_summary(samples, prob=0.89):
    """
    Prints a summary table displaying diagnostics of ``samples`` from the
    posterior. The diagnostics displayed are mean, standard deviation,
    the 89% Credibility Interval, :func:`~numpyro.diagnostics.effective_sample_size`
    :func:`~numpyro.diagnostics.split_gelman_rubin`.

    :param samples: a collection of input samples.
    :param float prob: the probability mass of samples within the HPDI interval.
    """
    if not isinstance(samples, dict):
        samples = {
            "Param:{}".format(i): v for i, v in enumerate(jax.tree_flatten(samples)[0])
        }
    # TODO: support summary for chains of samples
    for name, value in samples.items():
        value = jax.device_get(value)
        mean = value.mean(axis=0)
        sd = value.std(axis=0, ddof=1)
        hpd = numpyro.diagnostics.hpdi(value, prob=prob)
        n_eff = numpyro.diagnostics.effective_sample_size(value[None, ...])
        r_hat = numpyro.diagnostics.split_gelman_rubin(value[None, ...])
        shape = value.shape[1:]
        if len(shape) == 0:
            return [
                {
                    "var_name": name,
                    "mean": mean,
                    "std": sd,
                    "hpdi": (hpd[0], hpd[1]),
                    "n_eff": n_eff,
                    "r_hat": r_hat,
                }
            ]
        else:
            rows = []
            for idx in itertools.product(*map(range, shape)):
                idx_str = "[{}]".format(",".join(map(str, idx)))
                rows.append(  # NOTE: Currently untested
                    {
                        name + idx_str,
                        mean[idx],
                        sd[idx],
                        hpd[0][idx],
                        hpd[1][idx],
                        n_eff[idx],
                        r_hat[idx],
                    }
                )

            return rows


######################################################################
# Shared Beta Prior on p_infect                                      #
######################################################################
def p_infect_prior():
    p_infect = numpyro.sample("p_infect", dist.Beta(2, 5))
    return p_infect


########################################################################
# Multiple samples per parameter & Diagonal Covariance                 #
########################################################################
def diag_cov_ABM_model(
    abm, n_sims, grid_size, num_steps, n_initial_infected, data=None
):
    p_infect = p_infect_prior()

    _, _, I_t = abm(
        jax.random.split(numpyro.prng_key(), n_sims),
        grid_size,
        num_steps,
        p_infect,
        n_initial_infected,
    )

    # NOTE: Since the first count is always the same, we exclude it from the likelihood
    sample_mean = jnp.mean(I_t[:, 1:], axis=0)
    sample_std = jnp.std(I_t[:, 1:], axis=0, ddof=1) + 1e-7

    numpyro.sample("obs", dist.Normal(sample_mean, sample_std), obs=data)


def diag_cov_MCMLP_model(
    surrogate_fn, n_sims, grid_size, prop_initial_infected, data=None
):
    p_infect = p_infect_prior()

    I_counts = surrogate_fn(
        jnp.array([p_infect, prop_initial_infected / (grid_size**2)]),
        jax.random.split(numpyro.prng_key(), n_sims),
    )
    I_counts = I_counts * (grid_size**2)

    # NOTE: Since the first count is always the same, we exclude it from the likelihood
    sample_mean = jnp.mean(I_counts[:, 1:], axis=0)
    sample_std = jnp.std(I_counts[:, 1:], axis=0, ddof=1) + 1e-7

    numpyro.sample("obs", dist.Normal(sample_mean, sample_std), obs=data)


def diag_cov_PriorCVAE_model(
    surrogate_fn, n_sims, grid_size, prop_initial_infected, data=None
):
    p_infect = p_infect_prior()
    I_counts = surrogate_fn(
        key=numpyro.prng_key(),
        num_samples=n_sims,
        c=jnp.broadcast_to(
            jnp.array([p_infect, prop_initial_infected / (grid_size**2)]),
            (n_sims, 2),
        ),
    )
    I_counts = I_counts * (grid_size**2)

    # NOTE: Since the first count is always the same, we exclude it from the likelihood
    sample_mean = jnp.mean(I_counts[:, 1:], axis=0)
    sample_std = (
        jnp.std(I_counts[:, 1:], axis=0, ddof=1) + 1e-7
    )  # NOTE: Seems to be some instability

    numpyro.sample("obs", dist.Normal(sample_mean, sample_std), obs=data)


@hydra.main(version_base=None, config_path="../../configs/infer/spatial_compartmental")
def main(cfg: DictConfig):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    abm = jax.jit(
        get_abm(Neighbourhood(cfg.abm.neighbourhood), vmap=("n_sims" in cfg)),
        static_argnums=[1, 2, 4],
    )
    grid_size = cfg.abm.grid_size
    num_steps = cfg.abm.num_steps

    abm_numpyro_model = hydra.utils.get_method(cfg.abm.numpyro_model)
    abm_mcmc_alg = hydra.utils.get_method(cfg.abm.mcmc_algorithm)

    surrogate_numpyro_models = {}
    surrogate_mcmc_algs = {}
    surrogate_neural_networks = {}
    for surrogate_name, surrogate_settings in cfg.surrogates.items():
        surrogate_numpyro_models[surrogate_name] = hydra.utils.get_method(
            surrogate_settings.numpyro_model
        )
        surrogate_mcmc_algs[surrogate_name] = hydra.utils.get_method(
            surrogate_settings.mcmc_algorithm
        )
        surrogate_neural_networks[surrogate_name] = CheckPointer.load(
            surrogate_settings.checkpoint
        )[0]

    cartesian_product = list(
        itertools.product(cfg.params.p_infect, cfg.prop_initials.prop_initial_infected)
    )
    prior_predictive = numpyro.infer.Predictive(
        p_infect_prior, num_samples=cfg.abm.prior_num_samples
    )
    prior_predictions = prior_predictive(
        rng_key=jax.random.PRNGKey(cfg.prior_predictive_seed)
    )
    abm_post_dict = {}
    surrogates_post_dict = {}
    data_dict = {}

    data_gen_key = jax.random.PRNGKey(cfg.data_generation_seed)

    for p_infect, prop_initial_infected in tqdm(cartesian_product):
        data_gen_key, subkey = jax.random.split(data_gen_key)
        _, _, data = abm(
            subkey[None, :] if "n_sims" in cfg else subkey,
            grid_size,
            num_steps,
            p_infect,
            int(calc_start_n_initial(prop_initial_infected, grid_size)),
        )
        # NOTE: Since the first count is always the same, we exclude it from the likelihood
        data = data.squeeze()[1:]
        data_dict[(p_infect, prop_initial_infected)] = data

        ### ABM Inference ###
        ABM_kernel = abm_mcmc_alg(abm_numpyro_model)
        ABM_mcmc = MCMC(
            ABM_kernel,
            num_warmup=cfg.abm.num_warmup,
            num_samples=cfg.abm.num_samples,
            num_chains=cfg.abm.num_chains,
            chain_method=cfg.abm.chain_method,
            progress_bar=False,
        )
        ABM_kwargs = (
            {
                "rng_key": jax.random.PRNGKey(cfg.inference_seed),
                "abm": abm,
                "grid_size": grid_size,
                "num_steps": num_steps,
                "n_initial_infected": int(
                    calc_start_n_initial(prop_initial_infected, grid_size)
                ),
            }
            | (
                {"abm_key": jax.random.PRNGKey(cfg.abm_seed)}
                if "abm_seed" in cfg
                else {}
            )
            | ({"n_sims": cfg.n_sims} if "n_sims" in cfg else {})
        )
        start_time = time.perf_counter()

        jax.block_until_ready(
            ABM_mcmc.run(
                **ABM_kwargs,
                data=data,
            )
        )

        abm_time = time.perf_counter() - start_time
        ABM_posterior_samples = ABM_mcmc.get_samples()
        ABM_posterior_predictive = numpyro.infer.Predictive(
            abm_numpyro_model, ABM_posterior_samples
        )
        ABM_posterior_predictions = ABM_posterior_predictive(**ABM_kwargs)
        ABM_posterior_predictions["posterior_samples"] = {
            "p_infect": ABM_posterior_samples["p_infect"]
        }
        ABM_posterior_predictions["mcmc_time"] = abm_time
        ABM_posterior_predictions["mcmc_summary"] = mcmc_summary(ABM_posterior_samples)
        abm_post_dict[(p_infect, prop_initial_infected)] = ABM_posterior_predictions

        ### Surrogates ###
        for surrogate_name, surrogate_settings in cfg.surrogates.items():
            surrogate_kernel = surrogate_mcmc_algs[surrogate_name](
                surrogate_numpyro_models[surrogate_name]
            )

            surrogate_mcmc = MCMC(
                surrogate_kernel,
                num_warmup=surrogate_settings.num_warmup,
                num_samples=surrogate_settings.num_samples,
                num_chains=surrogate_settings.num_chains,
                chain_method=surrogate_settings.chain_method,
                progress_bar=False,
            )
            surrogate_kwargs = (
                {
                    "rng_key": jax.random.PRNGKey(cfg.inference_seed),
                    "surrogate_fn": nnx.jit(
                        surrogate_neural_networks[
                            surrogate_name
                        ].generate_decoder_samples,
                        static_argnames=["num_samples"],
                    )
                    if surrogate_settings.func == "generate_decoder_samples"
                    else (
                        nnx.jit(
                            partial(
                                nnx.vmap(
                                    lambda module, x, key: module(x, key),
                                    in_axes=(None, None, 0),
                                    out_axes=0,
                                ),
                                surrogate_neural_networks[surrogate_name],
                            )
                        )
                        if "n_sims" in cfg
                        else nnx.jit(surrogate_neural_networks[surrogate_name].__call__)
                    ),
                    "grid_size": grid_size,
                    "prop_initial_infected": int(
                        calc_start_n_initial(prop_initial_infected, grid_size)
                    ),
                }
                | (
                    {"abm_key": jax.random.PRNGKey(cfg.abm_seed)}
                    if "abm_seed" in cfg
                    else {}
                )
                | ({"n_sims": cfg.n_sims} if "n_sims" in cfg else {})
            )

            start_time = time.perf_counter()
            jax.block_until_ready(
                surrogate_mcmc.run(
                    **surrogate_kwargs,
                    data=data,
                )
            )
            surrogate_time = time.perf_counter() - start_time
            surrogate_posterior_samples = surrogate_mcmc.get_samples()

            surrogate_posterior_predictive = numpyro.infer.Predictive(
                surrogate_numpyro_models[surrogate_name], surrogate_posterior_samples
            )
            surrogate_posterior_predictions = surrogate_posterior_predictive(
                **surrogate_kwargs
            )
            surrogate_posterior_predictions["posterior_samples"] = {
                "p_infect": surrogate_posterior_samples["p_infect"]
            }
            surrogate_posterior_predictions["mcmc_time"] = surrogate_time
            surrogate_posterior_predictions["mcmc_summary"] = mcmc_summary(
                surrogate_posterior_samples
            )
            if surrogates_post_dict.get(surrogate_name) is None:
                surrogates_post_dict[surrogate_name] = {}

            surrogates_post_dict[surrogate_name][(p_infect, prop_initial_infected)] = (
                surrogate_posterior_predictions
            )
    with open(f"{output_dir}/prior_predictions.npy", "wb") as f:
        jnp.save(f, prior_predictions)
    with open(f"{output_dir}/abm_posterior.npy", "wb") as f:
        jnp.save(f, abm_post_dict)
    with open(f"{output_dir}/surrogates_posterior.npy", "wb") as f:
        jnp.save(f, surrogates_post_dict)
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
