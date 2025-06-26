# EDM Diffusion Sampler from https://github.com/eloialonso/diamond/blob/main/src/models/diffusion/diffusion_sampler.py

import os

# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true "
# )
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
from flax import nnx
import jax
import jax.numpy as jnp
from functools import partial
from abm.spatial_compartmental.sir import get_abm
from abm.spatial_compartmental.si import get_abm as get_old_abm
from abm.spatial_compartmental.utils import Neighbourhood
from surrogate.mc_dropout import MCMLP


from surrogate.diffusion.denoiser import (
    Denoiser,
    DenoiserConfig,
    SigmaDistributionConfig,
    map_image_to_palette,
)


# Diffusion Denoising Schedule
def build_sigmas(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: int,
):
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    discretised_time = jnp.linspace(0, 1, num_steps)
    sigmas = max_inv_rho + discretised_time * (min_inv_rho - max_inv_rho) ** rho
    return jnp.concat((sigmas, jnp.zeros((1,), dtype=sigmas.dtype)))


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int
    realisation_length: int
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1
    snap_after_diffusion_palette: Optional[jax.Array] = None


def euler_sample(
    cfg,
    gamma_,
    denoiser,
    x,
    sigma,
    next_sigma,
    noise_key,
    prev_obs,
    abm_params,
):
    gamma = jnp.where(
        jnp.logical_and(cfg.s_tmin <= sigma, sigma <= cfg.s_tmax), gamma_, 0
    )
    sigma_hat = sigma * (gamma + 1)
    x = jnp.where(
        gamma > 0,  # Whether we're using stochastic dynamics or not
        x
        + jax.random.normal(noise_key, x.shape)
        * cfg.s_noise
        * (sigma_hat**2 - sigma**2) ** 0.5,
        x,
    )
    #########################################################################
    # NOTE: This denoised output may be snapped to an exact rgb_palette.    #
    # However, we still retain the ability to iteratively improve since     #
    # it is combined with an x that can have values outside of the palette. #
    #########################################################################
    denoised = denoiser.denoise(x, sigma, prev_obs, abm_params)
    d = (x - denoised) / sigma_hat
    dt = next_sigma - sigma_hat
    return x + d * dt


def build_sampling(sample_fn, cfg, sigmas, denoiser):
    gamma_ = min(cfg.s_churn / (len(sigmas) - 1), 2**0.5 - 1)
    app_sample_fn = partial(
        sample_fn,
        cfg,
        gamma_,
        denoiser,
    )

    def _sample(
        sample_fn,
        sigmas,
        noise_key,
        prev_obs,
        abm_params,
    ):
        b, t, h, w, c = prev_obs.shape
        prev_obs = prev_obs.reshape(b, h, w, t * c)
        noise_key, subkey = jax.random.split(noise_key)
        # NOTE: This deviates from EDM where you would scale by first sigma,
        # the DIAMOND authors found that a lower variance starting point
        # helped to mitigate autoregressive drift
        # https://github.com/eloialonso/diamond/issues/40
        x = jax.random.normal(subkey, (b, h, w, c))
        trajectory = []
        for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
            noise_key, subkey = jax.random.split(noise_key)
            x = sample_fn(x, sigma, next_sigma, subkey, prev_obs, abm_params)
            trajectory.append(x)
        return x, trajectory

    ret_func = nnx.jit(partial(_sample, app_sample_fn, sigmas))
    return ret_func


def build_rollout(built_sample_func, cfg, num_steps_conditioning):
    def _scan_body(
        sample_fn,
        snap_after_diffusion_palette,
        # realisation_length,
        num_steps_conditioning,
        carry,
        idx,
    ):
        all_obs_current, noise_key, abm_params = carry
        noise_key, subkey = jax.random.split(noise_key)
        obs = jax.lax.dynamic_slice_in_dim(
            all_obs_current, idx, num_steps_conditioning, axis=1
        )
        denoised_pred = sample_fn(subkey, obs, abm_params)[0]
        #############################################################
        # Independently of whether the denoised x_0 predictions     #
        # are snapped during euler sampling, we may also snap       #
        # the final output of euler sampling. This final output     #
        # is the next step in the simulation and future predictions #
        # will condition on it.                                     #
        #############################################################
        if snap_after_diffusion_palette is not None:
            denoised_pred = (
                2
                * map_image_to_palette(
                    (denoised_pred + 1) / 2, snap_after_diffusion_palette
                )
                - 1
            )
        all_obs_current = jax.lax.dynamic_update_slice_in_dim(
            all_obs_current,
            denoised_pred[:, None, ...],
            idx + num_steps_conditioning,
            axis=1,
        )
        new_carry = (all_obs_current, noise_key, abm_params)
        return new_carry, None

    def _rollout_fn(
        scan_body_func,
        realisation_length,
        num_steps_conditioning,
        noise_key,
        initial_obs,
        abm_params,
    ):
        # initial obs shape:
        # B,Num_steps_cond,H,W,C
        sequence_length = realisation_length - num_steps_conditioning
        all_obs = jnp.pad(
            initial_obs.copy(), (((0, 0), (0, sequence_length), (0, 0), (0, 0), (0, 0)))
        )
        sequence_indices = jnp.arange(sequence_length)
        final_state, _ = jax.lax.scan(
            scan_body_func,
            (all_obs, noise_key, abm_params),
            sequence_indices,
        )
        return final_state[0]

    scan_body = partial(
        _scan_body,
        built_sample_func,
        cfg.snap_after_diffusion_palette,
        num_steps_conditioning,
    )
    ret_func = nnx.jit(
        partial(
            _rollout_fn,
            scan_body,
            cfg.realisation_length,
            num_steps_conditioning,
        )
    )
    return ret_func


class DiffusionSampler:
    def __init__(self, denoiser: Denoiser, cfg: DiffusionSamplerConfig):
        self.denoiser = denoiser
        self.cfg = cfg
        self.sigmas = build_sigmas(
            cfg.num_steps_denoising,
            cfg.sigma_min,
            cfg.sigma_max,
            cfg.rho,
        )
        self.sample = build_sampling(
            sample_fn=euler_sample,
            cfg=self.cfg,
            sigmas=self.sigmas,
            denoiser=self.denoiser,
        )
        self.sample_rollout = build_rollout(
            self.sample, cfg, denoiser.cfg.inner_model["num_steps_conditioning"]
        )
