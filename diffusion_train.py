from surrogate.diffusion.denoiser import (
    Denoiser,
    DenoiserConfig,
    SigmaDistributionConfig,
)
from surrogate.diffusion.diffusion_sampler import (
    DiffusionSampler,
    DiffusionSamplerConfig,
)
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from checkpointing import CheckPointer
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


def train_step(denoiser, batch):
    loss = denoiser(batch)
    return loss


def map_batch_grid_series_to_rgb(x, rgb_colours):
    # B, T, C, H, W
    indexed_x = 2 * x[:, :, 0, :, :] + x[:, :, 1, :, :] + 3 * x[:, :, 2, :, :]
    return rgb_colours[indexed_x]


def ImgEpochIterator(
    key,
    datasets,
    batch_size,
    indices,
    n_sim_steps,
    seq_length,
):
    for i in range(0, len(indices), batch_size):
        key, starts_key = jax.random.split(key, 2)
        idx = indices[i : i + batch_size]

        starts = jax.random.randint(
            starts_key, (batch_size,), 0, n_sim_steps - seq_length, dtype=jnp.int32
        )

        # stops = (starts + seq_length).astype(jnp.int32)
        # print("Starts:\n", starts)
        # print("Stops:\n", stops)

        batch_indices = jnp.arange(batch_size)

        time_offsets = jnp.arange(seq_length)
        # print("Indexing:")
        twoD_indices = starts[:, None] + time_offsets[None, :]

        yield {
            "abm_params": datasets["params"][idx],
            "obs": datasets["realisations"][idx][batch_indices[:, None], twoD_indices],
        }


if __name__ == "__main__":
    DATA_ON_HOST = bool(os.getenv("DATA_ON_HOST", False))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 100))
    N_EPOCHS = int(os.getenv("N_EPOCHS", 7))
    WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", 100))

    AUTOREG_STEPS = int(os.getenv("AUTOREG_STEPS", 1))
    SNAP_OUTPUTS = bool(os.getenv("SNAP_OUTPUTS", True))

    DATASET = str(os.getenv("DATASET"))
    NUM_ABM_PARAMS = int(os.getenv("NUM_ABM_PARAMS"))
    NUM_COND = int(os.getenv("NUM_COND", 1))
    MODEL_SIZE = int(os.getenv("MODEL_SIZE", 1))
    BF16 = bool(os.getenv("BF16", False))
    DWSC = bool(os.getenv("DWSC", False))

    GRID_SIZE = 10

    rgb_palette = jnp.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(
        jnp.float32
    )
    with open(f"{DATASET}/train_realisations.npy", "rb") as f:
        all_realisations = jax.device_get(np.load(f)) if DATA_ON_HOST else jnp.load(f)
    with open(f"{DATASET}/train_param_labels.npy", "rb") as f:
        all_params = jax.device_get(np.load(f)) if DATA_ON_HOST else jnp.load(f)

    DATASET_SIZE = all_realisations.shape[0]

    realisations_std = np.std(2 * all_realisations.astype(np.float32) - 1)
    print("Realisations std: ", realisations_std)
    rgb_colours = jnp.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(
        jnp.float32
    )

    if MODEL_SIZE == 0:
        den_cfg = DenoiserConfig(
            {
                "img_channels": 3,
                "num_steps_conditioning": NUM_COND,
                "cond_channels": 256,
                "depths": [2, 2],
                "channels": [32, 32],
                "attn_depths": [0, 0],
                "num_abm_params": NUM_ABM_PARAMS,
                "static_h": GRID_SIZE,
                "static_w": GRID_SIZE,
                "depthwise_sep_conv_unet": DWSC,
            },
            realisations_std,
            0.3,
        )
    elif MODEL_SIZE == 1:
        den_cfg = DenoiserConfig(
            {
                "img_channels": 3,
                "num_steps_conditioning": NUM_COND,  # 4,
                "cond_channels": 256,
                "depths": [2, 2, 2],  # [2, 2, 2, 2],
                "channels": [32, 32, 32],  # [64, 64, 64, 64],
                "attn_depths": [0, 0, 0],  # [0, 0, 0, 0],
                "num_abm_params": NUM_ABM_PARAMS,
                "static_h": GRID_SIZE,
                "static_w": GRID_SIZE,
                "depthwise_sep_conv_unet": DWSC,
            },
            realisations_std,  # 1.5,
            0.3,
        )

    else:
        den_cfg = DenoiserConfig(
            {
                "img_channels": 3,
                "num_steps_conditioning": NUM_COND,  # 4,
                "cond_channels": 256,
                "depths": [2, 2, 2, 2],  # [2, 2, 2, 2],
                "channels": [64, 64, 64, 64],  # [64, 64, 64, 64],
                "attn_depths": [0, 0, 0, 0],  # [0, 0, 0, 0],
                "num_abm_params": NUM_ABM_PARAMS,
                "static_h": GRID_SIZE,
                "static_w": GRID_SIZE,
                "depthwise_sep_conv_unet": DWSC,
            },
            realisations_std,  # 1.5,
            0.3,
        )
    print(den_cfg)
    denoiser = Denoiser(
        den_cfg,
        rgb_palette=rgb_palette if SNAP_OUTPUTS else None,
        dtype=jnp.bfloat16 if BF16 else jnp.float32,
        rngs=nnx.Rngs(0),
    )
    sigma_dist_conf = SigmaDistributionConfig(-0.4, 1.2, 2e-3, 20)
    denoiser.setup_training(sigma_dist_conf)

    lr_schedule = optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=0.0,
                end_value=1e-4,
                transition_steps=WARMUP_STEPS,
            ),
            optax.constant_schedule(1e-4),
        ],
        boundaries=[WARMUP_STEPS],
    )
    params = nnx.state(denoiser, nnx.Param)
    total_params = sum(
        [jnp.prod(jnp.array(x.shape)) for x in jax.tree.leaves(params)], 0
    )
    print("Number of parameters: ", total_params)

    optimizer = nnx.Optimizer(denoiser, optax.adamw(lr_schedule))

    grad_fn = nnx.value_and_grad(train_step)
    chkptr = CheckPointer("test_diff", max_to_keep=10)
    key = jax.random.key(0)

    SEQ_LENGTH = den_cfg.inner_model["num_steps_conditioning"] + 1 + AUTOREG_STEPS
    for epoch in range(N_EPOCHS):
        batches_key, perm_key = jax.random.split(jax.random.fold_in(key, epoch), 2)
        dataiter = ImgEpochIterator(
            batches_key,
            {"params": all_params, "realisations": all_realisations},
            BATCH_SIZE,
            jax.random.permutation(perm_key, jnp.arange(all_realisations.shape[0])),
            21,
            SEQ_LENGTH,
        )
        for step, batch in tqdm(enumerate(dataiter), total=DATASET_SIZE // BATCH_SIZE):
            # print(lr_schedule(epoch * 100_000 // BATCH_SIZE + idx))
            batch_obs = (
                batch["obs"]
                .reshape(BATCH_SIZE, SEQ_LENGTH, 10, 10, 3)
                .astype(jnp.float32)
            )

            batch_params = batch["abm_params"][:, :NUM_ABM_PARAMS]

            if DATA_ON_HOST:
                batch_obs = jax.device_put(batch_obs)
                batch_params = jax.device_put(batch_params)

            loss, grads = grad_fn(
                denoiser,
                {
                    "obs": 2 * batch_obs - 1,  # Scale into [-1,1]
                    "abm_params": batch_params,
                },
            )
            if step % 100 == 0:
                print(f"[Epoch {epoch}] Step {step} Loss: {loss:.3e}")

            optimizer.update(grads)

        res = chkptr.save(epoch, denoiser, jax.random.PRNGKey(0))
        print(f"Checkpointing at epoch {epoch} - {'success' if res else 'fail'}")
