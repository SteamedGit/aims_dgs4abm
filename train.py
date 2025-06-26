from flax import nnx
import hydra.conf
import jax
import surrogate.mc_dropout
import jax.numpy as jnp
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from dataloaders import DataLoader
from logger import LogEvent, MultiLogger
import time
import optax
from tqdm import tqdm
from functools import partial
from jax.typing import ArrayLike
from typing import Callable, Tuple, Dict, Union, TypeAlias
import time
import logging
from checkpointing import CheckPointer
from cloud_storage_utils import upload_many_blobs_with_transfer_manager
import os
import wandb
import tempfile


# TODO: Move type aliases to seperate file
# TODO: Reconsider having a Batch and Metrics Type
BatchType: TypeAlias = Dict[str, ArrayLike]
MetricsType: TypeAlias = Union[nnx.MultiMetric, nnx.Metric]
LossFn: TypeAlias = Callable[[nnx.Module, BatchType], jax.Array]
TrainStepFn: TypeAlias = Callable[
    [
        nnx.Module,
        nnx.Optimizer,
        MetricsType,
        BatchType,
        jax.random.PRNGKey,
    ],
    None,
]
EvalStepFn: TypeAlias = Callable[
    [nnx.Module, MetricsType, BatchType, jax.random.PRNGKey], None
]


def get_train_and_eval_steps(
    loss_fn: LossFn,
) -> Tuple[TrainStepFn, EvalStepFn]:
    def train_step(
        loss_fn: LossFn,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        metrics: MetricsType,
        batch: BatchType,
        key: jax.random.PRNGKey,
    ) -> None:
        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(model, batch, key)
        metrics.update(loss=loss)
        optimizer.update(grads)

    def eval_step(
        loss_fn: LossFn,
        model: nnx.Module,
        metrics: MetricsType,
        batch: BatchType,
        key: jax.random.PRNGKey,
    ) -> None:
        loss = loss_fn(model, batch, key)
        metrics.update(loss=loss)

    return nnx.jit(partial(train_step, loss_fn)), nnx.jit(partial(eval_step, loss_fn))


@hydra.main(
    version_base=None,
    config_path="configs/train",
    config_name="basic",
)
def main(cfg: DictConfig):
    print(cfg)

    logger = MultiLogger(cfg)
    if cfg.logger.use_wandb:
        yaml_str = OmegaConf.to_yaml(cfg)
        with tempfile.NamedTemporaryFile(
            mode="w+t",
            prefix="train_config",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        ) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(yaml_str)
            temp_file.flush()
            wandb.save(
                temp_file_path, base_path=os.path.dirname(temp_file_path), policy="now"
            )
    # TODO: Cleanup
    use_checkpointing = (
        bool(str.lower(cfg.logger.checkpointer.save_every) != "never")
        if isinstance(cfg.logger.checkpointer.save_every, str)
        else True
    )
    # TODO: Handle restoring from a different checkpoint
    checkpointer = None
    if use_checkpointing:
        assert cfg.logger.checkpointer.save_every % cfg.eval_every == 0, (
            "Checkpointing frequency must be divisible by eval frequency"
        )

        checkpointer = CheckPointer(
            model_name=str.split(cfg.model.arch._target_, ".")[-1],
            rel_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            save_interval_steps=cfg.logger.checkpointer.save_every,
            max_to_keep=cfg.logger.checkpointer.max_to_keep,
        )
    else:
        assert not cfg.cloud_save, "Cannot use cloud save if not saving checkpoints"

    train_rng, subkey = jax.random.split(jax.random.PRNGKey(cfg.train_rng_seed))
    train_dataloader = DataLoader(
        key=subkey,
        paths=cfg.datasets.train
        if not cfg.fuse_train_and_test
        else [cfg.datasets.train, cfg.datasets.test],
        batch_size=cfg.hyperparameters.train_batch_size,
        shuffle_every_epoch=cfg.hyperparameters.shuffle_every_epoch,
    )
    if not cfg.fuse_train_and_test:
        train_rng, subkey = jax.random.split(train_rng)
        test_dataloader = DataLoader(
            key=subkey,
            paths=cfg.datasets.test,
            batch_size=cfg.hyperparameters.eval_batch_size,
            shuffle_every_epoch=False,
        )
    # NOTE: We don't actually use NNX's stateful RNG system due to complications
    # with checkpointing. Instead we pass keys into the __call__ function
    model_rng_key = jax.random.PRNGKey(cfg.model_rng_seed)

    model = hydra.utils.instantiate(cfg.model.arch, rngs=nnx.Rngs(0))
    # TODO: Allow for chaining
    optimizer = nnx.Optimizer(
        model,
        hydra.utils.instantiate(cfg.hyperparameters.optax),
    )
    metrics = hydra.utils.instantiate(cfg.metrics)
    train_step, eval_step = get_train_and_eval_steps(
        hydra.utils.get_method(cfg.model.loss_fn)
    )

    params = nnx.state(model, nnx.Param)
    total_params = sum(
        [jnp.prod(jnp.array(x.shape)) for x in jax.tree.leaves(params)], 0
    )
    logger.log({"Number of parameters": total_params}, 0, LogEvent.MISC)
    # NOTE: Right now its assumed that test_metrics is just loss
    overall_train_step = 0
    test_metrics_history = []
    for epoch in range(cfg.hyperparameters.n_epochs):
        logger.log({"EPOCH": epoch}, overall_train_step, LogEvent.MISC)
        for idx, batch in enumerate(iter(train_dataloader)):
            model_rng_key, subkey = jax.random.split(model_rng_key)
            train_step(model, optimizer, metrics, batch, subkey)

            if overall_train_step % cfg.eval_every == 0:
                # Train metrics
                logger.log(metrics.compute(), overall_train_step, LogEvent.TRAIN)
                metrics.reset()

                if not cfg.fuse_train_and_test:
                    # Compute Test metrics
                    model.eval()
                    for test_batch in iter(test_dataloader):
                        model_rng_key, subkey = jax.random.split(model_rng_key)
                        eval_step(model, metrics, test_batch, subkey)

                    last_test_metrics = metrics.compute()
                    test_metrics_history.append(last_test_metrics)
                    # Log test metrics
                    logger.log(last_test_metrics, overall_train_step, LogEvent.TEST)
                    metrics.reset()
                    model.train()

            if use_checkpointing and checkpointer.should_save(overall_train_step):
                model_save_success = checkpointer.save(
                    overall_train_step,
                    model,
                    model_rng_key,
                )
                if not model_save_success:
                    # TODO: Handle
                    pass
                else:
                    logger.log(
                        {"Checkpoint Saved": overall_train_step},
                        overall_train_step,
                        LogEvent.MISC,
                    )

            overall_train_step += 1
    if use_checkpointing:
        checkpointer.save(overall_train_step, model, model_rng_key)
    if cfg.cloud_save:
        upload_many_blobs_with_transfer_manager(
            "dgs4abm",
            os.path.relpath(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            ),
        )
        logger.log(
            {
                "Uploaded train checkpoints": os.path.relpath(
                    hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
                )
            },
            overall_train_step,
            LogEvent.MISC,
        )
    logger.stop()
    if cfg.logger.use_wandb:
        os.remove(temp_file_path)


if __name__ == "__main__":
    main()
