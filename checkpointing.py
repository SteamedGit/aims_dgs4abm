# Adapted from https://github.com/EdanToledo/Stoix/blob/main/stoix/utils/checkpointing.py
import absl.logging

absl.logging.set_verbosity(absl.logging.WARNING)
import hydra.conf
import orbax.checkpoint
from datetime import datetime
from typing import Any, Dict, NamedTuple, Optional, Tuple, Type, Union
import jax
from omegaconf import DictConfig, OmegaConf
import os
from flax import nnx
import hydra
from hydra.core.global_hydra import GlobalHydra

# Keep track of the version of the checkpointer
# Any breaking API changes should be reflected in the major version (e.g. v0.1 -> v1.0)
# whereas minor versions (e.g. v0.1 -> v0.2) indicate backwards compatibility
CHECKPOINTER_VERSION = 1.0


class CheckPointer:
    def __init__(
        self,
        model_name: str,
        metadata: Optional[Dict] = None,
        rel_dir: str = "checkpoints",
        checkpoint_uid: Optional[str] = None,
        save_interval_steps: int = 1,
        max_to_keep: Optional[int] = 1,
        keep_period: Optional[int] = None,
    ):
        """Initialise the checkpointer tool

        Args:
            model_name (str): Name of the model to be saved.
            metadata (Optional[Dict], optional):
                For storing model metadata. Defaults to None.
            rel_dir (str, optional):
                Relative directory of checkpoints. Defaults to "checkpoints".
            checkpoint_uid (Optional[str], optional):
                Set the uniqiue id of the checkpointer, rel_dir/model_name/checkpoint_uid/...
                If not given, the timestamp is used.
            save_interval_steps (int, optional):
                The interval at which checkpoints should be saved. Defaults to 1.
            max_to_keep (Optional[int], optional):
                Maximum number of checkpoints to keep. Defaults to 1.
            keep_period (Optional[int], optional):
                If set, will not delete any checkpoint where
                checkpoint_step % keep_period == 0. Defaults to None.
        """
        # TODO: Maybe best_fn?
        options = orbax.checkpoint.CheckpointManagerOptions(
            create=True,
            save_interval_steps=save_interval_steps,
            max_to_keep=max_to_keep,
            keep_period=keep_period,
        )

        def get_json_ready(obj: Any) -> Any:
            if not isinstance(obj, (bool, str, int, float, type(None))):
                return str(obj)
            else:
                return obj

        # Convert metadata to JSON-ready format
        if metadata is not None and isinstance(metadata, DictConfig):
            metadata = OmegaConf.to_container(metadata, resolve=True)
        metadata_json_ready = jax.tree.map(get_json_ready, metadata)

        self._manager = orbax.checkpoint.CheckpointManager(
            directory=os.path.join(os.getcwd(), rel_dir, model_name, checkpoint_uid)
            if checkpoint_uid is not None
            else os.path.join(os.getcwd(), rel_dir, model_name),
            options=options,
            metadata={
                "checkpointer_version": CHECKPOINTER_VERSION,
                **(metadata_json_ready if metadata_json_ready is not None else {}),
            },
        )

    def get_latest(self) -> int:
        return self._manager.latest_step()

    # TODO: Save optimizer as well?
    def save(
        self,
        train_step: int,
        model: nnx.Module,
        train_rng_key: jax.random.PRNGKey,
    ) -> bool:
        # Need to seperate the model state (params) from the RNG
        # We don't actually use nnx's stateful randomness so no need to save
        rngstate, state = nnx.state(model, nnx.RngState, ...)

        model_save_success: bool = self._manager.save(
            step=train_step,
            args=orbax.checkpoint.args.Composite(
                state=orbax.checkpoint.args.StandardSave(state),
                train_rng_key=orbax.checkpoint.args.ArraySave(train_rng_key),
            ),
        )
        # TODO: Do i need this if I'm waiting for it to return?
        self._manager.wait_until_finished()
        return model_save_success

    def restore(self, train_step: int, abstract_state):
        return self._manager.restore(
            train_step,
            args=orbax.checkpoint.args.Composite(
                state=orbax.checkpoint.args.StandardRestore(abstract_state),
                train_rng_key=orbax.checkpoint.args.ArrayRestore(),
            ),
        )

    def should_save(self, train_step: int) -> bool:
        return self._manager.should_save(train_step)

    @staticmethod
    def load(
        rel_dir: str, train_step: Optional[Tuple[str, int]] = "last"
    ) -> Tuple[nnx.Module, jax.random.PRNGKey]:
        if not GlobalHydra.instance().is_initialized():
            with hydra.initialize(
                version_base=None, config_path=os.path.join(rel_dir, "hydra")
            ):
                cfg = hydra.compose("config")
        else:
            base = OmegaConf.load(os.path.join(rel_dir, "hydra/config.yaml"))
            override = OmegaConf.load(os.path.join(rel_dir, "hydra/overrides.yaml"))
            override = OmegaConf.from_dotlist(list(override))

            cfg = OmegaConf.merge(base, override)

        checkpointer = CheckPointer(
            model_name=str.split(cfg.model.arch._target_, ".")[-1], rel_dir=rel_dir
        )
        model = hydra.utils.instantiate(cfg.model.arch, rngs=nnx.Rngs(0))
        _, abstract_model = nnx.state(nnx.eval_shape(lambda: model), nnx.RngState, ...)

        restored_data = checkpointer.restore(
            train_step=train_step
            if isinstance(train_step, int)
            else checkpointer.get_latest(),
            abstract_state=abstract_model,
        )
        nnx.update(model, restored_data["state"])
        return model, jax.random.wrap_key_data(restored_data["train_rng_key"])
