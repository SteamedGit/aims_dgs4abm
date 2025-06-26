# https://github.com/instadeepai/Mava/blob/develop/mava/utils/logger.py
# https://github.com/EdanToledo/Stoix/blob/main/stoix/utils/logger.py
import logging
from colorama import Fore, Back, Style
from enum import Enum
import jax
import time
from pandas.io.json._normalize import _simple_json_normalize as flatten_dict
from omegaconf import DictConfig
from typing import Dict, List
import omegaconf
import wandb
import abc


class LogEvent(Enum):
    TRAIN = "train"
    TEST = "test"
    MISC = "misc"


class BaseLogger(abc.ABC):
    @abc.abstractmethod
    def __init__(self, cfg: DictConfig) -> None:
        pass

    @abc.abstractmethod
    def log_stat(self, key: str, value: float, step: int, event: LogEvent) -> None:
        """Log a single metric."""
        raise NotImplementedError

    def log_dict(self, data: Dict, step: int, event: LogEvent) -> None:
        """Log a dictionary of metrics."""
        # in case the dict is nested, flatten it.
        data = flatten_dict(data, sep="/")

        for key, value in data.items():
            self.log_stat(
                key,
                value,
                step,
                event,
            )

    def stop(self) -> None:
        """Stop the logger."""
        return None


class MultiLogger(BaseLogger):
    """Logger that can log to multiple loggers at oncce."""

    def __init__(self, cfg: DictConfig) -> None:
        self.loggers = []
        if cfg.logger.use_console:
            self.loggers.append(ConsoleLogger())
        if cfg.logger.use_wandb:
            self.loggers.append(WandbLogger(cfg))

    def log_stat(
        self, key: str, value: float, step: int, eval_step: int, event: LogEvent
    ) -> None:
        for logger in self.loggers:
            logger.log_stat(key, value, step, eval_step, event)

    def log(self, data: Dict, step: int, event: LogEvent) -> None:
        for logger in self.loggers:
            logger.log_dict(data, step, event)

    def stop(self) -> None:
        for logger in self.loggers:
            logger.stop()


class ConsoleLogger(BaseLogger):
    _EVENT_COLOURS = {
        LogEvent.TRAIN: Fore.BLUE,
        LogEvent.TEST: Fore.GREEN,
        LogEvent.MISC: Fore.WHITE,
    }

    def __init__(self) -> None:
        self.logger = logging.getLogger()

        self.logger.handlers = []

        ch = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(message)s",
            "%H:%M:%S",
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Set to info to suppress debug outputs.
        self.logger.setLevel("INFO")

    def log_stat(self, key: str, value: float, step: int, event: LogEvent) -> None:
        colour = self._EVENT_COLOURS[event]

        # Replace underscores with spaces and capitalise keys.
        key = key.replace("_", " ").capitalize()
        self.logger.info(
            f"{colour}{Style.BRIGHT}{event.value.upper()} - {key}: {value:.3f}{Style.RESET_ALL}"
        )

    def log_dict(self, data: Dict, step: int, event: LogEvent) -> None:
        # in case the dict is nested, flatten it.
        data = flatten_dict(data, sep=" ")

        colour = self._EVENT_COLOURS[event]
        # Replace underscores with spaces and capitalise keys.
        keys = [k.replace("_", " ").capitalize() for k in data.keys()]
        # Round values to 3 decimal places if they are floats.
        values = []
        for value in data.values():
            value = value.item() if isinstance(value, jax.Array) else value
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        log_str = " | ".join([f"{k}: {v}" for k, v in zip(keys, values, strict=True)])
        timestamp = time.strftime("%H:%M:%S")
        self.logger.info(
            f"{colour}{Style.BRIGHT}[{timestamp}] {event.value.upper()} - {log_str}{Style.RESET_ALL}"
        )

    def log(self, data, event):
        self.log_dict(data, event)


class WandbLogger(BaseLogger):
    def __init__(self, cfg: DictConfig) -> None:
        tags = list(cfg.logger.kwargs.tags)
        project = cfg.logger.kwargs.project

        wandb.init(
            project=project,
            tags=tags,
            config=omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
        )

    def log_stat(self, key: str, value: float, step: int, event: LogEvent) -> None:
        data_to_log = {f"{event.value}/{key}": value}
        wandb.log(data_to_log, step=step)

    def stop(self) -> None:
        wandb.finish()
