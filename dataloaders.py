# Inspired by https://github.com/BirkhoffG/jax-dataloader/blob/main/jax_dataloader/loaders/jax.py#L34
import jax
import jax.numpy as jnp
from typing import Dict, List, Iterator, Union
from jax.typing import ArrayLike


def EpochIterator(
    datasets: List[ArrayLike],
    names: List[str],
    batch_size: int,
    indices: ArrayLike,
) -> Iterator[Dict[str, jax.Array]]:
    for i in range(0, len(indices), batch_size):
        idx = indices[i : i + batch_size]
        yield {name: data_arr[idx] for data_arr, name in zip(datasets, names)}


class DataLoader:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        paths: Union[Dict[str, str], List[Dict[str, str]]],
        batch_size: int,
        shuffle_every_epoch: bool = False,
    ):
        self.key = key
        self.batch_size = batch_size
        self.shuffle_every_epoch = shuffle_every_epoch
        self.datasets = []
        self.names = []
        if not isinstance(paths, list):
            paths = [paths]
        else:
            print("Multiple paths were supplied. Attempting to fuse them.")
            assert paths[0].keys() == paths[1].keys(), (
                "Path dicts must have the same name mappings."
            )
        for name in paths[0].keys():
            tmp_datasets = []
            for path_dict in paths:
                with open(path_dict[name], "rb") as f:
                    tmp_datasets.append(jnp.load(f))
            self.datasets.append(jnp.concatenate(tmp_datasets, axis=0))
            self.names.append(name)
        assert all(
            self.datasets[0].shape[0] == arr.shape[0] for arr in self.datasets
        ), "All data arrays must have the same first dimension"
        self.indices = jax.random.permutation(
            self.next_key(), jnp.arange(self.datasets[0].shape[0])
        )

    def __iter__(self) -> EpochIterator:
        indices = (
            jax.random.permutation(self.next_key(), self.indices)
            if self.shuffle_every_epoch
            else self.indices
        )
        return EpochIterator(self.datasets, self.names, self.batch_size, indices)

    def next_key(self) -> jax.random.PRNGKey:
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def __len__(self) -> int:
        return jnp.ceil(len(self.indices) / self.batch_size).astype(jnp.int32)
