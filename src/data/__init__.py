from typing import Any, Dict, Union
from torch.utils.data import ConcatDataset
from omegaconf import DictConfig
from data.collators import (
    DataCollatorForSupervisedDataset,
)
from data.pretraining import CompletionDataset, PretrainingDataset
from data.qa import (
    QADataset,
    QAwithIdkDataset,
    QAwithRefusalStringDataset
)
from data.unlearn import ForgetRetainDataset, ForgetRetainDatasetTogether

DATASET_REGISTRY: Dict[str, Any] = {}
COLLATOR_REGISTRY: Dict[str, Any] = {}


def _register_data(data_class):
    DATASET_REGISTRY[data_class.__name__] = data_class


def _register_collator(collator_class):
    COLLATOR_REGISTRY[collator_class.__name__] = collator_class


def _load_single_dataset(dataset_name, dataset_cfg: DictConfig, **kwargs):
    dataset_handler_name = dataset_cfg.get("handler")
    assert dataset_handler_name is not None, ValueError(
        f"{dataset_name} handler not set"
    )
    dataset_handler = DATASET_REGISTRY.get(dataset_handler_name)
    if dataset_handler is None:
        raise NotImplementedError(
            f"{dataset_handler_name} not implemented or not registered"
        )
    dataset_args = dataset_cfg.args
    return dataset_handler(**dataset_args, **kwargs)


def get_datasets(dataset_cfgs: Union[Dict, DictConfig], **kwargs):
    """Build the dataset(s) for one split from config.

    Returns the single dataset when one is configured, or a `ConcatDataset` over all of them (used
    for combined multi-topic training). Concatenation keeps per-topic row alignment — each topic
    occupies a contiguous index range — so `forget[i]` and `retain[i]` stay paired.
    """
    datasets = []
    for dataset_name, dataset_cfg in dataset_cfgs.items():
        datasets.append(_load_single_dataset(dataset_name, dataset_cfg, **kwargs))
    if len(datasets) == 1:
        return datasets[0]
    # Multiple datasets (e.g. combined multi-topic training): concatenate them.
    # ConcatDataset preserves row-alignment per topic: forget[i] and retain[i]
    # remain paired since each topic's slice occupies a contiguous index range.
    return ConcatDataset(datasets)


def get_data(data_cfg: DictConfig, mode="train", **kwargs):
    """Build the datasets for every split in `data_cfg`.

    In `train` mode returns a dict of per-split datasets. In `unlearn` mode the non-eval splits are
    combined into one forget/retain dataset under the `train` key: by default
    `ForgetRetainDatasetTogether`, which keeps the hard alignment forget[i]↔retain[i] (pair order
    shuffled with seed 42) that the JensUn loss needs; with `random_pairing` set, `ForgetRetainDataset`
    instead samples a random retain example per forget example (the random-pairing ablation). `anchor`
    selects which split sets the dataset length in the random-pairing case (the default paired
    dataset requires equal lengths and ignores it).
    """
    data = {}
    data_cfg = dict(data_cfg)
    anchor = data_cfg.pop("anchor", "forget")
    random_pairing = data_cfg.pop("random_pairing", False)
    for split, dataset_cfgs in data_cfg.items():
        data[split] = get_datasets(dataset_cfgs, **kwargs)
    if mode == "train":
        return data
    elif mode == "unlearn":
        unlearn_splits = {k: v for k, v in data.items() if k not in ("eval", "test")}
        if random_pairing:
            unlearn_dataset = ForgetRetainDataset(**unlearn_splits, anchor=anchor)
        else:
            unlearn_dataset = ForgetRetainDatasetTogether(
                **unlearn_splits, anchor=anchor, seed=42,
            )
        data["train"] = unlearn_dataset
        for split in unlearn_splits:
            data.pop(split)

    return data


def _get_single_collator(collator_name: str, collator_cfg: DictConfig, **kwargs):
    collator_handler_name = collator_cfg.get("handler")
    assert collator_handler_name is not None, ValueError(
        f"{collator_name} handler not set"
    )
    collator_handler = COLLATOR_REGISTRY.get(collator_handler_name)
    if collator_handler is None:
        raise NotImplementedError(
            f"{collator_handler_name} not implemented or not registered"
        )
    collator_args = collator_cfg.args
    return collator_handler(**collator_args, **kwargs)


def get_collators(collator_cfgs, **kwargs):
    collators = {}
    for collator_name, collator_cfg in collator_cfgs.items():
        collators[collator_name] = _get_single_collator(
            collator_name, collator_cfg, **kwargs
        )
    if len(collators) == 1:
        # return a single collator
        return list(collators.values())[0]
    # return collators in a dict
    return collators


# Register datasets
_register_data(QADataset)
_register_data(QAwithRefusalStringDataset)
_register_data(QAwithIdkDataset)
_register_data(PretrainingDataset)
_register_data(CompletionDataset)

# Register composite datasets used in unlearning
_register_data(ForgetRetainDataset)

# Register collators
_register_collator(DataCollatorForSupervisedDataset)
