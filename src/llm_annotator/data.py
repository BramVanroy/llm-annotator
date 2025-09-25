import hashlib
import json
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm


def get_dataset(
    data_mixer: list[tuple[str, str] | str],
    label_columns: list[str],
    text_column: str = "text",
    splits: tuple[float, float, float] = (0.8, 0.1, 0.1),
):
    data_mixer = [item if isinstance(item, tuple) else (item, None) for item in data_mixer]
    dataset: Dataset = concatenate_datasets([load_dataset(name, config, split="train") for name, config in data_mixer])

    if "valid_fields" in dataset.column_names:
        dataset = dataset.filter(lambda item: item["valid_fields"], num_proc=36)

    keep_cols = label_columns + [text_column]
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in keep_cols])

    dataset = dataset.shuffle()

    dataset = dataset.train_test_split(test_size=splits[1] + splits[2])
    dataset_test_val = dataset["test"].train_test_split(test_size=splits[2] / (splits[1] + splits[2]))
    dataset = DatasetDict(
        {"train": dataset["train"], "validation": dataset_test_val["train"], "test": dataset_test_val["test"]}
    )

    return dataset


def auto_rebalance_dataset(
    ds: Dataset,
    label_column: str,
    strategy: Literal["upsample", "downsample", "mean", "balanced", "median", "downsample-to-multiple"]
    | int = "upsample",
    seed: int = 42,
    max_multiplicity: int | None = 15,
) -> Dataset:
    """
    Rebalance a HuggingFace Dataset by upsampling, downsampling, or balancing classes.

    Args:
        ds: The HuggingFace Dataset to rebalance.
        label_column: The column name containing class labels.
        strategy: "upsample", "downsample", or "balanced".
          - "upsample": Increase all classes to the size of the largest class.
          - "downsample": Reduce all classes to the size of the smallest class.
          - "balanced", "mean": Set all classes to the mean class size (rounded down).
          - "median": Set all classes to the median class size.
          - "downsample-to-multiple": Downsample majority classes to a multiple of the smallest class size.
          - int: If an integer is given, it will be used as the number of samples per class for upsampling/downsampling.
        seed: Random seed for reproducibility.
        max_multiplicity: Per-class maximum number of times to repeat a class when upsampling.

    Returns:
        A new, rebalanced Dataset.
    """
    # Convert label column to numpy array for easy indexing
    label_values = np.array(ds[label_column])
    # Find unique class labels and their counts
    unique_labels, counts = np.unique(label_values, return_counts=True)

    print("Original dataset class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} instances")

    # Get the indices of each class in the dataset
    indices_per_class = {label: np.where(label_values == label)[0] for label in unique_labels}
    rng = np.random.default_rng(seed)
    new_indices = []

    if strategy == "upsample":
        target_count = counts.max()
    elif strategy == "downsample":
        # Downsample: reduce all classes to the size of the smallest class
        target_count = counts.min()
    elif strategy in ("mean", "median", "balanced") or isinstance(strategy, int):
        # Balanced: set all classes to the mean class size (rounded down)
        if strategy in ("mean", "balanced"):
            target_count = int(np.mean(counts))
        elif strategy == "median":
            target_count = int(np.median(counts))
        elif isinstance(strategy, int):
            target_count = strategy
    elif strategy == "downsample-to-multiple":
        # Downsample majority classes to a multiple of the smallest class size
        if max_multiplicity is None:
            raise ValueError("max_multiplicity must be specified for 'downsample-to-multiple' strategy.")
        target_count = counts.min() * max_multiplicity
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    for label, indices in indices_per_class.items():
        if max_multiplicity:
            _target_count = min(target_count, len(indices) * max_multiplicity)
        else:
            _target_count = target_count
        if len(indices) < _target_count:
            # If class is smaller than target, upsample
            n_repeat = _target_count // len(indices)
            n_extra = _target_count % len(indices)
            sampled = np.concatenate(
                [np.tile(indices, n_repeat), rng.choice(indices, n_extra, replace=False) if n_extra > 0 else []]
            )
        elif len(indices) == _target_count:
            sampled = indices
        else:
            # If class is larger, downsample
            sampled = rng.choice(indices, _target_count, replace=False)
        new_indices.extend(sampled)

    # Shuffle the new indices to mix the classes
    rng = np.random.default_rng(seed)
    rng.shuffle(new_indices)
    ds = ds.select(new_indices)

    # Print new class distribution for verification
    unique_labels, counts = np.unique(ds[label_column], return_counts=True)
    print("Rebalanced dataset class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} instances")

    return ds


def get_hash(text):
    """Compute a SHA256 hash for a given text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def yield_jsonl_robust(
    pfiles: list[Path | str],
    keep_columns: list[str] | None = None,
    disable_tqdm: bool = False,
    deduplicate_on: str | None = None,
):
    """
    Given a set of .jsonl.gz files, this function reads them in a robust way, skipping incomplete lines,
    and yielding one sample at a time (parse-able JSON line).

    :param pfiles: A list of .jsonl.gz files
    :param keep_columns: A list of columns to keep in the output. If not given, all columns are kept.
    :param disable_tqdm: Whether to disable the progress bar
    :param deduplicate_on: Column name to use for deduplication (will be hashed)
    :return: A generator yielding the contents of the files
    """
    pfiles = [Path(pfile) for pfile in pfiles]
    seen = set()
    num_duplicates_removed = 0
    with tqdm(total=len(pfiles), desc="Reading", unit="file", disable=disable_tqdm) as pbar:
        for pfin in pfiles:
            if pfin.stat().st_size == 0:
                continue

            with pfin.open(encoding="utf-8") as fhin:
                num_failures = 0
                while True:
                    try:
                        line = fhin.readline()
                        if not line:
                            break
                        data = json.loads(line)
                        if deduplicate_on:
                            hashed_col = get_hash(data[deduplicate_on])
                            if hashed_col in seen:
                                num_duplicates_removed += 1
                                continue
                            seen.add(hashed_col)

                        if keep_columns:
                            data = {k: v for k, v in data.items() if k in keep_columns}

                        yield data
                    except json.JSONDecodeError:
                        # Handle partial or malformed JSON (incomplete writes)
                        num_failures += 1
                    except EOFError:
                        # Handle unexpected EOF in gzip
                        num_failures += 1
                        break
                if num_failures:
                    print(f"Skipped {num_failures:,} corrupt line(s) in {pfin}")
            pbar.update(1)

    if deduplicate_on:
        print(f"Removed {num_duplicates_removed:,} duplicates")


def count_lines(fname: str | PathLike) -> int:
    """Count the number of lines in a file."""
    with open(fname, "r", encoding="utf-8") as fhin:
        return sum([1 for _ in fhin])
