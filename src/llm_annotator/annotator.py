from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import shutil
import string
from collections import Counter
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from functools import wraps
from math import ceil
from os import cpu_count
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, Lock
from time import perf_counter
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    cast,
)
from typing import Counter as CounterType

from datasets import (
    Dataset,
    Features,
    concatenate_datasets,
    get_dataset_split_names,
    load_dataset,
)
from datasets.exceptions import DatasetGenerationError
from huggingface_hub import (
    create_branch,
    create_repo,
    delete_branch,
    list_repo_refs,
    upload_file,
    upload_folder,
)
from tqdm import tqdm

from llm_annotator.clients.base import (
    Client,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.clients.exceptions import (
    TooManyConsecutiveFailedBatchesError,
)
from llm_annotator.clients.vllm_offline_client import VLLMOfflineClient
from llm_annotator.logging_utils import get_logger
from llm_annotator.utils import (
    dataset_signature,
    drop_jsonl_rows,
    ensure_returns_bool,
    ensure_returns_dict,
    extract_prompt_prefix,
    get_hash,
    get_lib_versions,
    remove_empty_jsonl_files,
)


LOGGER = get_logger("annotator")

# Set a sensible default: cpu_count-1 cores
# but at least 1 at most 8 to avoid overloading the system
# (eg on SLURM, no need to use 128 cores for a small dataset)
DEFAULT_CPU_COUNT = min(8, max(1, (cpu_count() or 1) - 1))

BOOKKEEPING_SUFFIXES = (
    "error",
    "error_type",
    "finish_reason",
    "messages",
    "num_tokens",
    "reasoning",
    "response",
    "valid",
    "valid_fields",
)
"""Column names, after the task prefix, that the annotator writes itself."""

PREPARED_DS_BRANCH_SUFF = "prepared_dataset"
PREPARED_DS_LOCAL_SUBDIR = "prepared_dataset"
PROGRESS_BACKUP_BRANCH_SUFF = "progress_backup"
PROGRESS_DS_LOCAL_SUBDIR = "progress_backup"
SELECTION_RECORD_FILE = "selection.json"
# Where the bytes of the progress file that the writer has open are copied
# to while a background backup uploads them.
PROGRESS_UPLOAD_FILE = "progress_upload.jsonl"
METADATA_LOCAL_SUBDIR = "metadata"
METADATA_FILE_SUFF = "annotation_metadata.json"
VERSION_FILE = "_version.json"

# What `Dataset.save_to_disk` writes for the final dataset in the root of the
# output directory. Named here so that a run can remove its own result without
# clearing the directory, which holds the artifacts of the other tasks.
FINAL_DS_FILES = ("dataset_info.json", "state.json")
FINAL_DS_SHARD_GLOB = "data-*-of-*.arrow"

# How many batches a vLLM pool keeps queued per concurrent batch slot when
# `queue_size` is not given. One batch per slot would keep every server busy;
# the extra three absorb the time between a batch finishing and the next one
# being dispatched.
QUEUE_BATCHES_PER_SLOT = 4

# "auto" writes one progress file per this fraction of the run, so the number
# of files stays bounded no matter how large the dataset is.
AUTO_OUTPUT_FILE_FRACTION = 0.01
MIN_AUTO_SAMPLES_PER_OUTPUT_FILE = 1000


def _resolve_samples_per_output_file(
    max_samples_per_output_file: int | Literal["auto"] | None,
    *,
    num_rows: int,
) -> int:
    """Turn a configured progress-file size into a concrete sample count.

    ``"auto"`` is one percent of the run, with a floor of 1000 samples, which
    keeps the number of progress files at 100 or below.

    Args:
        max_samples_per_output_file: ``"auto"``, a positive sample count, or
            0 (``None`` is read as 0) for a single file of unlimited size.
        num_rows: Number of rows the run covers, used by ``"auto"``.

    Returns:
        Samples per progress file, or 0 for a single file of unlimited size.

    Raises:
        ValueError: If the value is neither ``"auto"`` nor a non-negative
            integer.

    Examples:
        >>> _resolve_samples_per_output_file("auto", num_rows=500_000)
        5000
        >>> _resolve_samples_per_output_file("auto", num_rows=2_000)
        1000
        >>> _resolve_samples_per_output_file(250, num_rows=500_000)
        250
    """
    if max_samples_per_output_file == "auto":
        return max(
            MIN_AUTO_SAMPLES_PER_OUTPUT_FILE,
            ceil(num_rows * AUTO_OUTPUT_FILE_FRACTION),
        )

    if max_samples_per_output_file is None:
        return 0

    if (
        isinstance(max_samples_per_output_file, bool)
        or not isinstance(max_samples_per_output_file, int)
        or max_samples_per_output_file < 0
    ):
        raise ValueError(
            "'max_samples_per_output_file' must be \"auto\", 0 (a single"
            " file of unlimited size) or a positive integer, but got"
            f" {max_samples_per_output_file!r}"
        )

    return max_samples_per_output_file


def _copy_file_prefix(*, src: Path, dest: Path, num_bytes: int) -> None:
    """Copy the first bytes of a file to another path.

    Args:
        src: File to read from.
        dest: File to write. An existing file is replaced.
        num_bytes: How many bytes to copy. A shorter source is copied whole.
    """
    remaining = num_bytes
    with src.open("rb") as fhin, dest.open("wb") as fhout:
        while remaining > 0:
            chunk = fhin.read(min(remaining, 1024 * 1024))
            if not chunk:
                break
            fhout.write(chunk)
            remaining -= len(chunk)


class _ProgressUploader:
    """Run the Hub progress backups of one run off the writer loop.

    The writer calls ``request`` when a cycle is due and goes on to the next
    batch while the upload runs on one background thread. A request that
    arrives while the previous upload still runs is dropped, since the next
    one carries the same rows and the ones after them. A failed upload is
    logged at warning level and the run continues: the progress files on
    disk are the copy that a resume reads, and the backup is caught up by
    the next cycle or by the upload at the end of the run.
    """

    def __init__(
        self,
        *,
        annotator: "Annotator",
        process_pdout: Path,
        hub_id: str,
        task_prefix: str,
    ) -> None:
        """Set up the single background thread that the uploads run on.

        Args:
            annotator: The annotator whose ``push_progress_to_hub`` is run.
            process_pdout: Directory that holds the ``*.jsonl`` files.
            hub_id: Hugging Face dataset ID to upload into.
            task_prefix: String prefix to use for branch naming.
        """
        self._annotator = annotator
        self._process_pdout = process_pdout
        self._hub_id = hub_id
        self._task_prefix = task_prefix
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="progress-upload"
        )
        self._future: Future[None] | None = None

    def request(self, *, active_path: Path, active_bytes: int) -> None:
        """Start an upload cycle unless the previous one still runs.

        Args:
            active_path: The progress file that the writer has open.
            active_bytes: How many bytes of that file are written.
        """
        if self._future is not None and not self._future.done():
            LOGGER.debug(
                "The previous Hub backup is still running, so this cycle is"
                " skipped. The next one uploads the same rows and more."
            )
            return

        self._future = self._executor.submit(
            self._upload, active_path, active_bytes
        )

    def _upload(self, active_path: Path, active_bytes: int) -> None:
        """Upload one cycle and turn a failure into a warning.

        Args:
            active_path: The progress file that the writer has open.
            active_bytes: How many bytes of that file are written.
        """
        try:
            self._annotator.push_progress_to_hub(
                self._process_pdout,
                hub_id=self._hub_id,
                task_prefix=self._task_prefix,
                active_path=active_path,
                active_bytes=active_bytes,
            )
        except Exception as exc:
            LOGGER.warning(
                f"Backing up the progress files to '{self._hub_id}' failed:"
                f" {exc}. The run continues; the files in"
                f" '{self._process_pdout}' are unaffected and the next"
                " backup uploads them again."
            )

    def close(self) -> None:
        """Wait for an upload that is in flight and stop the thread."""
        self._executor.shutdown(wait=True)


def _bookkeeping_columns(
    *, task_prefix: str, idx_column: str | None = None
) -> set[str]:
    """Name the columns that the annotator writes for every sample.

    Args:
        task_prefix: Prefix of the internal column names.
        idx_column: Name of the sample id column. ``None`` leaves it out, for
            a caller that does not know it.

    Returns:
        The column names that a schema property or a parsed key must not use.

    Examples:
        >>> columns = _bookkeeping_columns(task_prefix="qa_", idx_column="idx")
        >>> sorted(columns)[:3]
        ['idx', 'qa_error', 'qa_error_type']
    """
    columns = {f"{task_prefix}{name}" for name in BOOKKEEPING_SUFFIXES}
    if idx_column is not None:
        columns.add(idx_column)
    return columns


def is_retried_error(
    row: dict[str, Any],
    *,
    retry_errors: bool | Sequence[str],
    task_prefix: str = "",
) -> bool:
    """Check whether ``retry_errors`` selects a finished row for another attempt.

    Args:
        row: One row of a progress file.
        retry_errors: ``True`` selects every errored row. A sequence selects
            the errored rows whose ``error_type`` is in it.
        task_prefix: Prefix of the internal column names.

    Returns:
        Whether the row has an error that ``retry_errors`` selects.

    Examples:
        >>> row = {"error": "refused", "error_type": "ConnectError"}
        >>> is_retried_error(row, retry_errors=True)
        True
        >>> is_retried_error(row, retry_errors=["APITimeoutError"])
        False
        >>> is_retried_error({"error": None}, retry_errors=True)
        False
    """
    if not retry_errors or row.get(f"{task_prefix}error") is None:
        return False
    if retry_errors is True:
        return True
    if isinstance(retry_errors, str):
        retry_errors = [retry_errors]
    return row.get(f"{task_prefix}error_type") in retry_errors


_COMPONENT_CHANGES: dict[str, str] = {
    "prompt_template": "the prompt template changed",
    "system_message": "the system message changed",
    "sort_by_length": "'sort_by_length' changed",
    "idx_column": "'idx_column' changed",
    "reuse_idx_column": "'reuse_idx_column' changed",
    "shuffle_seed": "'shuffle_seed' changed",
    "preprocess_fn": "'preprocess_fn' changed",
    "dataset": "the source dataset changed",
    "output_schema": "the output schema changed",
    "upstream": "an earlier step was annotated again from scratch",
}
"""What each recorded component is called when it differs from the record."""

_REUSE_REMEDY = (
    "Restore the old value(s), use a new 'output_dir', or overwrite the run"
    " ('overwrite=True') to discard the finished rows and annotate every"
    " sample again."
)
"""What to do about finished rows that the request no longer matches."""


def _callable_component(func: Callable | None, *, setting: str) -> str:
    """Describe a user callable for a preparation record.

    Args:
        func: The callable, or ``None`` when the setting is unset.
        setting: Name of the setting the callable was passed as, used in the
            warning about a callable whose source cannot be read.

    Returns:
        The qualified name of the callable, followed by a hash of its source
        when the source is available.
    """
    if func is None:
        return "None"
    module = getattr(func, "__module__", "?")
    qualified = f"{module}.{getattr(func, '__qualname__', repr(func))}"
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        LOGGER.warning(
            f"The source of '{setting}' ({qualified}) cannot be read, so an"
            " edit to it is not detected when this run is resumed."
        )
        return qualified
    return f"{qualified}:{get_hash(source)}"


def _preparation_components(
    *,
    prompt_template: str,
    system_message: str | None,
    sort_by_length: bool | Literal["shortest_first", "longest_first"],
    idx_column: str,
    reuse_idx_column: bool,
    shuffle_seed: int | None,
    preprocess_fn: Callable | None,
    source_signature: str | None,
) -> dict[str, str]:
    """Describe every setting that decides what a prepared dataset holds.

    Long values are hashed, short ones are kept as they are, so a difference
    can be named without the record holding a copy of the prompt.
    ``max_num_samples`` is left out: a run may grow under a higher cap.

    Args:
        prompt_template: The prompt template.
        system_message: The system message, or ``None``.
        sort_by_length: The requested prompt ordering.
        idx_column: Name of the sample id column.
        reuse_idx_column: Whether the ids come from an existing column.
        shuffle_seed: The shuffle seed, or ``None``.
        preprocess_fn: The preprocessing callback, or ``None``.
        source_signature: Signature of the source dataset, or ``None`` for a
            source that is not loaded yet, which leaves it out.

    Returns:
        One short string per setting, keyed by the setting's name.

    Examples:
        >>> components = _preparation_components(
        ...     prompt_template="Q: {text}",
        ...     system_message=None,
        ...     sort_by_length=False,
        ...     idx_column="idx",
        ...     reuse_idx_column=False,
        ...     shuffle_seed=42,
        ...     preprocess_fn=None,
        ...     source_signature=None,
        ... )
        >>> components["shuffle_seed"]
        '42'
        >>> "dataset" in components
        False
    """
    components = {
        "prompt_template": get_hash(prompt_template),
        "system_message": (
            "None" if system_message is None else get_hash(system_message)
        ),
        "sort_by_length": repr(sort_by_length),
        "idx_column": idx_column,
        "reuse_idx_column": repr(reuse_idx_column),
        "shuffle_seed": repr(shuffle_seed),
        "preprocess_fn": _callable_component(
            preprocess_fn, setting="preprocess_fn"
        ),
    }
    if source_signature is not None:
        components["dataset"] = source_signature
    return components


def _schema_component(output_schema: dict[str, Any] | None) -> str:
    """Describe an output schema for a preparation record.

    Args:
        output_schema: The decoded JSON schema, or ``None``.

    Returns:
        A hash of the schema, or ``"None"``.

    Examples:
        >>> _schema_component(None)
        'None'
        >>> _schema_component({"type": "object"}) == _schema_component(
        ...     {"type": "object"}
        ... )
        True
    """
    if output_schema is None:
        return "None"
    return get_hash(json.dumps(output_schema, sort_keys=True, default=repr))


def _reuse_error(
    progress_dir: Path, causes: list[str], remedy: str = _REUSE_REMEDY
) -> ValueError:
    """Build the error for finished rows that the request no longer matches.

    Args:
        progress_dir: Directory that holds the progress files.
        causes: One phrase per setting that differs from the record.
        remedy: What the caller can do about it. A pipeline names the command
            that redoes the affected steps instead.

    Returns:
        The error to raise.
    """
    return ValueError(
        f"The finished rows in '{progress_dir}' cannot be reused:"
        f" {', '.join(causes)}. {remedy} A run can only grow through a higher"
        " 'max_num_samples' with the same settings, or through rows appended"
        " to a source that is not shuffled."
    )


@dataclass(frozen=True, slots=True)
class SelectionRecord:
    """What a prepared dataset was built from.

    [`prepare_data`][llm_annotator.annotator.Annotator.prepare_data] writes it
    to ``<output_dir>/<task_prefix>selection.json``. The file stays in place
    after a run finishes, so a later run can tell whether the finished rows
    were annotated with the settings that it asks for.

    Attributes:
        max_num_samples: The requested sample cap, ``None`` for no cap. It is
            outside ``components`` because a run may grow under a higher cap.
        source_rows: Number of rows in the source dataset.
        selected_rows: Number of rows that the cap left in the selection.
        components: One short string per setting that decides what the
            prepared dataset holds, keyed by the setting's name. A setting
            that the layer which owns it has not recorded yet is absent.

    Examples:
        >>> record = SelectionRecord(
        ...     max_num_samples=10,
        ...     source_rows=40,
        ...     selected_rows=10,
        ...     components={"prompt_template": "abc", "shuffle_seed": "42"},
        ... )
        >>> record.changed_components({"prompt_template": "abc"})
        []
        >>> record.changed_components({"prompt_template": "def"})
        ['prompt_template']
        >>> record.changed_components({"system_message": "def"})
        []
    """

    max_num_samples: int | None
    source_rows: int
    selected_rows: int
    components: dict[str, str] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """A single hash over every recorded component."""
        return get_hash(json.dumps(self.components, sort_keys=True))

    def changed_components(self, components: dict[str, str]) -> list[str]:
        """Name the requested settings that differ from the recorded ones.

        Args:
            components: The settings of the request, as
                ``_preparation_components`` builds them.

        Returns:
            The names of the settings that the record holds under another
            value, sorted. A setting the record does not hold is left out:
            the layer that owns it writes it only once it ran, so the output
            schema is absent until ``run_annotation`` recorded it, and a
            step's input version until the pipeline recorded it.
        """
        return sorted(
            name
            for name, value in components.items()
            if name in self.components and self.components[name] != value
        )

    @classmethod
    def path(cls, output_dir: str | Path, task_prefix: str = "") -> Path:
        """Build the path of the record file.

        Args:
            output_dir: The annotator's output directory.
            task_prefix: The task prefix of the run.

        Returns:
            Path of ``<output_dir>/<task_prefix>selection.json``.
        """
        return Path(output_dir) / f"{task_prefix}{SELECTION_RECORD_FILE}"

    @classmethod
    def read(
        cls, output_dir: str | Path, task_prefix: str = ""
    ) -> "SelectionRecord | None":
        """Read the record of an output directory.

        Args:
            output_dir: The annotator's output directory.
            task_prefix: The task prefix of the run.

        Returns:
            The record, or ``None`` when the directory has none, which is a
            run that never prepared data.

        Raises:
            ValueError: If the file holds no ``components``, which is a
                record from a release that did not describe the settings of
                its run.
        """
        record_path = cls.path(output_dir, task_prefix)
        if not record_path.is_file():
            return None
        stored = json.loads(record_path.read_text(encoding="utf-8"))
        components = stored.get("components")
        if not isinstance(components, dict):
            raise ValueError(
                f"The record in '{record_path}' does not describe the"
                " settings that its run was annotated with, so a resume"
                " cannot tell whether the prompt still matches. Finish the"
                " run with the release that wrote it, annotate it into a new"
                " 'output_dir', or overwrite it. See 'Migrating an output"
                " directory' in docs/growing-a-run.md."
            )
        return cls(
            max_num_samples=stored.get("max_num_samples"),
            source_rows=stored.get("source_rows", 0),
            selected_rows=stored.get("selected_rows", 0),
            components={str(k): str(v) for k, v in components.items()},
        )

    def write(self, output_dir: str | Path, task_prefix: str = "") -> None:
        """Write the record to an output directory.

        Args:
            output_dir: The annotator's output directory.
            task_prefix: The task prefix of the run.
        """
        payload = dataclasses.asdict(self) | {"fingerprint": self.fingerprint}
        self.path(output_dir, task_prefix).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


def destroy_on_error(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate an ``Annotator`` method to clean up on any exception.

    Calls [`destroy`][llm_annotator.annotator.Annotator.destroy] before
    re-raising. Catches ``BaseException`` (including ``KeyboardInterrupt`` and
    ``SystemExit``) so resources are freed even on forced termination. The
    original exception is always re-raised after the cleanup attempt.

    Should be used on methods that use the underlying client.

    Args:
        func: The instance method to wrap.

    Returns:
        The wrapped callable with automatic cleanup on failure.
    """

    @wraps(func)
    def wrapper(self: "Annotator", *args: Any, **kwargs: Any) -> Any:
        try:
            return func(self, *args, **kwargs)
        except BaseException as e:
            try:
                self.destroy()
            except Exception as clean_err:
                if hasattr(e, "__notes__"):
                    e.__notes__.append(f"Cleanup failed: {clean_err!r}")
            raise

    return wrapper


@dataclass(slots=True)
class Annotator:
    """Sensible base class for LLM-based dataset annotation.

    This class provides a framework for annotating datasets using LLMs
    via a pluggable [`Client`][llm_annotator.clients.base.Client]. It handles
    dataset loading, processing, and output generation with support for batching
    and uploading to the Hugging Face Hub.

    The ``Annotator`` class has four public entry points:

    - [`prepare_data`][llm_annotator.annotator.Annotator.prepare_data].
        Apply prompt templates, sorting, and caching without running
        inference. Backs-up prepared artifacts to Hugging Face Hub if
        ``hub_id`` is provided.
    - [`run_annotation`][llm_annotator.annotator.Annotator.run_annotation].
        Run inference only, using prepared data returned by
        ``prepare_data`` or loaded from a local path or Hub repo.
    - [`annotate_dataset`][llm_annotator.annotator.Annotator.annotate_dataset].
        Convenience wrapper that calls ``prepare_data`` and then
        ``run_annotation`` in one call.
    - [`generate_dataset`][llm_annotator.annotator.Annotator.generate_dataset].
        Generate a new dataset from scratch by calling ``annotate_dataset``
        over a synthetic prompt dataset.

    For large-scale annotation jobs, consider using
    [`VLLMQueueAnnotator`][llm_annotator.annotator.VLLMQueueAnnotator], which
    distributes inference across a pool of vLLM servers.

    Args:
        client: An initialised [`Client`][llm_annotator.clients.base.Client]
            instance that performs the actual generation.
        batch_size: Number of samples per inference batch. It depends on the
            client and its settings which batching is actually used. Batch
            size here is mostly intended for progress reporting. The client
            may split the given batch into smaller sub-batches if needed.
        num_proc: Number of processes for dataset preprocessing.
        verbose: Whether to print progress information.

    Examples:
        Basic usage with an OpenAI client:

        >>> from llm_annotator import Annotator, OpenAIClient
        >>> client = OpenAIClient(model="gpt-4o-mini")  # doctest: +SKIP
        >>> with Annotator(client=client) as anno:  # doctest: +SKIP
        ...     ds = anno.annotate_dataset(
        ...         output_dir="outputs/data",
        ...         prompt_template="Process: {text}",
        ...         dataset_name="my-dataset",
        ...     )

        Usage with vLLM offline client:

        >>> from llm_annotator import Annotator, VLLMOfflineClient
        >>> client = VLLMOfflineClient(  # doctest: +SKIP
        ...     model="meta-llama/Llama-3.2-3B-Instruct",
        ...     max_model_len=4096,
        ... )
        >>> try:  # doctest: +SKIP
        ...     ds = Annotator(client=client).annotate_dataset(
        ...         output_dir="outputs/data",
        ...         prompt_template="Process: {text}",
        ...         dataset_name="my-dataset",
        ...     )
        ... finally:
        ...     client.destroy()
    """

    client: Client
    batch_size: int = 256
    num_proc: int | None = DEFAULT_CPU_COUNT
    verbose: bool = False
    _logger: Any = field(init=False, repr=False)
    _ignored_keys: set[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the logger for annotator runtime messages."""
        self._logger = get_logger("annotator")
        self._ignored_keys = set()

    def __enter__(self) -> "Annotator":
        """Enter the context manager, returning the annotator instance."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Exit the context manager and free all client resources."""
        self.destroy()

    def destroy(self) -> None:
        """Clean up all resources used by the underlying client."""
        self.client.destroy()

    def _get_skip_idxs(
        self,
        *,
        process_pdout: Path,
        idx_column: str,
        dataset_split: str | None = None,
        dataset_config: str | None = None,
    ) -> set[int]:
        """Get indices of samples that have already been processed.

        Scans existing output files to determine which samples can be skipped
        in resumed processing.

        A hard crash can leave a partially written final line behind. The code here
        is robust so that it can delete the last non-parseable JSON line and still recover.

        Args:
            process_pdout: Output directory path to scan for existing files.
            idx_column: Column name used as unique identifier.
            dataset_split: Only count rows from this split, when recorded.
            dataset_config: Only count rows from this config, when recorded.

        Returns:
            Set of indices that have already been processed.

        Raises:
            ValueError: If an existing output file has no ``idx_column``.
        """
        ids_done: set[int] = set()
        if not (process_pdout.exists() and process_pdout.is_dir()):
            return ids_done

        for pfin in sorted(process_pdout.glob("*.jsonl")):
            # skip and remove empty files
            if pfin.stat().st_size == 0:
                pfin.unlink()
                continue

            valid_bytes = 0
            truncated = False
            with pfin.open("rb") as fhin:
                for raw_line in fhin:
                    if not raw_line.strip():
                        valid_bytes += len(raw_line)
                        continue

                    try:
                        row = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Only the very last line can legitimately be broken.
                        # If a "next" line exists, the file is corrupt (mid-file
                        # broken JSON is not expected) so truncation is not possible
                        if fhin.read(1):
                            raise
                        truncated = True
                        break

                    valid_bytes += len(raw_line)

                    if idx_column not in row:
                        raise ValueError(
                            f"Expected index column '{idx_column}' not found in existing output file '{pfin}'."
                            " Cannot determine which samples to skip on resume. Please check your configuration"
                            " and ensure the index column is included in the output."
                        )

                    # Filter on dataset split/config
                    if (
                        dataset_split
                        and "dataset_split" in row
                        and row["dataset_split"] != dataset_split
                    ):
                        continue

                    if (
                        dataset_config
                        and "dataset_config" in row
                        and row["dataset_config"] != dataset_config
                    ):
                        continue

                    ids_done.add(row[idx_column])

            if truncated:
                self._logger.warning(
                    f"Discarding an incomplete trailing line in '{pfin}'"
                    " (likely an interrupted write). The affected sample will"
                    " be annotated again."
                )
                with pfin.open("r+b") as fhout:
                    fhout.truncate(valid_bytes)

        return ids_done

    def _load_source(
        self,
        *,
        dataset_name: str | None = None,
        dataset: Dataset | None = None,
        dataset_config: str | None = None,
        data_dir: str | None = None,
        data_files: str | list[str] | dict[str, str | list[str]] | None = None,
        dataset_split: str | None = None,
    ) -> Dataset:
        """Load the source dataset, or return the one that was passed in.

        Args:
            dataset_name: Name or path of the dataset to load.
            dataset: Pre-loaded dataset to use instead of loading from name/path.
            dataset_config: Dataset configuration name (optional).
            data_dir: Data directory for local datasets (optional).
            data_files: Specific file(s) for local datasets (optional).
            dataset_split: Specific split to load (optional).

        Returns:
            The source dataset, without an index column or a selection.

        Raises:
            ValueError: If both or neither of ``dataset`` and ``dataset_name``
                are given, or if the split is ambiguous or unknown.
        """
        if dataset is not None and dataset_name is not None:
            raise ValueError(
                "Provide only one of 'dataset' or 'dataset_name', not both."
            )

        if dataset is not None:
            return dataset

        if dataset_name is None:
            raise ValueError(
                "Either 'dataset' or 'dataset_name' must be provided."
            )

        split_names = get_dataset_split_names(
            dataset_name,
            config_name=dataset_config,
            data_dir=data_dir,
            data_files=data_files,
        )
        if not dataset_split:
            if len(split_names) == 1:
                dataset_split = split_names[0]
            else:
                raise ValueError(
                    f"Dataset '{dataset_name}' has multiple splits {split_names}. "
                    "Please specify a split using the 'dataset_split' argument."
                )
        elif dataset_split not in split_names:
            raise ValueError(
                f"Dataset '{dataset_name}' does not have a split named '{dataset_split}'"
            )

        return load_dataset(
            dataset_name,
            name=dataset_config,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_split,
        )

    def _load_dataset(
        self,
        *,
        prompt_template: str,
        idx_column: str,
        dataset_name: str | None = None,
        dataset: Dataset | None = None,
        dataset_config: str | None = None,
        data_dir: str | None = None,
        data_files: str | list[str] | dict[str, str | list[str]] | None = None,
        dataset_split: str | None = None,
        max_num_samples: int | None = None,
        shuffle_seed: int | None = None,
        prompt_fields: Iterable[str] = (),
        task_prefix: str = "",
        sort_by_length: bool
        | Literal["shortest_first", "longest_first"] = False,
        system_message: str | None = None,
        preprocess_fn: Callable | None = None,
        reuse_idx_column: bool = False,
    ) -> Dataset:
        """Load and preprocess the dataset for annotation.

        Handles dataset loading, applies prompt templates, and manages
        caching for efficient resumption of interrupted jobs.

        Args:
            prompt_template: Prompt template used to build chat messages.
            idx_column: Column name used as unique identifier. Must not exist in the input dataset.
            dataset_name: Name or path of the dataset to load.
            dataset: Pre-loaded dataset to use instead of loading from name/path.
            dataset_config: Dataset configuration name (optional).
            data_dir: Data directory for local datasets (optional).
            data_files: Specific file(s) for local datasets (optional).
            dataset_split: Specific split to load (optional).
            max_num_samples: Maximum number of samples to process.
            shuffle_seed: Seed for dataset shuffling (optional).
            prompt_fields: Fields required by the prompt template.
            task_prefix: String prefix to use for internal column names and file operations.
            sort_by_length: Whether to sort the dataset by prompt length for more efficient batching.
                If set to "shortest_first", sort by shortest first, "longest_first" for longest first.
                If set to True, defaults to longest first as that makes most sense to avoid OOM errors
                down the line.
            system_message: Optional system message to add as "system" role in chat prompts.
            preprocess_fn: Optional function to preprocess the dataset after loading and before applying the prompt template.
            reuse_idx_column: Whether an existing ``idx_column`` is kept as the sample id.

        Returns:
            The loaded and preprocessed dataset ready for annotation.

        Raises:
            ValueError: If configuration is invalid or required fields are missing.
        """
        num_proc = self.num_proc
        pipeline_loaded = getattr(self.client, "_pipeline_loaded", False)

        if (
            num_proc is not None
            and isinstance(self.client, VLLMOfflineClient)
            and pipeline_loaded
        ):
            self._logger.warning(
                "num_proc>1 cannot be used with VLLMOfflineClient because the"
                " loaded model cannot be pickled for multiprocessing. This"
                " dataset is mapped in a single process."
            )
            num_proc = None

        if max_num_samples is not None and max_num_samples <= 0:
            raise ValueError(
                "'max_num_samples' must be a positive integer or None"
            )

        dataset = self._load_source(
            dataset_name=dataset_name,
            dataset=dataset,
            dataset_config=dataset_config,
            data_dir=data_dir,
            data_files=data_files,
            dataset_split=dataset_split,
        )

        if idx_column not in dataset.column_names:
            # Index column for tracking samples and resuming interrupted runs
            dataset = dataset.add_column(idx_column, list(range(len(dataset))))
        elif not reuse_idx_column:
            raise ValueError(
                f"Dataset already contains a column named '{idx_column}'."
                " Please specify a different 'idx_column' name that does not exist in the dataset,"
                " or set 'reuse_idx_column=True' when the column holds the sample ids of an earlier run."
            )
        elif len(set(dataset[idx_column])) != len(dataset):
            raise ValueError(
                f"Column '{idx_column}' cannot be reused as the sample id"
                " because it holds duplicate values."
            )

        if shuffle_seed is not None:
            dataset = dataset.shuffle(seed=shuffle_seed)

        if max_num_samples:
            dataset = dataset.select(range(min(max_num_samples, len(dataset))))

        missing = [
            fld for fld in prompt_fields if fld not in dataset.column_names
        ]
        if missing:
            raise ValueError(
                f"Template contains field '{missing[0]}' not present in dataset."
                f" Available columns: {dataset.column_names}"
            )

        if preprocess_fn is not None:
            dataset = preprocess_fn(dataset=dataset)

        dataset = dataset.map(
            _create_messages,
            num_proc=num_proc,
            fn_kwargs={
                "prompt_fields": prompt_fields,
                "prompt_template": prompt_template,
                "task_prefix": task_prefix,
                "system_message": system_message,
            },
            desc="Applying prompt template",
        )

        if sort_by_length:
            if self.verbose:
                self._logger.info(
                    "Sorting dataset roughly by prompt length for more efficient batching (longest first)..."
                )
            dataset = dataset.map(
                lambda msgs: {
                    f"{task_prefix}messages_chars": len(
                        json.dumps(msgs, default=str)
                    )
                },
                num_proc=num_proc,
                input_columns=[f"{task_prefix}messages"],
            )
            # Sort by longest first to trigger OOM as soon as possible
            if sort_by_length == "shortest_first":
                do_reverse = False
            else:
                do_reverse = True

            dataset = dataset.sort(
                f"{task_prefix}messages_chars", reverse=do_reverse
            ).remove_columns([f"{task_prefix}messages_chars"])

        return dataset

    def _process_output(
        self,
        *,
        response: Response,
        output_schema: dict | None = None,
        task_prefix: str = "",
    ) -> dict[str, Any]:
        """Process a single model response into the desired annotation format.

        Override this method to implement custom output parsing and validation.

        Args:
            response: The structured response from the client.
            output_schema: Optional JSON schema for structured output.
            task_prefix: String prefix to use for internal column names.

        Returns:
            - A key '{prefix}response' containing the raw model output text.
            - A key '{prefix}finish_reason' indicating why generation stopped.
            - A key '{prefix}num_tokens' indicating the number of tokens in the output.
            - A key '{prefix}error' and '{prefix}error_type' describing a failed request.
            - A key '{prefix}reasoning' containing the model's reasoning trace, or None when the provider did not return one separately.

            And if an output_schema is provided, also:
                - One key per top-level property of the schema. A property that
                  the response does not hold is None, so that every row has the
                  same keys. A parsed key that the schema does not declare is
                  ignored, with one warning per run.
                - A key '{prefix}valid_fields', False when the response errored,
                  did not parse as JSON, was not a JSON object, or left out a
                  property that the schema requires.
        """
        data: dict[str, Any] = {
            f"{task_prefix}response": response.text,
            f"{task_prefix}finish_reason": response.stop_reason,
            f"{task_prefix}num_tokens": response.num_output_tokens,
            f"{task_prefix}error": response.error,
            f"{task_prefix}error_type": response.error_type,
            # Always written, like error/error_type: the per-sample dicts are
            # stacked into one Dataset, so a key that is present on only some
            # rows would break the schema.
            f"{task_prefix}reasoning": response.reasoning,
        }

        if not output_schema:
            return data

        properties: dict[str, Any] = output_schema.get("properties", {})
        # Every row carries every property, so that the rows of one run stack
        # into a dataset with one set of columns.
        result: dict[str, Any] = dict.fromkeys(properties)
        invalid = {**data, f"{task_prefix}valid_fields": False, **result}

        if response.error is not None:
            return invalid

        try:
            parsed_response = json.loads(response.text)
        except json.JSONDecodeError:
            return invalid

        if not isinstance(parsed_response, dict):
            return invalid

        for key, value in parsed_response.items():
            if key in properties:
                result[key] = value
            else:
                self._warn_ignored_key(key=key, task_prefix=task_prefix)

        valid_fields = all(
            key in parsed_response for key in output_schema.get("required", [])
        )
        return {
            **data,
            f"{task_prefix}valid_fields": valid_fields,
            **result,
        }

    def _warn_ignored_key(self, *, key: str, task_prefix: str) -> None:
        """Report a parsed key that does not become a column.

        The same key is only reported once per run, since a model that returns
        it for one sample usually returns it for every sample.

        Args:
            key: The key of the parsed response.
            task_prefix: String prefix used for internal column names.
        """
        if key in self._ignored_keys:
            return
        self._ignored_keys.add(key)

        if key in _bookkeeping_columns(task_prefix=task_prefix):
            self._logger.warning(
                f"The model returned the key '{key}', which is the name of a"
                " column that the annotator writes for every sample. The key"
                " is ignored and the column keeps the annotator's value. This"
                " is reported once per run."
            )
        else:
            self._logger.warning(
                f"The model returned the key '{key}', which the output schema"
                " does not declare as a property. The key is ignored, so that"
                " every row has the same columns. This is reported once per"
                " run."
            )

    def _process_batch(
        self,
        *,
        batch: dict[str, list[Any]],
        options: ProviderRuntimeOptions | None,
        gen_kwargs: dict[str, Any] | None = None,
        task_prefix: str = "",
        validate_fn: Callable | None = None,
        postprocess_fn: Callable | None = None,
        client: Client | None = None,
    ) -> list[dict[str, Any]]:
        """Process a batch of samples through the client.

        Takes a batch of messages samples, runs inference, and processes
        the outputs using the `_process_output` method.

        Args:
            batch: Dictionary containing batch data with messages samples.
            options: Runtime options passed to the client.
            gen_kwargs: Extra request parameters merged over ``options``,
                for anything the options dataclass does not name.
            task_prefix: String prefix to use for internal column names.
            validate_fn: Optional custom validation function that takes a processed
                output dictionary and must return a boolean indicating validity. If a JSON schema
                was passed, and the fields were invalid, this function will not be called
                and `valid` will be set to False directly.
            postprocess_fn: Optional function to postprocess each sample after annotation.
            client: Optional client to run inference with. Defaults to
                ``self.client``. Subclasses that hold a pool of clients use this
                to dispatch a batch to a specific worker.

        Returns:
            List of processed output dictionaries for each sample in the batch,
            empty for a batch without samples.

        Raises:
            ValueError: If the client did not return exactly one response per
                input, which would silently drop or misalign samples.
        """
        output_schema = options.json_schema if options is not None else None
        messages = batch[f"{task_prefix}messages"]
        if not messages:
            return []

        client = client if client is not None else self.client
        responses = client.batch_generate(
            messages=messages,
            options=options,
            gen_kwargs=gen_kwargs,
        )

        if len(responses) != len(messages):
            raise ValueError(
                f"Client '{type(client).__name__}' returned {len(responses):,}"
                f" responses for {len(messages):,} inputs. Clients must return"
                " exactly one response per input, in order."
            )

        results = []
        for response in responses:
            res = self._process_output(
                response=response,
                output_schema=output_schema,
                task_prefix=task_prefix,
            )
            if postprocess_fn and response.error is None:
                res = ensure_returns_dict(postprocess_fn, res)

            if validate_fn:
                if response.error is not None:
                    is_valid = False
                elif (
                    f"{task_prefix}valid_fields" in res
                    and res[f"{task_prefix}valid_fields"] is False
                ):
                    # do not bother running the custom validation fn if the
                    # json schema validation already failed
                    is_valid = False
                else:
                    is_valid = ensure_returns_bool(validate_fn, res)
                res[f"{task_prefix}valid"] = is_valid
            results.append(res)

        if f"{task_prefix}valid_fields" in results[0]:
            n_invalid = sum(
                1 for res in results if not res[f"{task_prefix}valid_fields"]
            )
            if n_invalid == len(results) and self.verbose:
                self._logger.warning(
                    "Warning: All samples in the batch failed to produce valid JSON fields."
                    " This might be exceptional (esp. for smaller batches)"
                    " but if it happens often it suggests a deeper issue, such"
                    " as too few 'max_completion_tokens' in options."
                )

        if f"{task_prefix}valid" in results[0]:
            n_invalid = sum(
                1 for res in results if not res[f"{task_prefix}valid"]
            )
            if n_invalid == len(results) and self.verbose:
                self._logger.warning(
                    "Warning: All samples in the batch failed to produce valid outputs after"
                    " running the custom validation function."
                )

        return results

    def _invalid_indices(
        self, results: list[dict[str, Any]], task_prefix: str
    ) -> list[int]:
        """Return the positions of results that failed schema or custom validation.

        Args:
            results: Processed outputs for a single batch.
            task_prefix: String prefix used for internal column names.

        Returns:
            The indices in ``results`` that are marked invalid.
        """
        return [
            idx
            for idx, res in enumerate(results)
            if (
                (
                    f"{task_prefix}valid" in res
                    and not res[f"{task_prefix}valid"]
                )
                or (
                    f"{task_prefix}valid_fields" in res
                    and not res[f"{task_prefix}valid_fields"]
                )
            )
        ]

    @staticmethod
    def _all_errored(results: list[dict[str, Any]], task_prefix: str) -> bool:
        """Return whether a batch has results and every one of them errored."""
        return bool(results) and all(
            res.get(f"{task_prefix}error") is not None for res in results
        )

    def _annotate_batch(
        self,
        *,
        batch: dict[str, list[Any]],
        options: ProviderRuntimeOptions | None,
        gen_kwargs: dict[str, Any] | None = None,
        task_prefix: str = "",
        validate_fn: Callable | None = None,
        postprocess_fn: Callable | None = None,
        num_retries_invalid: int = 5,
        client: Client | None = None,
    ) -> list[dict[str, Any]]:
        """Annotate one batch, retrying the samples that come back invalid.

        Every attempt applies ``postprocess_fn`` before ``validate_fn``, so a
        sample that is answered on a retry has the same columns as one that
        was valid on the first attempt.

        Args:
            batch: Dictionary containing batch data with messages samples.
            options: Runtime options passed to the client.
            gen_kwargs: Extra request parameters merged over ``options``,
                for anything the options dataclass does not name.
            task_prefix: String prefix to use for internal column names.
            validate_fn: Optional custom validation function.
            postprocess_fn: Optional postprocessing function.
            num_retries_invalid: Number of retries for invalid outputs.
            client: Optional client to run inference with (defaults to
                ``self.client``).

        Returns:
            One processed output dictionary per sample in the batch, in order.
        """
        results = self._process_batch(
            batch=batch,
            options=options,
            gen_kwargs=gen_kwargs,
            task_prefix=task_prefix,
            validate_fn=validate_fn,
            postprocess_fn=postprocess_fn,
            client=client,
        )

        if num_retries_invalid <= 0:
            return results

        invalid_indices = self._invalid_indices(results, task_prefix)
        n_retries = 0
        while invalid_indices and n_retries < num_retries_invalid:
            n_retries += 1
            if self.verbose:
                self._logger.info(
                    f"Retrying {len(invalid_indices):,} invalid samples (attempt {n_retries}/{num_retries_invalid})..."
                )

            retry_batch = {
                k: [v[i] for i in invalid_indices] for k, v in batch.items()
            }
            retry_results = self._process_batch(
                batch=retry_batch,
                options=options,
                gen_kwargs=gen_kwargs,
                task_prefix=task_prefix,
                validate_fn=validate_fn,
                postprocess_fn=postprocess_fn,
                client=client,
            )

            for local_idx, global_idx in enumerate(invalid_indices):
                results[global_idx] = retry_results[local_idx]

            invalid_indices = self._invalid_indices(results, task_prefix)

            if (
                self.verbose
                and invalid_indices
                and n_retries == num_retries_invalid
            ):
                self._logger.warning(
                    f"After {n_retries}/{num_retries_invalid} attempts, {len(invalid_indices):,}"
                    " samples are still invalid. Skipping..."
                )

        return results

    def _iter_and_annotate_batches(
        self,
        *,
        prepared_dataset: Dataset,
        options: ProviderRuntimeOptions | None,
        gen_kwargs: dict[str, Any] | None = None,
        task_prefix: str = "",
        validate_fn: Callable | None = None,
        postprocess_fn: Callable | None = None,
        num_retries_invalid: int = 5,
    ) -> Iterator[tuple[dict[str, list[Any]], list[dict[str, Any]]]]:
        """Iterate over the dataset, yielding each batch with its annotations.

        Mostly intended for subclassing: the base implementation walks the
        dataset serially through a single client, while
        [`VLLMQueueAnnotator`][llm_annotator.annotator.VLLMQueueAnnotator]
        overrides it to keep several servers busy at once. Implementations may
        yield batches in any order, as long as
        every batch is yielded exactly once together with one result per sample.

        Args:
            prepared_dataset: The dataset still left to annotate.
            options: Runtime options passed to the client.
            gen_kwargs: Extra request parameters merged over ``options``,
                for anything the options dataclass does not name.
            task_prefix: String prefix to use for internal column names.
            validate_fn: Optional custom validation function.
            postprocess_fn: Optional postprocessing function.
            num_retries_invalid: Number of retries for invalid outputs.

        Yields:
            ``(batch, results)`` where ``batch`` is a column-oriented mapping
            such as ``{col_name: [value_0, value_1, ...]}`` for the current
            mini-batch, and ``results`` is a list of one processed result dict
            per sample in that batch. The two are aligned by position, so
            ``results[i]`` matches the sample in ``batch[col][i]`` for every
            column ``col``.
        """
        total_num_batches = ceil(len(prepared_dataset) / self.batch_size)
        for batch in tqdm(
            prepared_dataset.iter(self.batch_size),
            total=total_num_batches,
            desc=f"Annotating (max_bs={self.batch_size})",
            unit="batch",
        ):
            yield (
                batch,
                self._annotate_batch(
                    batch=batch,
                    options=options,
                    gen_kwargs=gen_kwargs,
                    task_prefix=task_prefix,
                    validate_fn=validate_fn,
                    postprocess_fn=postprocess_fn,
                    num_retries_invalid=num_retries_invalid,
                ),
            )

    def _warm_up(
        self,
        *,
        system_message: str | None = None,
        prompt_prefix: str | None = None,
        options: ProviderRuntimeOptions | None = None,
    ) -> None:
        """Warm up the inference backend before the first real batch.

        Only really relevant for vLLM which may do caching of the prompt
        as a warmup to speed up the first real batch. Other clients
        may implement just a no-op pass.

        Args:
            system_message: Optional system message shared across requests.
            prompt_prefix: Optional fixed prefix that starts every user turn.
            options: Optional generation options used for the warm-up call.
        """
        self.client.warm_up(
            system_message=system_message,
            prompt_prefix=prompt_prefix,
            options=options,
        )

    def prepare_data(
        self,
        output_dir: str | Path,
        prompt_template: str,
        *,
        dataset_name: str | None = None,
        dataset: Dataset | None = None,
        dataset_config: str | None = None,
        data_dir: str | None = None,
        data_files: str | list[str] | dict[str, str | list[str]] | None = None,
        dataset_split: str | None = None,
        max_num_samples: int | None = None,
        shuffle_seed: int | None = None,
        preprocess_fn: Callable | None = None,
        idx_column: str = "idx",
        task_prefix: str = "",
        sort_by_length: bool
        | Literal["shortest_first", "longest_first"] = False,
        system_message: str | None = None,
        hub_id: str | None = None,
        keep_columns: str | Iterable[str] | bool | None = None,
        force_data_preparation: bool = False,
        reuse_idx_column: bool = False,
        allow_selection_change: bool = False,
    ) -> tuple[Dataset, Path | None, str | None]:
        """Prepare input data for annotation without running generation.

        The method reuses local prepared data first, then optionally restores
        prepared data from Hugging Face Hub, and finally falls back to building
        the prepared dataset from source.

        Only the columns required for inference are retained in the cached
        artifact: ``idx_column`` and ``{task_prefix}messages``. Pass
        ``keep_columns`` to preserve additional source columns (e.g. those
        needed by ``run_annotation``'s ``keep_columns`` argument).

        Args:
            output_dir: Directory where prepared artifacts are stored.
            prompt_template: Prompt template used to build chat messages.
            dataset_name: Name or path of the dataset to load.
            dataset: Pre-loaded dataset to use instead of loading from name/path.
            dataset_config: Dataset configuration name (optional).
            data_dir: Data directory for local datasets (optional).
            data_files: Specific file(s) for local datasets (optional).
            dataset_split: Specific split to load (optional).
            max_num_samples: Maximum number of samples to prepare.
            shuffle_seed: Seed for dataset shuffling.
            preprocess_fn: Optional function to preprocess the dataset after loading and before applying the prompt template.
            idx_column: Column name used as unique identifier. Must not exist in the input dataset.
            task_prefix: Prefix for the internal column names and for the
                artifacts of this task inside ``output_dir``, so that several
                tasks can share one directory and one ``hub_id``.
            sort_by_length: Whether to sort prompts by length.
            system_message: Optional system message for chat prompts.
            hub_id: Optional Hugging Face dataset ID used for both prepared-data
                backup and restore. Will be stored in the PREPARED_DS_BRANCH_SUFF branch.
            keep_columns: Source columns to retain in the cached artifact in
                addition to the essential ``idx_column`` and messages column.
                ``True`` keeps all columns (logs a size warning). ``None`` or
                an empty collection keeps only the essential columns.
            force_data_preparation: Whether to rebuild prepared data even when
                local or Hub artifacts already exist.
            reuse_idx_column: Whether an ``idx_column`` that already exists in
                the dataset is kept as the sample id. Use it for a dataset that
                an earlier run produced with ``keep_idx_column=True``, so that
                a row keeps one id through several runs. A change of the
                source is then allowed, because the ids do not depend on the
                position of a row.
            allow_selection_change: Whether a request that the finished rows
                do not belong to is accepted. Only useful when those rows are
                deleted afterwards (``overwrite=True`` in ``run_annotation``).

        Returns:
            Tuple of prepared dataset, local prepared-data path when available,
            and Hugging Face dataset ID when available.

        Raises:
            ValueError: If progress files exist and the request no longer
                matches the settings that they were annotated with: an edited
                prompt template or system message, another ``shuffle_seed``, a
                lower ``max_num_samples``, or a source that changed in another
                way than appended rows without a shuffle.
        """
        pdout = Path(output_dir)
        pdout.mkdir(exist_ok=True, parents=True)

        prepared_data_path = pdout / f"{task_prefix}{PREPARED_DS_LOCAL_SUBDIR}"

        components = _preparation_components(
            prompt_template=prompt_template,
            system_message=system_message,
            sort_by_length=sort_by_length,
            idx_column=idx_column,
            reuse_idx_column=reuse_idx_column,
            shuffle_seed=shuffle_seed,
            preprocess_fn=preprocess_fn,
            source_signature=(
                dataset_signature(dataset) if dataset is not None else None
            ),
        )
        previous = SelectionRecord.read(pdout, task_prefix)
        changed: list[str] = []
        if previous is not None:
            # A recorded component that this method does not produce belongs
            # to another layer (the output schema to `run_annotation`, the
            # step bookkeeping to the pipeline), so a rebuild keeps it.
            components = {
                **{
                    name: value
                    for name, value in previous.components.items()
                    if name not in components
                },
                **components,
            }
            changed = previous.changed_components(components)
            causes = [_COMPONENT_CHANGES[name] for name in changed]
            if previous.max_num_samples != max_num_samples:
                causes.append(
                    "'max_num_samples' changed from"
                    f" {previous.max_num_samples} to {max_num_samples}"
                )
            # A cache built with other settings holds the old messages or the
            # old selection, so it is rebuilt like a forced preparation.
            if causes:
                self._logger.info(
                    "The prepared data was built with other settings than the"
                    f" ones given now ({', '.join(causes)}), so it is rebuilt."
                )
                force_data_preparation = True

        has_local_cache = prepared_data_path.is_dir() and any(
            prepared_data_path.glob("*")
        )
        if has_local_cache and not force_data_preparation:
            cached_ds = Dataset.load_from_disk(prepared_data_path)
            self._adopt_record(
                previous=previous,
                components=components,
                cached_rows=len(cached_ds),
                max_num_samples=max_num_samples,
                pdout=pdout,
                task_prefix=task_prefix,
                origin=f"the cache at '{prepared_data_path}'",
            )
            return cached_ds, prepared_data_path, hub_id

        if hub_id and not force_data_preparation:
            try:
                cached_ds = load_dataset(
                    hub_id,
                    revision=f"{task_prefix}{PREPARED_DS_BRANCH_SUFF}",
                    split="train",
                )
            except Exception:
                pass
            else:
                self._logger.info(
                    f"Restoring prepared data from Hub to local cache at '{prepared_data_path}'..."
                )
                cached_ds.save_to_disk(prepared_data_path)
                self._adopt_record(
                    previous=previous,
                    components=components,
                    cached_rows=len(cached_ds),
                    max_num_samples=max_num_samples,
                    pdout=pdout,
                    task_prefix=task_prefix,
                    origin=f"the Hub backup in '{hub_id}'",
                )
                return cached_ds, prepared_data_path, hub_id

        # ... and if all of that fails, prepare the dataset from the source
        source = self._load_source(
            dataset_name=dataset_name,
            dataset=dataset,
            dataset_config=dataset_config,
            data_dir=data_dir,
            data_files=data_files,
            dataset_split=dataset_split,
        )
        if "dataset" not in components:
            components["dataset"] = dataset_signature(source)
            if previous is not None:
                changed = previous.changed_components(components)
        record = SelectionRecord(
            max_num_samples=max_num_samples,
            source_rows=len(source),
            selected_rows=min(max_num_samples or len(source), len(source)),
            components=components,
        )
        progress_dir = pdout / f"{task_prefix}{PROGRESS_DS_LOCAL_SUBDIR}"
        if (
            previous is not None
            and not allow_selection_change
            and any(progress_dir.glob("*.jsonl"))
        ):
            self._check_reuse(
                previous=previous,
                record=record,
                changed=changed,
                source=source,
                shuffle_seed=shuffle_seed,
                reuse_idx_column=reuse_idx_column,
                progress_dir=progress_dir,
            )

        # Only now, so that a rejected selection leaves the old artifacts intact
        if has_local_cache:
            shutil.rmtree(prepared_data_path, ignore_errors=True)
        if hub_id and force_data_preparation:
            try:
                delete_branch(
                    hub_id,
                    branch=f"{task_prefix}{PREPARED_DS_BRANCH_SUFF}",
                    repo_type="dataset",
                )
            except Exception:
                pass

        _str_formatter = string.Formatter()
        prompt_fields = tuple(
            [
                fld[1]
                for fld in _str_formatter.parse(prompt_template)
                if fld[1] is not None and not fld[2]
            ]
        )

        prepared_dataset: Dataset = self._load_dataset(
            prompt_template=prompt_template,
            idx_column=idx_column,
            dataset=source,
            max_num_samples=max_num_samples,
            shuffle_seed=shuffle_seed,
            prompt_fields=prompt_fields,
            task_prefix=task_prefix,
            sort_by_length=sort_by_length,
            system_message=system_message,
            preprocess_fn=preprocess_fn,
            reuse_idx_column=reuse_idx_column,
        )

        essential_cols = {idx_column, f"{task_prefix}messages"}
        if keep_columns is True:
            self._logger.warning(
                "keep_columns=True: the full prepared dataset will be cached, which may use significant disk space."
            )
        else:
            if isinstance(keep_columns, str):
                essential_cols.add(keep_columns)
            elif keep_columns:
                essential_cols |= set(keep_columns)
            cols_to_drop = [
                c
                for c in prepared_dataset.column_names
                if c not in essential_cols
            ]
            if cols_to_drop:
                prepared_dataset = prepared_dataset.remove_columns(
                    cols_to_drop
                )

        self._logger.info(
            f"Saving prepared data to local cache at '{prepared_data_path}' for faster resumption on failure..."
        )
        prepared_dataset.save_to_disk(prepared_data_path)
        record.write(pdout, task_prefix)
        if hub_id:
            self._logger.info(
                f"Uploading prepared data to Hugging Face Hub at '{hub_id}' for backup and easy restore..."
            )
            prepared_dataset.push_to_hub(
                hub_id,
                revision=f"{task_prefix}{PREPARED_DS_BRANCH_SUFF}",
                split="train",
                private=True,
            )

        return prepared_dataset, prepared_data_path, hub_id

    def _adopt_record(
        self,
        *,
        previous: SelectionRecord | None,
        components: dict[str, str],
        cached_rows: int,
        max_num_samples: int | None,
        pdout: Path,
        task_prefix: str,
        origin: str,
    ) -> None:
        """Record the current settings for prepared data that has no record.

        A backup restored from the Hub arrives on a machine that never ran
        the preparation, so there is nothing to compare the prepared data
        against and it is taken at face value. Recording the settings of the
        request makes a later edit to them detectable.

        Args:
            previous: The record of the prepared data, or ``None``.
            components: The settings of the request.
            cached_rows: Number of rows in the reused prepared data.
            max_num_samples: The requested sample cap.
            pdout: The annotator's output directory.
            task_prefix: The task prefix of the run.
            origin: Where the reused prepared data comes from, for the
                warning.
        """
        if previous is not None:
            return
        self._logger.warning(
            f"The prepared data in {origin} has no record of the settings it"
            " was built with, so it is reused as it is. The settings of this"
            " run are recorded now, so a later edit to them is detected."
        )
        SelectionRecord(
            max_num_samples=max_num_samples,
            source_rows=cached_rows,
            selected_rows=cached_rows,
            components=components,
        ).write(pdout, task_prefix)

    def _record_output_schema(
        self,
        *,
        output_dir: Path,
        task_prefix: str,
        output_schema: dict[str, Any] | None,
        overwrite: bool,
    ) -> None:
        """Compare the output schema against the record and update it.

        The schema decides which columns a finished row has, so two schemas
        must not be mixed in one output. It is recorded here rather than in
        [`prepare_data`][llm_annotator.annotator.Annotator.prepare_data]
        because it has no effect on the prepared data.

        Args:
            output_dir: The annotator's output directory.
            task_prefix: The task prefix of the run.
            output_schema: The decoded JSON schema, or ``None``.
            overwrite: Whether the finished rows are discarded anyway.

        Raises:
            ValueError: If the schema changed and progress files exist.
        """
        record = SelectionRecord.read(output_dir, task_prefix)
        if record is None:
            return

        component = _schema_component(output_schema)
        stored = record.components.get("output_schema")
        if stored == component:
            return

        if stored is not None:
            progress_dir = (
                output_dir / f"{task_prefix}{PROGRESS_DS_LOCAL_SUBDIR}"
            )
            if not overwrite and any(progress_dir.glob("*.jsonl")):
                raise _reuse_error(
                    progress_dir, [_COMPONENT_CHANGES["output_schema"]]
                )
            self._logger.info(
                "The output schema changed since the last run; it is"
                " recorded and every sample is annotated with the new one."
            )

        dataclasses.replace(
            record,
            components={**record.components, "output_schema": component},
        ).write(output_dir, task_prefix)

    def _check_reuse(
        self,
        *,
        previous: SelectionRecord,
        record: SelectionRecord,
        changed: list[str],
        source: Dataset,
        shuffle_seed: int | None,
        reuse_idx_column: bool,
        progress_dir: Path,
    ) -> None:
        """Reject a request that the finished rows do not belong to.

        With the same seed and source, the first N rows of the shuffled source
        are a prefix of the first M rows for every M > N, so a higher cap only
        adds rows. Without a shuffle, rows that are appended to the source
        leave the ids of the old rows unchanged as well. Every other change
        gives the finished rows another meaning.

        Args:
            previous: The record that the progress files were written for.
            record: The record of the request.
            changed: Names of the settings that differ from ``previous``.
            source: The source dataset of the request.
            shuffle_seed: The requested shuffle seed.
            reuse_idx_column: Whether the ids come from an existing column,
                which makes a changed source harmless.
            progress_dir: Directory that holds the progress files.

        Raises:
            ValueError: If a setting changed that gives the finished rows
                another meaning, or if fewer rows are selected than before.
        """
        causes = []
        for name in changed:
            if name == "dataset" and (
                reuse_idx_column
                or (
                    shuffle_seed is None
                    and record.source_rows >= previous.source_rows
                    and dataset_signature(
                        source.select(range(previous.source_rows))
                    )
                    == previous.components.get("dataset")
                )
            ):
                continue
            causes.append(_COMPONENT_CHANGES[name])

        if (
            not reuse_idx_column
            and record.selected_rows < previous.selected_rows
        ):
            causes.append(
                f"the selection shrank from {previous.selected_rows:,} to"
                f" {record.selected_rows:,} rows"
            )

        if causes:
            raise _reuse_error(progress_dir, causes)

        if record.selected_rows > previous.selected_rows:
            self._logger.info(
                f"The selection grows from {previous.selected_rows:,} to"
                f" {record.selected_rows:,} rows. Finished rows are reused."
            )

    def _remove_task_output(
        self, *, root_pdout: Path, task_prefix: str, hub_id: str | None
    ) -> None:
        """Remove what an earlier run of this task left in the output directory.

        Every artifact is named, never globbed on the bare prefix, so that a
        task with an empty ``task_prefix`` does not remove the files of the
        other tasks that share the directory. The prepared data and the record
        that describes it are kept, so a run that is overwritten does not have
        to prepare its data again. The final dataset in the root belongs to no
        single task, and is removed because the run that starts writes it
        again.

        Args:
            root_pdout: The annotator's output directory.
            task_prefix: The task prefix of the run.
            hub_id: Hugging Face dataset ID of the run, or ``None``.
        """
        shutil.rmtree(
            root_pdout / f"{task_prefix}{PROGRESS_DS_LOCAL_SUBDIR}",
            ignore_errors=True,
        )
        (
            root_pdout
            / METADATA_LOCAL_SUBDIR
            / f"{task_prefix}{METADATA_FILE_SUFF}"
        ).unlink(missing_ok=True)

        for name in FINAL_DS_FILES:
            (root_pdout / name).unlink(missing_ok=True)
        for shard in root_pdout.glob(FINAL_DS_SHARD_GLOB):
            shard.unlink()

        if hub_id:
            branch = f"{task_prefix}{PROGRESS_BACKUP_BRANCH_SUFF}"
            try:
                delete_branch(hub_id, branch=branch, repo_type="dataset")
            except Exception as exc:
                self._logger.debug(
                    f"Could not delete the branch '{branch}' on '{hub_id}',"
                    f" which usually means it does not exist: {exc}"
                )

    def _check_progress_backup_branch(
        self, *, hub_id: str, output_dir: Path, task_prefix: str
    ) -> None:
        """Refuse to overwrite a Hub backup with a run that starts from zero.

        A run with no local progress files writes its first progress file
        under the same name as the one on the backup branch, so the next
        upload replaces the backed-up rows with fewer ones. Called only when
        the local progress directory is empty.

        Args:
            hub_id: The dataset repository that the run backs up to.
            output_dir: The annotator's output directory.
            task_prefix: The task prefix of the run.

        Raises:
            ValueError: If the repository has a backup branch for this task.
        """
        branch = f"{task_prefix}{PROGRESS_BACKUP_BRANCH_SUFF}"
        try:
            refs = list_repo_refs(hub_id, repo_type="dataset")
        except Exception as exc:
            self._logger.debug(
                f"Could not list the branches of '{hub_id}', so the progress"
                f" backup on '{branch}' is not checked: {exc}"
            )
            return

        if branch not in {ref.name for ref in refs.branches}:
            return

        prefix_flag = f" --task-prefix {task_prefix}" if task_prefix else ""
        raise ValueError(
            f"'{hub_id}' has a progress backup on the branch '{branch}',"
            f" while '{output_dir}' holds no progress files. This run would"
            " annotate every row again and its first upload would replace"
            " the backup with fewer rows. Restore the backup first:\n"
            "    python scripts/restore_progress_from_hub.py --hub-id"
            f" {hub_id} --output-dir {output_dir}{prefix_flag}\n"
            "Pass overwrite=True to delete the backup branch and annotate"
            " every row again."
        )

    @destroy_on_error
    def run_annotation(
        self,
        output_dir: str | Path,
        prompt_template: str | None = None,
        *,
        prepared_dataset: Dataset | None = None,
        prepared_data_path: str | Path | None = None,
        hub_id: str | None = None,
        overwrite: bool = False,
        dataset_split: str | None = None,
        dataset_config: str | None = None,
        keep_columns: str | Iterable[str] | bool | None = None,
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
        output_schema: str | dict[str, Any] | None = None,
        idx_column: str = "idx",
        upload_every_n_samples: int | None = 10_000,
        max_samples_per_output_file: int | Literal["auto"] = "auto",
        task_prefix: str = "",
        validate_fn: Callable | None = None,
        postprocess_fn: Callable | None = None,
        num_retries_invalid: int = 5,
        system_message: str | None = None,
        keep_idx_column: bool = False,
        max_consecutive_failed_batches: int = 10,
        retry_errors: bool | Sequence[str] = False,
    ) -> Dataset:
        """Run model generation on already prepared annotation inputs.

        Args:
            output_dir: Directory where annotation output is written.
            prompt_template: Prompt template used for warm-up metadata. Optional
                because the prepared dataset already carries the rendered
                messages; when given, its static prefix is used to prime the
                prefix cache.
            prepared_dataset: Pre-prepared dataset with messages column.
            prepared_data_path: Local path to prepared data on disk.
            hub_id: Hugging Face dataset ID used for prepared-data cache and
                JSONL progress backup.
            overwrite: Whether to discard the finished rows of this task and
                annotate every sample again. It removes
                ``<output_dir>/<task_prefix>progress_backup/``, the Hub branch
                of the same name, ``metadata/<task_prefix>annotation_metadata.json``
                and the final dataset in the root of ``output_dir``. It keeps
                the prepared data (``<task_prefix>prepared_dataset/``, its Hub
                branch and ``<task_prefix>selection.json``), so a crashed run
                resumes without preparing its data again; pass
                ``force_data_preparation=True`` to
                [`prepare_data`][llm_annotator.annotator.Annotator.prepare_data]
                to rebuild it. It also keeps the artifacts of every other
                ``task_prefix`` in the same directory.
            dataset_split: Dataset split used for skip filtering.
            dataset_config: Dataset config used for skip filtering.
            keep_columns: Columns to keep in output. ``True`` for all.
            options: Runtime options passed to the client.
            gen_kwargs: Extra request parameters merged over ``options``,
                for anything the options dataclass does not name.
            output_schema: Convenience JSON schema input. When provided, it is
                injected into ``options.json_schema``.
            idx_column: Column name used as unique identifier.
            upload_every_n_samples: Upload to Hub every N samples.
            max_samples_per_output_file: Samples per JSONL progress
                file. ``"auto"`` is one percent of the rows with a floor
                of 1000, so at most 100 files are written and a resume
                stays cheap. A fixed number trades the samples lost at a
                crash against the cost of rescanning the files on every
                resume; 0 writes a single file of unlimited size.
            task_prefix: Prefix for the internal column names and for the
                artifacts of this task inside ``output_dir``, so that several
                tasks can share one directory and one ``hub_id``. The final
                dataset in the root of ``output_dir`` and on Hub ``main`` is
                shared by design: the task that finishes last replaces it, and
                with ``keep_columns=True`` it holds the columns of the tasks
                that ran before it.
            validate_fn: Optional custom validation function.
            postprocess_fn: Optional postprocessing function that takes in a sample and must return a dict.
            num_retries_invalid: Number of retries for invalid outputs.
            system_message: Optional system message for chat prompts.
            keep_idx_column: Whether to keep idx column in final dataset.
            max_consecutive_failed_batches: Abort the run once this many
                batches in a row come back with every sample errored (e.g. a
                vLLM server that died mid-run), instead of continuing to
                dispatch batches against a backend that isn't responding.
                The rows of such batches are only written once a later batch
                succeeds, so a run that aborts leaves them to the resumed
                run. Set to 0 to disable.
            retry_errors: Annotate rows again that finished with an error in
                an earlier run. By default such rows are final, so that an
                error caused by the sample (e.g. a prompt longer than the
                context) is not repeated on every resume. ``True`` redoes
                every errored row, a sequence redoes only those error types,
                e.g. ``["ConnectError", "APITimeoutError"]``. The selected
                rows are removed from the progress files before the run.

        Returns:
            Final concatenated annotation dataset.

        Raises:
            ValueError: If no prepared data source can be resolved, if a
                top-level property of the schema has the name of a column that
                the annotator writes itself, if ``output_schema`` differs
                from the one that the finished rows were annotated with while
                ``overwrite`` is off, or if ``hub_id`` has a progress backup
                while the local progress directory is empty. Restore that
                backup with
                [`restore_progress_from_hub`][llm_annotator.hub.restore_progress_from_hub],
                or pass ``overwrite`` to discard it.
            TooManyConsecutiveFailedBatchesError: If
                ``max_consecutive_failed_batches`` consecutive batches fail
                entirely.
        """
        upload_every_n_samples = upload_every_n_samples or 0
        # Rejected here so a bad value fails before any data is loaded. The
        # concrete size needs the prepared dataset's row count, so it is
        # resolved again once that dataset is in hand.
        _resolve_samples_per_output_file(
            max_samples_per_output_file, num_rows=0
        )

        if max_consecutive_failed_batches < 0:
            raise ValueError(
                "'max_consecutive_failed_batches' must be 0 or a positive"
                " integer"
            )

        if upload_every_n_samples < 0 or not isinstance(
            upload_every_n_samples, int
        ):
            raise ValueError(
                "upload_every_n_samples must be a positive integer or 0"
            )
        if upload_every_n_samples > 0 and not hub_id:
            upload_every_n_samples = 0

        if output_schema is not None:
            if isinstance(output_schema, str):
                output_schema = json.loads(output_schema)
            if not isinstance(output_schema, dict):
                raise TypeError("'output_schema' must decode to a dictionary.")
            if options is not None and options.json_schema is not None:
                raise ValueError(
                    "Provide 'output_schema' OR set 'options.json_schema', not both."
                )
            # Inject the output_schema into options for use in _process_output
            options = dataclasses.replace(
                options or ProviderRuntimeOptions(),
                json_schema=output_schema,
            )

        schema = options.json_schema if options is not None else None
        if schema:
            reserved = _bookkeeping_columns(
                task_prefix=task_prefix, idx_column=idx_column
            )
            taken = sorted(set(schema.get("properties", {})) & reserved)
            if taken:
                names = ", ".join(f"'{name}'" for name in taken)
                raise ValueError(
                    f"The output schema property names {names} are also the"
                    " names of columns that the annotator writes for every"
                    " sample. Pick other names in the schema, or give the run"
                    " another 'task_prefix' or 'idx_column'."
                )

        self._ignored_keys = set()

        if not keep_columns:
            keep_columns = set()
        elif isinstance(keep_columns, str):
            keep_columns = {keep_columns}
        elif keep_columns is True:
            keep_columns = True
        else:
            try:
                keep_columns = set(keep_columns)
            except TypeError as exc:
                raise TypeError(
                    "keep_columns must be None, True, a string, or a collection of strings"
                ) from exc

        if isinstance(keep_columns, set):
            keep_columns.add(idx_column)

        root_pdout = Path(output_dir)
        self._record_output_schema(
            output_dir=root_pdout,
            task_prefix=task_prefix,
            output_schema=output_schema,
            overwrite=overwrite,
        )

        prepared_path = (
            Path(prepared_data_path) if prepared_data_path else None
        )
        if prepared_dataset is None and prepared_data_path:
            try:
                prepared_dataset = Dataset.load_from_disk(prepared_path)
            except Exception as exc:
                self._logger.warning(
                    f"Failed to load prepared dataset from local path '{prepared_data_path}'."
                    f" This might be because the file does not exist or is not a valid dataset. Error: {exc}"
                )

        if prepared_dataset is None and hub_id:
            try:
                prepared_dataset = load_dataset(
                    hub_id,
                    revision=f"{task_prefix}{PREPARED_DS_BRANCH_SUFF}",
                    split="train",
                )
            except Exception as exc:
                self._logger.warning(
                    f"Failed to load prepared dataset from Hub ID '{hub_id}' with revision '{PREPARED_DS_BRANCH_SUFF}'."
                    f" This might be because the dataset or revision does not exist, or due to network issues. Error: {exc}"
                )

        if prepared_dataset is None:
            raise ValueError(
                "No prepared data found. Provide 'prepared_dataset', "
                "'prepared_data_path' (locally saved dataset), or 'hub_id' (cloud-saved dataset)."
                " If needed, first run 'prepare_data' to create the prepared dataset."
            )

        if idx_column not in prepared_dataset.column_names:
            raise ValueError(
                f"Expected index column '{idx_column}' not found in prepared dataset."
                " This column is required for tracking processed samples and resuming on failure."
                " Please ensure the prepared dataset includes the index column with name matching 'idx_column' argument."
            )

        samples_per_output_file = _resolve_samples_per_output_file(
            max_samples_per_output_file, num_rows=len(prepared_dataset)
        )
        if max_samples_per_output_file == "auto" and self.verbose:
            self._logger.info(
                f"Writing progress files of {samples_per_output_file:,}"
                f" samples over {len(prepared_dataset):,} rows. Set"
                " 'max_samples_per_output_file' to a fixed number to change"
                " that."
            )

        # Only discard the finished rows after potentially reading the cached
        # input.
        if root_pdout.is_dir() and overwrite:
            self._remove_task_output(
                root_pdout=root_pdout,
                task_prefix=task_prefix,
                hub_id=hub_id,
            )

        root_pdout.mkdir(exist_ok=True, parents=True)
        process_pdout = root_pdout / f"{task_prefix}{PROGRESS_DS_LOCAL_SUBDIR}"
        process_pdout.mkdir(exist_ok=True, parents=True)

        if (
            hub_id
            and upload_every_n_samples > 0
            and not any(process_pdout.glob("*.jsonl"))
        ):
            self._check_progress_backup_branch(
                hub_id=hub_id,
                output_dir=root_pdout,
                task_prefix=task_prefix,
            )

        if retry_errors:
            retried_rows = drop_jsonl_rows(
                process_pdout,
                lambda row: is_retried_error(
                    row, retry_errors=retry_errors, task_prefix=task_prefix
                ),
            )
            self._logger.info(
                f"Removed {len(retried_rows):,} errored sample(s) from the"
                " progress files; they are annotated again ('retry_errors')."
            )

        # Get indices from the local
        skip_idxs = self._get_skip_idxs(
            process_pdout=process_pdout,
            idx_column=idx_column,
            dataset_split=dataset_split,
            dataset_config=dataset_config,
        )
        processed_n_samples = len(skip_idxs)

        if processed_n_samples == len(prepared_dataset):
            self._logger.info(
                "All samples in the prepared dataset have already been processed according to existing output files."
            )
            return self._post_annotate(
                process_pdout=process_pdout,
                idx_column=idx_column,
                hub_id=hub_id,
                keep_idx_column=keep_idx_column,
                task_prefix=task_prefix,
            )

        if skip_idxs:
            prepared_dataset = prepared_dataset.filter(
                lambda sample_idxs: [
                    sidx not in skip_idxs for sidx in sample_idxs
                ],
                num_proc=self.num_proc,
                input_columns=[idx_column],
                batched=True,
                desc="Filtering done idxs",
            )
            if self.verbose:
                self._logger.info(
                    f"Skipping {len(skip_idxs):,} already-processed samples"
                )

        prompt_template_prefix = (
            extract_prompt_prefix(prompt_template) if prompt_template else None
        )

        pfout = self.get_pfout_name(
            process_pdout=process_pdout,
            max_samples_per_output_file=samples_per_output_file,
            processed_n_samples=processed_n_samples,
        )
        fhout = pfout.open("a", encoding="utf-8")

        uploader = (
            _ProgressUploader(
                annotator=self,
                process_pdout=process_pdout,
                hub_id=hub_id,
                task_prefix=task_prefix,
            )
            if hub_id and upload_every_n_samples > 0
            else None
        )

        run_started = perf_counter()
        annotated_n_rows = 0
        annotated_n_tokens = 0

        self._warm_up(
            system_message=system_message,
            prompt_prefix=prompt_template_prefix,
            options=options,
        )

        annotated_batches = self._iter_and_annotate_batches(
            prepared_dataset=prepared_dataset,
            options=options,
            gen_kwargs=gen_kwargs,
            task_prefix=task_prefix,
            validate_fn=validate_fn,
            postprocess_fn=postprocess_fn,
            num_retries_invalid=num_retries_invalid,
        )

        def write_rows(rows: list[dict[str, Any]]) -> None:
            """Append rows to the progress files and upload when it is time."""
            nonlocal fhout, processed_n_samples
            nonlocal annotated_n_rows, annotated_n_tokens
            for row in rows:
                fhout.write(json.dumps(row, default=str) + "\n")
                fhout.flush()
                processed_n_samples += 1
                annotated_n_rows += 1
                annotated_n_tokens += row.get(f"{task_prefix}num_tokens") or 0

                time_to_upload = (
                    upload_every_n_samples > 0
                    and processed_n_samples % upload_every_n_samples == 0
                )
                # Cap the file size even when nothing is pushed to the Hub:
                # otherwise a long run leaves one unbounded JSONL that every
                # restart has to parse in full.
                file_is_full = (
                    samples_per_output_file > 0
                    and processed_n_samples % samples_per_output_file == 0
                )

                if time_to_upload or file_is_full:
                    fhout.close()
                    remove_empty_jsonl_files(process_pdout)
                    pfout = self.get_pfout_name(
                        process_pdout=process_pdout,
                        max_samples_per_output_file=samples_per_output_file,
                        processed_n_samples=processed_n_samples,
                    )
                    fhout = pfout.open("a", encoding="utf-8")
                    if time_to_upload and uploader is not None:
                        uploader.request(
                            active_path=pfout,
                            active_bytes=pfout.stat().st_size,
                        )

        # Rows of batches in which every sample errored. They are written once
        # a later batch succeeds and dropped when the run aborts, so that the
        # resumed run annotates them instead of keeping the errors.
        held_back_rows: list[dict[str, Any]] = []
        consecutive_failed_batches = 0
        try:
            for batch, results in annotated_batches:
                rows = [
                    {
                        **{
                            k: v[i]
                            for k, v in batch.items()
                            if keep_columns is True or k in keep_columns  # type: ignore[operator]
                        },
                        **res,
                    }
                    for i, res in enumerate(results)
                ]

                if max_consecutive_failed_batches and self._all_errored(
                    results, task_prefix
                ):
                    held_back_rows.extend(rows)
                    consecutive_failed_batches += 1
                    if (
                        consecutive_failed_batches
                        >= max_consecutive_failed_batches
                    ):
                        raise TooManyConsecutiveFailedBatchesError(
                            f"{consecutive_failed_batches} consecutive batches"
                            " failed entirely; aborting instead of continuing"
                            " to burn compute against a backend that isn't"
                            " responding. Their rows were not written, so a"
                            " resumed run annotates them again. Last error:"
                            f" {results[-1].get(f'{task_prefix}error')}"
                        )
                    continue

                consecutive_failed_batches = 0
                write_rows(held_back_rows + rows)
                held_back_rows = []

            write_rows(held_back_rows)
        finally:
            # The uploader is joined first: its thread is not a daemon, and
            # closing the batch generator may raise.
            if uploader is not None:
                uploader.close()
            # Closing the generator lets alternative execution strategies
            # (e.g. the multi-server queue) shut their workers down when the
            # writer stops early because of an error or interruption.
            close_batches = getattr(annotated_batches, "close", None)
            if callable(close_batches):
                close_batches()
            fhout.close()

        elapsed_seconds = perf_counter() - run_started

        remove_empty_jsonl_files(process_pdout)
        if hub_id and upload_every_n_samples > 0:
            self.push_progress_to_hub(
                process_pdout, hub_id=hub_id, task_prefix=task_prefix
            )

        return self._post_annotate(
            process_pdout=process_pdout,
            idx_column=idx_column,
            hub_id=hub_id,
            keep_idx_column=keep_idx_column,
            task_prefix=task_prefix,
            num_rows_annotated=annotated_n_rows,
            num_output_tokens=annotated_n_tokens,
            elapsed_seconds=elapsed_seconds,
        )

    @destroy_on_error
    def annotate_dataset(
        self,
        output_dir: str | Path,
        prompt_template: str,
        *,
        dataset_name: str | None = None,
        dataset: Dataset | None = None,
        dataset_config: str | None = None,
        data_dir: str | None = None,
        data_files: str | list[str] | dict[str, str | list[str]] | None = None,
        dataset_split: str | None = None,
        max_num_samples: int | None = None,
        shuffle_seed: int | None = None,
        preprocess_fn: Callable | None = None,
        idx_column: str = "idx",
        task_prefix: str = "",
        sort_by_length: bool
        | Literal["shortest_first", "longest_first"] = False,
        system_message: str | None = None,
        hub_id: str | None = None,
        force_data_preparation: bool = False,
        overwrite: bool = False,
        keep_columns: str | Iterable[str] | bool | None = None,
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
        output_schema: str | dict[str, Any] | None = None,
        upload_every_n_samples: int | None = 10_000,
        max_samples_per_output_file: int | Literal["auto"] = "auto",
        validate_fn: Callable | None = None,
        postprocess_fn: Callable | None = None,
        num_retries_invalid: int = 5,
        keep_idx_column: bool = False,
        max_consecutive_failed_batches: int = 10,
        reuse_idx_column: bool = False,
        retry_errors: bool | Sequence[str] = False,
    ) -> Dataset:
        """Annotate an existing dataset in one call.

        This is a convenience wrapper around
        [`prepare_data`][llm_annotator.annotator.Annotator.prepare_data] and
        [`run_annotation`][llm_annotator.annotator.Annotator.run_annotation]
        for callers that prefer a single entry point.

        Args:
            output_dir: Directory where annotation output is written. Every
                argument below is passed straight through, so the full
                description of each is in
                [`prepare_data`][llm_annotator.annotator.Annotator.prepare_data]
                (data selection and prompting) or in
                [`run_annotation`][llm_annotator.annotator.Annotator.run_annotation]
                (inference and output).
            prompt_template: Prompt template with dataset fields.
            dataset_name: Name or path of the dataset to load.
            dataset: Pre-loaded dataset to annotate instead of loading one.
            dataset_config: Dataset configuration name.
            data_dir: Data directory for local datasets.
            data_files: Specific file(s) for local datasets.
            dataset_split: Dataset split to load.
            max_num_samples: Maximum number of samples to annotate.
            shuffle_seed: Seed for dataset shuffling.
            preprocess_fn: Optional preprocessing callback.
            idx_column: Column name used as the stable sample identifier.
            task_prefix: Prefix for this task's columns and artifacts.
            sort_by_length: Whether to sort prompts by length.
            system_message: Optional system message for the chat prompt.
            hub_id: Optional Hub dataset ID for prepared-data cache and
                JSONL progress backup.
            force_data_preparation: Rebuild prepared data even if cached.
            overwrite: Whether to discard the finished rows of this task and
                annotate every sample again.
            keep_columns: Columns to keep in the final dataset.
            options: Runtime options passed to the client.
            gen_kwargs: Extra request parameters merged over ``options``.
            output_schema: Optional JSON schema for structured output.
            upload_every_n_samples: Upload checkpoint cadence.
            max_samples_per_output_file: Samples per JSONL progress file.
            validate_fn: Optional validation callback.
            postprocess_fn: Optional postprocessing callback.
            num_retries_invalid: Number of retries for invalid outputs.
            keep_idx_column: Whether to keep the index column in the result.
            max_consecutive_failed_batches: Abort the run once this many
                batches in a row come back with every sample errored.
            reuse_idx_column: Whether an ``idx_column`` that already exists in
                the dataset is kept as the sample id.
            retry_errors: Annotate rows again that finished with an error in
                an earlier run.

        Returns:
            The concatenated annotation dataset.

        Raises:
            ValueError: If finished rows exist that were annotated with
                other settings than the ones given now, and ``overwrite`` is
                off.
            TooManyConsecutiveFailedBatchesError: If
                ``max_consecutive_failed_batches`` consecutive batches fail
                entirely.
        """
        prepared_dataset, _, _ = self.prepare_data(
            output_dir=output_dir,
            prompt_template=prompt_template,
            dataset_name=dataset_name,
            dataset=dataset,
            dataset_config=dataset_config,
            data_dir=data_dir,
            data_files=data_files,
            dataset_split=dataset_split,
            max_num_samples=max_num_samples,
            shuffle_seed=shuffle_seed,
            preprocess_fn=preprocess_fn,
            idx_column=idx_column,
            task_prefix=task_prefix,
            sort_by_length=sort_by_length,
            system_message=system_message,
            hub_id=hub_id,
            keep_columns=keep_columns,
            force_data_preparation=force_data_preparation,
            reuse_idx_column=reuse_idx_column,
            allow_selection_change=overwrite,
        )

        return self.run_annotation(
            output_dir=output_dir,
            prompt_template=prompt_template,
            prepared_dataset=prepared_dataset,
            hub_id=hub_id,
            overwrite=overwrite,
            keep_columns=keep_columns,
            options=options,
            gen_kwargs=gen_kwargs,
            output_schema=output_schema,
            idx_column=idx_column,
            upload_every_n_samples=upload_every_n_samples,
            max_samples_per_output_file=max_samples_per_output_file,
            task_prefix=task_prefix,
            validate_fn=validate_fn,
            postprocess_fn=postprocess_fn,
            num_retries_invalid=num_retries_invalid,
            system_message=system_message,
            keep_idx_column=keep_idx_column,
            max_consecutive_failed_batches=max_consecutive_failed_batches,
            retry_errors=retry_errors,
        )

    @destroy_on_error
    def generate_dataset(
        self,
        output_dir: str | Path,
        prompts: str | Sequence[str],
        *,
        prompt_prefix: str | None = None,
        hub_id: str | None = None,
        force_data_preparation: bool = False,
        overwrite: bool = False,
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
        max_num_samples: int | None = None,
        output_schema: str | dict[str, Any] | None = None,
        idx_column: str = "idx",
        upload_every_n_samples: int | None = 10_000,
        max_samples_per_output_file: int | Literal["auto"] = "auto",
        task_prefix: str = "",
        validate_fn: Callable | None = None,
        postprocess_fn: Callable | None = None,
        num_retries_invalid: int = 5,
        keep_idx_column: bool = False,
        max_consecutive_failed_batches: int = 10,
        retry_errors: bool | Sequence[str] = False,
    ) -> Dataset:
        """Generate a new dataset from prompts.

        Args:
            output_dir: Directory where annotation output is written. Every
                argument below is passed straight through, so the full
                description of each is in
                [`prepare_data`][llm_annotator.annotator.Annotator.prepare_data]
                (data selection and prompting) or in
                [`run_annotation`][llm_annotator.annotator.Annotator.run_annotation]
                (inference and output).
            prompts: A single prompt or a sequence of prompts.
            prompt_prefix: Optional shared prefix used for prefix caching.
            hub_id: Optional Hub dataset ID for prepared-data cache and
                JSONL progress backup.
            force_data_preparation: Rebuild prepared data even if cached.
            overwrite: Whether to discard the finished rows of this task and
                annotate every sample again.
            options: Runtime options passed to the client.
            gen_kwargs: Extra request parameters merged over ``options``.
            max_num_samples: Number of times to repeat a single prompt.
            output_schema: Optional JSON schema for structured output.
            idx_column: Column name used as the stable sample identifier.
            upload_every_n_samples: Upload checkpoint cadence.
            max_samples_per_output_file: Samples per JSONL progress file.
            task_prefix: Prefix for this task's columns and artifacts.
            validate_fn: Optional validation callback.
            postprocess_fn: Optional postprocessing callback.
            num_retries_invalid: Number of retries for invalid outputs.
            keep_idx_column: Whether to keep the index column in the result.
            max_consecutive_failed_batches: Abort the run once this many
                batches in a row come back with every sample errored.
            retry_errors: Annotate rows again that finished with an error in
                an earlier run.

        Returns:
            The concatenated annotation dataset.

        Raises:
            ValueError: If no prompts are provided.
            TooManyConsecutiveFailedBatchesError: If
                ``max_consecutive_failed_batches`` consecutive batches fail
                entirely.
        """
        if isinstance(prompts, str):
            if max_num_samples is None:
                max_num_samples = 1
            prompt_list = [prompts] * max_num_samples
        else:
            prompt_list = list(prompts)
            max_num_samples = len(prompt_list)

        if not prompt_list:
            raise ValueError("At least one prompt must be provided.")

        prompt_dataset = Dataset.from_dict({"prompt": prompt_list})
        prompt_template = f"{prompt_prefix or ''}{{prompt}}"

        prepared_dataset, _, _ = self.prepare_data(
            output_dir=output_dir,
            prompt_template=prompt_template,
            dataset=prompt_dataset,
            max_num_samples=max_num_samples,
            idx_column=idx_column,
            task_prefix=task_prefix,
            hub_id=hub_id,
            force_data_preparation=force_data_preparation,
        )

        return self.run_annotation(
            output_dir=output_dir,
            prompt_template=prompt_template,
            prepared_dataset=prepared_dataset,
            hub_id=hub_id,
            overwrite=overwrite,
            options=options,
            gen_kwargs=gen_kwargs,
            output_schema=output_schema,
            idx_column=idx_column,
            upload_every_n_samples=upload_every_n_samples,
            max_samples_per_output_file=max_samples_per_output_file,
            task_prefix=task_prefix,
            validate_fn=validate_fn,
            postprocess_fn=postprocess_fn,
            num_retries_invalid=num_retries_invalid,
            keep_idx_column=keep_idx_column,
            max_consecutive_failed_batches=max_consecutive_failed_batches,
            retry_errors=retry_errors,
        )

    def _load_progress_files(self, process_pdout: Path) -> Dataset:
        """Read every progress file in a directory back into one dataset.

        The fast path hands the whole directory to ``load_dataset``, which
        needs every file to share one schema. Files written by different
        library versions do not: a release that adds a bookkeeping column
        leaves a run that was already in flight with older files that lack it.
        Rather than making such a run unresumable, fall back to reading each
        file on its own and padding the missing columns with nulls.

        Args:
            process_pdout: Directory holding the ``*.jsonl`` progress files.

        Returns:
            One dataset with the union of every file's columns.
        """
        try:
            return load_dataset(
                "json", data_dir=str(process_pdout), split="train"
            )
        except DatasetGenerationError:
            self._logger.warning(
                f"The progress files in '{process_pdout}' do not all have the"
                " same columns, which happens when a run is resumed by a"
                " different version of this library. Reading them one by one"
                " and filling the missing columns with nulls."
            )

        parts = [
            load_dataset("json", data_files=str(pfin), split="train")
            for pfin in sorted(process_pdout.glob("*.jsonl"))
        ]
        # A column that happens to be null in every row of one file is typed
        # "null" there, so prefer any file that gives it a concrete type.
        features = Features()
        for part in parts:
            for column, feature in part.features.items():
                known = features.get(column)
                if known is None or getattr(known, "dtype", None) == "null":
                    features[column] = feature

        padded = []
        for part in parts:
            missing = [c for c in features if c not in part.column_names]
            for column in missing:
                part = part.add_column(column, [None] * part.num_rows)
            padded.append(part.cast(features) if missing else part)

        return concatenate_datasets(padded)

    def _post_annotate(
        self,
        *,
        process_pdout: Path,
        idx_column: str,
        hub_id: str | None = None,
        keep_idx_column: bool = False,
        task_prefix: str = "",
        num_rows_annotated: int = 0,
        num_output_tokens: int = 0,
        elapsed_seconds: float | None = None,
    ) -> Dataset:
        """Build the final dataset out of the progress files and clean up.

        Concatenates the progress files, sorts them by ``idx_column`` and
        keeps the first row of every repeated id. The result is written to the
        root of the output directory and, with a ``hub_id``, pushed to the
        ``main`` branch of that repository. Afterwards the local prepared-data
        cache and the two temporary Hub branches are removed and the metadata
        of the run is written.

        Args:
            process_pdout: Directory that holds the ``*.jsonl`` progress files.
            idx_column: Column name used as unique identifier.
            hub_id: Optional Hugging Face dataset ID for uploads and cleanup.
            keep_idx_column: Whether to keep the idx_column in the final dataset before uploading and returning.
            task_prefix: Prefix used for the local cache directory name and the upload branch names.
            num_rows_annotated: How many rows this invocation annotated.
            num_output_tokens: How many output tokens this invocation
                generated.
            elapsed_seconds: How long this invocation generated for.

        Returns:
            The concatenated dataset of all annotation results (invalid samples are NOT removed)
        """
        ds = self._load_progress_files(process_pdout).sort(idx_column)

        # A sample can only be written twice if two processes wrote to the same
        # progress directory (e.g. a requeued job whose predecessor was still
        # draining). Keep the first row per idx so the final dataset holds
        # exactly one row per sample.
        seen: set[Any] = set()
        unique_rows: list[int] = []
        for row_idx, sample_idx in enumerate(ds[idx_column]):
            if sample_idx not in seen:
                seen.add(sample_idx)
                unique_rows.append(row_idx)

        if len(unique_rows) < ds.num_rows:
            self._logger.warning(
                f"Found {ds.num_rows - len(unique_rows):,} duplicate"
                f" '{idx_column}' values in '{process_pdout}'. Keeping the"
                " first row of each; this usually means two annotation"
                " processes shared one output directory."
            )
            ds = ds.select(unique_rows)

        if not keep_idx_column:
            ds = ds.remove_columns([idx_column])

        # Save final dataset to root directory
        ds.save_to_disk(process_pdout.parent)

        if hub_id:
            ds.push_to_hub(hub_id, private=True)
            if self.verbose:
                self._logger.info(
                    f"Uploaded final dataset to the HF Hub: https://huggingface.co/datasets/{hub_id}!"
                )

        ds.cleanup_cache_files()

        # Clean up the local prepared-data cache
        cached_input_ds = (
            process_pdout.parent / f"{task_prefix}{PREPARED_DS_LOCAL_SUBDIR}"
        )
        if cached_input_ds.exists():
            shutil.rmtree(cached_input_ds, ignore_errors=True)

        if hub_id:
            # Clean up the prepared-data branch on the Hub
            try:
                delete_branch(
                    hub_id,
                    branch=f"{task_prefix}{PREPARED_DS_BRANCH_SUFF}",
                    repo_type="dataset",
                )
            except Exception as exc:
                self._logger.warning(
                    "Failed to delete prepared-data branch"
                    f" '{task_prefix}{PREPARED_DS_BRANCH_SUFF}' on"
                    f" '{hub_id}': {exc}"
                )

            # Clean up the progress upload branch used for JSONL progress backup
            # These branches can take up a lot of space and are easily forgotten,
            # so best to clean up
            try:
                delete_branch(
                    hub_id,
                    branch=f"{task_prefix}{PROGRESS_BACKUP_BRANCH_SUFF}",
                    repo_type="dataset",
                )
            except Exception as exc:
                self._logger.warning(
                    "Failed to delete progress branch"
                    f" '{task_prefix}{PROGRESS_BACKUP_BRANCH_SUFF}' on"
                    f" '{hub_id}': {exc}"
                )

        self._add_metadata(
            root_pdout=process_pdout.parent,
            dataset=ds,
            task_prefix=task_prefix,
            hub_id=hub_id,
            num_rows_annotated=num_rows_annotated,
            num_output_tokens=num_output_tokens,
            elapsed_seconds=elapsed_seconds,
        )

        return ds

    def _add_metadata(
        self,
        root_pdout: Path,
        dataset: Dataset,
        task_prefix: str,
        hub_id: str | None = None,
        *,
        num_rows_annotated: int = 0,
        num_output_tokens: int = 0,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Write counts and library versions to the metadata subdirectory.

        The counts of the run go to
        ``<output_dir>/metadata/<task_prefix>annotation_metadata.json``, one
        file per task of the directory. The library versions go to
        ``_version.json``, which is shared because it does not depend on the
        task. Both are uploaded to the ``metadata`` folder of the Hub
        repository when ``hub_id`` is given.

        The ``run_summary`` of that file covers this invocation only: the
        rows and the output tokens that it generated, and the seconds from
        the warm-up to its last written row. A resumed run therefore reports
        its own throughput and not the average over every invocation. It is
        ``null`` when the invocation generated nothing, which is what a run
        that finds every row already annotated does.

        Args:
            root_pdout: The root output directory path.
            dataset: The final annotated dataset.
            task_prefix: String prefix to use for internal column names.
            hub_id: Optional Hugging Face dataset ID to upload metadata to.
            num_rows_annotated: How many rows this invocation annotated.
            num_output_tokens: How many output tokens this invocation
                generated. An errored row reports no tokens and counts as 0.
            elapsed_seconds: How long this invocation generated for.
        """
        mtd_dir = root_pdout / METADATA_LOCAL_SUBDIR
        mtd_dir.mkdir(exist_ok=True)

        # Add version info
        mtd_dir.joinpath(VERSION_FILE).write_text(
            json.dumps(get_lib_versions(), indent=4, default=str),
            encoding="utf-8",
        )

        # Get counts for finish_reason and valid_fields
        finish_reason_counts: CounterType[str] = Counter()
        valid_fields_counts: CounterType[str] = Counter()
        valid_res = {None: "none", True: "valid", False: "invalid"}
        error_type_counts: CounterType[str] = Counter()

        # Iterate to avoid OOM
        for batch in dataset.iter(batch_size=10_000):
            if f"{task_prefix}finish_reason" in batch:
                reasons = [
                    "none" if item is None else item
                    for item in batch[f"{task_prefix}finish_reason"]
                ]
                finish_reason_counts.update(reasons)

            if f"{task_prefix}valid_fields" in batch:
                valids = [
                    valid_res.get(item, "unknown")
                    for item in batch[f"{task_prefix}valid_fields"]
                ]
                valid_fields_counts.update(valids)

            if f"{task_prefix}error_type" in batch:
                error_types = [
                    "none" if item is None else item
                    for item in batch[f"{task_prefix}error_type"]
                ]
                error_type_counts.update(error_types)

        run_summary: dict[str, float] | None = None
        if num_rows_annotated and elapsed_seconds and elapsed_seconds > 0:
            run_summary = {
                "num_rows": num_rows_annotated,
                "num_output_tokens": num_output_tokens,
                "elapsed_seconds": round(elapsed_seconds, 2),
                "rows_per_second": round(
                    num_rows_annotated / elapsed_seconds, 2
                ),
                "output_tokens_per_second": round(
                    num_output_tokens / elapsed_seconds, 2
                ),
            }
            self._logger.info(
                f"This run annotated {num_rows_annotated:,} row(s) and"
                f" {num_output_tokens:,} output token(s) in"
                f" {elapsed_seconds:,.1f}s:"
                f" {run_summary['rows_per_second']:,.2f} row(s) per second"
                f" and {run_summary['output_tokens_per_second']:,.2f} output"
                " token(s) per second. Rows that an earlier run of the same"
                " output directory annotated are not counted."
            )

        mtd = {
            "finish_reason_counts": dict(finish_reason_counts),
            "valid_fields_counts": dict(valid_fields_counts),
            "error_type_counts": dict(error_type_counts),
            "run_summary": run_summary,
        }

        mtd_dir.joinpath(f"{task_prefix}{METADATA_FILE_SUFF}").write_text(
            json.dumps(mtd, indent=4, default=str), encoding="utf-8"
        )

        errors = {k: v for k, v in error_type_counts.items() if k != "none"}
        num_invalid = valid_fields_counts["invalid"]
        summary = (
            f"Annotated {len(dataset):,} sample(s): {sum(errors.values()):,}"
            f" with an error, {num_invalid:,} with invalid fields."
        )
        if errors:
            summary += (
                f" Errors per type: {errors}. Errored rows are final. To"
                " annotate them again, run with 'retry_errors=True'"
                " ('llm-annotate --retry-errors'), or name the error types"
                " to redo only those."
            )
        self._logger.log(
            logging.WARNING if errors or num_invalid else logging.INFO,
            summary,
        )

        if hub_id:
            upload_folder(
                repo_id=hub_id,
                repo_type="dataset",
                folder_path=mtd_dir,
                path_in_repo=METADATA_LOCAL_SUBDIR,
            )

    def get_pfout_name(
        self,
        *,
        process_pdout: Path,
        max_samples_per_output_file: int,
        processed_n_samples: int | None = None,
    ) -> Path:
        """Generate the output file name based on configuration.

        Creates appropriate file names for output files, handling both
        single-file and multi-file output modes.

        Args:
            process_pdout: The output directory path.
            max_samples_per_output_file: Maximum samples per output file (0 for unlimited).
            processed_n_samples: The number of samples processed so far.

        Returns:
            Path object for the output file name.
        """
        processed_n_samples = processed_n_samples or 0
        stem = process_pdout.stem
        if not max_samples_per_output_file:
            return process_pdout.joinpath(f"{stem}.jsonl")
        else:
            count_idx = processed_n_samples // max_samples_per_output_file
            return process_pdout.joinpath(f"{stem}_{count_idx}.jsonl")

    def push_progress_to_hub(
        self,
        dir_path: Path | str,
        hub_id: str | None = None,
        *,
        task_prefix: str = "",
        active_path: Path | None = None,
        active_bytes: int = 0,
    ) -> None:
        """Upload the progress files of a run to the Hugging Face Hub.

        Creates the dataset repository and its
        ``<task_prefix>progress_backup`` branch, and uploads the ``*.jsonl``
        files of ``dir_path`` plus the selection record next to it, which is
        what ``llm_annotator.hub.restore_progress_from_hub`` reads back.

        ``active_path`` names the progress file that the writer has open, so
        that a background upload does not read a file while it grows. That
        file is left out of the folder upload and its first ``active_bytes``
        bytes are copied out and uploaded from the copy, which ends on a
        line boundary. Uploading the file itself would hash it and then read
        it again, and for a file that grew in between the stored object no
        longer matches the checksum of the commit.

        Args:
            dir_path: Directory that holds the ``*.jsonl`` progress files.
            hub_id: Optional Hugging Face dataset ID to upload into.
            task_prefix: String prefix to use for branch naming.
            active_path: The progress file that the writer has open, when
                the writer is running.
            active_bytes: How many bytes of ``active_path`` were written
                when the upload was requested.

        Raises:
            ValueError: If no ``hub_id`` is given.
        """
        if not hub_id:
            raise ValueError(
                "'hub_id' must be set to push data to the HuggingFace Hub"
            )

        pdout = Path(dir_path)
        branch = f"{task_prefix}{PROGRESS_BACKUP_BRANCH_SUFF}"
        create_repo(hub_id, repo_type="dataset", exist_ok=True, private=True)
        create_branch(
            hub_id,
            repo_type="dataset",
            branch=branch,
            exist_ok=True,
        )

        upload_folder(
            repo_id=hub_id,
            repo_type="dataset",
            folder_path=str(pdout),
            revision=branch,
            allow_patterns=["*.jsonl"],
            ignore_patterns=(
                [active_path.name] if active_path is not None else None
            ),
        )

        if active_path is not None and active_bytes > 0:
            staged = pdout.parent / f"{task_prefix}{PROGRESS_UPLOAD_FILE}"
            try:
                _copy_file_prefix(
                    src=active_path, dest=staged, num_bytes=active_bytes
                )
                upload_file(
                    path_or_fileobj=str(staged),
                    path_in_repo=active_path.name,
                    repo_id=hub_id,
                    repo_type="dataset",
                    revision=branch,
                )
            finally:
                staged.unlink(missing_ok=True)

        record_path = SelectionRecord.path(pdout.parent, task_prefix)
        if record_path.is_file():
            upload_file(
                path_or_fileobj=str(record_path),
                path_in_repo=record_path.name,
                repo_id=hub_id,
                repo_type="dataset",
                revision=branch,
            )

        if self.verbose:
            self._logger.info(
                "Backed-up data to the HF Hub:"
                f" https://huggingface.co/datasets/{hub_id}/tree/{branch}"
            )


def _create_messages(
    sample: dict,
    prompt_fields: Iterable[str],
    prompt_template: str,
    task_prefix: str,
    system_message: str | None = None,
) -> dict[str, Any]:
    """Restructure the sample into a "messages" format. Fills in the prompt template with values from the sample,
    based on the prompt_fields.

    Args:
        sample: The dataset sample to process.
        prompt_fields: Fields required by the prompt template.
        prompt_template: The prompt template string with placeholders.
        task_prefix: String prefix to use for internal column names.
        system_message: Optional system message to add as "system" role in chat prompts.

    Returns:
        A dictionary with the filled-in prompt and the sample index.
    """
    if system_message is not None:
        return {
            f"{task_prefix}messages": [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": prompt_template.format(
                        **{fld: sample[fld] for fld in prompt_fields}
                    ),
                },
            ]
        }
    else:
        return {
            f"{task_prefix}messages": [
                {
                    "role": "user",
                    "content": prompt_template.format(
                        **{fld: sample[fld] for fld in prompt_fields}
                    ),
                }
            ]
        }


@dataclass(slots=True, kw_only=True)
class VLLMQueueAnnotator(Annotator):
    """Annotator that spreads one workload over several vLLM servers/clients.

    Instead of walking the prepared dataset batch-by-batch through a single client
    it keeps a bounded queue of batches in flight over a pool of vLLM server clients,
    handing each batch to whichever server is free. The process can be simplified as:

    - add each client to a queue once per allowed concurrent batch;
    - for each batch:
        - pop a client from the queue;
        - send the batch to that client;

    Everything else -- prompt templating,
    JSONL progress snapshots keyed by ``idx``, resumption, Hub backups, and the
    final concatenation -- is inherited from
    [`Annotator`][llm_annotator.annotator.Annotator], so all four public entry
    points behave exactly as documented there.

    Because batches finish out of order, results are written in completion
    order and sorted by ``idx`` at the end (as they already are for the base
    annotator, whose JSONL files are concatenated and sorted in
    `Annotator._post_annotate`).

    Args:
        clients: vLLM server clients used as the worker pool. Keyword-only.
            The first client doubles as ``Annotator.client`` for inherited
            helpers, so ``client`` is derived here rather than passed in.
        queue_size: Maximum number of batches in flight (dispatched but not yet
            written out). This bounds memory, *not* the amount of work: the
            full dataset is always annotated. ``None`` resolves to four batches
            per concurrent batch slot, and any lower value is raised to the
            number of slots, since a smaller queue would leave servers idle.
            After initialisation the attribute always holds the resolved value.
            A config-driven run rejects a too-small value when the config
            loads instead of raising it here.
        max_concurrent_batches_per_client: Maximum number of batches each
            server handles at once. This is independent of ``batch_size``.
            Defaults to four for high throughput.
        max_workers: Maximum worker threads used for batch annotation. It can
            exceed the initially available batch slots when additional servers
            are expected to join, allowing workers to wait for and immediately
            use those late-ready servers.
        batch_size: Maximum number of samples sent to a worker in one request.
        num_proc: Number of processes for dataset preprocessing.
        verbose: Whether to print progress information.

    Raises:
        ValueError: If no clients are given or ``queue_size`` is not positive.
        TypeError: If a client is not a vLLM server client.

    Examples:
        >>> from llm_annotator import VLLMOnlineClient, VLLMQueueAnnotator
        >>> clients = [  # doctest: +SKIP
        ...     VLLMOnlineClient(
        ...         model="Qwen/Qwen3-8B", base_url=f"http://{host}:8000/v1"
        ...     )
        ...     for host in ("node01", "node02")
        ... ]
        >>> with VLLMQueueAnnotator(
        ...     clients=clients, batch_size=64
        ... ) as anno:  # doctest: +SKIP
        ...     ds = anno.annotate_dataset(
        ...         output_dir="outputs/data",
        ...         prompt_template="Classify: {text}",
        ...         dataset_name="my-dataset",
        ...     )
    """

    clients: Sequence[Client[Any]]
    queue_size: int | None = None
    max_workers: int | None = None
    max_concurrent_batches_per_client: int = 4
    # Required in the base class but set to init=False here
    # since we derive it from the first client in the pool
    client: Client = field(init=False, repr=False)
    _client_pool: SimpleQueue[Client[Any]] = field(init=False, repr=False)
    _requested_queue_size: int | None = field(init=False, repr=False)
    _shutdown_started: Event = field(init=False, repr=False)
    _destroyed: Event = field(init=False, repr=False)
    _clients_lock: Lock = field(init=False, repr=False)
    _checked_out_clients: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the pool, derive the defaults, and fill the client queue.

        Raises:
            ValueError: If no clients are given, ``queue_size`` is not
                positive, or ``max_concurrent_batches_per_client`` is not
                positive.
            TypeError: If a client is not a vLLM server client.
        """
        # Explicit unbound call rather than a zero-argument `super()`:
        # `@dataclass(slots=True)` builds a *new* class object, and on Python
        # 3.12 the `__class__` cell captured by this method still points at the
        # pre-slots class, so `super()` raises TypeError. CPython fixed that in
        # 3.13; we support 3.12, so name the base class directly.
        Annotator.__post_init__(self)
        self._logger = get_logger("annotator.vllm_queue")

        if not self.clients:
            raise ValueError(
                "'clients' must contain at least one vLLM server client."
            )

        self.clients = list(self.clients)
        for client in self.clients:
            if getattr(client, "provider_type", None) != Provider.VLLM_ONLINE:
                raise TypeError(
                    "VLLMQueueAnnotator only supports vLLM server clients"
                    " (provider 'vllm_online'), got"
                    f" '{type(client).__name__}'."
                )

        # not used here but to satisfy the base class and type-checer
        self.client = self.clients[0]
        self.max_concurrent_batches_per_client = (
            self._resolve_max_concurrent_batches_per_client(
                self.max_concurrent_batches_per_client
            )
        )
        self.max_workers = max(self._max_workers, self.max_workers or 0)
        self._requested_queue_size = self.queue_size
        self.queue_size: int = self._resolve_queue_size(self.queue_size)

        # Load balancing: a batch is only dispatched once a client is free, so
        # a slow server never gets a backlog while another one idles.
        self._client_pool = SimpleQueue()
        self._shutdown_started = Event()
        self._destroyed = Event()
        self._clients_lock = Lock()
        self._checked_out_clients = 0
        self._rebuild_client_pool()

    @property
    def _max_workers(self) -> int:
        """Return the total number of batches the pool runs at once."""
        return len(self.clients) * self.max_concurrent_batches_per_client

    def _resolve_max_concurrent_batches_per_client(
        self, max_concurrent_batches_per_client: int
    ) -> int:
        """Validate the per-server concurrent-request limit."""
        if max_concurrent_batches_per_client < 1:
            raise ValueError(
                "'max_concurrent_batches_per_client' must be a positive"
                " integer."
            )
        return max_concurrent_batches_per_client

    @staticmethod
    def resolve_queue_size(queue_size: int | None, num_slots: int) -> int:
        """Turn a requested queue size into the effective one for a pool.

        This is the arithmetic behind the ``queue_size`` key, exposed so that
        a caller can report the effective value (as
        ``llm-annotate --describe-steps`` does) without building a pool first.

        Args:
            queue_size: Requested number of batches in flight, or ``None`` to
                derive it from the number of slots.
            num_slots: Concurrent batch slots in the pool, that is, servers
                times ``max_concurrent_batches_per_client``.

        Returns:
            The number of batches to keep in flight: ``num_slots`` at the very
            least, and ``QUEUE_BATCHES_PER_SLOT`` times that when nothing is
            requested.

        Raises:
            ValueError: If ``queue_size`` is given but not positive.

        Examples:
            >>> VLLMQueueAnnotator.resolve_queue_size(None, 8)
            32
            >>> VLLMQueueAnnotator.resolve_queue_size(64, 8)
            64
            >>> VLLMQueueAnnotator.resolve_queue_size(2, 8)
            8
        """
        if queue_size is None:
            return QUEUE_BATCHES_PER_SLOT * num_slots

        if queue_size < 1:
            raise ValueError(
                "'queue_size' must be a positive integer or None."
            )

        return max(queue_size, num_slots)

    def _resolve_queue_size(self, queue_size: int | None) -> int:
        """Resolve a queue size against this pool and report any clamping.

        Args:
            queue_size: Requested number of batches in flight, or ``None`` to
                derive it from the pool size.

        Returns:
            The number of batches to keep in flight.

        Raises:
            ValueError: If ``queue_size`` is given but not positive.
        """
        resolved = self.resolve_queue_size(queue_size, self._max_workers)
        if queue_size is not None and resolved > queue_size:
            self._logger.warning(
                f"'queue_size' ({queue_size}) is smaller than the number of"
                f" batches the pool runs at once ({self._max_workers}), which"
                " would leave servers idle. We're raising it to that number as a"
                " sensible minimal value."
            )
        return resolved

    def set_queue_size(self, queue_size: int | None) -> None:
        """Change how many batches are kept in flight.

        Assigning to ``queue_size`` directly would break the invariant that it
        always holds a *resolved* value, since ``None`` and values below the
        pool size are only normalised on the way in. Use this instead when a
        pool is reused for another workload.

        Args:
            queue_size: Requested number of batches in flight, or ``None`` to
                derive it from the pool size.

        Raises:
            ValueError: If ``queue_size`` is given but not positive.
        """
        self._requested_queue_size = queue_size
        self.queue_size = self._resolve_queue_size(queue_size)

    def add_client(self, client: Client[Any]) -> None:
        """Add a vLLM server that became ready after annotation started.

        Args:
            client: Ready vLLM server client to make available to workers.

        Raises:
            TypeError: If ``client`` is not a vLLM server client.
        """
        with self._clients_lock:
            self._add_client_locked(client)

    def add_client_for_base_url(
        self,
        base_url: str,
        client_factory: Callable[[str], Client[Any]],
    ) -> None:
        """Construct and add a late-ready server under the pool lock."""
        with self._clients_lock:
            if (
                self._shutdown_started.is_set()
                or self._destroyed.is_set()
                or self._has_client_base_url_locked(base_url)
            ):
                return
        client = client_factory(base_url)
        with self._clients_lock:
            if (
                self._shutdown_started.is_set()
                or self._destroyed.is_set()
                or self._has_client_base_url_locked(base_url)
            ):
                client.destroy()
                return
            self._add_client_locked(client)

    def _has_client_base_url_locked(self, base_url: object) -> bool:
        return base_url is not None and any(
            getattr(existing, "base_url", None) == base_url
            for existing in self.clients
        )

    def _add_client_locked(self, client: Client[Any]) -> None:
        if getattr(client, "provider_type", None) != Provider.VLLM_ONLINE:
            raise TypeError(
                "VLLMQueueAnnotator only supports vLLM server clients"
                " (provider 'vllm_online'), got"
                f" '{type(client).__name__}'."
            )
        if self._shutdown_started.is_set() or self._destroyed.is_set():
            client.destroy()
            return
        if self._has_client_base_url_locked(getattr(client, "base_url", None)):
            client.destroy()
            return
        cast(list[Client[Any]], self.clients).append(client)
        self.max_workers = max(self.max_workers or 0, self._max_workers)
        for _ in range(self.max_concurrent_batches_per_client):
            self._client_pool.put(client)
        self.set_queue_size(self._requested_queue_size)

    @property
    def is_shutting_down(self) -> bool:
        """Whether the annotator has begun releasing its clients."""
        return self._shutdown_started.is_set()

    def wait_for_shutdown(self, timeout: float) -> bool:
        """Block until the pool is shutting down or the timeout elapses."""
        return self._shutdown_started.wait(timeout)

    def client_count(self) -> int:
        """Return the current number of pool members."""
        with self._clients_lock:
            return len(self.clients)

    def client_base_urls(self) -> set[str]:
        """Return the base URLs currently registered in the pool."""
        with self._clients_lock:
            return {
                str(base_url)
                for client in self.clients
                if (base_url := getattr(client, "base_url", None)) is not None
            }

    def set_max_concurrent_batches_per_client(
        self, max_concurrent_batches_per_client: int
    ) -> None:
        """Change the concurrent-request limit for every server.

        This may only be called between annotation runs because it rebuilds
        the available-client queue.

        Args:
            max_concurrent_batches_per_client: Maximum simultaneous batch
                requests sent to each server.

        Raises:
            ValueError: If the requested limit is not positive.
            RuntimeError: If called while annotation is in progress.
        """
        with self._clients_lock:
            total_slots = (
                len(self.clients) * self.max_concurrent_batches_per_client
            )
            if (
                self._checked_out_clients
                or self._client_pool.qsize() != total_slots
            ):
                raise RuntimeError(
                    "'max_concurrent_batches_per_client' can only be changed"
                    " between annotation runs."
                )
            self.max_concurrent_batches_per_client = (
                self._resolve_max_concurrent_batches_per_client(
                    max_concurrent_batches_per_client
                )
            )
            self.queue_size = self._resolve_queue_size(self.queue_size)
            self._rebuild_client_pool()

    def _rebuild_client_pool(self) -> None:
        """Recreate the available-client queue in round-robin order."""
        self._client_pool = SimpleQueue()
        for _ in range(self.max_concurrent_batches_per_client):
            for client in self.clients:
                self._client_pool.put(client)

    def destroy(self) -> None:
        """Clean up the resources of every client in the pool. Since clients
        can only be ``VLLMOnlineClient``s, the impact is likely minimal:
        that class has no meaningful ``destroy`` of its own. It inherits
        ``OpenAIClient``'s, which only does batch-related cleanup, and vLLM
        does not support the OpenAI Batch API.

        Every client is destroyed even if some of them raise; the first error
        is re-raised afterwards.

        Raises:
            BaseException: The first error raised by a client, if any.
        """
        self._shutdown_started.set()
        with self._clients_lock:
            self._destroyed.set()
            clients = list(self.clients)
        first_error: BaseException | None = None
        for client in clients:
            try:
                client.destroy()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                self._logger.warning(
                    f"Failed to destroy client '{type(client).__name__}': {exc}"
                )
                if first_error is None:
                    first_error = exc

        if first_error is not None:
            raise first_error

    def _warm_up(
        self,
        *,
        system_message: str | None = None,
        prompt_prefix: str | None = None,
        options: ProviderRuntimeOptions | None = None,
    ) -> None:
        """Warm up every server in the pool in parallel.

        Args:
            system_message: Optional system message shared across requests.
            prompt_prefix: Optional fixed prefix that starts every user turn.
            options: Optional generation options used for the warm-up call.
        """
        with ThreadPoolExecutor(max_workers=len(self.clients)) as pool:
            futures = [
                pool.submit(
                    client.warm_up,
                    system_message=system_message,
                    prompt_prefix=prompt_prefix,
                    options=options,
                )
                for client in self.clients
            ]
            for future in futures:
                future.result()

    def _annotate_batch_on_free_client(
        self,
        batch: dict[str, list[Any]],
        *,
        options: ProviderRuntimeOptions | None,
        gen_kwargs: dict[str, Any] | None,
        task_prefix: str,
        validate_fn: Callable | None,
        postprocess_fn: Callable | None,
        num_retries_invalid: int,
    ) -> tuple[dict[str, list[Any]], list[dict[str, Any]]]:
        """Annotate one batch on the first available client in the pool.

        Args:
            batch: Dictionary containing batch data with messages samples.
            options: Runtime options passed to the client.
            gen_kwargs: Extra request parameters merged over ``options``,
                for anything the options dataclass does not name.
            task_prefix: String prefix to use for internal column names.
            validate_fn: Optional custom validation function.
            postprocess_fn: Optional postprocessing function.
            num_retries_invalid: Number of retries for invalid outputs.

        Returns:
            The batch together with one result per sample, in order.

        Raises:
            RuntimeError: If the pool is shutting down.
            TooManyConsecutiveFailedBatchesError: If the last server of the
                pool was evicted.
        """
        while True:
            client = self._acquire_client()
            is_dead = False
            try:
                results = self._annotate_batch(
                    batch=batch,
                    client=client,
                    options=options,
                    gen_kwargs=gen_kwargs,
                    task_prefix=task_prefix,
                    validate_fn=validate_fn,
                    postprocess_fn=postprocess_fn,
                    num_retries_invalid=num_retries_invalid,
                )
                # The errors of an entirely failed batch are only kept when
                # the server is healthy. A server that stopped answering
                # fails every batch regardless of the samples, so its batch
                # is sent to another server.
                is_dead = (
                    self._all_errored(results, task_prefix)
                    and not client.is_healthy()
                )
            finally:
                with self._clients_lock:
                    self._checked_out_clients -= 1
                    if is_dead:
                        self._evict_client_locked(client)
                    elif not self._destroyed.is_set() and any(
                        client is member for member in self.clients
                    ):
                        self._client_pool.put(client)

            if not is_dead:
                return batch, results

    def _acquire_client(self) -> Client[Any]:
        """Block until a client has a free batch slot and check it out.

        Returns:
            The checked-out client.

        Raises:
            RuntimeError: If the pool is shutting down.
        """
        while True:
            if self._shutdown_started.is_set():
                raise RuntimeError(
                    "Cannot start a new batch after shutdown begins."
                )
            try:
                client = self._client_pool.get(timeout=1)
                break
            except Empty:
                continue
        with self._clients_lock:
            if self._shutdown_started.is_set():
                raise RuntimeError(
                    "Cannot start a new batch after shutdown begins."
                )
            self._checked_out_clients += 1
        return client

    def _evict_client_locked(self, client: Client[Any]) -> None:
        """Remove a server that stopped answering from the pool.

        The pool watcher of a config-driven run admits the server again once
        its ``/health`` endpoint answers.

        Args:
            client: The client of the server to remove.

        Raises:
            TooManyConsecutiveFailedBatchesError: If no server is left.
        """
        clients = cast(list[Client[Any]], self.clients)
        if not any(client is member for member in clients):
            return
        self.clients = [member for member in clients if member is not client]

        free_slots = []
        while True:
            try:
                free_slots.append(self._client_pool.get_nowait())
            except Empty:
                break
        for slot in free_slots:
            if slot is not client:
                self._client_pool.put(slot)

        base_url = getattr(client, "base_url", None)
        self._logger.warning(
            f"The vLLM server at '{base_url}' failed a whole batch and does"
            " not answer '/health'. Removed it from the pool;"
            f" {len(self.clients)} server(s) left. Its batches are sent to"
            " the other servers."
        )
        if not self.clients:
            raise TooManyConsecutiveFailedBatchesError(
                "No vLLM server in the pool answers '/health' any more, so"
                " the run stops. The rows of the failed batches were not"
                " written, so a resumed run annotates them."
            )

    def _iter_and_annotate_batches(
        self,
        *,
        prepared_dataset: Dataset,
        options: ProviderRuntimeOptions | None,
        gen_kwargs: dict[str, Any] | None = None,
        task_prefix: str = "",
        validate_fn: Callable | None = None,
        postprocess_fn: Callable | None = None,
        num_retries_invalid: int = 5,
    ) -> Iterator[tuple[dict[str, list[Any]], list[dict[str, Any]]]]:
        """Annotate the dataset over all clients, yielding batches as they finish.

        At most ``queue_size`` batches are in flight at any time and a new one
        is dispatched as soon as a finished batch has been handed to the
        caller, so memory stays bounded while every server stays busy until
        the dataset is exhausted. A server that joins the pool mid-run raises
        ``queue_size``, and the next top-up dispatches the extra batches, so a
        run that started on a partial pool grows into the full one.

        Args:
            prepared_dataset: The dataset still left to annotate.
            options: Runtime options passed to the clients.
            gen_kwargs: Extra request parameters merged over ``options``,
                for anything the options dataclass does not name.
            task_prefix: String prefix to use for internal column names.
            validate_fn: Optional custom validation function.
            postprocess_fn: Optional postprocessing function.
            num_retries_invalid: Number of retries for invalid outputs.

        Yields:
            ``(batch, results)`` in completion order, where ``batch`` is a
            column-oriented mini-batch dict like
            ``{col_name: [value_0, value_1, ...]}`` and ``results`` is a list
            of result dicts with one item per sample. They remain aligned by
            position, so ``results[i]`` corresponds to the sample at
            ``batch[col][i]`` for each column ``col``. Note that
            [`run_annotation`][llm_annotator.annotator.Annotator.run_annotation]'s
            consecutive-failed-batch count is therefore also in completion
            order here, not dataset order.
        """
        batches = prepared_dataset.iter(self.batch_size)

        pool = ThreadPoolExecutor(
            max_workers=cast(int, self.max_workers),
            thread_name_prefix="vllm-queue-worker",
        )
        pending: set[Future[Any]] = set()
        pbar = tqdm(
            total=ceil(len(prepared_dataset) / self.batch_size),
            desc=(
                f"Annotating (max_bs={self.batch_size},"
                f" servers={len(self.clients)})"
            ),
            unit="batch",
        )

        def _submit_next() -> bool:
            """Dispatch the next batch, if any is left.

            Returns:
                Whether a batch was dispatched.
            """
            batch = next(batches, None)
            if batch is None:
                return False
            pending.add(
                pool.submit(
                    self._annotate_batch_on_free_client,
                    batch,
                    options=options,
                    gen_kwargs=gen_kwargs,
                    task_prefix=task_prefix,
                    validate_fn=validate_fn,
                    postprocess_fn=postprocess_fn,
                    num_retries_invalid=num_retries_invalid,
                )
            )
            return True

        def _fill_queue() -> None:
            """Dispatch batches until the current queue size is reached."""
            # `__post_init__` always resolves `queue_size` to a positive int,
            # and a server that joins mid-run raises it, so it is read again
            # on every top-up instead of once.
            while len(pending) < cast(int, self.queue_size) and _submit_next():
                pass

        try:
            _fill_queue()

            # start retrieving first results and replacing the completed
            # jobs with new ones until the work is done
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    batch, results = future.result()
                    pbar.update(1)
                    yield batch, results
                _fill_queue()
        finally:
            pbar.close()
            pool.shutdown(wait=False, cancel_futures=True)


__all__ = [
    "Annotator",
    "SelectionRecord",
    "VLLMQueueAnnotator",
    "destroy_on_error",
    "is_retried_error",
]
