from __future__ import annotations

import dataclasses
import json
import shutil
import string
from dataclasses import dataclass, field
from functools import wraps
from math import ceil
from os import cpu_count
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

from datasets import (
    Dataset,
    concatenate_datasets,
    get_dataset_split_names,
    load_dataset,
)
from huggingface_hub import (
    create_branch,
    create_repo,
    delete_branch,
    upload_large_folder,
)
from tqdm import tqdm

from llm_annotator.clients.base import Client, ProviderRuntimeOptions, Response
from llm_annotator.clients.vllm_offline_client import VLLMOfflineClient
from llm_annotator.logging_utils import get_logger
from llm_annotator.utils import (
    ensure_returns_bool,
    ensure_returns_dict,
    extract_prompt_prefix,
    get_lib_versions,
    remove_empty_jsonl_files,
    retry,
)


# Set a sensible default: cpu_count-1 cores
# but at least 1 at most 8 to avoid overloading the system
DEFAULT_CPU_COUNT = min(8, max(1, (cpu_count() or 1) - 1))

CACHED_DATASET_BRANCH = "prepared_cache"


def destroy_on_error(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate an ``Annotator`` method to call :meth:`~Annotator.destroy` on any exception.

    Catches ``BaseException`` (including ``KeyboardInterrupt`` and ``SystemExit``)
    so resources are freed even on forced termination. The original exception
    is always re-raised after the cleanup attempt.

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

    This class provides a framework for annotating datasets using large language
    models via a pluggable :class:`~llm_annotator.clients.base.Client`. It handles
    dataset loading, processing, and output generation with support for batching
    and uploading to Hugging Face Hub.

    The ``Annotator`` class has four public entry points:

        * :meth:`prepare_data`. Apply prompt templates, sorting, and caching
            without running inference. Backs prepared artifacts up to Hugging Face
            Hub if ``prepared_hub_id`` is provided.
        * :meth:`run_annotation`. Run inference only, consuming data returned by
            :meth:`prepare_data` or loaded from a local path or Hub repo.
        * :meth:`annotate_dataset`. Convenience wrapper that calls
            :meth:`prepare_data` and then :meth:`run_annotation` in one call.
        * :meth:`generate_dataset`. Generate a new dataset from scratch by calling
            :meth:`annotate_dataset` over a synthetic prompt dataset.

    The staged :meth:`prepare_data` + :meth:`run_annotation` pattern is
    recommended for large-scale or cluster (SLURM) workflows. When
    ``prepared_hub_id`` is provided, prepared artifacts are stored on
    Hugging Face Hub and restored automatically on the next call, so a
    failed generation job can be restarted without repeating the
    preparation step.

    Args:
        client: An initialised :class:`~llm_annotator.clients.base.Client`
            instance that performs the actual generation.
        batch_size: Number of samples per inference batch.
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

    def __post_init__(self) -> None:
        """Initialize a package-scoped logger for annotator runtime messages."""
        self._logger = get_logger("annotator")

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
        pdout: Path,
        idx_column: str,
        dataset_split: str | None = None,
        dataset_config: str | None = None,
    ) -> set[int]:
        """Get indices of samples that have already been processed.

        Scans existing output files to determine which samples can be skipped
        in resumed processing.

        Args:
            pdout: Output directory path to scan for existing files.

        Returns:
            Set of indices that have already been processed.
        """
        ids_done = set()
        if pdout.exists() and pdout.is_dir():
            for pfin in pdout.glob("*.jsonl"):
                if pfin.stat().st_size == 0:
                    continue
                ds = Dataset.from_json(str(pfin))

                if dataset_split and "dataset_split" in ds.column_names:
                    ds = ds.filter(
                        lambda s: s["dataset_split"] == dataset_split
                    )

                if dataset_config and "dataset_config" in ds.column_names:
                    ds = ds.filter(
                        lambda s: s["dataset_config"] == dataset_config
                    )

                if idx_column not in ds.column_names:
                    raise ValueError(
                        f"Expected index column '{idx_column}' not found in existing output file '{pfin}'."
                        " Cannot determine which samples to skip on resume. Please check your configuration and ensure the index column is included in the output."
                    )
                ids_done.update(ds.unique(idx_column))

        return ids_done

    def _load_dataset(
        self,
        *,
        prompt_template: str,
        idx_column: str,
        dataset_name: str | None = None,
        dataset: Dataset | None = None,
        dataset_config: str | None = None,
        data_dir: str | None = None,
        dataset_split: str | None = None,
        max_num_samples: int | None = None,
        shuffle_seed: int | None = None,
        prompt_fields: Iterable[str] = (),
        task_prefix: str = "",
        sort_by_length: bool
        | Literal["shortest_first", "longest_first"] = False,
        system_message: str | None = None,
        preprocess_fn: Callable | None = None,
    ) -> Dataset:
        """Load and preprocess the dataset for annotation.

        Handles dataset loading from various sources, applies prompt templates,
        and manages caching for efficient resumption of interrupted jobs.

        Args:
            dataset_name: Name or path of the dataset to load.
            dataset: Pre-loaded dataset to use instead of loading from name/path.
            dataset_config: Dataset configuration name (optional).
            data_dir: Data directory for local datasets (optional).
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

        Returns:
            The loaded and preprocessed dataset ready for annotation.

        Raises:
            ValueError: If configuration is invalid or required fields are missing.
        """

        pipeline_loaded = getattr(self.client, "_pipeline_loaded", False)

        if (
            self.num_proc is not None
            and isinstance(self.client, VLLMOfflineClient)
            and pipeline_loaded
        ):
            self._logger.warning(
                "num_proc>1 cannot be used with VLLMOfflineClient because the loaded model "
                "cannot be pickled for multiprocessing. Setting num_proc=None."
            )
            self.num_proc = None

        if dataset is not None and dataset_name is not None:
            raise ValueError(
                "Provide only one of 'dataset' or 'dataset_name', not both."
            )

        if dataset is None and dataset_name is None:
            raise ValueError(
                "Either 'dataset' or 'dataset_name' must be provided."
            )

        if max_num_samples is not None and max_num_samples <= 0:
            raise ValueError(
                "'max_num_samples' must be a positive integer or None"
            )

        # Split verification and defaulting
        if dataset_name:
            split_names = get_dataset_split_names(
                dataset_name, config_name=dataset_config
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

        if dataset is None:
            dataset = load_dataset(
                dataset_name,
                name=dataset_config,
                data_dir=data_dir,
                split=dataset_split,
            )

        # Add index column for tracking samples and resuming interrupted runs
        if idx_column in dataset.column_names:
            raise ValueError(
                f"Dataset already contains a column named '{idx_column}'."
                " Please specify a different 'idx_column' name that does not exist in the dataset."
            )

        dataset = dataset.add_column(idx_column, list(range(len(dataset))))

        if shuffle_seed is not None:
            dataset = dataset.shuffle(seed=shuffle_seed)

        if max_num_samples:
            dataset = dataset.select(range(min(max_num_samples, len(dataset))))

        # Validate that the dataset contains all fields required by the
        # prompt template. Tests expect a ValueError when a required
        # field is missing.
        if dataset is not None and prompt_fields:
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
            self._create_messages,
            num_proc=self.num_proc,
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
                    f"{task_prefix}messages_chars": len(json.dumps(msgs))
                },
                num_proc=self.num_proc,
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

    def _create_messages(
        self,
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

            And if an output_schema is provided, also:
                - Keys from the output_schema with their parsed values (or None if parsing failed).
                - A key '{prefix}valid_fields' indicating if all required fields were valid.
        """
        data: dict[str, Any] = {
            f"{task_prefix}response": response.text,
            f"{task_prefix}finish_reason": response.stop_reason,
            f"{task_prefix}num_tokens": response.num_output_tokens,
            f"{task_prefix}error": response.error,
            f"{task_prefix}error_type": response.error_type,
        }

        if response.error is not None:
            if not output_schema:
                return data

            result = dict.fromkeys(output_schema.get("properties", {}).keys())
            return {
                **data,
                f"{task_prefix}valid_fields": False,
                **result,
            }

        if not output_schema:
            return data

        valid_fields = None
        result = dict.fromkeys(output_schema.get("properties", {}).keys())
        try:
            parsed_response = json.loads(response.text)
        except json.JSONDecodeError:
            valid_fields = False
        else:
            result = parsed_response

            if "required" in output_schema:
                required_keys = output_schema["required"]
                valid_fields = all(key in result for key in required_keys)

        return {
            **data,
            f"{task_prefix}valid_fields": valid_fields,
            **result,
        }

    def _process_batch(
        self,
        *,
        batch: dict[str, list[Any]],
        options: ProviderRuntimeOptions | None,
        task_prefix: str = "",
        validate_fn: Callable | None = None,
        postprocess_fn: Callable | None = None,
    ) -> list[dict[str, Any]]:
        """Process a batch of samples through the client.

        Takes a batch of messages samples, runs inference, and processes
        the outputs using the :meth:`_process_output` method.

        Args:
            batch: Dictionary containing batch data with messages samples.
            options: Runtime options passed to the client.
            task_prefix: String prefix to use for internal column names.
            validate_fn: Optional custom validation function that takes a processed
                output dictionary and must return a boolean indicating validity. If a JSON schema
                was passed, and the fields were invalid, this function will not be called
                and `valid` will be set to False directly.
            postprocess_fn: Optional function to postprocess each sample after annotation.

        Returns:
            List of processed output dictionaries for each sample in the batch.
        """
        output_schema = options.json_schema if options is not None else None
        responses = self.client.batch_generate(
            messages=batch[f"{task_prefix}messages"],
            options=options,
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
                    " but if it happens often it suggests a deeper issue,"
                    " such as too few 'max_tokens' in options."
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

    def prepare_data(
        self,
        output_dir: str | Path,
        prompt_template: str,
        *,
        dataset_name: str | None = None,
        dataset: Dataset | None = None,
        dataset_config: str | None = None,
        data_dir: str | None = None,
        dataset_split: str | None = None,
        max_num_samples: int | None = None,
        shuffle_seed: int | None = None,
        preprocess_fn: Callable | None = None,
        prompt_field_swapper: dict[str, str] | None = None,
        idx_column: str = "idx",
        task_prefix: str = "",
        sort_by_length: bool
        | Literal["shortest_first", "longest_first"] = False,
        system_message: str | None = None,
        prepared_hub_id: str | None = None,
        force_data_preparation: bool = False,
    ) -> tuple[Dataset, Path | None, str | None]:
        """Prepare input data for annotation without running generation.

        The method reuses local prepared data first, then optionally restores
        prepared data from Hugging Face Hub, and finally falls back to building
        the prepared dataset from source.

        Args:
            output_dir: Directory where prepared artifacts are stored.
            prompt_template: Prompt template used to build chat messages.
            dataset_name: Name or path of the dataset to load.
            dataset: Pre-loaded dataset to use instead of loading from name/path.
            dataset_config: Dataset configuration name (optional).
            data_dir: Data directory for local datasets (optional).
            dataset_split: Specific split to load (optional).
            max_num_samples: Maximum number of samples to prepare.
            shuffle_seed: Seed for dataset shuffling.
            preprocess_fn: Optional function to preprocess the dataset after loading and before applying the prompt template.
            prompt_field_swapper: Optional mapping to replace template fields.
            idx_column: Column name used as unique identifier. Must not exist in the input dataset.
            task_prefix: Prefix for internal columns and artifact names.
            sort_by_length: Whether to sort prompts by length.
            system_message: Optional system message for chat prompts.
            prepared_hub_id: Optional Hugging Face dataset ID for prepared-data
                backup and restore. Will be stored in the "prepared_cache" configuration of the repo if provided.
            force_data_preparation: Whether to rebuild prepared data even when
                local or Hub artifacts already exist.

        Returns:
            Tuple of prepared dataset, local prepared-data path when available,
            and prepared Hugging Face dataset ID when available.
        """
        pdout = Path(output_dir)
        pdout.mkdir(exist_ok=True, parents=True)

        cached_input_path = pdout / f"{task_prefix}cached_input_dataset"

        prompt_field_swapper = prompt_field_swapper or {}
        for fld, value in prompt_field_swapper.items():
            prompt_template = prompt_template.replace(
                f"{{{fld}}}", f"{{{value}}}"
            )

        # Attempt loading from local cache at
        # pdout / f"{task_prefix}cached_input_dataset"
        if (
            cached_input_path.exists()
            and cached_input_path.is_dir()
            and any(cached_input_path.glob("*"))
        ):
            if force_data_preparation:
                shutil.rmtree(cached_input_path)
            else:
                cached_ds = Dataset.load_from_disk(cached_input_path)
                return cached_ds, cached_input_path, prepared_hub_id

        # Attempt loading from the hub
        if prepared_hub_id:
            if force_data_preparation:
                try:
                    delete_branch(
                        prepared_hub_id,
                        branch=CACHED_DATASET_BRANCH,
                        repo_type="dataset",
                    )
                except Exception:
                    pass
            else:
                cached_ds = load_dataset(
                    prepared_hub_id,
                    revision=CACHED_DATASET_BRANCH,
                    split="train",
                )
                return cached_ds, None, prepared_hub_id

        # ... and if all of that fails, prepare the dataset from the source
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
            dataset_name=dataset_name,
            dataset=dataset,
            dataset_config=dataset_config,
            data_dir=data_dir,
            dataset_split=dataset_split,
            max_num_samples=max_num_samples,
            shuffle_seed=shuffle_seed,
            prompt_fields=prompt_fields,
            task_prefix=task_prefix,
            sort_by_length=sort_by_length,
            system_message=system_message,
            preprocess_fn=preprocess_fn,
        )

        self._logger.info(
            f"Saving prepared data to local cache at '{cached_input_path}' for faster resumption on failure..."
        )
        prepared_dataset.save_to_disk(cached_input_path)
        if prepared_hub_id:
            self._logger.info(
                f"Uploading prepared data to Hugging Face Hub at '{prepared_hub_id}' for backup and easy restore..."
            )
            prepared_dataset.push_to_hub(
                prepared_hub_id,
                revision=CACHED_DATASET_BRANCH,
                split="train",
                private=True,
            )

        return prepared_dataset, cached_input_path, prepared_hub_id

    @destroy_on_error
    def run_annotation(
        self,
        output_dir: str | Path,
        prompt_template: str,
        *,
        prepared_dataset: Dataset | None = None,
        prepared_data_path: str | Path | None = None,
        prepared_hub_id: str | None = None,
        resume_from_hub_id: str | None = None,
        new_hub_id: str | None = None,
        overwrite: bool = False,
        dataset_split: str | None = None,
        dataset_config: str | None = None,
        keep_columns: str | Iterable[str] | bool | None = None,
        options: ProviderRuntimeOptions | None = None,
        output_schema: str | dict[str, Any] | None = None,
        idx_column: str = "idx",
        upload_every_n_samples: int | None = 0,
        max_samples_per_output_file: int = 0,
        task_prefix: str = "",
        validate_fn: Callable | None = None,
        postprocess_fn: Callable | None = None,
        num_retries_invalid: int = 5,
        system_message: str | None = None,
        keep_idx_column: bool = False,
    ) -> Dataset:
        """Run model generation on already prepared annotation inputs.

        Args:
            output_dir: Directory where annotation output is written.
            prompt_template: Prompt template used for warm-up metadata.
            prepared_dataset: Pre-prepared dataset with messages column.
            prepared_data_path: Local path to prepared data on disk.
            prepared_hub_id: Hugging Face dataset ID with prepared data.
            resume_from_hub_id: Optional Hugging Face dataset ID to restore
                generation checkpoints from.
            new_hub_id: Optional Hugging Face dataset ID for generation uploads.
            overwrite: Whether to overwrite existing output directory.
            dataset_split: Dataset split used for skip filtering.
            dataset_config: Dataset config used for skip filtering.
            keep_columns: Columns to keep in output. ``True`` for all.
            options: Runtime options passed to the client.
            output_schema: Convenience JSON schema input. When provided, it is
                injected into ``options.json_schema``.
            idx_column: Column name used as unique identifier.
            upload_every_n_samples: Upload to Hub every N samples.
            max_samples_per_output_file: Maximum samples per output file.
            task_prefix: Prefix for internal columns and file names.
            validate_fn: Optional custom validation function.
            postprocess_fn: Optional postprocessing function that takes in a sample and must return a dict.
            num_retries_invalid: Number of retries for invalid outputs.
            system_message: Optional system message for chat prompts.
            keep_idx_column: Whether to keep idx column in final dataset.

        Returns:
            Final concatenated annotation dataset.

        Raises:
            ValueError: If no prepared data source can be resolved.
        """
        if resume_from_hub_id is not None:
            if not resume_from_hub_id.strip() or "/" not in resume_from_hub_id:
                raise ValueError(
                    "'resume_from_hub_id' must be a Hugging Face dataset ID like "
                    "'owner/name'."
                )
            if overwrite:
                raise ValueError(
                    "Cannot set 'overwrite=True' when 'resume_from_hub_id' is set."
                )

        upload_every_n_samples = upload_every_n_samples or 0
        if (
            max_samples_per_output_file is not None
            and max_samples_per_output_file < 0
        ):
            raise ValueError(
                "'max_samples_per_output_file' must be None or 0 or a positive integer"
            )
        max_samples_per_output_file = max_samples_per_output_file or 0

        if upload_every_n_samples < 0 or not isinstance(
            upload_every_n_samples, int
        ):
            raise ValueError(
                "upload_every_n_samples must be a positive integer or 0"
            )
        if upload_every_n_samples > 0 and not new_hub_id:
            raise ValueError(
                "If upload_every_n_samples is set, new_hub_id must be provided"
            )

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

        pdout = Path(output_dir)

        if prepared_dataset is None and prepared_hub_id:
            try:
                prepared_dataset = load_dataset(
                    prepared_hub_id,
                    revision=CACHED_DATASET_BRANCH,
                    split="train",
                )
            except Exception as exc:
                self._logger.warning(
                    f"Failed to load prepared dataset from Hub ID '{prepared_hub_id}' with revision '{CACHED_DATASET_BRANCH}'."
                    f" This might be because the dataset or revision does not exist, or due to network issues. Error: {exc}"
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

        if prepared_dataset is None:
            raise ValueError(
                "No prepared data found. Provide 'prepared_dataset', "
                "'prepared_data_path' (locally saved dataset), or 'prepared_hub_id' (cloud-saved dataset)."
            )

        if idx_column not in prepared_dataset.column_names:
            raise ValueError(
                f"Expected index column '{idx_column}' not found in prepared dataset."
                " This column is required for tracking processed samples and resuming on failure."
                " Please ensure the prepared dataset includes the index column with name matching 'idx_column' argument."
            )

        # Only empty the output directory after potentially reading the cached input
        if pdout.is_dir() and overwrite:
            # Remove everything except the prepared data
            for item in pdout.glob("*"):
                if item.is_dir() and item != prepared_data_path:
                    shutil.rmtree(item)
                else:
                    item.unlink()

        pdout.mkdir(exist_ok=True, parents=True)

        # Add version info
        pdout.joinpath("_version.json").write_text(
            json.dumps(get_lib_versions(), indent=4), encoding="utf-8"
        )

        # Get indices from the local
        skip_idxs = self._get_skip_idxs(
            pdout=pdout,
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
                pdout=pdout,
                idx_column=idx_column,
                new_hub_id=new_hub_id,
                keep_idx_column=keep_idx_column,
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

        prompt_template_prefix = extract_prompt_prefix(prompt_template)

        if new_hub_id:
            self.push_dir_to_hub(pdout, new_hub_id=new_hub_id)

        pfout = self.get_pfout_name(
            pdout=pdout,
            max_samples_per_output_file=max_samples_per_output_file,
            processed_n_samples=processed_n_samples,
        )
        fhout = pfout.open("a", encoding="utf-8")

        self.client.warm_up(
            system_message=system_message,
            prompt_prefix=prompt_template_prefix,
            options=options,
        )

        total_num_batches = ceil(len(prepared_dataset) / self.batch_size)
        for batch in tqdm(
            prepared_dataset.iter(self.batch_size),
            total=total_num_batches,
            desc=f"Annotating (max_bs={self.batch_size})",
            unit="batch",
        ):
            results = self._process_batch(
                batch=batch,
                options=options,
                task_prefix=task_prefix,
                validate_fn=validate_fn,
                postprocess_fn=postprocess_fn,
            )

            if num_retries_invalid > 0:
                invalid_indices = [
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

                n_retries = 0
                while invalid_indices and n_retries < num_retries_invalid:
                    n_retries += 1
                    if self.verbose:
                        self._logger.info(
                            f"Retrying {len(invalid_indices):,} invalid samples (attempt {n_retries}/{num_retries_invalid})..."
                        )

                    retry_batch = {
                        k: [v[i] for i in invalid_indices]
                        for k, v in batch.items()
                    }
                    retry_results = self._process_batch(
                        batch=retry_batch,
                        options=options,
                        task_prefix=task_prefix,
                        validate_fn=validate_fn,
                    )

                    for local_idx, global_idx in enumerate(invalid_indices):
                        results[global_idx] = retry_results[local_idx]

                    invalid_indices = [
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

                    if self.verbose:
                        if (
                            invalid_indices
                            and n_retries == num_retries_invalid
                        ):
                            self._logger.warning(
                                f"After {n_retries}/{num_retries_invalid} attempts, {len(invalid_indices):,} samples are still invalid. Skipping..."
                            )

            batch_size = len(batch[idx_column])
            if keep_columns is True:
                inputs = [
                    {k: v[i] for k, v in batch.items()}
                    for i in range(batch_size)
                ]
            else:
                inputs = [
                    {
                        k: v[i]
                        for k, v in batch.items()
                        if k in keep_columns  # type: ignore[operator]
                    }
                    for i in range(batch_size)
                ]

            for result_idx, res in enumerate(results):
                inp = inputs[result_idx]
                data_sample = {**inp, **res}
                fhout.write(json.dumps(data_sample) + "\n")
                fhout.flush()
                processed_n_samples += 1

                if (
                    upload_every_n_samples > 0
                    and processed_n_samples % upload_every_n_samples == 0
                ):
                    fhout.close()
                    remove_empty_jsonl_files(pdout)
                    if new_hub_id:
                        self.push_dir_to_hub(pdout, new_hub_id=new_hub_id)
                    pfout = self.get_pfout_name(
                        pdout=pdout,
                        max_samples_per_output_file=max_samples_per_output_file,
                        processed_n_samples=processed_n_samples,
                    )
                    fhout = pfout.open("a", encoding="utf-8")

        fhout.close()
        remove_empty_jsonl_files(pdout)
        if new_hub_id and upload_every_n_samples > 0:
            self.push_dir_to_hub(pdout, new_hub_id=new_hub_id)

        return self._post_annotate(
            pdout=pdout,
            idx_column=idx_column,
            new_hub_id=new_hub_id,
            keep_idx_column=keep_idx_column,
        )

    @destroy_on_error
    def annotate_dataset(
        self,
        output_dir: str | Path,
        prompt_template: str | None = None,
        *,
        full_prompt_template: str | None = None,
        dataset_name: str | None = None,
        dataset: Dataset | None = None,
        dataset_config: str | None = None,
        data_dir: str | None = None,
        dataset_split: str | None = None,
        max_num_samples: int | None = None,
        shuffle_seed: int | None = None,
        preprocess_fn: Callable | None = None,
        prompt_field_swapper: dict[str, str] | None = None,
        idx_column: str = "idx",
        task_prefix: str = "",
        sort_by_length: bool
        | Literal["shortest_first", "longest_first"] = False,
        system_message: str | None = None,
        prepared_hub_id: str | None = None,
        force_data_preparation: bool = False,
        new_hub_id: str | None = None,
        overwrite: bool = False,
        keep_columns: str | Iterable[str] | bool | None = None,
        options: ProviderRuntimeOptions | None = None,
        output_schema: str | dict[str, Any] | None = None,
        upload_every_n_samples: int | None = 0,
        max_samples_per_output_file: int = 0,
        validate_fn: Callable | None = None,
        postprocess_fn: Callable | None = None,
        num_retries_invalid: int = 5,
        keep_idx_column: bool = False,
    ) -> Dataset:
        """Annotate an existing dataset in one call.

        This is a convenience wrapper around :meth:`prepare_data` and
        :meth:`run_annotation` for callers that prefer a single entry point.

        Args:
            output_dir: Directory where annotation output is written.
            prompt_template: Prompt template with dataset fields. Defaults to
                ``full_prompt_template`` when provided.
            full_prompt_template: Backwards-compatible alias for
                ``prompt_template``.
            dataset_name: Name or path of the dataset to load.
            dataset: Pre-loaded dataset to annotate instead of loading one.
            dataset_config: Dataset configuration name.
            data_dir: Data directory for local datasets.
            dataset_split: Dataset split to load.
            max_num_samples: Maximum number of samples to annotate.
            shuffle_seed: Seed for dataset shuffling.
            preprocess_fn: Optional preprocessing callback.
            prompt_field_swapper: Optional mapping that renames prompt fields.
            idx_column: Column name used as the stable sample identifier.
            task_prefix: Prefix for internal column names and output files.
            sort_by_length: Whether to sort prompts by length.
            system_message: Optional system message for the chat prompt.
            prepared_hub_id: Optional Hub dataset ID for prepared-data cache.
            force_data_preparation: Rebuild prepared data even if cached.
            new_hub_id: Optional Hub dataset ID for annotation outputs.
            overwrite: Whether to overwrite the output directory.
            keep_columns: Columns to keep in the final dataset.
            options: Runtime options passed to the client.
            output_schema: Optional JSON schema for structured output.
            upload_every_n_samples: Upload checkpoint cadence.
            max_samples_per_output_file: Maximum samples per output file.
            validate_fn: Optional validation callback.
            postprocess_fn: Optional postprocessing callback.
            num_retries_invalid: Number of retries for invalid outputs.
            keep_idx_column: Whether to keep the index column in the result.

        Returns:
            The concatenated annotation dataset.

        Raises:
            TypeError: If no prompt template is provided.
        """
        if prompt_template is None:
            prompt_template = full_prompt_template
        elif (
            full_prompt_template is not None
            and full_prompt_template != prompt_template
        ):
            raise ValueError(
                "Provide only one of 'prompt_template' or 'full_prompt_template'."
            )

        if prompt_template is None:
            raise TypeError(
                "'prompt_template' or 'full_prompt_template' must be provided."
            )

        prepared_dataset, _, _ = self.prepare_data(
            output_dir=output_dir,
            prompt_template=prompt_template,
            dataset_name=dataset_name,
            dataset=dataset,
            dataset_config=dataset_config,
            data_dir=data_dir,
            dataset_split=dataset_split,
            max_num_samples=max_num_samples,
            shuffle_seed=shuffle_seed,
            preprocess_fn=preprocess_fn,
            prompt_field_swapper=prompt_field_swapper,
            idx_column=idx_column,
            task_prefix=task_prefix,
            sort_by_length=sort_by_length,
            system_message=system_message,
            prepared_hub_id=prepared_hub_id,
            force_data_preparation=force_data_preparation,
        )

        return self.run_annotation(
            output_dir=output_dir,
            prompt_template=prompt_template,
            prepared_dataset=prepared_dataset,
            new_hub_id=new_hub_id,
            overwrite=overwrite,
            keep_columns=keep_columns,
            options=options,
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
        )

    @destroy_on_error
    def generate_dataset(
        self,
        output_dir: str | Path,
        prompts: str | Sequence[str],
        *,
        prompt_prefix: str | None = None,
        new_hub_id: str | None = None,
        overwrite: bool = False,
        options: ProviderRuntimeOptions | None = None,
        max_num_samples: int | None = None,
        output_schema: str | dict[str, Any] | None = None,
        idx_column: str = "idx",
        upload_every_n_samples: int | None = 0,
        max_samples_per_output_file: int = 0,
        task_prefix: str = "",
        validate_fn: Callable | None = None,
        postprocess_fn: Callable | None = None,
        num_retries_invalid: int = 5,
        keep_idx_column: bool = False,
    ) -> Dataset:
        """Generate a new dataset from prompts.

        Args:
            output_dir: Directory where annotation output is written.
            prompts: A single prompt or a sequence of prompts.
            prompt_prefix: Optional shared prefix used for prefix caching.
            new_hub_id: Optional Hub dataset ID for annotation outputs.
            overwrite: Whether to overwrite the output directory.
            options: Runtime options passed to the client.
            max_num_samples: Number of times to repeat a single prompt.
            output_schema: Optional JSON schema for structured output.
            idx_column: Column name used as the stable sample identifier.
            upload_every_n_samples: Upload checkpoint cadence.
            max_samples_per_output_file: Maximum samples per output file.
            task_prefix: Prefix for internal column names and output files.
            validate_fn: Optional validation callback.
            postprocess_fn: Optional postprocessing callback.
            num_retries_invalid: Number of retries for invalid outputs.
            keep_idx_column: Whether to keep the index column in the result.

        Returns:
            The concatenated annotation dataset.

        Raises:
            ValueError: If no prompts are provided.
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
        )

        return self.run_annotation(
            output_dir=output_dir,
            prompt_template=prompt_template,
            prepared_dataset=prepared_dataset,
            new_hub_id=new_hub_id,
            overwrite=overwrite,
            options=options,
            output_schema=output_schema,
            idx_column=idx_column,
            upload_every_n_samples=upload_every_n_samples,
            max_samples_per_output_file=max_samples_per_output_file,
            task_prefix=task_prefix,
            validate_fn=validate_fn,
            postprocess_fn=postprocess_fn,
            num_retries_invalid=num_retries_invalid,
            keep_idx_column=keep_idx_column,
        )

    def _post_annotate(
        self,
        *,
        pdout: Path,
        idx_column: str,
        new_hub_id: str | None = None,
        keep_idx_column: bool = False,
    ) -> Dataset:
        """Clean up after annotation is complete.

        Removes empty output files and performs any final cleanup operations.

        Args:
            pdout: Output directory path to clean up.
            new_hub_id: Optional Hugging Face dataset ID for uploads.
            idx_column: Column name used as unique identifier.
            keep_idx_column: Whether to keep the idx_column in the final dataset before uploading and returning

        Returns:
            The concatenated dataset of all annotation results (JSON-invalid samples are NOT removed)
        """
        ds: Dataset = concatenate_datasets(
            [
                Dataset.from_json(str(pfin))
                for pfin in pdout.glob("*.jsonl")
                if pfin.stat().st_size > 0
            ]
        ).sort(idx_column)
        if not keep_idx_column:
            ds = ds.remove_columns([idx_column])

        ds.save_to_disk(pdout / "done_disk_dataset")

        if new_hub_id:
            ds.push_to_hub(new_hub_id, private=True)
            if self.verbose:
                self._logger.info(
                    f"Uploaded final dataset to the HF Hub: https://huggingface.co/datasets/{new_hub_id}!"
                )

        ds.cleanup_cache_files()

        cached_input_ds = pdout / "cached_input_dataset"
        if cached_input_ds.exists():
            shutil.rmtree(cached_input_ds)

        return ds

    def get_pfout_name(
        self,
        *,
        pdout: Path | str,
        max_samples_per_output_file: int,
        processed_n_samples: int | None = None,
    ) -> Path:
        """Generate the output file name based on configuration.

        Creates appropriate file names for output files, handling both
        single-file and multi-file output modes.

        Args:
            pdout: The output directory path.
            max_samples_per_output_file: Maximum samples per output file (0 for unlimited).
            processed_n_samples: The number of samples processed so far.

        Returns:
            Path object for the output file name.
        """
        pdout = Path(pdout)
        processed_n_samples = processed_n_samples or 0
        stem = pdout.stem
        if not max_samples_per_output_file:
            return pdout.joinpath(f"{stem}.jsonl")
        else:
            count_idx = processed_n_samples // max_samples_per_output_file
            return pdout.joinpath(f"{stem}_{count_idx}.jsonl")

    @retry()
    def push_dir_to_hub(
        self,
        dir_path: Path | str,
        new_hub_id: str | None = None,
        *,
        task_prefix: str = "",
        revision: str | None = None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
    ) -> None:
        """Upload the output directory to Hugging Face Hub.

        Creates a dataset repository and uploads all annotation files,
        excluding cached input data. Uses a separate branch for uploads.

        Args:
            dir_path: Path to the directory containing annotation files.
            new_hub_id: Optional Hugging Face dataset ID to override the instance's new_hub_id.
            task_prefix: String prefix to use for branch naming.
            revision: Optional explicit branch name. Defaults to
                ``{task_prefix}jsonl_upload``.
            allow_patterns: Optional include patterns for upload.
            ignore_patterns: Optional ignore patterns for upload.

        Raises:
            Exception: If upload fails after retries (handled by @retry decorator).
        """
        if not new_hub_id:
            raise ValueError(
                "'new_hub_id' must be set to push data to the HuggingFace Hub"
            )

        revision = revision or f"{task_prefix}jsonl_upload"
        allow_patterns = allow_patterns or ["*.jsonl", "*.json"]
        ignore_patterns = ignore_patterns or [
            f"{task_prefix}cached_input_dataset/*",
            ".cache/*",
        ]

        create_repo(
            new_hub_id, repo_type="dataset", exist_ok=True, private=True
        )
        create_branch(
            new_hub_id,
            repo_type="dataset",
            branch=revision,
            exist_ok=True,
        )

        upload_large_folder(
            repo_id=new_hub_id,
            repo_type="dataset",
            folder_path=str(dir_path),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            private=True,
            revision=revision,
            print_report=False,
        )
        if self.verbose:
            self._logger.info(
                "Backed-up data to the HF Hub:"
                f" https://huggingface.co/datasets/{new_hub_id}/tree/{revision}"
            )


__all__ = ["Annotator", "destroy_on_error"]
