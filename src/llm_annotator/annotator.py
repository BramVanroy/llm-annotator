import gc
import json
import shutil
import string
from dataclasses import dataclass, field
from functools import wraps
from math import ceil
from os import PathLike
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, IterableDataset, concatenate_datasets, get_dataset_split_names, load_dataset
from huggingface_hub import create_branch, create_repo, upload_large_folder
from torch import cuda
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm import LLM, RequestOutput, SamplingParams
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.sampling_params import StructuredOutputsParams

from llm_annotator.utils import remove_empty_jsonl_files, retry


def destroy_model_on_error(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except BaseException as e:
            # Catch BaseException so we also clean on KeyboardInterrupt/SystemExit
            try:
                self.destroy_model()
            except Exception as clean_err:
                # Don't hide the original error; attach a note (Python 3.11+)
                if hasattr(e, "__notes__"):
                    e.__notes__.append(f"Cleanup failed: {clean_err!r}")
            raise  # re-raise the original exception

    return wrapper


@dataclass(slots=True)
class Annotator:
    """Sensible base class for LLM-based dataset annotation.

    This class provides a framework for annotating datasets using large language models
    through the vLLM library. It handles dataset loading, processing, and output generation
    with support for streaming, batching, and uploading to Hugging Face Hub.

    Args:
        model_id: The Hugging Face model identifier or local path.
        num_proc: Number of processes for dataset operations.
        tensor_parallel_size: Number of GPUs for tensor parallelism. Especially useful if running on
            multiple GPUs; set to the number of GPUs available.
        max_num_seqs: Maximum number of sequences to process in parallel (~batch size).
        gpu_memory_utilization: Max. GPU memory utilization goal.
        enforce_eager: Whether to enforce eager execution mode. Eager mode is safer but may be slower.
        quantization: Quantization method to use (optional).
        verbose: Whether to enable verbose logging.
        max_model_len: Maximum model sequence length.
        enable_thinking: Whether to enable thinking mode for chat templates.
        verbose: Whether to enable verbose logging.
    """

    model_id: str
    num_proc: int | None = None
    tensor_parallel_size: int = 1
    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.95
    enforce_eager: bool = False
    quantization: str | None = None
    verbose: bool = False
    max_model_len: int | None = None
    enable_thinking: bool = False
    verbose: bool = False

    pipe: LLM | None = field(default=None, init=False)
    tokenizer: PreTrainedTokenizer | None = field(default=None, init=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.destroy_model()

    def _get_skip_idxs(
        self, *, pdout: Path, idx_column: str, dataset_split: str | None, dataset_config: str | None
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
        if pdout.exists() and pdout.stat().st_size > 0:
            for pfin in pdout.glob("*.jsonl"):
                if pfin.stat().st_size == 0:
                    continue
                ds = Dataset.from_json(str(pfin))

                if dataset_split and "dataset_split" in ds.column_names:
                    ds = ds.filter(lambda s: s["dataset_split"] == dataset_split)

                if dataset_config and "dataset_config" in ds.column_names:
                    ds = ds.filter(lambda s: s["dataset_config"] == dataset_config)

                ids_done.update(ds.unique(idx_column))

        return ids_done

    def _load_dataset(
        self,
        *,
        prompt_template: str,
        pdout: Path,
        idx_column: str,
        dataset_name: str | None = None,
        dataset: Dataset | None = None,
        dataset_config: str = None,
        data_dir: str | None = None,
        dataset_split: str | None = None,
        streaming: bool = False,
        max_num_samples: int | None = None,
        shuffle_seed: int | None = None,
        cache_input_dataset: bool = True,
        use_cached_input_dataset: bool = True,
        prompt_fields: Iterable[str] = (),
        prefix: str = "",
    ) -> tuple[Dataset, int]:
        """Load and preprocess the dataset for annotation.

        Handles dataset loading from various sources, applies prompt templates,
        and manages caching for efficient resumption of interrupted jobs.

        Args:
            dataset_name: Name or path of the dataset to load.
            dataset: Pre-loaded dataset to use instead of loading from name/path.
            pdout: Output directory for caching and results.
            dataset_config: Dataset configuration name (optional).
            data_dir: Data directory for local datasets (optional).
            dataset_split: Specific split to load (optional).
            streaming: Whether to use streaming mode for large datasets.
            max_num_samples: Maximum number of samples to process.
            shuffle_seed: Seed for dataset shuffling (optional).
            cache_input_dataset: Whether to cache the input dataset.
                Especially useful if using streaming + max_num_samples.
            use_cached_input_dataset: Whether to use a cached input dataset if available.
            prompt_fields: Fields required by the prompt template.
            prefix: String prefix to use for internal column names and file operations.

        Returns:
            A tuple containing:
                - The loaded and preprocessed dataset ready for annotation.
                - The number of samples that were skipped because they were already processed.

        Raises:
            ValueError: If configuration is invalid or required fields are missing.
        """
        if max_num_samples is not None and max_num_samples <= 0:
            raise ValueError("'max_num_samples' must be a positive integer or None")

        if not dataset_name and dataset is None:
            raise ValueError("Either 'dataset_name' or 'dataset' must be provided")

        # Split verification and defaulting
        if dataset_name:
            split_names = get_dataset_split_names(dataset_name)
            if not dataset_split:
                if len(split_names) == 1:
                    dataset_split = split_names[0]
                else:
                    raise ValueError(
                        f"Dataset '{dataset_name}' has multiple splits {split_names}. "
                        "Please specify a split using the 'dataset_split' argument."
                    )
            elif dataset_split not in split_names:
                raise ValueError(f"Dataset '{dataset_name}' does not have a split named '{dataset_split}'")

        pdout = Path(pdout)
        p_cached_input_ds = pdout / f"{prefix}cached_input_dataset"

        # If exists and not empty, try to load from cache. If loading the
        # cached dataset fails (corrupted cache), fall back to loading from
        # the original source.
        loaded_ds = None
        if use_cached_input_dataset and p_cached_input_ds.exists() and p_cached_input_ds.stat().st_size > 0:
            loaded_ds = Dataset.load_from_disk(p_cached_input_ds)

        # Always prefer a locally cached dataset if available
        if loaded_ds is not None:
            dataset = loaded_ds
        else:
            if streaming and not max_num_samples:
                raise ValueError(
                    "Streaming mode requires max_num_samples to be set."
                    " The dataset itself will be streamed and stored up to"
                    " the requested number of samples."
                )

            # No dataset provided, so got to load it from dataset_name
            if dataset is None and streaming:
                ds_iter: IterableDataset = load_dataset(
                    dataset_name, name=dataset_config, data_dir=data_dir, split=dataset_split, streaming=True
                )

                if shuffle_seed is not None:
                    # IterableDataset.shuffle does not accept buffer_size in some
                    # versions; call with only seed to be compatible.
                    try:
                        ds_iter = ds_iter.shuffle(seed=shuffle_seed, buffer_size=10_000)
                    except TypeError:
                        ds_iter = ds_iter.shuffle(seed=shuffle_seed)

                def yield_fn():
                    num_samples = 0
                    for sample in ds_iter:
                        yield sample
                        num_samples += 1
                        if max_num_samples and num_samples >= max_num_samples:
                            break

                # Convert to Dataset
                dataset = Dataset.from_generator(yield_fn, split=dataset_split)
            else:
                # Use the provided dataset if available
                if dataset is not None:
                    dataset = dataset
                else:
                    dataset = load_dataset(dataset_name, name=dataset_config, data_dir=data_dir, split=dataset_split)

                if shuffle_seed is not None:
                    dataset = dataset.shuffle(seed=shuffle_seed)

                if max_num_samples:
                    dataset = dataset.select(range(min(max_num_samples, len(dataset))))

            # Validate that the dataset contains all fields required by the
            # prompt template. Tests expect a ValueError when a required
            # field is missing.
            if dataset is not None and prompt_fields:
                missing = [fld for fld in prompt_fields if fld not in dataset.column_names]
                if missing:
                    raise ValueError(f"Template contains field '{missing[0]}' not present in dataset")

            dataset = self._preprocess_dataset(dataset=dataset)

            dataset = dataset.map(
                self.apply_prompt_template,
                with_indices=True,
                num_proc=self.num_proc,
                fn_kwargs={
                    "prompt_fields": prompt_fields,
                    "prompt_template": prompt_template,
                    "idx_column": idx_column,
                    "prefix": prefix,
                },
                desc="Applying prompt template",
            )

            if cache_input_dataset:
                dataset.save_to_disk(p_cached_input_ds)

        skip_idxs = self._get_skip_idxs(
            pdout=pdout,
            idx_column=idx_column,
            dataset_split=dataset_split,
            dataset_config=dataset_config,
        )
        processed_n_samples = 0
        if skip_idxs:
            dataset = dataset.filter(
                lambda s: s[idx_column] not in skip_idxs,
                num_proc=self.num_proc,
                desc="Filtering done idxs",
            )
            processed_n_samples = len(skip_idxs)
            if self.verbose:
                print(f"Skipping {len(skip_idxs)} already-processed samples")

        dataset = self._postprocess_dataset(dataset=dataset)
        return dataset, processed_n_samples

    def apply_prompt_template(
        self, sample: dict, idx: int, prompt_fields: Iterable[str], prompt_template: str, idx_column: str, prefix: str
    ) -> dict[str, str | int]:
        """Apply the prompt template to a single dataset sample. Fills in the prompt template with values from the sample,
        based on the prompt_fields.

        Args:
            sample: The dataset sample to process.
            idx: The index of the sample in the dataset.
            prompt_fields: Fields required by the prompt template.
            prompt_template: The prompt template string with placeholders.
            idx_column: Column name to use as unique identifier.
            prefix: String prefix to use for internal column names.

        Returns:
            A dictionary with the filled-in prompt and the sample index.
        """
        return {
            f"{prefix}prompted": self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": prompt_template.format(**{fld: sample[fld] for fld in prompt_fields}),
                    }
                ],
                tokenize=False,
                add_generation_template=True,
                enable_thinking=self.enable_thinking,
            ),
            idx_column: idx,
        }

    def _preprocess_dataset(self, *, dataset: Dataset) -> Dataset:
        """Preprocess the dataset before applying prompt templates.

        Override this method to add custom preprocessing logic such as
        filtering, transforming columns, or adding metadata.

        Args:
            dataset: The loaded dataset to preprocess.

        Returns:
            The preprocessed dataset.
        """
        return dataset

    def _postprocess_dataset(self, *, dataset: Dataset) -> Dataset:
        """Postprocess the dataset after applying prompt templates.

        Override this method to add final processing steps before annotation
        such as additional filtering or column transformations.

        Args:
            dataset: The dataset with applied prompt templates.

        Returns:
            The postprocessed dataset ready for annotation.
        """
        return dataset

    def _load_tokenizer(self) -> None:
        """Load and configure the tokenizer for the model.

        Sets up the tokenizer with appropriate padding settings and ensures
        a pad token is available.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.tokenizer.padding_side = "left"

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

    def _load_pipeline(self) -> None:
        """Load and initialize the vLLM pipeline for inference.

        Configures the LLM with the specified parameters including tensor
        parallelism, quantization, and memory settings.
        """
        self.pipe = LLM(
            model=self.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            quantization=self.quantization,
            max_model_len=self.max_model_len,
            enforce_eager=self.enforce_eager,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )

    def _process_output(
        self, *, output: RequestOutput, output_schema: dict | None = None, prefix: str = ""
    ) -> dict[str, Any]:
        """Process a single model output into the desired annotation format.

        Override this method to implement custom output parsing and validation.

        Args:
            output: The raw output from the model for a single input.
            output_schema: Optional JSON schema for guided decoding.
            prefix: String prefix to use for internal column names.
        Returns:
            - A key '{prefix}_response' containing the raw model output text.
            - A key '{prefix}_finish_reason' indicating why generation stopped.
            - A key '{prefix}_num_tokens' indicating the number of tokens in the output.

            And if an output_schema is provided, also:
                - Keys from the output_schema with their parsed values (or None if parsing failed).
                - A key '{prefix}_valid_fields' indicating if all required fields were valid.
        """
        raw_response = output.outputs[0].text

        data = {
            f"{prefix}response": raw_response,
            f"{prefix}finish_reason": output.outputs[0].finish_reason if output.outputs else "unknown",
            f"{prefix}num_tokens": len(output.outputs[0].token_ids) if output.outputs else 0,
        }
        if not output_schema:
            return data

        valid_fields = None
        result = dict.fromkeys(output_schema.get("properties", {}).keys())
        try:
            parsed_response = json.loads(raw_response)
        except json.JSONDecodeError:
            valid_fields = False
        else:
            result = parsed_response

            if "required" in output_schema:
                required_keys = output_schema["required"]
                valid_fields = all(key in result for key in required_keys)

        return {
            **data,
            f"{prefix}valid_fields": valid_fields,
            **result,
        }

    def destroy_model(self) -> None:
        """Clean up model resources to free memory.

        Destroys the distributed environment, clears GPU cache, and resets internal state.
        """
        if isinstance(self.pipe, LLM):
            destroy_model_parallel()
            destroy_distributed_environment()

            try:
                self.pipe.llm_engine.model_executor.shutdown()
                del self.pipe.llm_engine.model_executor
            except Exception:
                pass
            try:
                self.pipe.llm_engine.engine_core.shutdown()
                del self.pipe.llm_engine.engine_core
            except Exception:
                pass
            del self.pipe.llm_engine
            del self.pipe
            cuda.empty_cache()
            gc.collect()

            self.pipe = None

    def _process_batch(
        self,
        *,
        batch: dict[str, list[Any]],
        sampling_params: SamplingParams,
        prefix: str = "",
    ) -> list[dict[str, Any]]:
        """Process a batch of samples through the model.

        Takes a batch of prompted samples, runs inference, and processes
        the outputs using the `_process_output` method.

        Args:
            batch: Dictionary containing batch data with prompted samples.
            sampling_params: Sampling parameters for model generation.
            prefix: String prefix to use for internal column names.

        Returns:
            List of processed output dictionaries for each sample in the batch.
        """
        output_schema = sampling_params.structured_outputs.json if sampling_params.structured_outputs else None
        outputs = self.pipe.generate(batch[f"{prefix}prompted"], sampling_params, use_tqdm=False)
        results = [self._process_output(output=outp, output_schema=output_schema, prefix=prefix) for outp in outputs]

        if f"{prefix}valid_fields" in results[0]:
            n_invalid = sum([1 for res in results if not res[f"{prefix}valid_fields"]])
            if n_invalid == len(results) and self.verbose:
                print(
                    "Warning: All samples in the batch failed to produce valid fields."
                    " This might be exceptional but if it happens often it suggest a deeper issue,"
                    " such as too few 'max_tokens' in sampling_params."
                )

        return results

    @destroy_model_on_error
    def annotate_dataset(
        self,
        output_dir: str | Path,
        *,
        dataset_name: str | None = None,
        dataset: Dataset | None = None,
        new_hub_id: str | None = None,
        overwrite: bool = False,
        dataset_config: str | None = None,
        data_dir: str | None = None,
        dataset_split: str | None = None,
        keep_columns: str | Iterable[str] | bool | None = None,
        shuffle_seed: int | None = None,
        streaming: bool = False,
        sampling_params: dict[str, Any] | None = None,
        max_num_samples: int | None = None,
        cache_input_dataset: bool = True,
        use_cached_input_dataset: bool = True,
        prompt_template_file: str | PathLike | None = None,
        prompt_template: str | None = None,
        prompt_field_swapper: dict[str, str] | None = None,
        output_schema_file: str | PathLike | None = None,
        output_schema: str | dict[str, Any] | None = None,
        whitespace_pattern: str | None = r"[ ]?",
        idx_column: str = "idx",
        upload_every_n_samples: int = 0,
        max_samples_per_output_file: int = 0,
        prefix: str = "",
    ) -> Dataset:
        """Annotate an entire dataset using the configured model and prompt.

        Main entry point for dataset annotation. Handles the complete pipeline
        from dataset loading through model inference to output generation.

        Args:
            output_dir: Directory to save annotation results.
            dataset_name: Name or path of the dataset to annotate.
            dataset: Pre-loaded dataset to use instead of loading from name/path.
            new_hub_id: Optional Hugging Face dataset ID for uploads.
            overwrite: Whether to overwrite existing output directory.
            dataset_config: Dataset configuration name (optional).
            data_dir: Data directory for local datasets (optional).
            dataset_split: Specific split to annotate (optional).
            keep_columns: Columns to keep in output. True for all, None/false-y for none. Available default columns are
                {prefix}prompted (filled-in prompt), {prefix}response (raw model output).
                If a JSON schema is given, also {prefix}valid_fields (boolean if all required fields were valid
                according to output_schema) and output columns according to the JSON schema if given.
                During processing, the idx_column is always kept to ensure proper mapping, but it is
                removed again in the final output dataset.
            shuffle_seed: Seed for dataset shuffling (optional).
            streaming: Whether to use streaming mode for large datasets.
            sampling_params: Parameters for model generation (optional).
            max_num_samples: Maximum number of samples to annotate.
            cache_input_dataset: Whether to cache the input dataset. Especially useful if
                using streaming + max_num_samples.
            use_cached_input_dataset: Whether to use a cached input dataset if available.
            prompt_template_file: Path to the prompt template file. Can/should contain fields in `{}`
            that match dataset column names, e.g. "Analyze the following text: {text}".
            prompt_template: Prompt template string (alternative to prompt_template_file). Can/should
                contain fields in `{}` that match dataset column names, e.g. "Analyze the
                following text: {text}".
            prompt_field_swapper: Optional mapping to replace template fields. Useful if you want to use
                the same template with different datasets that use different field names.
            output_schema_file: Path to a JSON schema file for guided decoding (optional).
            output_schema: JSON schema as a dictionary or string (alternative to output_schema_file).
            whitespace_pattern: Regex pattern for whitespace handling in guided decoding.
            idx_column: Column name to use as unique identifier.
            upload_every_n_samples: Upload to hub every N samples (0 to disable).
            max_samples_per_output_file: Maximum samples per output file (0 for unlimited).
            prefix: String prefix to use for internal column names and file operations.
        """
        # Verify 'max_samples_per_output_file'
        if max_samples_per_output_file is not None and max_samples_per_output_file < 0:
            raise ValueError("'max_samples_per_output_file' must be None or 0 or a positive integer")
        max_samples_per_output_file = max_samples_per_output_file or 0

        # Verify 'upload_every_n_samples' and 'new_hub_id'
        if upload_every_n_samples < 0 or not isinstance(upload_every_n_samples, int):
            raise ValueError("upload_every_n_samples must be a positive integer or 0")
        elif upload_every_n_samples > 0 and not new_hub_id:
            raise ValueError("If upload_every_n_samples is set, new_hub_id must be provided")

        # Verify and normalize 'keep_columns'
        if not keep_columns:
            keep_columns = set()
        elif isinstance(keep_columns, str):
            keep_columns = {keep_columns}
        elif keep_columns is True:
            # Redundant but makes it clearer that the value can be True
            keep_columns = True
        else:
            try:
                keep_columns = set(keep_columns)
            except TypeError as exc:
                raise TypeError("keep_columns must be None, True, a string, or a collection of strings") from exc

        # Always keep idx_column
        if isinstance(keep_columns, set):
            keep_columns.add(idx_column)

        # Verify prompt template inputs
        if not prompt_template_file and not prompt_template:
            raise ValueError("Either prompt_template_file or prompt_template must be provided")

        if prompt_template_file and prompt_template:
            raise ValueError("Only one of prompt_template_file or prompt_template should be provided")

        if prompt_template_file:
            prompt_template = Path(prompt_template_file).read_text(encoding="utf-8")

        prompt_field_swapper = prompt_field_swapper or {}

        for fld, value in prompt_field_swapper.items():
            prompt_template = prompt_template.replace(f"{{{fld}}}", value)

        _str_formatter = string.Formatter()
        prompt_fields = tuple(
            [fld[1] for fld in _str_formatter.parse(prompt_template) if fld[1] is not None and not fld[2]]
        )

        # Verify output schema inputs
        if output_schema_file and output_schema:
            raise ValueError("Only one of output_schema_file or output_schema should be provided")

        if output_schema_file:
            output_schema = json.loads(Path(output_schema_file).read_text(encoding="utf-8"))

        # Set up output directory
        pdout = Path(output_dir)
        if pdout.is_dir() and overwrite:
            shutil.rmtree(pdout)
        pdout.mkdir(exist_ok=True, parents=True)

        # We need to clear the model before doing self._load_dataset because the model
        # cannot be be pickled (which is needed for multiprocessing in dataset.map)
        if self.pipe is not None and self.num_proc is not None:
            self.destroy_model()

        if self.tokenizer is None:
            self._load_tokenizer()

        dataset, processed_n_samples = self._load_dataset(
            prompt_template=prompt_template,
            idx_column=idx_column,
            pdout=pdout,
            dataset_name=dataset_name,
            dataset=dataset,
            dataset_config=dataset_config,
            data_dir=data_dir,
            dataset_split=dataset_split,
            streaming=streaming,
            max_num_samples=max_num_samples,
            shuffle_seed=shuffle_seed,
            cache_input_dataset=cache_input_dataset,
            use_cached_input_dataset=use_cached_input_dataset,
            prompt_fields=prompt_fields,
            prefix=prefix,
        )
        if len(dataset) > 0:
            pfout = self.get_pfout_name(
                pdout=pdout,
                max_samples_per_output_file=max_samples_per_output_file,
                processed_n_samples=processed_n_samples,
            )
            fhout = pfout.open("a", encoding="utf-8")

            self._load_pipeline()

            sampling_params = sampling_params or {}
            if output_schema:
                if "structured_outputs" in sampling_params:
                    raise ValueError(
                        "If 'output_schema' is provided, do not set 'structured_outputs' in sampling_params yourself"
                    )
                sampling_params["structured_outputs"] = StructuredOutputsParams(
                    json=output_schema,
                    whitespace_pattern=whitespace_pattern,
                )
            sampling_params = SamplingParams(**sampling_params)

            total_num_batches = ceil(len(dataset) / self.max_num_seqs)
            for batch in tqdm(
                dataset.iter(self.max_num_seqs),
                total=total_num_batches,
                desc=f"Annotating (max_bs={self.max_num_seqs})",
                unit="batch",
            ):
                results = self._process_batch(batch=batch, sampling_params=sampling_params, prefix=prefix)

                batch_size = len(batch[idx_column])
                if keep_columns is True:
                    # Keep all columns
                    inputs = [{k: v[i] for k, v in batch.items()} for i in range(batch_size)]
                else:
                    inputs = [{k: v[i] for k, v in batch.items() if k in keep_columns} for i in range(batch_size)]

                # Iterate over results and write them out in order
                for result_idx, res in enumerate(results):
                    inp = inputs[result_idx]
                    data_sample = {**inp, **res}
                    fhout.write(json.dumps(data_sample) + "\n")
                    fhout.flush()
                    processed_n_samples += 1

                    # Handle hub upload checkpointing and output file rotation
                    if upload_every_n_samples > 0 and processed_n_samples % upload_every_n_samples == 0:
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

        return self._post_annotate(pdout=pdout, idx_column=idx_column, new_hub_id=new_hub_id)

    def _post_annotate(self, *, pdout: Path, idx_column: str, new_hub_id: str | None = None) -> Dataset:
        """Clean up after annotation is complete.

        Removes empty output files and performs any final cleanup operations.

        Args:
            pdout: Output directory path to clean up.
            new_hub_id: Optional Hugging Face dataset ID for uploads.

        Returns:
            The concatenated dataset of all annotation results (JSON-invalid samples are NOT removed)
        """
        ds_parts = []
        for pfin in pdout.glob("*.jsonl"):
            if pfin.stat().st_size > 0:
                ds_parts.append(Dataset.from_json(str(pfin)))

        ds = concatenate_datasets(ds_parts).remove_columns(idx_column)

        if new_hub_id:
            ds.push_to_hub(new_hub_id, private=True)
            if self.verbose:
                print(f"Uploaded final dataset to the HF Hub: https://huggingface.co/datasets/{new_hub_id}!")

        ds.cleanup_cache_files()

        cached_input_ds = pdout / "cached_input_dataset"
        if cached_input_ds.exists():
            shutil.rmtree(cached_input_ds)

        return ds

    def get_pfout_name(
        self, *, pdout: Path | str, max_samples_per_output_file: int, processed_n_samples: int | None = None
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
    def push_dir_to_hub(self, dir_path: Path | str, new_hub_id: str | None = None, *, prefix: str = "") -> None:
        """Upload the output directory to Hugging Face Hub.

        Creates a dataset repository and uploads all annotation files,
        excluding cached input data. Uses a separate branch for uploads.

        Args:
            dir_path: Path to the directory containing annotation files.
            new_hub_id: Optional Hugging Face dataset ID to override the instance's new_hub_id.
            prefix: String prefix to use for branch naming.

        Raises:
            Exception: If upload fails after retries (handled by @retry decorator).
        """
        if not new_hub_id:
            raise ValueError("'new_hub_id' must be set to push data to the HuggingFace Hub")

        create_repo(new_hub_id, repo_type="dataset", exist_ok=True, private=True)
        create_branch(new_hub_id, repo_type="dataset", branch=f"{prefix}jsonl_upload", exist_ok=True)

        upload_large_folder(
            repo_id=new_hub_id,
            repo_type="dataset",
            folder_path=str(dir_path),
            allow_patterns=["*.jsonl", "*.json"],  # Include data files (jsonl) and config files (json)
            ignore_patterns=[f"{prefix}cached_input_dataset/*", ".cache/*"],  # Ignore cached input dataset
            private=True,
            revision=f"{prefix}jsonl_upload",
            print_report=False,
        )
        if self.verbose:
            url = f"https://huggingface.co/datasets/{new_hub_id}/tree/{prefix}jsonl_upload"
            print(f"Backed-up data to the HF Hub: {url}")
