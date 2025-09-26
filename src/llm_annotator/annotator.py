import gc
import json
import shutil
import string
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any, Literal, TextIO

import torch
from datasets import Dataset, IterableDataset, load_dataset
from huggingface_hub import create_branch, create_repo, upload_large_folder
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm import LLM, RequestOutput, SamplingParams
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.sampling_params import GuidedDecodingParams

from llm_annotator.utils import retry


@dataclass
class Annotator:
    """Sensible base class for LLM-based dataset annotation.

    This class provides a framework for annotating datasets using large language models
    through the vLLM library. It handles dataset loading, processing, and output generation
    with support for streaming, batching, and uploading to Hugging Face Hub.

    Args:
        model_id: The Hugging Face model identifier or local path.
        prompt_template_file: Path to the prompt template file.
        prompt_field_swapper: Optional mapping to replace template fields.
        output_schema: JSON schema for guided decoding (optional).
        whitespace_pattern: Regex pattern for whitespace handling in guided decoding.
        idx_column: Column name to use as unique identifier.
        num_proc: Number of processes for dataset operations.
        tensor_parallel_size: Number of GPUs for tensor parallelism. Especially useful if running on
          multiple GPUs; set to the number of GPUs available.
        max_num_seqs: Maximum number of sequences to process in parallel (~batch size).
        gpu_memory_utilization: Max. GPU memory utilization goal.
        enforce_eager: Whether to enforce eager execution mode. Eager mode is safer but may be slower.
        quantization: Quantization method to use (optional).
        verbose: Whether to enable verbose logging.
        keep_columns: Columns to keep in output. True for all, None/empty for none.
        upload_every_n_samples: Upload to hub every N samples (0 to disable).
        max_samples_per_output_file: Maximum samples per output file (0 for unlimited).
        new_hub_id: Hugging Face dataset ID for uploads.
        max_model_len: Maximum model sequence length.
        enable_thinking: Whether to enable thinking mode for chat templates.
        cache_input_dataset: Whether to cache the input dataset. Especially useful if using streaming + max_num_samples.
        use_cached_input_dataset: Whether to use a cached input dataset if available.
        prefix: String prefix to use for internal column names and file operations.

    Example:
        Create a prompt template file (sentiment_prompt.txt):

        ```
        Analyze the sentiment of the following text and classify it as positive, negative, or neutral.

        Text: {text}

        Please respond with only one word: positive, negative, or neutral.
        ```

        The dataset should have a 'text' column matching the {text} field in the template:

        ```
        Dataset({
            features: ['text', 'label'],  # 'text' is required, 'label' is optional
            num_rows: 1000
        })
        # Example row: {'text': 'I love this movie!', 'label': 'positive'}
        ```

        Then implement the annotator:

        >>> class SentimentAnnotator(Annotator):
        ...     def _process_output(self, output: RequestOutput) -> dict[str, Any]:
        ...         return {"predicted_sentiment": output.outputs[0].text.strip()}
        >>>
        >>> annotator = SentimentAnnotator(
        ...     model_id="meta-llama/Llama-2-7b-chat-hf",
        ...     prompt_template_file="sentiment_prompt.txt"
        ... )
        >>> annotator.annotate_dataset("stanfordnlp/imdb", "output_dir")
    """

    model_id: str
    prompt_template_file: str | Path
    prompt_field_swapper: dict[str, str] | None = None
    output_schema: str | None = None
    whitespace_pattern: str | None = None
    idx_column: str = "llm_annotator_idx"
    num_proc: int | None = None
    tensor_parallel_size: int = 1
    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.95
    enforce_eager: bool = False
    quantization: str | None = None
    verbose: bool = False
    keep_columns: str | list[str] | None | set[str] | Literal[True] = None
    upload_every_n_samples: int = 0
    max_samples_per_output_file: int = 0
    new_hub_id: str | None = None
    max_model_len: int | None = None
    enable_thinking: bool = False
    cache_input_dataset: bool = True
    use_cached_input_dataset: bool = True
    prefix: str = "llm_annotator"

    prompt_template: str = field(init=False)
    pipe: LLM | None = field(init=False)
    dataset: Dataset | None = field(init=False)
    dataset_config: str = field(init=False)
    dataset_split: str = field(init=False)
    tokenizer: PreTrainedTokenizer | None = field(init=False)
    prompt_fields: tuple[str, ...] = field(init=False)
    processed_n_samples: int = 0

    def __post_init__(self) -> None:
        """Initialize the annotator after dataclass creation.

        Validates configuration, loads prompt template, and prepares internal state.

        Raises:
            ValueError: If upload_every_n_samples is negative or new_hub_id is missing
                when upload_every_n_samples > 0.
        """
        if self.upload_every_n_samples < 0:
            raise ValueError("upload_every_n_samples must be a positive integer or 0")
        elif self.upload_every_n_samples > 0 and not self.new_hub_id:
            raise ValueError("If upload_every_n_samples is set, new_hub_id must be provided")

        self.max_samples_per_output_file = (
            0 if self.max_samples_per_output_file is None else max(0, self.max_samples_per_output_file)
        )

        self.prompt_template = Path(self.prompt_template_file).read_text(encoding="utf-8")
        self.prompt_field_swapper = self.prompt_field_swapper or {}

        for fld, value in self.prompt_field_swapper.items():
            self.prompt_template = self.prompt_template.replace(f"{{{fld}}}", value)

        str_formatter = string.Formatter()
        self.prompt_fields = tuple(
            [fld[1] for fld in str_formatter.parse(self.prompt_template) if fld[1] is not None and not fld[2]]
        )
        if not self.keep_columns:
            self.keep_columns = set()
        elif isinstance(self.keep_columns, str):
            self.keep_columns = {self.keep_columns}
        elif isinstance(self.keep_columns, list):
            self.keep_columns = set(self.keep_columns)

        if isinstance(self.keep_columns, set):
            self.keep_columns.add(self.idx_column)

        # Initialize runtime fields
        self.pipe = None
        self.dataset = None
        self.tokenizer = None

    def _get_skip_idxs(self, pdout: Path) -> set[int]:
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

                if self.dataset_split and "dataset_split" in ds.column_names:
                    ds = ds.filter(lambda s: s["dataset_split"] == self.dataset_split)

                if self.dataset_config and "dataset_config" in ds.column_names:
                    ds = ds.filter(lambda s: s["dataset_config"] == self.dataset_config)

                ids_done.update(ds.unique(self.idx_column))

        return ids_done

    def _load_dataset(
        self,
        dataset_name: str,
        pdout: Path,
        dataset_config: str | None = None,
        data_dir: str | None = None,
        dataset_split: str | None = None,
        streaming: bool = False,
        max_num_samples: int | None = None,
        shuffle_seed: int | None = None,
    ) -> None:
        """Load and preprocess the dataset for annotation.

        Handles dataset loading from various sources, applies prompt templates,
        and manages caching for efficient resumption of interrupted jobs.

        Args:
            dataset_name: Name or path of the dataset to load.
            pdout: Output directory for caching and results.
            dataset_config: Dataset configuration name (optional).
            data_dir: Data directory for local datasets (optional).
            dataset_split: Specific split to load (optional).
            streaming: Whether to use streaming mode for large datasets.
            max_num_samples: Maximum number of samples to process.
            shuffle_seed: Seed for dataset shuffling (optional).

        Raises:
            ValueError: If streaming mode is used without max_num_samples.
        """
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split
        self.streaming = streaming

        cached_input_ds = pdout / f"{self.prefix}_cached_input_dataset"

        dataset = None

        # If exists and not empty, try to load from cache. If loading the
        # cached dataset fails (corrupted cache), fall back to loading from
        # the original source.
        if self.use_cached_input_dataset and cached_input_ds.exists() and cached_input_ds.stat().st_size > 0:
            try:
                dataset = Dataset.load_from_disk(cached_input_ds)
            except Exception:
                dataset = None

        if dataset is None:
            if streaming and not max_num_samples:
                raise ValueError(
                    "Streaming mode requires max_num_samples to be set."
                    " The dataset itself will be streamed and stored up to"
                    " the requested number of samples."
                )

            if streaming:
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
                dataset = load_dataset(dataset_name, name=dataset_config, data_dir=data_dir, split=dataset_split)
                if shuffle_seed is not None:
                    dataset = dataset.shuffle(seed=shuffle_seed)

                # If max_num_samples is provided and 0, treat as explicit
                # request to process zero samples.
                if max_num_samples is not None and max_num_samples <= 0:
                    # Create an empty dataset that still contains the required
                    # prompt fields so template validation won't fail.
                    empty_data = {self.idx_column: []}
                    for fld in self.prompt_fields:
                        empty_data[fld] = []
                    dataset = Dataset.from_dict(empty_data)
                elif max_num_samples:
                    dataset = dataset.select(range(min(max_num_samples, len(dataset))))

            # Validate that the dataset contains all fields required by the
            # prompt template. Tests expect a ValueError when a required
            # field is missing.
            if dataset is not None and self.prompt_fields:
                missing = [fld for fld in self.prompt_fields if fld not in dataset.column_names]
                if missing:
                    raise ValueError(f"Template contains field '{missing[0]}' not present in dataset")

            dataset = self._preprocess_dataset(dataset)

            dataset = dataset.map(
                lambda sample, idx: {
                    f"{self.prefix}_prompted": self.tokenizer.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": self.prompt_template.format(
                                    **{fld: sample[fld] for fld in self.prompt_fields}
                                ),
                            }
                        ],
                        tokenize=False,
                        add_generation_template=True,
                        enable_thinking=self.enable_thinking,
                    ),
                    self.idx_column: idx,
                },
                with_indices=True,
                num_proc=self.num_proc,
                desc="Applying prompt template",
            )
            if self.cache_input_dataset:
                dataset.save_to_disk(cached_input_ds)

        skip_idxs = self._get_skip_idxs(pdout)

        if skip_idxs:
            dataset = dataset.filter(
                lambda s: s[self.idx_column] not in skip_idxs,
                num_proc=self.num_proc,
                desc="Filtering done idxs",
            )
            self.processed_n_samples = len(skip_idxs)
            if self.verbose:
                print(f"Skipping {len(skip_idxs)} already-processed samples")

        dataset = self._postprocess_dataset(dataset)
        self.dataset = dataset

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Preprocess the dataset before applying prompt templates.

        Override this method to add custom preprocessing logic such as
        filtering, transforming columns, or adding metadata.

        Args:
            dataset: The loaded dataset to preprocess.

        Returns:
            The preprocessed dataset.
        """
        return dataset

    def _postprocess_dataset(self, dataset: Dataset) -> Dataset:
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

    def _process_output(self, output: RequestOutput) -> dict[str, Any]:
        """Process a single model output into the desired annotation format.

        Override this method to implement custom output parsing and validation.

        Args:
            output: The raw output from the model for a single input.
        Returns:
            If output_schema is provided, returns a dictionary with:
                - Keys from the output_schema with their parsed values (or None if parsing failed).
                - A key '{prefix}_valid_fields' indicating if all required fields were valid.
                - A key '{prefix}_response' containing the raw model output text.
            If output_schema is not provided, returns a dictionary with:
                - A key '{prefix}_valid_fields' set to None.
                - A key '{prefix}_response' containing the raw model output text.
        """
        raw_response = output.outputs[0].text

        if not self.output_schema:
            return {f"{self.prefix}_valid_fields": None, f"{self.prefix}_response": raw_response}

        required_keys = self.output_schema["properties"].keys()
        result = dict.fromkeys(required_keys)

        valid_fields = True
        try:
            parsed_response = json.loads(raw_response)
        except json.JSONDecodeError:
            valid_fields = False
        else:
            result.update(parsed_response)

        valid_fields = valid_fields and all(result[key] is not None for key in required_keys)

        return {
            f"{self.prefix}_valid_fields": valid_fields,
            **result,
            f"{self.prefix}_response": raw_response,
        }

    def _reset_model_and_dataset(self) -> None:
        """Clean up model and dataset resources to free memory.

        Destroys the distributed environment, clears GPU cache, and resets
        internal state. Useful for processing multiple datasets sequentially.
        """
        # Best-effort cleanup: many attributes may be missing depending on
        # how far initialization progressed. Catch AttributeError and
        # continue to ensure cleanup does not raise during tests.
        try:
            destroy_model_parallel()
        except Exception:
            pass
        try:
            destroy_distributed_environment()
        except Exception:
            pass

        try:
            # Remove nested attributes if present
            if hasattr(self.pipe, "llm_engine") and hasattr(self.pipe.llm_engine, "model_executor"):
                del self.pipe.llm_engine.model_executor
        except Exception:
            pass

        try:
            del self.pipe
        except Exception:
            pass

        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        self.pipe = None
        self.dataset = None

    def _process_batch(
        self, batch: dict[str, list[Any]], sampling_params: SamplingParams, fhout: TextIO
    ) -> list[dict[str, Any]]:
        """Process a batch of samples through the model.

        Takes a batch of prompted samples, runs inference, and processes
        the outputs using the _process_output method.

        Args:
            batch: Dictionary containing batch data with prompted samples.
            sampling_params: Sampling parameters for model generation.
            fhout: File handle for output writing (currently unused).

        Returns:
            List of processed output dictionaries for each sample in the batch.
        """
        outputs = self.pipe.generate(batch[f"{self.prefix}_prompted"], sampling_params, use_tqdm=False)
        results = [self._process_output(outp) for outp in outputs]

        return results

    def annotate_dataset(
        self,
        dataset_name: str,
        output_dir: str | Path,
        *,
        overwrite: bool = False,
        dataset_config: str | None = None,
        data_dir: str | None = None,
        dataset_split: str | None = None,
        shuffle_seed: int | None = None,
        streaming: bool = False,
        sampling_params: dict[str, Any] | None = None,
        max_num_samples: int | None = None,
    ) -> None:
        """Annotate an entire dataset using the configured model and prompt.

        Main entry point for dataset annotation. Handles the complete pipeline
        from dataset loading through model inference to output generation.

        Args:
            dataset_name: Name or path of the dataset to annotate.
            output_dir: Directory to save annotation results.
            overwrite: Whether to overwrite existing output directory.
            dataset_config: Dataset configuration name (optional).
            data_dir: Data directory for local datasets (optional).
            dataset_split: Specific split to annotate (optional).
            shuffle_seed: Seed for dataset shuffling (optional).
            streaming: Whether to use streaming mode for large datasets.
            sampling_params: Parameters for model generation (optional).
            max_num_samples: Maximum number of samples to annotate.

        Example:
            >>> annotator.annotate_dataset(
            ...     "imdb",
            ...     "output/imdb_annotations",
            ...     dataset_split="test",
            ...     max_num_samples=1000,
            ...     sampling_params={"temperature": 0.7, "max_tokens": 512}
            ... )
        """
        pdout = Path(output_dir)
        if pdout.is_dir() and overwrite:
            shutil.rmtree(pdout)

        pdout.mkdir(exist_ok=True, parents=True)

        self._load_tokenizer()
        self._load_dataset(
            dataset_name,
            pdout,
            dataset_config=dataset_config,
            data_dir=data_dir,
            dataset_split=dataset_split,
            streaming=streaming,
            max_num_samples=max_num_samples,
            shuffle_seed=shuffle_seed,
        )
        if len(self.dataset) > 0:
            pfout = self.get_fhout_name(pdout)
            # Use built-in open so tests that patch pathlib.Path.open do not
            # interfere with dataset loading. Converting to str ensures a
            # filesystem path is passed.
            fhout = open(str(pfout), "a", encoding="utf-8")

            self._load_pipeline()

            sampling_params = sampling_params or {}
            if self.output_schema:
                ws_pattern = self.whitespace_pattern or None
                sampling_params["guided_decoding"] = GuidedDecodingParams(
                    json=self.output_schema,
                    whitespace_pattern=ws_pattern,
                )
            sampling_params = SamplingParams(**sampling_params)

            total_num_batches = ceil(len(self.dataset) / self.max_num_seqs)
            for batch in tqdm(
                self.dataset.iter(self.max_num_seqs),
                total=total_num_batches,
                desc=f"Annotating (max_bs={self.max_num_seqs})",
                unit="batch",
            ):
                # Prepare inputs list of dicts, keeping requested columns
                batch_size = len(batch[f"{self.prefix}_prompted"])
                if self.keep_columns is True:  # Keep all columns
                    inputs = [{k: v[i] for k, v in batch.items()} for i in range(batch_size)]
                else:
                    inputs = [{k: v[i] for k, v in batch.items() if k in self.keep_columns} for i in range(batch_size)]

                results = self._process_batch(inputs, sampling_params, fhout)

                # Iterate over results and write them out in order
                for result_idx, res in enumerate(results):
                    inp = inputs[result_idx]
                    data_sample = {**inp, **res}
                    fhout.write(json.dumps(data_sample) + "\n")
                    fhout.flush()
                    self.processed_n_samples += 1

                    # Handle hub upload checkpointing
                    if (
                        self.new_hub_id
                        and self.upload_every_n_samples > 0
                        and self.processed_n_samples % self.upload_every_n_samples == 0
                    ):
                        fhout.close()
                        self.push_dir_to_hub(pdout)
                        pfout = self.get_fhout_name(pdout)
                        fhout = open(str(pfout), "a", encoding="utf-8")

                    # Rotate output files when threshold reached
                    if (
                        self.max_samples_per_output_file > 0
                        and self.processed_n_samples % self.max_samples_per_output_file == 0
                    ):
                        fhout.close()
                        pfout = self.get_fhout_name(pdout)
                        fhout = open(str(pfout), "a", encoding="utf-8")

            fhout.close()

        self._post_annotate(pdout)

    def _post_annotate(self, pdout: Path) -> None:
        """Clean up after annotation is complete.

        Removes empty output files and performs any final cleanup operations.

        Args:
            pdout: Output directory path to clean up.
        """
        for pfin in pdout.glob("*.jsonl"):
            if pfin.stat().st_size == 0:
                pfin.unlink()

    def get_fhout_name(self, output_dir: Path | str) -> Path:
        """Generate the output file name based on configuration.

        Creates appropriate file names for output files, handling both
        single-file and multi-file output modes.

        Args:
            output_dir: The output directory path.

        Returns:
            Path object for the output file name.
        """
        stem = Path(output_dir).stem
        if self.max_samples_per_output_file == 0:
            return output_dir.joinpath(f"{stem}.jsonl")
        else:
            count_idx = self.processed_n_samples // self.max_samples_per_output_file
            return output_dir.joinpath(f"{stem}_{count_idx}.jsonl")

    @retry()
    def push_dir_to_hub(self, dir_path: Path | str) -> None:
        """Upload the output directory to Hugging Face Hub.

        Creates a dataset repository and uploads all annotation files,
        excluding cached input data. Uses a separate branch for uploads.

        Args:
            dir_path: Path to the directory containing annotation files.

        Raises:
            Exception: If upload fails after retries (handled by @retry decorator).
        """
        if not self.new_hub_id:
            raise ValueError("new_hub_id must be set to push data to the HuggingFace Hub")

        create_repo(self.new_hub_id, repo_type="dataset", exist_ok=True, private=True)
        create_branch(self.new_hub_id, repo_type="dataset", branch=f"{self.prefix}_jsonl_upload", exist_ok=True)

        upload_large_folder(
            repo_id=self.new_hub_id,
            repo_type="dataset",
            folder_path=str(dir_path),
            allow_patterns=["*.jsonl", "*.json"],  # Include data files (jsonl) and config files (json)
            ignore_patterns=[f"{self.prefix}_cached_input_dataset/*", ".cache/*"],  # Ignore cached input dataset
            private=True,
            revision=f"{self.prefix}_jsonl_upload",
        )
