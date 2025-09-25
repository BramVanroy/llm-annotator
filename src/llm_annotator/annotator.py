import abc
import gc
import json
import shutil
import string
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Literal, TextIO, Union

import torch
from datasets import Dataset, IterableDataset, load_dataset
from huggingface_hub import create_branch, create_repo, upload_large_folder
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from vllm import LLM, RequestOutput, SamplingParams
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.sampling_params import GuidedDecodingParams

from dataq.utils import USABLE_CPU_COUNT, retry


@dataclass
class Annotator(abc.ABC):
    model_id: str
    prompt_template_file: str | Path
    prompt_field_swapper: dict[str, str] | None = None
    output_schema: str | None = None
    whitespace_pattern: str | None = None
    idx_column: str = "idx"
    num_proc: int = USABLE_CPU_COUNT
    tensor_parallel_size: int = 1
    max_num_seqs: int = 256
    enforce_eager: bool = True
    quantization: str | None = None
    verbose: bool = False
    keep_columns: Union[str, list[str], None, set[str], Literal[True]] = None
    upload_every_n_samples: int = 0
    max_samples_per_output_file: int = 0
    new_hub_id: str | None = None
    max_model_len: int | None = None
    enable_thinking: bool = False
    no_dataset_cache: bool = False

    prompt_template: str = field(init=False)
    pipe: LLM | None = field(init=False)
    dataset: Dataset | None = field(init=False)
    dataset_config: str = field(init=False)
    dataset_split: str = field(init=False)
    tokenizer: PreTrainedTokenizer | None = field(init=False)
    prompt_fields: tuple[str, ...] = field(init=False)
    processed_n_samples: int = 0

    def __post_init__(self):
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

    def _get_skip_idxs(self, pdout: Path) -> set[int]:
        ids_done = set()
        if pdout.exists() and pdout.stat().st_size > 0:
            for pfin in pdout.glob("*.jsonl"):
                if pfin.stat().st_size == 0:
                    continue
                ds = Dataset.from_json(str(pfin))

                if "dataset_split" in ds.column_names:
                    ds = ds.filter(lambda s: s["dataset_split"] == self.dataset_split)

                if "dataset_config" in ds.column_names:
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
    ):
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split
        self.streaming = streaming

        cached_input_ds = pdout / "cached_input_dataset"

        # If exists and not empty, load from cache
        if cached_input_ds.exists() and cached_input_ds.stat().st_size > 0 and not self.no_dataset_cache:
            dataset = Dataset.load_from_disk(cached_input_ds)
        else:
            if streaming and not max_num_samples:
                raise ValueError(
                    "Streaming mode requires max_num_samples to be set. The dataset itself will be streamed and stored up to the requested number of samples."
                )

            if streaming:
                ds_iter: IterableDataset = load_dataset(
                    dataset_name, name=dataset_config, data_dir=data_dir, split=dataset_split, streaming=True
                )

                if shuffle_seed is not None:
                    ds_iter = ds_iter.shuffle(seed=shuffle_seed, buffer_size=10_000)

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

                if max_num_samples:
                    dataset = dataset.select(range(min(max_num_samples, len(dataset))))

            dataset = self._preprocess_dataset(dataset)

            dataset = dataset.map(
                lambda sample, idx: {
                    "dataq_prompted": self.tokenizer.apply_chat_template(
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
            dataset.save_to_disk(cached_input_ds)

        skip_idxs = self._get_skip_idxs(pdout)

        if skip_idxs:
            dataset = dataset.filter(
                lambda s: s[self.idx_column] not in skip_idxs,
                num_proc=self.num_proc,
                desc="Filtering done idxs",
            )
            self.processed_n_samples = len(skip_idxs)

        dataset = self._postprocess_dataset(dataset)
        self.dataset = dataset

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        return dataset

    def _postprocess_dataset(self, dataset: Dataset) -> Dataset:
        return dataset

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.tokenizer.padding_side = "left"

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

    def _load_pipeline(self):
        self.pipe = LLM(
            model=self.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            quantization=self.quantization,
            max_model_len=self.max_model_len,
            enforce_eager=self.enforce_eager,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=0.95,
        )

    @abc.abstractmethod
    def _process_output(self, output: RequestOutput) -> dict:
        raise NotImplementedError

    def _reset_model_and_dataset(self):
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.pipe.llm_engine.model_executor
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()

        self.pipe = None
        self.dataset = None

    def _process_batch(self, batch: dict[str, list], sampling_params: SamplingParams, fhout: TextIO):
        outputs = self.pipe.generate(batch["dataq_prompted"], sampling_params, use_tqdm=False)
        results = [self._process_output(outp) for outp in outputs]

        return results

    def annotate_dataset(
        self,
        dataset_name: str,
        output_dir: str | Path,
        *,
        overwrite: bool = False,
        dataset_config: str = None,
        data_dir: str = None,
        dataset_split: str = None,
        shuffle_seed: int | None = None,
        streaming: bool = False,
        sampling_params: dict = None,
        max_num_samples: int | None = None,
    ):
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
            fhout = pfout.open("a", encoding="utf-8")

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
                results = self._process_batch(batch, sampling_params, fhout)

                # Merge input batch and output results and then write them to the open JSONL file handle fhout
                if self.keep_columns is True:  # Keep all columns
                    inputs = [{k: v[i] for k, v in batch.items()} for i in range(len(batch["dataq_prompted"]))]
                else:  # Keep only specified columns or none (always includes idx_column)
                    inputs = [
                        {k: v[i] for k, v in batch.items() if k in self.keep_columns}
                        for i in range(len(batch["dataq_prompted"]))
                    ]

                merged = [{**inp, **res} for inp, res in zip(inputs, results)]

                for data_sample in merged:
                    fhout.write(json.dumps(data_sample) + "\n")
                    fhout.flush()
                    self.processed_n_samples += 1

                    if (
                        self.new_hub_id
                        and self.upload_every_n_samples > 0
                        and self.processed_n_samples % self.upload_every_n_samples == 0
                    ):
                        fhout.close()
                        self.push_dir_to_hub(pdout)
                        pfout = self.get_fhout_name(pdout)
                        fhout = pfout.open("a", encoding="utf-8")

                    if (
                        self.max_samples_per_output_file > 0
                        and self.processed_n_samples % self.max_samples_per_output_file == 0
                    ):
                        fhout.close()
                        pfout = self.get_fhout_name(pdout)
                        fhout = pfout.open("a", encoding="utf-8")

        self._post_annotate(pdout)

    def _post_annotate(self, pdout: Path):
        for pfin in pdout.glob("*.jsonl"):
            if pfin.stat().st_size == 0:
                pfin.unlink()

    def get_fhout_name(self, output_dir: Path | str) -> Path:
        stem = Path(output_dir).stem
        if self.max_samples_per_output_file == 0:
            return output_dir.joinpath(f"{stem}.jsonl")
        else:
            count_idx = self.processed_n_samples // self.max_samples_per_output_file
            return output_dir.joinpath(f"{stem}_{count_idx}.jsonl")

    @retry()
    def push_dir_to_hub(self, dir_path: Path | str):
        create_repo(self.new_hub_id, repo_type="dataset", exist_ok=True, private=True)
        create_branch(self.new_hub_id, repo_type="dataset", branch="dataq_jsonl_upload", exist_ok=True)

        upload_large_folder(
            repo_id=self.new_hub_id,
            repo_type="dataset",
            folder_path=str(dir_path),
            allow_patterns=["*.jsonl", "*.json"],  # Include data files (jsonl) and config files (json)
            ignore_patterns=["cached_input_dataset/*", ".cache/*"],  # Ignore cached input dataset
            private=True,
            revision="dataq_jsonl_upload",  # Stored in a separate branch as "backup"
        )
