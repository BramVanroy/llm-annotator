# LLM Annotator

LLM Annotator is a Python library for robust, resumable annotation and
generation workflows powered by large language models.

It provides a common interface for multiple providers:

- `VLLMOfflineClient` for local vLLM inference.
- `VLLMClient` for vLLM server endpoints.
- `OpenAIClient` for OpenAI-compatible APIs.
- `ClaudeClient` for Anthropic APIs.

Provider setup details, extras, and auth variables are listed on
[Provider setup](provider-info.md).

## Install

With uv:

```bash
uv add llm-annotator
```

With pip:

```bash
pip install llm-annotator
```

Install provider extras when needed:

```bash
uv add "llm-annotator[vllm]"
uv add "llm-annotator[vllm-flashinfer]"  # Faster if your hardware supports it
uv add "llm-annotator[openai]"
uv add "llm-annotator[anthropic]"
```

## Quickstart

### One-step convenience

Annotate a dataset end-to-end with a single call:

```python
from llm_annotator import Annotator, VLLMOfflineClient

client = VLLMOfflineClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_model_len=4096,
)

with Annotator(client=client) as anno:
    ds = anno.annotate_dataset(
        output_dir="outputs/imdb-sentiment",
        prompt_template="Classify the sentiment: {text}",
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        max_num_samples=100,
    )
```

Generate a dataset from scratch:

```python
from llm_annotator import Annotator, OpenAIClient

client = OpenAIClient(model="gpt-4o-mini")

with Annotator(client=client) as anno:
    ds = anno.generate_dataset(
        output_dir="outputs/generated",
        prompts="Create one short NER training sentence.",
        max_num_samples=50,
    )
```

### Two-step staged workflow

For large datasets or SLURM-style pipelines, separate data preparation
from model inference. `prepare_data` handles template application and
optional sorting, then uploads the result to Hugging Face Hub. On
inference failures, `run_annotation` can reload the prepared data from
Hub without repeating the expensive preparation step.

```python
from llm_annotator import Annotator, VLLMOfflineClient

client = VLLMOfflineClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_model_len=4096,
)

HUB_ID = "my-org/imdb-prepared"

with Annotator(client=client, verbose=True) as anno:
    # Step 1: prepare:  reuses local cache, falls back to Hub, builds
    # from source if neither exists.
    prepared_dataset, local_path, hub_id = anno.prepare_data(
        output_dir="outputs/imdb-sentiment",
        prompt_template="Classify the sentiment: {text}",
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        max_num_samples=100,
        sort_by_length=True,
        prepared_hub_id=HUB_ID,         # back up prepared data to Hub
    )

    # Step 2: run generation against the prepared data.
    # If this step fails, re-run it with prepared_hub_id=HUB_ID and the
    # same output_dir:  the prepared data is restored from Hub automatically.
    ds = anno.run_annotation(
        output_dir="outputs/imdb-sentiment",
        prompt_template="Classify the sentiment: {text}",
        prepared_dataset=prepared_dataset,
        new_hub_id="my-org/imdb-annotated",
        upload_every_n_samples=500,
    )
```

To force a fresh preparation even when local or Hub artifacts exist, pass
`force_data_preparation=True` to `prepare_data` (or to `annotate_dataset`).

## Why use it

- Staged `prepare_data` + `run_annotation` pipeline for SLURM and
  cluster workflows:  expensive data preparation is done once and stored.
- Resume interrupted generation runs from JSONL checkpoints.
- Validate and post-process outputs with custom callables.
- Enforce structured responses through JSON schemas.
- Upload incrementally to the Hugging Face Hub.

## Development

```bash
git clone https://github.com/BramVanroy/llm-annotator.git
cd llm-annotator
uv sync --dev
```

Run checks:

```bash
make style
make quality
make test
make typecheck
```

Local docs preview with mike:

```bash
make serve-docs
```

The API reference section is generated from source code docstrings.
