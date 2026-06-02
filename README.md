# Robust, resumable LLM dataset annotation

[![CI](https://github.com/BramVanroy/llm-annotator/actions/workflows/ci.yml/badge.svg)](https://github.com/BramVanroy/llm-annotator/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BramVanroy/llm-annotator/branch/main/graph/badge.svg)](https://codecov.io/gh/BramVanroy/llm-annotator)
![PyPI version](https://img.shields.io/pypi/v/llm-annotator)
[![Python versions](https://img.shields.io/pypi/pyversions/llm-annotator.svg)](https://pypi.org/project/llm-annotator/)
[![License](https://img.shields.io/github/license/BramVanroy/llm-annotator)](LICENSE)


`llm-annotator` is a Python 3.12+ library for robust, resumable
LLM-driven dataset annotation and generation.

It supports multiple providers through pluggable clients:

- vLLM offline inference: `VLLMOfflineClient`
- vLLM server API: `VLLMClient`
- OpenAI API: `OpenAIClient`
- Anthropic API: `ClaudeClient`

Key capabilities:

- **Staged pipeline**:  `prepare_data` + `run_annotation` separates expensive
  template application and sorting from model inference, enabling SLURM and
  cluster restart workflows.
- Resumable processing with JSONL checkpoints.
- Annotation of existing datasets and generation from scratch.
- Structured outputs via JSON schema.
- Retry and validation hooks for robust pipelines.
- Optional Hugging Face Hub upload cadence for both prepared data and outputs.
- Context-manager cleanup of client resources.

It is not intended for parallel, multi-node, multi-instance generation.
If that is what you are after, maybe [`datatrove`](https://github.com/huggingface/datatrove/tree/main/examples/inference)
is something for you.

## Documentation

Read the full documentation at
[bramvanroy.github.io/llm-annotator](https://bramvanroy.github.io/llm-annotator/).

Provider setup reference:
[docs/provider-info.md](docs/provider-info.md)

## Installation

Recommended:

```sh
uv add llm-annotator
```

or

```sh
pip install llm-annotator
```

Install provider extras as needed:

```sh
uv add "llm-annotator[vllm]"
uv add "llm-annotator[vllm-flashinfer]"  # Faster if your hardware supports it
uv add "llm-annotator[openai]"
uv add "llm-annotator[anthropic]"
```

See [docs/provider-info.md](docs/provider-info.md) for auth environment
variables and provider-specific setup notes.

## Usage

### One-step convenience

Annotate an existing dataset:

```python
from llm_annotator import Annotator, VLLMOfflineClient

client = VLLMOfflineClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_model_len=4096,
)

with Annotator(client=client, verbose=True) as anno:
    ds = anno.annotate_dataset(
        output_dir="outputs/sentiment",
        prompt_template="Classify the sentiment of this text: {text}",
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
        output_dir="outputs/generated-qa",
        prompts="Write a short geography quiz question with answer.",
        max_num_samples=200,
    )
```

### Two-step staged workflow

For large datasets or cluster (SLURM) environments, split the pipeline
explicitly into a preparation step and a generation step. `prepare_data`
applies prompt templates, optional sorting, and saves the prepared
artifacts locally and to Hugging Face Hub. `run_annotation` then handles
only model inference. If generation fails, re-run `run_annotation` with
`prepared_hub_id` pointing to the Hub backup:  preparation is skipped.

```python
from llm_annotator import Annotator, VLLMOfflineClient

client = VLLMOfflineClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_model_len=4096,
)

HUB_ID = "my-org/imdb-prepared"  # Hub repo for prepared data backup

with Annotator(client=client, verbose=True) as anno:
    # Step 1: prepare data (reuses local cache or Hub backup if available)
    prepared_dataset, local_path, hub_id = anno.prepare_data(
        output_dir="outputs/imdb-sentiment",
        prompt_template="Classify the sentiment of this text: {text}",
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        max_num_samples=100,
        sort_by_length=True,
        prepared_hub_id=HUB_ID,
    )

    # Step 2: run generation against the prepared data
    ds = anno.run_annotation(
        output_dir="outputs/imdb-sentiment",
        prompt_template="Classify the sentiment of this text: {text}",
        prepared_dataset=prepared_dataset,
        new_hub_id="my-org/imdb-annotated",
        upload_every_n_samples=500,
    )
```

To force a fresh preparation (ignoring any cached or Hub-stored artifacts),
pass `force_data_preparation=True` to `prepare_data` or to `annotate_dataset`.

See the documentation for more examples, including:
- Structured output with JSON schemas
- Custom validation and post-processing
- Generating datasets from scratch

Or check out the [examples/](examples/) directory for complete working examples.


## Testing

Install development dependencies first:

```sh
uv sync --dev
```

Run the default checks:

```sh
make style
make quality
make test
make typecheck
```

Pytest marker targets:

```sh
# Fast tests (same as `make test`)
make test-fast

# Slow tests only
make test-slow

# Integration tests only
make test-integration

# Entire suite (fast + slow)
make test-all
```

You can also run markers directly with pytest:

```sh
uv run pytest -m "not slow"
uv run pytest -m "slow"
uv run pytest -m "integration"
```

Slow and integration tests may load local models, require more runtime, or depend on optional components.

## Building documentation

Local versioned docs preview (uses mike on a temporary local branch):

```sh
make serve-docs
```

Override version metadata when needed:

```sh
make serve-docs DOCS_VERSION=0.4.0 DOCS_ALIAS=latest DOCS_SOURCE_REF=v0.4.0
```

Docs are published with mike on release tags through
`.github/workflows/docs.yml`.
