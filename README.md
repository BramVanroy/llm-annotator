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

- **No-code config runs**:  describe prompts, schemas, model, dataset and
  multiple chained annotation steps in one JSON/YAML file and run it with
  `llm-annotate my-pipeline.yaml`.
- **Staged pipeline**:  `prepare_data` + `run_annotation` separates expensive
  template application and sorting from model inference, enabling SLURM and
  cluster restart workflows.
- **Multi-server vLLM**:  `VLLMQueueAnnotator` runs one workload over a pool of
  vLLM servers (e.g. one per GPU of a multi-node allocation); see `slurm/` for
  ready-made job scripts.
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
only model inference. If generation fails, re-run it with the same
`output_dir` and `hub_id`:  the prepared data is restored and the samples
already recorded in the progress files are skipped.

A single `hub_id` drives every Hub destination: the prepared data and the
JSONL progress backup live on temporary branches of that repo, the final
dataset is pushed to its `main` branch, and both temporary branches are
deleted once the run completes.

```python
from llm_annotator import Annotator, VLLMOfflineClient

client = VLLMOfflineClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_model_len=4096,
)

HUB_ID = "my-org/imdb-sentiment"  # backups *and* the final dataset

with Annotator(client=client, verbose=True) as anno:
    # Step 1: prepare data (reuses local cache or Hub backup if available)
    prepared_dataset, local_path, hub_id = anno.prepare_data(
        output_dir="outputs/imdb-sentiment",
        prompt_template="Classify the sentiment of this text: {text}",
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        max_num_samples=100,
        sort_by_length=True,
        hub_id=HUB_ID,
    )

    # Step 2: run generation against the prepared data
    ds = anno.run_annotation(
        output_dir="outputs/imdb-sentiment",
        prompt_template="Classify the sentiment of this text: {text}",
        prepared_dataset=prepared_dataset,
        hub_id=HUB_ID,
        upload_every_n_samples=500,
    )
```

To force a fresh preparation (ignoring any cached or Hub-stored artifacts),
pass `force_data_preparation=True` to `prepare_data` or to `annotate_dataset`.

### Run from a config file

The same work can be described in a single JSON or YAML file and run without
writing any Python:

```sh
llm-annotate my-pipeline.yaml
# or, from a checkout: python scripts/annotate.py my-pipeline.yaml
```

A config lists one or more **steps** that run in order, each annotating the
dataset the previous one produced. That is what makes generate-then-judge
workflows possible: one model writes question-answer pairs, a second rates them.

```yaml
output_dir: outputs/pipeline-qa

dataset:
  name: stanfordnlp/imdb
  split: test
  max_num_samples: 20

client:
  provider: vllm_offline
  model: Qwen/Qwen3-8B
  options:
    max_tokens: 512

steps:
  - name: write-qa
    prompt_file: prompts/write_qa.md
    output_schema_file: schemas/qa.json    # produces `question`, `answer`
    filter_invalid: true
    rename:
      question: question_v1

  - name: rate-qa
    prompt: "Rate this question about the text.\n\n{text}\n\nQ: {question_v1}"
    output_schema_file: schemas/rating.json
    client:
      provider: anthropic                  # a different judge
      model: claude-haiku-4-5
```

Paths inside the config resolve relative to the config file, so a config
directory is self-contained. Finished steps write a snapshot and are skipped on
a re-run, so an interrupted pipeline resumes rather than starting over.

A complete, runnable example lives in [examples/pipeline-qa/](examples/pipeline-qa/),
and the full key reference is in [docs/pipeline.md](docs/pipeline.md).

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
