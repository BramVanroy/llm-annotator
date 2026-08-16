# LLM Annotator

LLM Annotator is a Python library for robust, resumable annotation and
generation workflows powered by large language models.

It provides a common interface for multiple providers:

- `VLLMOfflineClient` for in-process vLLM inference (`vllm_offline`).
- `VLLMOnlineClient` for vLLM server endpoints (`vllm_online`).
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
uv add "llm-annotator[openai]"
uv add "llm-annotator[anthropic]"
```

## Quickstart

### No Python at all

Describe the whole run -- prompts, schema, model, dataset, and any number of
chained annotation steps -- in one JSON or YAML file:

```yaml title="my-pipeline.yaml"
output_dir: outputs/imdb-sentiment

dataset:
  name: stanfordnlp/imdb
  split: test
  max_num_samples: 100

client:
  provider: vllm_offline
  model: meta-llama/Llama-3.2-3B-Instruct
  init:
    max_model_len: 4096

steps:
  - name: sentiment
    prompt: "Classify the sentiment: {text}"
```

```bash
llm-annotate my-pipeline.yaml
```

See [Annotating from a config file](pipeline.md) for multi-step pipelines,
where one model's output becomes the next model's input.

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

One `hub_id` drives every Hub destination: the prepared data and the JSONL
progress backup go to temporary branches of that repo, the final dataset is
pushed to its `main` branch, and both temporary branches are deleted once
the run completes.

```python
from llm_annotator import Annotator, VLLMOfflineClient

client = VLLMOfflineClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_model_len=4096,
)

HUB_ID = "my-org/imdb-sentiment"

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
        hub_id=HUB_ID,                  # back up prepared data to Hub
    )

    # Step 2: run generation against the prepared data.
    # If this step fails, re-run it with hub_id=HUB_ID and the same
    # output_dir:  the prepared data is restored from Hub automatically and
    # the samples already in the progress files are skipped.
    ds = anno.run_annotation(
        output_dir="outputs/imdb-sentiment",
        prompt_template="Classify the sentiment: {text}",
        prepared_dataset=prepared_dataset,
        hub_id=HUB_ID,
        upload_every_n_samples=500,
    )
```

To force a fresh preparation even when local or Hub artifacts exist, pass
`force_data_preparation=True` to `prepare_data` (or to `annotate_dataset`).

### Many vLLM servers at once

`VLLMQueueAnnotator` spreads one workload over a pool of vLLM servers -- for
instance one server per GPU of a multi-node SLURM allocation. It is a drop-in
`Annotator`: the same four entry points, the same JSONL progress files, the same
resume behaviour. The only difference is that batches are dispatched to whichever
server is free, with at most `queue_size` batches in flight at a time.

```python
from llm_annotator import (
    VLLMOnlineClient,
    VLLMOnlineRuntimeOptions,
    VLLMQueueAnnotator,
)

clients = [
    VLLMOnlineClient(
        model="Qwen/Qwen3.5-4B", base_url=f"http://{host}:8000/v1"
    )
    for host in ("gcn1", "gcn2", "gcn3", "gcn4")
]

with VLLMQueueAnnotator(clients=clients, batch_size=64, verbose=True) as anno:
    ds = anno.annotate_dataset(
        output_dir="outputs/imdb-sentiment",
        prompt_template="Classify the sentiment: {text}",
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        options=VLLMOnlineRuntimeOptions(max_tokens=128, temperature=0.0),
    )
```

Because results are written per sample and keyed by `idx`, re-running the exact
same call after a crash, a timeout or a preemption picks up where the previous
attempt stopped.

On a cluster you do not have to write this at all: `slurm/` holds ready-made job
scripts that take a [config file](pipeline.md) and submit one job chain per step,
starting the right servers for each step's own model and releasing them when that
step is done. `examples/vllm-server-pool/` has both forms side by side.

## Why use it

- Run a whole annotation, or a chain of them, from one JSON/YAML config file
  with the `llm-annotate` CLI.
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
