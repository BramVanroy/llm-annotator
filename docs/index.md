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
  engine:
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

`prepare_data` records the settings that decide what the prepared data holds (the prompt template,
the system message, `sort_by_length`, the source dataset and the rest) next to it. A later call
with an edited prompt template is refused instead of reused, so one output never holds answers to
two prompts. See [Growing a run](growing-a-run.md) for what may change, what is rejected, and the
way out.

### Several tasks in one output directory

`task_prefix` is put in front of every column that the annotator writes and in front of every
artifact it stores, so two tasks can annotate the same data into one `output_dir` (and one
`hub_id`) without reading each other's progress files:

```python
for task, prompt in (("sentiment_", "Sentiment: {text}"), ("topic_", "Topic: {text}")):
    ds = anno.annotate_dataset(
        output_dir="outputs/imdb",
        prompt_template=prompt,
        dataset=ds,
        task_prefix=task,
        keep_columns=True,
    )
```

Each task gets its own `<task_prefix>prepared_dataset/`, `<task_prefix>progress_backup/`,
`<task_prefix>selection.json` and `metadata/<task_prefix>annotation_metadata.json`, and its own
pair of Hub branches.

The final dataset is shared: it is written to the root of `output_dir` and pushed to the `main`
branch of `hub_id`, and the task that finishes last replaces it. That is the point of the example
above. Each call reads the previous result and, with `keep_columns=True`, carries its columns
along, so the dataset that stays behind holds the columns of every task. Two tasks that must end in
two datasets need two `output_dir` values.

`overwrite=True` discards the finished rows of one task: its progress files, its Hub progress
branch, its metadata file and the shared final dataset in the root, which the run writes again. It
keeps that task's prepared data (so a crashed run does not prepare it a second time) and everything
that belongs to another `task_prefix`.

### Errors and retries

A client's `on_error` setting (`"raise"`, `"warn"` or `"ignore"`) decides what
happens when a request fails. With `"warn"` or `"ignore"` the client returns a
`Response` with `error` and `error_type` set instead of raising, and the
annotator records those two fields on the sample and marks it invalid.

An errored row is final once it is written: a run that resumes the same
`output_dir` does not send it to the model again, so an error that the sample
itself causes (a prompt longer than the model's context, for instance) is not
repeated on every resume. Pass `retry_errors=True` to annotate every errored
row again, or a list of `error_type` values to annotate only those. The
selected rows are removed from the progress files before the run starts, so
they are annotated like rows that never ran.

```python
ds = anno.run_annotation(
    output_dir="outputs/imdb-sentiment",
    prompt_template="Classify the sentiment: {text}",
    prepared_dataset=prepared_dataset,
    retry_errors=["ConnectError", "APITimeoutError"],
)
```

The same option exists for a config-driven pipeline as `--retry-errors`, see
[Command line](pipeline.md#command-line):

```bash
llm-annotate my-pipeline.yaml --retry-errors ConnectError APITimeoutError
```

When every sample of a batch errors, the backend is probably down. The rows
of such a batch are held back and written once a later batch succeeds. After
`max_consecutive_failed_batches` (default 10) such batches in a row, the run
stops with `TooManyConsecutiveFailedBatchesError`. The rows that are held back
at that point are never written, so the resumed run annotates them again. Set
it to 0 to disable both the abort and the hold-back.

Every run ends with a log line that says how many samples finished with an
error (with a count per `error_type`, which are the names that `retry_errors`
takes) and how many have invalid fields. The same counts are written to
`<output_dir>/metadata/<task_prefix>annotation_metadata.json`.

### Many vLLM servers at once

`VLLMQueueAnnotator` spreads one workload over a pool of vLLM servers -- for
instance one server per GPU of a multi-node SLURM allocation. It is a drop-in
`Annotator`: the same four entry points, the same JSONL progress files, the same
resume behaviour. Batches are dispatched to whichever server is free, with at
most `queue_size` batches in flight at a time. Set
`max_concurrent_batches_per_client` to send more than one request to each
server without increasing the request `batch_size`. The two multiply: with four
servers, four concurrent requests each and a `batch_size` of 64, a server holds
256 prompts and the pool 1024. `queue_size` never drops below the number of
concurrent requests, since a smaller queue would leave servers idle, and
defaults to four batches per request slot.

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

with VLLMQueueAnnotator(
    clients=clients,
    batch_size=64,
    max_concurrent_batches_per_client=4,
    verbose=True,
) as anno:
    ds = anno.annotate_dataset(
        output_dir="outputs/imdb-sentiment",
        prompt_template="Classify the sentiment: {text}",
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        options=VLLMOnlineRuntimeOptions(
            max_completion_tokens=128, temperature=0.0
        ),
    )
```

Because results are written per sample and keyed by `idx`, re-running the exact
same call after a crash, a timeout or a preemption picks up where the previous
attempt stopped.

A server that fails a whole batch and then does not answer `/health` is
removed from the pool, and its batch is sent to another server (no errored
rows are written for it). The run stops only once no server is left. A
config-driven run also re-admits a server that recovers, see
[Many vLLM servers](pipeline.md#many-vllm-servers).

A cluster job submitter needs nothing beyond the [config file](pipeline.md) and
four CLI flags to drive this: `--describe-steps` to plan the allocation,
`--serve-args` to start each step's servers with its own model, and
`--hosts-file` or `--url-glob` to hand them back in. `examples/vllm-server-pool/` has both the
Python-API and config-driven forms side by side, and [`slurm/`](slurm.md) is a
ready-made submitter built on those flags:

```bash
cp slurm/cluster.env.example slurm/cluster.env   # once, per cluster
./slurm/submit_pipeline.sh examples/vllm-server-pool/pipeline.yaml
```

## Why use it

- Run a whole annotation, or a chain of them, from one JSON/YAML config file
  with the `llm-annotate` CLI.
- Staged `prepare_data` + `run_annotation` pipeline for SLURM and
  cluster workflows:  expensive data preparation is done once and stored.
- Resume interrupted generation runs from JSONL checkpoints, and grow a finished run by raising
  `dataset.max_num_samples` and re-running; see [Growing a run](growing-a-run.md).
- Validate and post-process outputs with custom callables.
- Enforce structured responses through JSON schemas.
- Keep a thinking model's reasoning trace in its own column, separated from the
  answer; see [Annotating from a config file](pipeline.md#how-steps-see-each-others-output).
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
