# Robust, resumable LLM dataset annotation

[![CI](https://github.com/BramVanroy/llm-annotator/actions/workflows/ci.yml/badge.svg)](https://github.com/BramVanroy/llm-annotator/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BramVanroy/llm-annotator/branch/main/graph/badge.svg)](https://codecov.io/gh/BramVanroy/llm-annotator)
![PyPI version](https://img.shields.io/pypi/v/llm-annotator)
[![Python versions](https://img.shields.io/pypi/pyversions/llm-annotator.svg)](https://pypi.org/project/llm-annotator/)
[![License](https://img.shields.io/github/license/BramVanroy/llm-annotator)](LICENSE)


`llm-annotator` is a Python 3.12+ library that runs an LLM over a dataset and
writes the answers back as columns. A run is resumable: every finished sample is
appended to a JSONL progress file, so a crashed, preempted or timed-out job
continues where it stopped.

Four providers share one interface:

- vLLM in-process (`VLLMOfflineClient`, config name `vllm_offline`)
- vLLM server (`VLLMOnlineClient`, config name `vllm_online`)
- OpenAI (`OpenAIClient`, config name `openai`)
- Anthropic (`ClaudeClient`, config name `claude`)

## Install

<!-- --8<-- [start:install] -->
```sh
uv add llm-annotator
```

or

```sh
pip install llm-annotator
```

A provider needs its extra:

```sh
uv add "llm-annotator[vllm]"       # vllm_offline
uv add "llm-annotator[openai]"     # openai and vllm_online
uv add "llm-annotator[anthropic]"  # claude
```
<!-- --8<-- [end:install] -->

The online vLLM client speaks the OpenAI protocol, so it takes the `openai`
extra rather than the much heavier `vllm` one. Authentication variables and
per-provider notes are in [docs/provider-info.md](docs/provider-info.md), which
also has the two FlashInfer wheels to install next to the `vllm` extra wherever
you serve models, so that vLLM does not JIT-compile its kernels at start-up.

## Quickstart

Describe the run in one YAML (or JSON) file and start it. No Python needed.

<!-- --8<-- [start:config-quickstart] -->
```yaml title="my-pipeline.yaml"
output_dir: outputs/imdb-sentiment

dataset:
  name: stanfordnlp/imdb
  split: test
  max_num_samples: 20

client:
  provider: vllm_offline
  model: HuggingFaceTB/SmolLM2-135M-Instruct

steps:
  - name: sentiment
    prompt: "Classify the sentiment: {text}"
```

```sh
llm-annotate my-pipeline.yaml
```

The result is a dataset with a `sentiment_response` column next to the original
`text`, written to `outputs/imdb-sentiment/final/`.
<!-- --8<-- [end:config-quickstart] -->

A config can hold several steps that run in order, each annotating what the
previous one produced, which is what a generate-then-judge workflow needs.
[examples/pipeline-qa/](examples/pipeline-qa/) is a complete two-step example
and [docs/pipeline.md](docs/pipeline.md) is the key reference.

The same run from Python:

<!-- --8<-- [start:one-step] -->
```python
from llm_annotator import Annotator, VLLMOfflineClient

client = VLLMOfflineClient(model="HuggingFaceTB/SmolLM2-135M-Instruct")

with Annotator(client=client, verbose=True) as anno:
    ds = anno.annotate_dataset(
        output_dir="outputs/imdb-sentiment",
        prompt_template="Classify the sentiment: {text}",
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        max_num_samples=20,
    )
```

Or build a dataset from scratch instead of annotating one:

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
<!-- --8<-- [end:one-step] -->

## What it does

- No-code config runs: prompts, schemas, model, dataset and any number of
  chained annotation steps in one JSON or YAML file, started with
  `llm-annotate my-pipeline.yaml`.
- Staged pipeline: `prepare_data` applies the templates and sorts, and
  `run_annotation` does the inference, so a crashed GPU job restarts without
  repeating the preparation.
- Resumable runs: results are streamed to JSONL checkpoints per sample. Raising
  `dataset.max_num_samples` and re-running annotates only the new rows, see
  [docs/growing-a-run.md](docs/growing-a-run.md).
- Multi-server vLLM: `VLLMQueueAnnotator` runs one workload over a pool of vLLM
  servers, one per GPU of a multi-node allocation. See
  [examples/vllm-server-pool/](examples/vllm-server-pool/) for the config-driven
  and the Python form.
- SLURM out of the box: `slurm/submit_pipeline.sh` turns a config into one job
  chain per step, with everything cluster-specific in one cluster file. See
  [slurm/README.md](slurm/README.md).
- Annotation of an existing dataset and generation from scratch.
- Structured output through a JSON schema, with the schema's properties as
  columns.
- A thinking model's reasoning trace in its own column, on either vLLM provider
  and on Claude.
- Retry and validation hooks, and per-sample error columns instead of a dead
  run.
- Hugging Face Hub backup of the prepared data and the progress files while the
  run continues, and of the final dataset when it ends.

## Two-step staged workflow

For a large dataset or a cluster job, split the work: `prepare_data` loads the
dataset, applies the prompt template, optionally sorts by length and caches the
result; `run_annotation` only does inference against that prepared data.

<!-- --8<-- [start:two-step] -->
```python
from llm_annotator import Annotator, VLLMOfflineClient

client = VLLMOfflineClient(model="Qwen/Qwen3-8B", max_model_len=4096)

HUB_ID = "my-org/imdb-sentiment"  # backups and the final dataset
PROMPT = "Classify the sentiment: {text}"

with Annotator(client=client, verbose=True) as anno:
    prepared_dataset, local_path, hub_id = anno.prepare_data(
        output_dir="outputs/imdb-sentiment",
        prompt_template=PROMPT,
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        max_num_samples=100,
        sort_by_length=True,
        hub_id=HUB_ID,
    )

    ds = anno.run_annotation(
        output_dir="outputs/imdb-sentiment",
        prompt_template=PROMPT,
        prepared_dataset=prepared_dataset,
        hub_id=HUB_ID,
        upload_every_n_samples=500,
    )
```

If inference fails, run the second call again with the same `output_dir` and
`hub_id`: the prepared data comes back from the Hub and the samples already in
the local progress files are skipped.

One `hub_id` drives all three Hub destinations. The prepared data and the JSONL
progress files go to temporary branches of that repository, the final dataset is
pushed to its `main` branch, and both temporary branches are deleted once the
run finishes.

On a machine with no local progress files (a purged scratch directory, or a run
that moves to another cluster), restore the progress backup first:

```sh
llm-annotate-restore --hub-id my-org/imdb-sentiment --output-dir outputs/imdb-sentiment
```

`run_annotation` refuses to start when the repository has a progress backup
while the local progress directory is empty, so a forgotten restore cannot
replace the backup with a run that starts from zero.
<!-- --8<-- [end:two-step] -->

The settings that produced the prepared data are recorded next to it, so a later
call with an edited prompt template is refused instead of reused. What may
change between two runs, and what that costs, is in
[docs/growing-a-run.md](docs/growing-a-run.md).

## Run it on SLURM

The same config runs on a cluster without a scheduler-specific rewrite. Fill in
one small cluster file (partitions, accounting, cores per GPU) and submit:

```sh
cp slurm/cluster.env.example slurm/cluster.env
./slurm/submit_pipeline.sh --dry-run my-pipeline.yaml   # inspect the jobs
./slurm/submit_pipeline.sh my-pipeline.yaml
```

Each step becomes its own job chain: a step served by vLLM gets a GPU server
array plus a client, a step on a hosted API gets a CPU-only job, and GPUs are
released as soon as the step that needed them is done. Details in
[slurm/README.md](slurm/README.md).

## Documentation

The full documentation is at
[bramvanroy.github.io/llm-annotator](https://bramvanroy.github.io/llm-annotator/).

- [Choosing a provider](docs/choosing-a-provider.md): which client fits the
  hardware you have.
- [Annotating from a config file](docs/pipeline.md): every config key, the
  multi-step workflow and the CLI.
- [Python API guide](docs/python-api.md): the staged workflow, several tasks in
  one directory, errors and retries, server pools.
- [Growing a run](docs/growing-a-run.md): resuming, raising the sample cap,
  editing a prompt mid-run.
- [Troubleshooting](docs/troubleshooting.md): what an error message means and
  what to do about it.
- [Provider setup](docs/provider-info.md): extras, authentication, vLLM tuning.
- [SLURM](slurm/README.md): the job submitter.
- [Migrating from 0.16](docs/migration.md): what changed since the last
  release.

[examples/](examples/) holds runnable examples, and
[case-studies/](case-studies/) holds two complete research projects built on the
library.

## Contributing

Development setup, the make targets, the test markers and the docs layout are
in [CONTRIBUTING.md](CONTRIBUTING.md).

```sh
uv sync --dev
make style
make quality
make typecheck
make test
```
