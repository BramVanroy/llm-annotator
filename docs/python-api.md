# Python API guide

Everything a config file describes is also reachable from Python, plus the
three hooks a config cannot express (`preprocess_fn`, `postprocess_fn` and
`validate_fn`, which take callables).

[`Annotator`][llm_annotator.annotator.Annotator] has four entry points. Two do
the work and two are wrappers around them:

| Method | What it does |
| --- | --- |
| `prepare_data` | load the dataset, add the id column, apply the prompt template, optionally sort by length, cache the result and back it up to the Hub. No inference. |
| `run_annotation` | inference over prepared data. |
| `annotate_dataset` | `prepare_data` followed by `run_annotation`. |
| `generate_dataset` | build a dataset from a list of prompts, then the same path. |

## The staged workflow

The split exists for cluster work: preparation is expensive and needs no GPU, so
a crashed inference job restarts without repeating it.

--8<-- "README.md:two-step"

`prepare_data` records the settings that decide what the prepared data holds
(the prompt template, the system message, `sort_by_length`, the source dataset
and the rest) next to it, in `<output_dir>/<task_prefix>selection.json`. A later
call with an edited prompt template is refused instead of reused, so one output
never holds answers to two prompts. [Growing a run](growing-a-run.md) has the
full list of what may change, what is rejected and the way out.

`force_data_preparation=True` on `prepare_data` (or on `annotate_dataset`)
prepares again even when a local cache or a Hub backup exists.

### The background Hub backup

Every `upload_every_n_samples` rows the progress files are pushed to the
`<task_prefix>progress_backup` branch. That upload runs on a background thread,
so the next batch is dispatched while it is in flight. A cycle that comes due
while the previous upload still runs is skipped, since the next one carries the
same rows and the ones after them. An upload that fails is logged at warning
level and the run continues, because the progress files on disk are the copy
that a resume reads. The upload at the end of the run waits for the background
one and is not skipped: a failure there ends the run, since that is the upload a
restore on another machine depends on.

## Several tasks in one output directory

`task_prefix` is put in front of every column that the annotator writes and in
front of every artifact it stores, so two tasks can annotate the same data into
one `output_dir` (and one `hub_id`) without reading each other's progress files:

```python
sentiment = anno.annotate_dataset(
    output_dir="outputs/imdb",
    prompt_template="Sentiment: {text}",
    dataset=ds,
    task_prefix="sentiment_",
    keep_columns=True,
    keep_idx_column=True,
)

topic = anno.annotate_dataset(
    output_dir="outputs/imdb",
    prompt_template="Topic: {text}",
    dataset=sentiment,      # the sentiment columns travel along
    task_prefix="topic_",
    keep_columns=True,
    keep_idx_column=True,
    reuse_idx_column=True,  # keep the ids that the first call handed out
)
```

Each task gets its own `<task_prefix>prepared_dataset/`,
`<task_prefix>progress_backup/`, `<task_prefix>selection.json` and
`metadata/<task_prefix>annotation_metadata.json`, and its own pair of Hub
branches.

The final dataset is shared: it is written to the root of `output_dir` and
pushed to the `main` branch of `hub_id`, and the task that finishes last
replaces it. That is the point of the example above. The second call annotates
the result of the first and, with `keep_columns=True`, carries its columns
along, so the dataset that stays behind holds the columns of both tasks. Two
tasks whose results must both survive on disk need two `output_dir` values.

`overwrite=True` discards the finished rows of one task: its progress files, its
Hub progress branch, its metadata file and the shared final dataset in the root,
which the run writes again. It keeps that task's prepared data (so a crashed run
does not prepare it a second time) and everything that belongs to another
`task_prefix`.

## Errors and retries

A client's `on_error` setting (`"raise"`, `"warn"` or `"ignore"`) decides what
happens when a request fails. With `"warn"` or `"ignore"` the client returns a
`Response` with `error` and `error_type` set instead of raising, and the
annotator records those two fields on the sample and marks it invalid.
[Reading the error columns](troubleshooting.md#reading-the-error-columns) says
what each column holds.

An errored row is final once it is written: a run that resumes the same
`output_dir` does not send it to the model again, so an error that the sample
itself causes (a prompt longer than the model's context, for instance) is not
repeated on every resume. Pass `retry_errors=True` to annotate every errored row
again, or a list of `error_type` values to annotate only those. The selected
rows are removed from the progress files before the run starts, so they are
annotated like rows that never ran.

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

When every sample of a batch errors, the backend is probably down. The rows of
such a batch are held back and written once a later batch succeeds. After
`max_consecutive_failed_batches` (default 10) such batches in a row, the run
stops with `TooManyConsecutiveFailedBatchesError`. The rows that are held back
at that point are never written, so the resumed run annotates them again. Set it
to 0 to disable both the abort and the hold-back.

### What a run reports

Every run ends with a log line that says how many samples finished with an error
(with a count per `error_type`, which are the names that `retry_errors` takes)
and how many have invalid fields. The same counts are written to
`<output_dir>/metadata/<task_prefix>annotation_metadata.json`.

A second log line gives the throughput, and the same numbers go to the
`run_summary` key of that file:

```json
"run_summary": {
    "num_rows": 8000,
    "num_output_tokens": 1536000,
    "elapsed_seconds": 412.7,
    "rows_per_second": 19.39,
    "output_tokens_per_second": 3721.83
}
```

Those numbers cover the invocation that wrote them and nothing else: the rows
that it annotated, the output tokens that they hold, and the seconds from the
warm-up to its last written row. A run that resumes another one therefore
reports its own throughput, not the average over every attempt, and a run that
finds every row already annotated writes `"run_summary": null`. An errored row
has no token count and adds 0 to `num_output_tokens`.

## Many vLLM servers at once

[`VLLMQueueAnnotator`][llm_annotator.annotator.VLLMQueueAnnotator] spreads one
workload over a pool of vLLM servers, for instance one server per GPU of a
multi-node SLURM allocation. It is a drop-in `Annotator`: the same four entry
points, the same JSONL progress files, the same resume behaviour. Batches are
dispatched to whichever server is free, with at most `queue_size` batches in
flight at a time. Set `max_concurrent_batches_per_client` to give each server
more than one batch at a time without raising `batch_size`. The two multiply:
with four servers, four concurrent batches each and a `batch_size` of 64, a
server holds 256 prompts and the pool 1024. `queue_size` never drops below the
number of concurrent batches, since a smaller queue would leave servers idle,
and defaults to four batches per batch slot.

```python
from llm_annotator import (
    VLLMOnlineClient,
    VLLMOnlineRuntimeOptions,
    VLLMQueueAnnotator,
)

clients = [
    VLLMOnlineClient(
        model="Qwen/Qwen3-8B", base_url=f"http://{host}:8000/v1"
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
attempt stopped. See [Growing a run](growing-a-run.md).

A server that fails a whole batch and then does not answer `/health` is removed
from the pool, and its batch is sent to another server (no errored rows are
written for it). The run stops only once no server is left. A config-driven run
also re-admits a server that recovers, see
[Many vLLM servers](pipeline.md#many-vllm-servers).

A cluster job submitter needs nothing beyond the [config file](pipeline.md) and
four CLI flags to drive this: `--describe-steps` to plan the allocation,
`--serve-args` to start each step's servers with its own model, and
`--hosts-file` or `--url-glob` to hand them back in. `examples/vllm-server-pool/`
has both the Python-API and config-driven forms side by side, and
[`slurm/`](slurm.md) is a ready-made submitter built on those flags:

```bash
cp slurm/cluster.env.example slurm/cluster.env   # once, per cluster
./slurm/submit_pipeline.sh examples/vllm-server-pool/pipeline.yaml
```

## What the package exports

`llm_annotator` exports what the workflow above uses: `Annotator`,
`VLLMQueueAnnotator`, the four clients with their runtime options classes,
`Response`, the exceptions (`LLMClientError`, `ProviderError`,
`TooManyConsecutiveFailedBatchesError`), `run_pipeline`,
`load_pipeline_config`, `PipelineConfig`, `restore_progress_from_hub`, and
`configure_logging`, `set_log_level`, `get_logger`.

Everything else is importable from the module that defines it, which
[Migrating from 0.16](migration.md#python-api) lists.
