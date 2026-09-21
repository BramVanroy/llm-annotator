# Migrating from 0.16

Everything that changed between `v0.16.0` and this release, grouped by where it
bites. A section that says nothing about your run needs nothing from you.

## Output directories and records

### A record without settings

`prepare_data` writes `<output_dir>/<task_prefix>selection.json` next to the
prepared data. The 0.16 record held the sample selection only, so it says
nothing about the prompt, and a resume could not tell whether the finished rows
answer the prompt that is asked for now. Reading such a record raises:

```text
The record in 'outputs/imdb-sentiment/selection.json' does not describe the
settings that its run was annotated with, so a resume cannot tell whether the
prompt still matches. Finish the run with the release that wrote it, annotate it
into a new 'output_dir', or overwrite it. See 'A record without settings' in
docs/migration.md.
```

Three ways forward:

- Finish the run with the release that wrote it, and let the next run start
  under this one.
- Point `output_dir` at a new directory, which annotates every row again and
  leaves the old result on disk.
- Pass `--overwrite` (`overwrite: true`, or `overwrite=True` from Python) to
  discard the finished rows in place and annotate every row again.

A fourth way keeps the finished rows, at your own risk: delete the
`selection.json` files under `output_dir` (inside a pipeline,
`<NN>-<name>/annotate/<name>_selection.json`). Only the rows that never ran are
then sent to the model. Nothing is compared, and the settings of that run are
recorded as the ones the whole directory was annotated with. If the prompt
changed since the finished rows were written, the output holds answers to both
prompts and no later run can tell. Use it only when you know the settings did
not change.

```bash
rm outputs/imdb-sentiment/*selection.json
llm-annotate pilot.yaml
```

### A finished step with no record

Inside a pipeline the same rule applies per step, and a step that already
finished needs one extra move. Its `output/` snapshot is what a re-run loads
instead of running the step, and with no record next to it nothing can say
whether that snapshot belongs to the config that is being run now, so the run
stops:

```text
Step 'sentiment' finished into 'outputs/imdb-sentiment/01-sentiment/output', but
there is no record of the settings it was annotated with, so this run cannot
tell whether its result still matches the config. Remove
'outputs/imdb-sentiment/01-sentiment/output' to run the step again: the rows in
its progress files are not sent to the model a second time. See 'A finished step
with no record' in docs/migration.md.
```

Removing the `output/` directory is cheap: the step runs again, reads the ids it
already finished out of its JSONL progress files, annotates only what is missing
(nothing, for a step that had finished) and writes the snapshot again, this time
with a record next to it.

```bash
rm -r outputs/imdb-sentiment/*/output
llm-annotate pilot.yaml
```

### The metadata file name

The counts of a run are written to
`<output_dir>/metadata/<task_prefix>annotation_metadata.json`. An output
directory from 0.16 holds them under `annotation_metadata.json`, without the
prefix and without the `metadata/` directory. Nothing in the library reads that
file, so an old one is left where it is. Delete it, or point your own scripts at
the new name.

### The Hub progress backup

A run that starts with an empty progress directory while its `hub_id` has a
`<task_prefix>progress_backup` branch now raises instead of replacing the backup
with a run that starts from zero. Restore the backup first:

```sh
python scripts/restore_progress_from_hub.py --hub-id my-org/imdb-sentiment --output-dir outputs/imdb-sentiment
```

Or pass `overwrite=True` (`--overwrite` in a pipeline) to delete the branch and
redo the run. A backup branch written by 0.16 holds no selection record; the
restore logs one line about that, and the next `prepare_data` records its own
settings.

`overwrite=True` also deletes that branch, so the rows of a discarded run do not
mix with the new backup.

## Config files

### Keys that no longer load

| Key | What to do |
| --- | --- |
| `init.batch_size` on a `vllm_offline` step | remove it and set the client block's own `batch_size` |
| `init.min_batch_size` on a `vllm_offline` step | remove it |
| `options.n` on either vLLM provider | remove it, or set it to 1 |
| `pool.gpus_per_vllm_server` | write it as `engine.tensor_parallel_size` |

The first two fail at load time with a message that names them, and the third
with the unknown-option error. One response per sample is read, so `n` above 1
was only ever paid for and dropped.

### `init` is checked against the constructor

Every `init` key is validated against the client constructor's signature, and
every `options` key against the runtime-options dataclass, when the config
loads. A typo that 0.16 accepted and ignored is now an error before the first
request, with the accepted names listed. An engine setting written under `init`
is rejected the same way, with the move spelled out.

### A step that switches provider inherits nothing

A step whose `client` block names a different `provider` than the top-level
block no longer inherits that block's `init`, `options`, `engine` or pool keys.
Those name fields of the other provider's constructor and dataclass, so an
inherited `api_key` would send one provider's key to another. Write out what
such a step needs.

### Schema properties may not shadow a column

An `output_schema` whose top-level properties include the `idx_column` or a
bookkeeping column name (`response`, `finish_reason`, `num_tokens`, `error`,
`error_type`, `reasoning`, `valid_fields`, `valid`, `messages`, each behind the
task prefix) is rejected before the step's first request. Rename the property,
or use another `task_prefix` or `idx_column`.

Keys that a model returns outside the schema are no longer written as columns.
That can only happen on the vLLM providers, which pass a schema on without
`additionalProperties: false`. Add such a key to the schema's `properties` if
you need it.

### Prompt files are read verbatim

A `prompt_file` or `system_prompt_file` reaches the model as the bytes it holds,
whatever its suffix. A `.json` system prompt is sent as the JSON text it holds,
where 0.16 rewrote some files as Markdown bullets.

### Relative data paths

`dataset.data_dir` and `dataset.data_files` resolve against the config file's
directory when `dataset.name` is a local builder or a directory, like every
other path in a config. A config that relied on them resolving against the
current working directory needs the paths rewritten, or an absolute path.

### Config errors on the command line

`llm-annotate` reports a config that does not load as one
`error: <location>: <message>` line per problem on stderr and exits with status
2, instead of printing a traceback. `--debug` keeps the traceback. Only config
loading is reported this way.

## Python API

### Renamed and removed arguments

| Was | Now |
| --- | --- |
| `full_prompt_template=` | `prompt_template=`, which is required |
| `prompt_field_swapper={"content": "body"}` | apply the rename yourself: `template.replace("{content}", "{body}")` |
| `VLLMOfflineClient(batch_size=..., min_batch_size=...)` | the annotator's own `batch_size`, at or above `max_num_seqs` |
| `client.batch_generate(..., use_batch_api=True, poll_interval=30)` | `OpenAIClient(..., use_batch_api=True, batch_poll_interval=30)` |

```python
template = "Summarize this document: {content}"

# before
anno.prepare_data(
    output_dir=out,
    full_prompt_template=template,
    prompt_field_swapper={"content": "body"},
)
# after
anno.prepare_data(
    output_dir=out,
    prompt_template=template.replace("{content}", "{body}"),
)
```

### Imports

The package root exports 21 names instead of 36. Everything else is importable
from the module that defines it:

| Name | Import from |
| --- | --- |
| `SelectionRecord` | `llm_annotator.annotator` |
| `OnError`, `Provider`, `ProviderRuntimeOptions` | `llm_annotator.clients.base` |
| `VLLMBaseRuntimeOptions` | `llm_annotator.clients.vllm_online_client` |
| `ClientConfig`, `DatasetConfig`, `StepConfig`, `load_config_file` | `llm_annotator.config` |
| `build_annotator`, `build_client`, `wait_for_servers` | `llm_annotator.pool` |
| `extract_prompt_prefix`, `get_hash` | `llm_annotator.utils` |

<!-- docs-test: fragment -->
```python
# before
from llm_annotator import ClientConfig
# after
from llm_annotator.config import ClientConfig
```

`build_client` and `build_annotator` were methods of `ClientConfig` and are now
functions that take the config as their first argument, in the new
`llm_annotator.pool` module:

```python
from llm_annotator.pool import build_annotator

# before: annotator = client_config.build_annotator(root, verbose=True)
annotator = build_annotator(client_config, root, verbose=True)
```

### Removed names

- `auto_reduce_batch_size`, the decorator that halved a chunk on a CUDA
  out-of-memory error. vLLM sizes its own KV cache, so there is nothing to
  halve.
- `ConfigurationError` and `ParsingError`, which nothing raised.
  `LLMClientError` is the base class to catch for every error of a client.
- `SelectionRecord.is_stale()` and the attributes `shuffle_seed`,
  `source_signature` and `reuse_idx_column`. Those values are entries of
  `SelectionRecord.components`, and
  [`changed_components`][llm_annotator.annotator.SelectionRecord.changed_components]
  is what compares a request against a record.
- `llm_annotator.external.propella`, which left the installed package. Copy
  `examples/propella/propella_schema.py` out of the repository if you used it.

### Moved directories

`examples/wiki-nl-persona-qa/` and `examples/model-comparison/` are now
`case-studies/wiki-nl-persona-qa/` and `case-studies/model-comparison/`.

## Clients

### Errors from a direct `generate` call

`client.generate(...)` follows `on_error` the way `batch_generate` always did.
With `"warn"` or `"ignore"` it returns a `Response` that carries `error` and
`error_type`; with `"raise"` it raises `ProviderError` rather than the SDK's own
exception. A malformed message, or `n` above 1, is still a `ValueError` for the
caller whatever `on_error` says, because the request payload is built outside
that path.

### `batch_generate` takes three arguments

Every client's `batch_generate` takes `messages`, `options` and `gen_kwargs`.
The OpenAI Batch API is a constructor setting now:

```yaml
client:
  provider: openai
  model: gpt-4o-mini
  init:
    use_batch_api: true
    batch_poll_interval: 30
```

`VLLMOnlineClient` does not accept `use_batch_api`; as an `init` key of a
`vllm_online` step it fails when the config loads, as an unknown key.

### A vLLM server batch is one request per sample

`VLLMOnlineClient` no longer posts to vLLM's `/v1/chat/completions/batch` route.
No config key or Python argument changes, but two behaviours do:

- `{prefix}num_tokens` is filled on a `vllm_online` step. It used to be `None`,
  because the batch route reported one `usage` block for the whole batch. Code
  that treated the column as always empty on this provider (a filter, a
  throughput report) now gets real numbers.
- A failing sample no longer takes its batch down with it. A batch in which one
  prompt is too long used to write errors for every row of that batch.

`timeout` (3600 seconds), `max_retries` (2) and `max_workers` (`None`, which
sends every request of a batch at once) are constructor settings of that client,
reachable from a config under `init`. See
[Provider setup](provider-info.md#vllm-online-server).

## SLURM

Nothing in `slurm/cluster.env` has to change: no variable was removed or
renamed. Six optional ones were added, and `cluster.env.example` documents each
of them:

- `MAX_CONCURRENT_SERVERS` throttles a server array at a site with a per-user
  GPU limit.
- `SERVER_HOST_CMD` picks the command a server uses to publish its address.
- `MODEL_DOWNLOAD`, `DOWNLOAD_PARTITION` and `DOWNLOAD_TIME` fetch the weights
  in one CPU job per model before any GPU is allocated.
- `CANCEL_SERVERS_ON_EXIT=0` keeps a step's servers alive after its client ends.

`submit_pipeline.sh` also takes `--max-resubmits N`, which queues follow-up
attempts per step in advance. [The SLURM guide](slurm.md) has all of them.
