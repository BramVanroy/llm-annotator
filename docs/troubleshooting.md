# Troubleshooting

Every message below is quoted from the source, with `<...>` where the library
fills in a value.

## How an error is reported

A config that does not load is reported as one line per problem on stderr, and
`llm-annotate` exits with status 2:

```console
$ llm-annotate cfg.yaml
error: client: Unknown 'init' keys for provider 'openai': ['on_eror']. OpenAIClient takes ['api_key', 'base_url', 'max_retries', 'max_workers', 'on_error', 'timeout'].
```

The part before the message is the key the problem belongs to, or the config
file itself when the problem names no key. `--debug` prints the full traceback
instead, which is what a bug report needs. Only config loading is reported this
way; an error raised while the pipeline runs keeps its traceback.

## Authentication and access

### 403 or "gated repo" from the Hub

Symptom: the run stops while it loads the model or the dataset, with a 403 from
`huggingface.co` or a message about a gated repository.

Cause: the repository needs an accepted licence and a token. `Qwen/Qwen3-8B`
and `HuggingFaceTB/SmolLM2-135M-Instruct` are ungated; `meta-llama/*` and many
other repositories are not.

Fix: accept the licence on the model page while logged in, then give the machine
a token.

```sh
hf auth login          # or: export HF_TOKEN="..."
```

On a cluster the token has to reach the compute node. `slurm/submit_pipeline.sh`
submits with `--export=ALL`, so `HF_TOKEN` in the submitting shell is carried
along; `hf auth login` writes it to `HF_HOME`, which every node has to be able
to read.

### No API key

Symptom: an `openai` or `claude` step fails on the first request with an
authentication error from the provider's SDK.

Cause: the client reads `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` from the
environment unless a config gives `init.api_key`.

Fix:

```sh
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

See [Provider setup](provider-info.md#environment-variables).

## Errors when the config loads

### The file

```text
Config file '<path>' does not exist.
```

```text
Config file '<path>' must contain a mapping at the top level, got <type>.
```

A config is a mapping with `output_dir` and `steps` at the top level. A file
that starts with a list is usually a `steps` block that lost its key.

### Command-line overrides

```text
--set expects 'key=value', got '<text>'. Use a dotted key to reach a nested value, as in --set dataset.max_num_samples=2000.
```

```text
Override '<key>': '<segment>' is not a list index. A list in a config is addressed by position, as in 'steps.0.name'.
```

```text
Override '<key>': position <n> is out of range for a list of <n>.
```

```text
Override '<key>': '<segment>' cannot be set inside a <type>.
```

```text
'<key>' is set twice on the command line. Give it once, either through its own flag or through --set.
```

```text
Config '<path>' has no 'dataset' block, so --max-num-samples and --shuffle-seed have nothing to apply to. A pipeline whose first step generates its own data sizes it with 'num_samples' on that step: --set steps.0.num_samples=N.
```

See [Overriding config keys](pipeline.md#overriding-config-keys).

### The dataset block

```text
Provide exactly one of 'name' or 'path' in the dataset block.
```

```text
'data_dir' and 'data_files' only apply to 'name', not 'path'.
```

```text
The pipeline needs a 'dataset' block because its first step ('<name>') annotates an existing dataset.
```

```text
The first step generates its own data, so the 'dataset' block would be ignored. Remove one of the two.
```

`name` is a Hub id or a packaged builder (`json`, `csv`, `parquet`, ...);
`path` is a directory written with `save_to_disk`. A `generate` step brings its
own prompts and must be the first step, so a pipeline has either that or a
`dataset` block.

### Provider and model

```text
Unknown provider '<name>'. Choose one of <providers>.
```

```text
Provider '<name>' needs an explicit 'model'.
```

```text
Provider '<name>' needs its optional dependency. Install it with `uv sync --extra <extra>`.
```

```text
Step '<name>' has no client to run on. Add a top-level 'client' block to share one across steps, or a 'client' block to this step.
```

```text
Step '<name>': invalid 'client' block. <problem>
```

The four spellings are `vllm_offline`, `vllm_online`, `openai` and `claude`.
Outside a checkout, install the extra with `uv add "llm-annotator[vllm]"` or
`pip install "llm-annotator[vllm]"`. A `vllm_online` step may leave `model` out
only when a running server can be asked what it serves.

### Where a setting goes

```text
Move <keys> from 'init' to 'engine'. Engine settings live there so that both vLLM providers spell them the same way.
```

```text
'engine' configures a vLLM engine, so provider '<name>' has no use for it. Hosted providers are configured with 'init' and 'options'.
```

```text
Unknown 'init' keys for provider '<name>': <keys>. <ClientClass> takes <names>.
```

```text
Unknown 'options' for provider '<name>': <keys>. Valid options are <names>.
```

```text
'init' sets 'model', which the client block names itself. Write it as 'model' next to 'provider'.
```

```text
'init' sets 'base_url', but each server of a pool gets its own. The pool's servers are named by 'base_urls', 'hosts_file' or 'url_glob'.
```

```text
'init' sets 'batch_size'. A run has one batch size, the client block's own 'batch_size', which decides how many samples go to the provider per call.
```

Each message lists the names that are accepted where it fires.
[Where a setting goes](pipeline.md#where-a-setting-goes) is the table of which
block a key belongs in.

### Pools of vLLM servers

```text
Provide at most one of 'base_urls', 'hosts_file' or 'url_glob', got <keys>.
```

```text
'<key>' describes a pool of vLLM servers and needs provider 'vllm_online', not '<name>'.
```

```text
<keys> size a pool of vLLM servers, so provider '<name>' has no use for them: it sends one request at a time. Use 'batch_size' to change how much work that request carries.
```

```text
'pool.min_servers' cannot exceed 'pool.servers'.
```

```text
'pool.gpus_per_vllm_server' moved to 'engine.tensor_parallel_size', which both vLLM providers read, so a step states its GPU count once.
```

```text
'queue_size' is <n>, below the minimum of <n> for this step: <n> server(s) times <n> concurrent batch(es) each ('max_concurrent_batches_per_client'). A queue smaller than that leaves servers idle. Set 'queue_size' to at least <n>, or remove it to keep <n> batches queued per batch slot (<n>).
```

```text
hosts_file '<path>' does not exist.
```

```text
url_glob '<pattern>' matched no files under <directory>.
```

```text
No vLLM server URLs found for the client pool.
```

An empty `url_glob` on a cluster means no server published its `.url` file. The
`vllm-<step>_*.err` logs say why. See
[Many vLLM servers](pipeline.md#many-vllm-servers).

### Steps

```text
Step 'name' must not be empty.
```

```text
Step '<name>': provide either '<inline key>' or '<file key>', not both.
```

```text
Step '<name>': an 'annotate' step needs 'prompt' or 'prompt_file'.
```

```text
Step '<name>': 'prompts' and 'num_samples' only apply to a 'generate' step.
```

```text
Step '<name>': a 'generate' step needs 'prompts'.
```

```text
Step '<name>': 'filter_invalid' relies on schema validation, so it needs 'output_schema' or 'output_schema_file'.
```

```text
Step '<name>': the template of a 'generate' step must contain the '{prompt}' placeholder, which is filled in with each entry of 'prompts'.
```

```text
Step names must be unique, found duplicates: <names>.
```

```text
Step task prefixes must be unique, found duplicates: <prefixes>.
```

```text
Step '<name>' renames or drops '<column>', the 'idx_column'. It identifies a row in every step and is removed from the final dataset automatically.
```

```text
Step '<name>' is a 'generate' step, which replaces the dataset instead of annotating it, so it can only be the first step.
```

```text
Set the schema either as the step's 'output_schema' or as client options 'json_schema', not both.
```

### Prompt and schema files

```text
File '<path>' referenced from the config does not exist (resolved to '<path>').
```

Every path in a config resolves against the config file's directory, never the
current working directory, which the resolved path in the message shows.

```text
Schema file '<path>' must contain a JSON object, got <type>.
```

```text
Prompts file '<path>' is not valid JSON: <problem>. A '.json' prompts file holds a list of strings, one per prompt.
```

```text
Prompts file '<path>' holds a <type>. A '.json' prompts file holds a list of strings, one per prompt.
```

```text
Prompts file '<path>' holds an empty list.
```

```text
Prompts file '<path>' holds entries that are not strings, at position(s) <positions>. A '.json' prompts file holds a list of strings, one per prompt.
```

```text
Step '<name>': 'prompts' resolved to an empty list.
```

A prompts file whose suffix is not `.json` is read as one prompt per line, and
blank lines are skipped.

### Step selection and the pool flags

```text
Unknown step(s) <names>. This pipeline defines <names>.
```

```text
Steps must be selected contiguously; <names> would be skipped in the middle, and later steps read the columns they produce.
```

```text
Cannot override the client of unknown step(s) <names>. This config defines <names>.
```

```text
--hosts-file and --url-glob both name the pool's servers; give one of them.
```

```text
<flag> points at vLLM servers, but none of the steps being run (<steps>) uses provider 'vllm_online'.
```

```text
Steps <names> want different models <models>, but one set of vLLM servers serves a single model. Run them as separate --steps invocations, each against its own servers.
```

```text
Config defines no step '<name>'. It has <names>.
```

```text
Step '<name>' <reason>, so there is no vLLM server to start for it.
```

```text
Step '<name>' names no 'model', so there is nothing to serve. A client can ask a running server what it serves, but nothing can ask a server that does not exist yet.
```

## Errors when a run starts or resumes

### An edited prompt with finished rows

Symptom: a re-run stops with a `ValueError` before anything is annotated or
deleted.

```text
The finished rows in '<progress dir>' cannot be reused: <what changed>. <remedy> A run can only grow through a higher 'max_num_samples' with the same settings, or through rows appended to a source that is not shuffled.
```

Cause: the prompt template, the system message, `sort_by_length`, `idx_column`,
`reuse_idx_column`, `shuffle_seed`, `preprocess_fn`, the output schema or the
source dataset differs from the record next to the prepared data, and rows are
already finished. Keeping them would put answers to two questions in one
output.

Fix: one of the three the remedy names. From Python it reads:

```text
Restore the old value(s), use a new 'output_dir', or overwrite the run ('overwrite=True') to discard the finished rows and annotate every sample again.
```

A pipeline names the command instead, which covers the edited step and every
step that reads it:

```text
Restore the old value(s), use a new 'output_dir', or re-run with '--steps <names> --overwrite' to annotate that step and the ones that read it again from scratch.
```

A higher `max_num_samples` with everything else unchanged is not a conflict: it
resumes and annotates only the new rows. See
[Growing a run](growing-a-run.md).

### A step that reads an out-of-date step

```text
Step '<name>' has <reason>, so there is no input for '<name>'. Run it first, or select it too.
```

Cause: `--steps` selected a step whose input step has not run, or finished with
other settings than the config now asks for.

Fix: add that step to the selection, or restore its old settings.

### Schema properties that shadow a column

```text
The output schema property names <names> are also the names of columns that the annotator writes for every sample. Pick other names in the schema, or give the run another 'task_prefix' or 'idx_column'.
```

Cause: a top-level property of `output_schema` is called `idx` or carries the
name of a bookkeeping column of that step (`response`, `finish_reason`,
`num_tokens`, `error`, `error_type`, `reasoning`, `valid_fields`, `valid`,
`messages`, each behind the task prefix). The check runs before the first
request.

Fix: rename the property, or give the step another `task_prefix` or
`idx_column`.

### A Hub backup and an empty progress directory

```text
'<hub id>' has a progress backup on the branch '<branch>', while '<output dir>' holds no progress files. This run would annotate every row again and its first upload would replace the backup with fewer rows. Restore the backup first:
```

Cause: the run has a `hub_id` whose `<task_prefix>progress_backup` branch
exists, and the local progress directory is empty. A purged scratch directory
and a move to another cluster both look like this.

Fix: run the restore command that the message prints, then start the run again.

```sh
python scripts/restore_progress_from_hub.py --hub-id user/my-dataset --output-dir outputs/qa/02-judge/annotate --task-prefix judge_
```

`--force` merges the backup into a progress directory that already holds files;
rows are merged per sample id and a local row wins. Pass `overwrite=True`
(`--overwrite`) instead to delete the branch and annotate every row again. See
[Resuming on another machine](pipeline.md#resuming-on-another-machine).

The restore itself fails when the branch is gone, which is what a finished run
leaves behind:

```text
The dataset '<hub id>' has no branch '<branch>', so there is no progress backup to restore. That branch is deleted once a run finished, and the final dataset is then on the 'main' branch of the same repository.
```

### A finished step or record from an older release

```text
Step '<name>' finished into '<path>', but there is no record of the settings it was annotated with, so this run cannot tell whether its result still matches the config. Remove '<path>' to run the step again: the rows in its progress files are not sent to the model a second time.
```

```text
The record in '<path>' does not describe the settings that its run was annotated with, so a resume cannot tell whether the prompt still matches. Finish the run with the release that wrote it, annotate it into a new 'output_dir', or overwrite it.
```

Both come from an output directory that 0.16 wrote.
[Migrating from 0.16](migration.md#output-directories-and-records) lists the
ways forward.

## Errors while a run goes

### `RuntimeError: can't start new thread`

Symptom: a pool client dies with that message, usually soon after the first
batch goes out.

Cause: a `vllm_online` batch sends one request per sample and each request in
flight holds a thread. The count is `servers` times
`max_concurrent_batches_per_client` times `batch_size`: four servers with the
defaults (four batches of 256) is 4096 threads, which is over a common
`ulimit -u` or a cgroup `pids.max` on a CPU job. The client logs the limit it
found at the top of its log:

```console
Thread limit (ulimit -u): 4096
```

Fix: lower `batch_size`, or set `init.max_workers` to cap how many requests of
one batch go out at once. Raising the limit works too where the site allows it.

### The run stops on failed batches

```text
No vLLM server in the pool answers '/health' any more, so the run stops.
```

Cause: every server of the pool failed a whole batch and then failed its health
probe, so there is nothing left to annotate on. On SLURM the usual reason is
that the array hit `SERVER_TIME` or was preempted.

Fix: the finished rows are on disk, so resubmitting the same command resumes.
`--max-resubmits` queues that in advance, see [SLURM](slurm.md).

A single-client run stops for a related reason:

```text
<n> consecutive batches failed entirely; aborting instead of continuing
```

Cause: `max_consecutive_failed_batches` (10 by default) batches in a row in
which every sample errored. The rows of those batches are held back rather than
written, so a resumed run annotates them again instead of keeping them as
errors. Set the option to 0 to disable both the abort and the hold-back.

### `'n' is <n>, but one response per sample is read`

```text
'n' is <n>, but one response per sample is read and the others are dropped. Remove 'n' or set it to 1.
```

Cause: `options.n` or `options.extra_body.n` above 1. This is a `ValueError` for
the caller whatever `on_error` says, because the payload is built before the
error handling starts.

## Reading the error columns

Four columns say what happened to a sample. `{prefix}` is the task prefix,
which in a pipeline is the step's name followed by an underscore.

| Column | Holds |
| --- | --- |
| `{prefix}error` | the message of the failed request, or `None` when it succeeded |
| `{prefix}error_type` | the exception class name, such as `ConnectError` or `APITimeoutError`. These are the names `retry_errors` takes |
| `{prefix}valid_fields` | whether the response parsed into the schema. Only written when the step has an `output_schema` |
| `{prefix}valid` | what your own `validate_fn` returned. Only written when you passed one |

`{prefix}valid_fields` is `False` for an errored response, a response that does
not parse as JSON, one that parses to something other than an object, and one
that leaves out a property named in the schema's `required` list. A schema with
no `required` list counts as valid as soon as the response parses to an object.

Every run ends with a log line that counts the errored and the invalid samples,
with a count per `error_type`, and writes the same counts to
`<output_dir>/metadata/<task_prefix>annotation_metadata.json`.

## Redoing failed rows

An errored row is final once it is written: a resumed run does not send it to
the model again, so an error the sample itself causes (a prompt longer than the
context, for instance) is not repeated on every resume.

`retry_errors` selects rows to annotate again. They are removed from the JSONL
progress files before the run starts, so they are annotated as if they had never
run.

```python
ds = anno.run_annotation(
    output_dir="outputs/imdb-sentiment",
    prompt_template="Classify the sentiment: {text}",
    prepared_dataset=prepared_dataset,
    retry_errors=["ConnectError", "APITimeoutError"],
)
```

`retry_errors=True` takes every errored row. From the command line:

```bash
llm-annotate my-pipeline.yaml --retry-errors                              # all of them
llm-annotate my-pipeline.yaml --retry-errors ConnectError APITimeoutError # by type
```

A row that is redone in one step is redone in every selected step after it,
because those steps read what it produces. A finished step that is affected is
resumed rather than loaded from its snapshot, and every row that is not selected
keeps its result.

An invalid row (`valid_fields` or `valid` is `false`) is not an errored row and
`retry_errors` does not select it. Raise `num_retries_invalid` for the next run,
or drop those rows between steps with `filter_invalid: true`. See
[What a retry costs](choosing-a-provider.md#what-a-retry-costs).
