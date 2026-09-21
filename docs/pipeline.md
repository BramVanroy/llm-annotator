# Annotating from a config file

Everything the library can do to a dataset can also be described in a single
JSON or YAML file and run without writing any Python:

```bash
llm-annotate my-pipeline.yaml
```

or, from a checkout that is not installed:

```bash
python scripts/annotate.py my-pipeline.yaml
```

A config describes one or more **steps**. Steps run in order, and each one
annotates the dataset the previous step produced, so a later prompt can read
columns that an earlier model wrote. That is what makes generate-then-judge
workflows possible: one model writes question-answer pairs, another rates them.

!!! note "What a config cannot express"

    `preprocess_fn`, `postprocess_fn` and `validate_fn` take Python callables
    and are deliberately unavailable here. If you need them, use
    [`Annotator`][llm_annotator.annotator.Annotator] directly. Validity in a
    config-driven run means "the model returned JSON containing every
    `required` property of the step's schema".

## A complete example

The pipeline below is shipped as `examples/pipeline-qa/`. Step 1 writes a
question-answer pair about each text; step 2 has a different model rate the pair
that step 1 produced.

```yaml title="examples/pipeline-qa/config.yaml"
--8<-- "examples/pipeline-qa/config.yaml"
```

Run it with:

```bash
llm-annotate examples/pipeline-qa/config.yaml
```

The same pipeline is also provided as `config.json`; the two formats are
interchangeable and the file suffix decides how it is parsed.

## Paths are relative to the config

Every path inside a config file -- `output_dir`, `prompt_file`,
`system_prompt_file`, `output_schema_file`, `hosts_file`, `dataset.path` --
resolves against the directory holding the config file, never against your
current working directory. A config directory is therefore self-contained and
can be copied to a cluster or shared with a colleague as a unit.

The `--output-dir` CLI flag is the one exception: since it is typed at the
shell rather than written into the config, it resolves against your current
working directory instead, the same as the `config` argument itself.

## Prompts and schemas: inline or in a file

Each of the three text inputs has an inline form and a file form. Giving both is
an error, so there is never any doubt about which one won:

| Inline | From a file | Purpose |
| --- | --- | --- |
| `prompt` | `prompt_file` | Prompt template, with `{column}` placeholders |
| `system_prompt` | `system_prompt_file` | System message for the chat turn |
| `output_schema` | `output_schema_file` | JSON schema for structured output |

Short prompts read well inline; anything longer belongs in a `.md` file next to
the config, which also keeps the prompt reviewable in a diff.

## How steps see each other's output

Each step writes several kinds of column:

* **Schema properties.** Every top-level property of `output_schema` becomes a
  column under its own name. A schema with `question` and `answer` produces
  exactly those two columns, which is what the next step's prompt refers to.
* **Bookkeeping columns**, namespaced by the step's `task_prefix` (which
  defaults to `<name>_`): `{prefix}response`, `{prefix}finish_reason`,
  `{prefix}num_tokens`, `{prefix}error`, `{prefix}error_type`,
  `{prefix}reasoning` and, when a schema is set, `{prefix}valid_fields`.
* The **`idx_column`** (`idx` by default). It identifies a row from the first
  step onward, so every step's `output/` keeps it. It is removed from
  `final/` and from the dataset pushed to the Hub. See
  [Growing a run](growing-a-run.md) for what this makes possible.

`{prefix}reasoning` holds a reasoning model's trace, separated from the answer
in `{prefix}response`. For either vLLM provider it is filled when the step names
a parser:

```yaml
client:
  engine:
    reasoning_parser: qwen3   # or deepseek_r1, granite, gemma4, glm45, ...
```

A served step passes that to `vllm serve` as `--reasoning-parser`, which splits
the trace off before it reaches the client. An offline step splits it with the
same vLLM parser inside the client, since `vllm.LLM` returns the trace inline.
A `claude` step needs no parser: it fills the column from the thinking blocks
whenever the request carries a thinking budget. Without a parser a reasoning
model returns its trace inside `{prefix}response`, tags and all, and this column
stays `None`.

!!! note "`{prefix}num_tokens` on a `vllm_online` step"

    A `vllm_online` step sends each batch to vLLM's batch endpoint, which
    reports one `usage` block for the whole batch instead of one per sample, so
    `{prefix}num_tokens` is `None` on that provider. Every other column,
    `{prefix}reasoning` included, is per sample as usual. Use `vllm_offline` if
    you need the counts, for instance to see how much of the budget a reasoning
    trace is eating.

Because schema properties are *not* prefixed, two steps that use the same
property name would collide. Use `rename` to give a step's output its final
name:

```yaml
steps:
  - name: write-qa
    output_schema_file: schemas/qa.json   # produces `question`, `answer`
    rename:
      question: question_v1
      answer: answer_v1

  - name: rate-qa
    prompt: |
      Rate this pair.
      Q: {question_v1}
      A: {answer_v1}
```

Renaming onto a column that already exists is refused rather than silently
overwriting it.

Two further knobs tidy up between steps:

* `drop_columns` removes columns you no longer need.
* `filter_invalid: true` drops rows whose `{prefix}valid_fields` is still
  `false` after all retries, so a broken generation is not carried into the next
  step. It requires a schema, and it fails loudly if *every* row was invalid --
  usually a sign that `max_completion_tokens` is too small for the schema.

The rendered `{prefix}messages` column is dropped once a step finishes, so an
N-step pipeline does not accumulate N copies of every prompt. Set
`keep_messages: true` on a step to keep it for debugging.

## Providers and models

A client can be described at the top level, per step, or both:

* **Top level only** -- every step runs on it. Best when one model does all the
  work.
* **Top level plus a step block** -- the step's keys are merged over the
  defaults. Merging is one level deep: `init` and `options` are merged
  key-by-key, so a step that only changes `max_completion_tokens` need not
  repeat the rest. A step that switches `provider` is the exception, described
  below.
* **Per step only** -- omit the top-level block entirely. Best when every step
  uses a different model and there is no sensible shared default; each step's
  block then has to name its own `provider` and `model`.

Every step must end up with a client one way or the other, and a step that has
neither is reported by name when the config loads.

```yaml
client:
  provider: vllm_offline
  model: Qwen/Qwen3-8B
  batch_size: 256
  num_proc: 8
  engine:            # how the vLLM engine itself is built
    max_model_len: 8192
  options:           # fields of the provider's runtime-options dataclass
    temperature: 0.7
    max_completion_tokens: 1024

steps:
  - name: judge
    prompt_file: prompts/judge.md
    client:
      options:
        max_completion_tokens: 256   # temperature is inherited
```

With no top-level block, each step carries its own complete client:

```yaml
steps:
  - name: write
    prompt_file: prompts/write.md
    client:
      provider: vllm_offline
      model: Qwen/Qwen3-8B

  - name: judge
    prompt_file: prompts/judge.md
    client:
      provider: claude
      model: claude-haiku-4-5
```

`provider` accepts exactly `openai`, `claude`, `vllm_online` (a running vLLM
server) or `vllm_offline` (in-process vLLM) — no other spellings are
recognized. See [Provider setup](provider-info.md) for authentication.

### Where a setting goes

A `client` block has five groups, split by *when* a setting is used rather than
by what it configures. That is the rule to remember: a setting belongs to
whichever moment it takes effect.

| Group | Key | Used when | Providers |
| --- | --- | --- | --- |
| Execution | `batch_size`, `num_proc` | the annotator drives the run | all |
| Execution (pool) | `queue_size`, `max_concurrent_batches_per_client`, `wait_for_servers` | the annotator drives a pool of servers | `vllm_online` |
| Connection | `init` | the client object is constructed | all |
| Engine | `engine` | the vLLM engine is built | `vllm_offline`, `vllm_online` |
| Pool | `pool` | a job submitter starts servers | `vllm_online` |
| Request | `options`, `gen_kwargs` | every generation call | all |

```yaml
client:
  provider: vllm_offline
  model: Qwen/Qwen3-8B
  batch_size: 256          # execution
  num_proc: 8
  init:                    # connection: the client constructor
    on_error: warn
  engine:                  # engine: how vLLM itself is built
    tensor_parallel_size: 2
    max_model_len: 8192
  options:                 # request: sent with every prompt
    temperature: 0.7
    max_completion_tokens: 1024
```

`init` and `options` are passed straight through to the matching client
constructor and `*RuntimeOptions` dataclass, so every provider-specific setting
is reachable; unknown option names are rejected at load time with the valid
names listed. When a dataclass does not name what you need, `options.extra_body`
(vLLM) and `gen_kwargs` (any provider) are merged into the request as written.

The groups do not overlap, and the config says so rather than letting a value
sit in two places: an engine setting written under `init` is rejected at load
time, and `engine` on a hosted provider is too.

`engine` is the same block for both vLLM providers — same field names, same
meaning. A `vllm_offline` step turns it into `vllm.LLM(...)` keyword arguments;
a `vllm_online` step whose servers still have to be started turns it into
`vllm serve` flags, which `llm-annotate <config> --serve-args <step>` prints for
a job submitter. So moving a step between the two changes only `provider`, and a
step states its GPU count once, in `engine.tensor_parallel_size`.

Steps whose provider, model, `init` and `engine` all match share one live
client, so a pipeline that uses the same local model twice loads it only once.
Changing only `options` never triggers a reload, because options are per
request.

One exception to the merging above: a step that names a *different* `provider`
than the top-level block inherits no `options` from it at all. They name fields
of the previous provider's runtime-options dataclass — `top_k` means nothing to
Claude — and would be rejected as unknown. The step's own `options` are kept
exactly as written:

```yaml
client:
  provider: vllm_offline
  model: Qwen/Qwen3-8B
  options:
    temperature: 0.7
    top_k: 20

steps:
  - name: judge
    prompt_file: prompts/judge.md
    client:
      provider: claude
      model: claude-haiku-4-5
      options:
        max_completion_tokens: 256   # and *only* that; nothing is inherited
```

## Many vLLM servers

Point the `vllm_online` provider at several servers and the pipeline uses a
[`VLLMQueueAnnotator`][llm_annotator.annotator.VLLMQueueAnnotator] instead of a
single client. Three ways to say where the servers are, matching how a job
submitter publishes them:

```yaml
client:
  provider: vllm_online
  model: Qwen/Qwen3-8B
  base_urls:                       # explicit
    - http://node01:8000/v1
    - http://node02:8000/v1
  # hosts_file: logs/pool_123/hosts.txt   # one URL per line, read once
  # url_glob: logs/pool_*/*.url           # one URL per file, re-read during the run
  queue_size: 8                    # batches kept in flight over the pool
  max_concurrent_batches_per_client: 4  # requests per server, independent of batch_size
  wait_for_servers: 300            # poll /health first; 0 disables
```

`max_concurrent_batches_per_client` is how many requests each server is asked to
handle at once, and every one of them carries `batch_size` samples, so one
server holds up to `max_concurrent_batches_per_client * batch_size` prompts. The
whole pool runs `servers * max_concurrent_batches_per_client` requests at once,
which is the floor for `queue_size`: a smaller queue cannot fill every server,
so the config is rejected with the minimum spelled out. Leave `queue_size` out
to keep four batches queued per request slot. Both keys size a pool, so both are
`vllm_online`-only; on the other providers one request is in flight at a time
and `batch_size` is the only knob.

The block above works out to 2 servers * 4 requests = 8 requests in flight, so
its `queue_size` of 8 is exactly the minimum. With the default `batch_size` of
256, each of those servers holds 1024 prompts, which is what its
`engine.max_num_seqs` has to cover. `--describe-steps` prints all of these
numbers, so a pool can be sized before a single GPU is allocated.

`base_urls` and `hosts_file` are read once, at the start of the run.
`url_glob` is re-read while the run continues, so a server whose file
appears later is admitted into the pool and takes over part of the work.

When a batch fails entirely on one server, that server's `/health` endpoint
is probed. A server that does not answer is removed from the pool, and its
batch is sent to another server (no errored rows are written for it). A batch
that fails entirely on a healthy server is recorded as errors. Every
configured URL that is not in the pool is probed once a second, so a server
that recovers, or a requeued SLURM server job that publishes its `.url` file
again, joins the run the same way a late starter does. The run stops with
`TooManyConsecutiveFailedBatchesError` once no server is left.

When the servers do not exist yet and something has to start them, say how many
you want in `pool` and what each one is in `engine`. Both blocks are reported
to a job submitter rather than acted on by the library, apart from
`pool.min_servers`, which is also how many servers a pooled run waits for
before it starts:

```yaml
client:
  provider: vllm_online
  model: Qwen/Qwen3-8B
  engine:
    tensor_parallel_size: 2   # GPUs per server
    max_model_len: 8192
    gpu_memory_utilization: 0.90
  pool:
    servers: 4                # four such servers
    min_servers: 2            # start once two are ready
```

`min_servers` is how many servers have to answer before annotation starts. It
defaults to one, so a step begins on the first server that is ready and the
rest join the run as they come up. A submitter reads it from
`--describe-steps` and releases its own wait loop at the same threshold.

Because the profile is per step, one pipeline can serve a different model with
different serving flags at each step.

## Running one step at a time

`--steps` runs part of a pipeline. Earlier steps must already have finished:
their saved `output/` snapshot is loaded as the input, which is exactly the
resume path, so running

```bash
llm-annotate cfg.yaml --steps write-qa
llm-annotate cfg.yaml --steps rate-qa
```

produces the same dataset as one `llm-annotate cfg.yaml`. The selection has to be
contiguous, since skipping a step in the middle would drop the columns the next
prompt reads. Only the run that includes the **last** step writes
`<output_dir>/final/` and pushes to the Hub — a partial run has a partial
dataset and must not publish it as finished.

This is what lets a scheduler give each step its own resources while one config
file stays the source of truth. `--describe-steps` is the machine-readable half:
it prints one JSON object per step and annotates nothing.

```console
$ llm-annotate cfg.yaml --describe-steps
{"index": 1, "name": "write-qa", "kind": "vllm_pool", "model": "Qwen/Qwen3-8B", "servers": 4, "min_servers": 1, "gpus_per_vllm_server": 2, "batch_size": 64, "max_concurrent_batches_per_client": 4, "queue_size": 32, "max_requests_per_server": 256, "max_requests_in_flight": 1024, ...}
{"index": 2, "name": "rate-qa", "kind": "api", "model": "claude-haiku-4-5", "batch_size": 256, "max_concurrent_batches_per_client": null, "queue_size": null, "max_requests_per_server": null, "max_requests_in_flight": null, ...}
```

`kind` says what the step needs to run: `vllm_pool` (servers must be started for
it), `vllm_online` (they already exist), `vllm_offline` (loads the model
in-process) or `api` (a hosted provider, no accelerator at all).

The last five keys are the step's concurrency, and they are what a pool is sized
against:

| Key | Meaning |
| --- | --- |
| `batch_size` | samples in one request |
| `max_concurrent_batches_per_client` | requests each server handles at once |
| `queue_size` | batches in flight over the pool, after the default and the minimum have been applied |
| `max_requests_per_server` | `max_concurrent_batches_per_client * batch_size`, the prompts one server holds, so what its `max_num_seqs` has to cover |
| `max_requests_in_flight` | that number times the server count |

The four pool keys are `null` for a provider that has no pool, which sends one
request at a time.

`--serve-args` is the other half: it prints the `vllm serve` argument list for
one step, one argument per line, so a server job reads its own serving profile
out of the config instead of being handed one through the environment.

```bash
llm-annotate cfg.yaml --serve-args write-qa
```

```
Qwen/Qwen3-8B
--served-model-name
Qwen/Qwen3-8B
--tensor-parallel-size
2
--max-model-len
8192
```

One argument per line is what keeps a value containing spaces intact —
`--speculative-config` takes a JSON object. `--host` and `--port` are absent by
design: the port has to be probed on the node, because two servers of one pool
can land on the same machine.

`--hosts-file` completes the picture for a scheduler: it attaches a file of
server URLs to the selected step that runs on vLLM, and to that step only, so a
hosted step in the same pipeline is unaffected. `--url-glob` is its sibling for
a pool that is still filling up: it attaches a glob of one-URL-per-file server
addresses instead, and re-reads it while the run continues, so servers that
become ready later are admitted into the pool. Giving both flags is an error.

```bash
llm-annotate cfg.yaml --steps write-qa --hosts-file logs/pool_123/hosts.txt
llm-annotate cfg.yaml --steps write-qa --url-glob 'logs/pool_123/*.url'
```

A cluster job submitter is built entirely out of these flags: it reads
`--describe-steps` to plan the allocation, `--serve-args` to start each step's
servers, and attaches `--hosts-file` once they are up. `slurm/` ships such a
submitter for SLURM, with everything cluster-specific in one small
[cluster file](slurm.md); it submits one job chain per step of a config.

## Resuming

Long pipelines are restartable at two levels:

* **Within a step**, the usual JSONL progress files under the step's
  `annotate/` directory mean an interrupted step continues where it stopped.
* **Between steps**, a finished step writes its result to
  `<output_dir>/<NN>-<name>/output/`. Re-running the same config loads that
  snapshot and skips the step, so a pipeline that dies in step three does not
  repeat steps one and two.

Re-run the identical command to resume. Pass `--overwrite` (or set
`overwrite: true`) to delete the existing step directories, including every
finished generation in them, and run those steps from scratch.

Raising `dataset.max_num_samples` and re-running the identical command also
resumes every step, without `--overwrite`: only the new rows are sent to the
model, in every step. See [Growing a run](growing-a-run.md) for the full
workflow, what is allowed to change, and what is rejected.

The layout under `output_dir` is:

```text
outputs/pipeline-qa/
├── pipeline.json          # the fully resolved config that produced this run
├── 01-write-qa/
│   ├── annotate/          # prepared data + JSONL progress for this step
│   └── output/            # the step's finished dataset (its "done" marker)
├── 02-rate-qa/
│   ├── annotate/
│   └── output/
└── final/                 # the last step's dataset
```

### How large a progress file is

A step appends every finished sample to a JSONL file under
`<NN>-<name>/annotate/<task_prefix>progress_backup/`, and opens a new file
every `max_samples_per_output_file` samples. A resume reads all of those files
back to learn which ids are already done, so the value sets two costs against
each other:

- Many small files: every resume opens and parses each one, which is slow on a
  shared network filesystem, and some filesystems limit how many files a
  directory may hold.
- Few large files: the file that is currently open is re-uploaded in full on
  every Hub progress push, and one corrupt file costs more rows. (No finished
  sample is lost at a crash either way, because every line is flushed as it is
  written. A half-written last line is detected and dropped on the next run.)

The default, `auto`, is one percent of the rows of the step's prepared
dataset, with a floor of 1000 samples, so a step writes at most 100 progress
files: 12 files of 1000 samples for 12,000 rows, 100 files of 5000 samples for
500,000 rows. Set a number to fix the size instead, or `0` to write a single
file of unlimited size:

```yaml
steps:
  - name: write-qa
    max_samples_per_output_file: 20000   # "auto" (the default), or 0 for one file
```

The same key is a keyword argument of `annotate_dataset`, `run_annotation` and
`generate_dataset` in the Python API, with the same default.

## Pushing to the Hub

The top-level `hub_id` is the **final** dataset only; it is pushed once, after
the last step:

```yaml
hub_id: your-username/wiki-qa-rated
```

Per-step Hub backup of prepared data and progress is separate, because it exists
for crash recovery rather than publication. Set it on the step that needs it:

```yaml
steps:
  - name: write-qa
    hub_id: your-username/wiki-qa-scratch
    upload_every_n_samples: 10000
```

## Generating data from scratch

A step with `type: generate` builds its own dataset from a list of prompts
instead of annotating an existing one, so the pipeline needs no `dataset` block.
It must be the first step, since it replaces the data rather than adding to it.

```yaml
output_dir: outputs/synthetic
client:
  provider: openai
  model: gpt-4o-mini

steps:
  - name: make-questions
    type: generate
    prompts: ["Write a short geography quiz question with its answer."]
    num_samples: 200
    output_schema_file: schemas/qa.json
```

`prompts` may be a list, or a path to a file with one prompt per line. A single
prompt with `num_samples` is repeated that many times; a list is truncated to
`num_samples` when both are given. To wrap every prompt in a shared prefix, add
a template containing the `{prompt}` placeholder:

```yaml
    prompt: "Answer in Dutch.\n\n{prompt}"
```

## Command line

```text
llm-annotate [-h] [--output-dir OUTPUT_DIR] [--hub-id HUB_ID]
             [--log-level LOG_LEVEL] [--overwrite]
             [--max-num-samples MAX_NUM_SAMPLES] [--shuffle-seed SHUFFLE_SEED]
             [--set KEY=VALUE] [--steps STEPS]
             [--retry-errors [ERROR_TYPE ...]] [--hosts-file HOSTS_FILE]
             [--url-glob URL_GLOB] [--serve-args STEP] [--describe-steps]
             config
```

`--output-dir`, `--hub-id`, `--log-level` and `--overwrite` override the matching
config keys, which is handy for pointing one config at a scratch directory or
resuming with a different log level without editing the file. `--steps`,
`--hosts-file`, `--url-glob`, `--serve-args` and `--describe-steps` are
described under [Running one step at a time](#running-one-step-at-a-time).

`--retry-errors` without a value annotates every errored row of the selected
steps again. With one or more `ERROR_TYPE` values only the rows with that
`error_type` are redone, as in
`llm-annotate cfg.yaml --retry-errors ConnectError APITimeoutError`. A row that
is redone in one step is also redone in every selected step after it, because
those steps read what it produces. Finished steps that are affected are
resumed: every other row keeps its result. See
[Errors and retries](index.md#errors-and-retries) for when a row counts as
errored.

### Overriding config keys

`--max-num-samples` and `--shuffle-seed` override `dataset.max_num_samples` and
`dataset.shuffle_seed`, so a pilot, the full run and a later extension share one
config file and differ only in the command that starts them:

```bash
llm-annotate cfg.yaml --max-num-samples 2000     # pilot
llm-annotate cfg.yaml --max-num-samples 50000    # same output_dir, grows it
llm-annotate cfg.yaml --max-num-samples "$N"     # from a job script
```

`--set KEY=VALUE` reaches every other key. A dotted key descends into a nested
block, an integer segment indexes a list, and the value is read as YAML, so it
gets the type it looks like:

```bash
llm-annotate cfg.yaml --set client.options.temperature=0.2
llm-annotate cfg.yaml --set steps.0.client.batch_size=8 --set idx_column=row_id
llm-annotate cfg.yaml --set 'steps.1.drop_columns=[rate_response, rate_error]'
```

A key that both a named flag and `--set` would set is an error, so there is no
question of which one wins. The overrides are applied before validation, which
means a typo is rejected the same way a typo in the file is, and the resolved
values are written to `<output_dir>/pipeline.json`, so the run records what it
actually used.

The same mapping is available from Python, keyed the same way:

```python
config = load_pipeline_config("cfg.yaml", overrides={"dataset.max_num_samples": 50_000})
```

## Full reference

Every key, with its type and default, is documented on the
[configuration API page](api/config.md); the executor is on the
[pipeline API page](api/pipeline.md).
