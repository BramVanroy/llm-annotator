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
| Execution | `batch_size`, `num_proc`, `queue_size`, `wait_for_servers` | the annotator drives the run | all |
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
  # hosts_file: logs/pool_123/hosts.txt   # one URL per line
  # url_glob: logs/pool_*/*.url           # one URL per file
  queue_size: 8
  wait_for_servers: 300            # poll /health first; 0 disables
```

When the servers do not exist yet and something has to start them, say how many
you want in `pool` and what each one is in `engine`. Neither block is acted on
by the library itself; both are reported to a job submitter, and a local run
ignores them entirely:

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
```

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
{"index": 1, "name": "write-qa", "kind": "vllm_pool", "model": "Qwen/Qwen3-8B", "servers": 4, "gpus_per_vllm_server": 2, ...}
{"index": 2, "name": "rate-qa", "kind": "api", "model": "claude-haiku-4-5", ...}
```

`kind` says what the step needs to run: `vllm_pool` (servers must be started for
it), `vllm_online` (they already exist), `vllm_offline` (loads the model
in-process) or `api` (a hosted provider, no accelerator at all).

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
hosted step in the same pipeline is unaffected.

```bash
llm-annotate cfg.yaml --steps write-qa --hosts-file logs/pool_123/hosts.txt
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
`overwrite: true`) to throw the existing step directories away and start over.

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
             [--log-level LOG_LEVEL] [--overwrite] [--steps STEPS]
             [--hosts-file HOSTS_FILE] [--describe-steps]
             config
```

`--output-dir`, `--hub-id`, `--log-level` and `--overwrite` override the matching
config keys, which is handy for pointing one config at a scratch directory or
resuming with a different log level without editing the file. `--steps`,
`--hosts-file` and `--describe-steps` are described under
[Running one step at a time](#running-one-step-at-a-time).

## Full reference

Every key, with its type and default, is documented on the
[configuration API page](api/config.md); the executor is on the
[pipeline API page](api/pipeline.md).
