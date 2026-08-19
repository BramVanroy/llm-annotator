# SLURM job scripts (Snellius)

You describe the whole annotation in one config file — prompts, schemas, models,
providers, however many steps — and submit it:

```sh
ANNOTATE_CONFIG=examples/vllm-server-pool/pipeline.yaml ./slurm/submit_pipeline.sh
```

`submit_pipeline.sh` asks the config what each step needs and submits **one job
chain per step**, so a step only starts once the one before it has succeeded.
Nothing about the allocation lives in an `#SBATCH` header you have to edit, and
nothing about the annotation is repeated on the command line.

Submitting per step is what makes a real pipeline work on a cluster: each step
gets servers for **its own** model, and GPUs are released as soon as that step is
done rather than being held for the whole run. A step that calls a hosted API
gets no GPU at all.

| File | Role |
| --- | --- |
| `submit_pipeline.sh` | Run on a login node. Reads the config and submits the job chain. |
| `vllm_annotate.sh` | Runs **one step** of the config. Waits for its pool if it has one. |
| `vllm_server.sh` | One array task = one vLLM server. Publishes its base URL once healthy. |
| `vllm_common.sh` | Sourced by both jobs: environment setup, port selection, health polling. |

## What each step becomes

The shape of a job is derived from the step's client, never configured twice.
Check it before submitting anything:

```sh
uv run llm-annotate examples/vllm-server-pool/pipeline.yaml --describe-steps
```

```json
{"index": 1, "name": "write-qa", "kind": "vllm_pool", "model": "Qwen/Qwen3-8B", "servers": 4, "gpus_per_vllm_server": 2, ...}
{"index": 2, "name": "rate-qa",  "kind": "api", "model": "claude-haiku-4-5", ...}
```

| `kind` | When | Jobs submitted |
| --- | --- | --- |
| `vllm_pool` | `provider: vllm_online`, servers not given | GPU server array + CPU client |
| `vllm_online` | `provider: vllm_online` with `base_urls`/`hosts_file`/`url_glob` | CPU client only |
| `vllm_offline` | `provider: vllm_offline` | one GPU job, model loaded in-process |
| `api` | `openai` / `claude` | CPU-only job |

A `vllm_pool` step must name a `model`. It is optional for `provider:
vllm_online` in general — a client can ask a running server what it serves —
but a submitter that has to *start* those servers has nothing to ask, so
`submit_pipeline.sh` rejects such a step on the login node rather than after the
GPUs have been allocated.

A step sizes its own pool in the config, next to the model it belongs to:

```yaml
client:
  provider: vllm_online
  model: Qwen/Qwen3-8B
  pool:
    servers: 4           # four servers for this step
    gpus_per_vllm_server: 2   # tensor-parallel size, max 4 (one server, one node)
```

Four small jobs schedule far sooner than one 8-GPU allocation, because each one
fits on a partially used node. They also start at different times, which is
fine: the client waits for the pool to fill before it begins.

## How the two jobs of a step find each other

The server array writes into `logs/pool_<array-job-id>/`, one `<task>.url` file
per server containing that server's `http://<host>:<port>/v1`. A file appears
only **after** the server answers `/health`, and is removed when the job ends, so
every URL in the directory belongs to a server that is up right now. The client
polls that directory, concatenates it into `hosts.txt` and passes it to the CLI
as `--hosts-file`, which attaches it to that step alone — a step on another
provider is left untouched.

Ports are `VLLM_PORT + array task id`, then probed upward for the first free one.
Two array tasks can land on the same node (a 4-GPU node fits two
`gpus_per_vllm_server: 2` servers), so a fixed port would collide.

When a step finishes, its client `scancel`s that step's server array instead of
leaving GPU jobs idling until their time limit. Set `CANCEL_SERVERS_ON_EXIT=0` to
keep them alive.

## Configuration

Everything about the annotation — models, prompts, schemas, batch sizes, pool
sizes — lives in the config file. The environment variables below only cover the
allocation and are passed on the `submit_pipeline.sh` command line; both job
types are submitted with `--export=ALL`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ANNOTATE_CONFIG` | *required* | JSON/YAML pipeline config to run |
| `GPU_PARTITION`, `SERVER_TIME` | `gpu_a100`, `04:00:00` | Server allocation. CPUs per GPU are derived from `GPU_PARTITION` (18 on `gpu_a100`, 16 on `gpu_h100`, 16 otherwise), not a separate variable. |
| `CLIENT_PARTITION`, `CLIENT_TIME` | `rome`, `05:00:00` | Client allocation. `CLIENT_PARTITION` must be `rome` or `genoa`; cpus-per-task is set to each partition's minimal whole-node request (16 on `rome`, 24 on `genoa`), not a separate variable. |
| `SLURM_ACCOUNT` | `tnsr72764` | Accounting |
| `MAX_GPUS_PER_NODE` | `4` | Upper bound checked against each step's `gpus_per_vllm_server` |
| `MIN_SERVERS` | `1` | Servers a step insists on before the client starts. Raise it to require more of the pool up front. |
| `POOL_WAIT` | `3600` | Seconds a client waits for its servers to register |
| `VLLM_PORT`, `MAX_MODEL_LEN`, `GPU_MEM_UTIL` | `8000`, `8192`, `0.90` | Passed to `vllm serve` |
| `VLLM_EXTRA_ARGS` | – | Extra flags appended to `vllm serve` |
| `OUTPUT_DIR`, `HUB_ID`, `OVERWRITE` | from the config | Override the config's `output_dir` / `hub_id`, or discard existing step output |

## Resuming

Two levels, both automatic:

- **Within a step**, every annotated sample is appended to
  `<output_dir>/<NN>-<step>/annotate/<prefix>progress_backup/*.jsonl` and flushed
  immediately; a restart re-reads those files and skips the ids already present.
  A half-written final line from a killed job is detected and re-annotated.
- **Between steps**, a finished step writes `<output_dir>/<NN>-<step>/output/`,
  which a later run loads instead of recomputing.

So after a crash, a timeout or a preemption you run **the same
`submit_pipeline.sh` command again**. Finished steps are skipped, and the step
that died continues where it stopped.

## Running a step yourself

The scripts add nothing the CLI cannot do, so any step can be run by hand — from
a login node, an interactive session, or your laptop:

```sh
# one step, against servers that already exist
llm-annotate my-pipeline.yaml --steps write-qa --hosts-file logs/pool_<id>/hosts.txt

# the next step, wherever you like
llm-annotate my-pipeline.yaml --steps rate-qa
```

Running steps one at a time produces exactly the dataset a single
`llm-annotate my-pipeline.yaml` would, so you can start locally and finish on the
cluster, or vice versa. Only the run that includes the last step writes
`<output_dir>/final/` and pushes to the Hub.

To start servers without a client:

```sh
sbatch --array=1-4 --gres=gpu:2 --cpus-per-task=36 \
  --export=ALL,MODEL="$MODEL",GPUS_PER_VLLM_SERVER=2 slurm/vllm_server.sh   # note the id

cat logs/pool_<id>/*.url > logs/pool_<id>/hosts.txt
```

See [../docs/pipeline.md](../docs/pipeline.md) for the full set of config keys.
