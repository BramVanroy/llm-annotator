# SLURM job scripts

Run a whole annotation pipeline on a SLURM cluster from one config file.

First ensure that all variables relevant to your SLURM custer are set:

```sh
cp slurm/cluster.env.example slurm/cluster.env
$EDITOR slurm/cluster.env
# then simply submit
./slurm/submit_pipeline.sh my-pipeline.yaml
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
| `cluster.env.example` | Template for the one file you edit: partitions, accounting, cores per GPU, modules. |
| `submit_pipeline.sh` | Run on a login node. Reads the config and submits the job chain. |
| `vllm_annotate.sh` | Runs **one step** of the config. Waits for its pool if it has one. |
| `vllm_server.sh` | One array task = one vLLM server. Publishes its base URL once healthy. |
| `vllm_common.sh` | Sourced by all of the above: cluster file, environment setup, ports, health polling. |

Three layers, kept apart: the **pipeline config** says what is annotated, the
**cluster file** says what your cluster calls things, and the scripts hold the
job shapes, which are the same everywhere. Nothing is configured twice.

## Try it before you submit

`--dry-run` prints the `sbatch` command lines it would run and submits nothing.
It is the fastest way to check a new cluster file:

```sh
./slurm/submit_pipeline.sh --dry-run my-pipeline.yaml
```

```console
Step 1 'write-qa' (vllm_pool)
sbatch --account=my_project --time=04:00:00 --job-name=vllm-write-qa --array=1-4 \
  --gres=gpu:2 --cpus-per-task=36 --partition=gpu_a100 ... slurm/vllm_server.sh
```

## The cluster file

Copy `cluster.env.example` to `cluster.env` and fill it in; it is read
automatically. To keep several clusters side by side, write one file each and
pick one per run:

```sh
./slurm/submit_pipeline.sh --cluster-env slurm/clusters/leonardo.env my-pipeline.yaml
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `SLURM_ACCOUNT` | – | Project to charge. Empty means no `--account`. |
| `GPU_PARTITION`, `CPU_PARTITION` | – | Partitions for the GPU and CPU-only jobs. Empty means the cluster's default. |
| `SERVER_TIME`, `CLIENT_TIME` | `04:00:00`, `05:00:00` | Wall time per job kind. |
| `CPUS_PER_GPU` | `8` | Cores a GPU job asks for per GPU. |
| `CLIENT_CPUS` | `8` | Cores a CPU-only annotation job asks for. |
| `MAX_GPUS_PER_NODE` | `8` | Ceiling a step's `engine.tensor_parallel_size` is checked against before anything is submitted. |
| `GPU_REQUEST`, `GPU_TYPE` | `gres`, – | How GPUs are requested: `--gres=gpu:N` or `--gpus-per-node=N`, optionally typed (`gpu:a100:N`). |
| `SERVER_SBATCH_ARGS`, `CLIENT_SBATCH_ARGS` | `()` | Extra `sbatch` flags per job kind: QoS, memory, constraints, reservations. |
| `CLUSTER_MODULES` | – | Environment modules to load inside a job. |
| `VENV_PATH`, `UV_SYNC` | `<repo>/.venv`, `0` | Python environment to activate, and whether to `uv sync` first. |
| `CUDA_MODULE` | – | Toolkit module loaded only when `nvcc` is missing and vLLM has to JIT-compile a kernel. |
| `LOG_DIR` | `<repo>/logs` | Where job logs and pool directories go. |

Every scalar can be overridden for one submission by setting it on the command
line, which wins over the file and reaches the jobs through `--export=ALL`:

```sh
SERVER_TIME=08:00:00 GPU_PARTITION=gpu_h100 ./slurm/submit_pipeline.sh my-pipeline.yaml
```

If modules plus a virtualenv do not describe your cluster — conda, a container,
a wrapper script of your own — define `cluster_setup_env` in the cluster file
instead. It replaces the environment handling entirely and only has to leave
`llm-annotate` and `vllm` on `PATH`:

```sh
cluster_setup_env() {
  module load Python/3.12 CUDA/12.8.0
  source /projects/shared/llm-annotator/.venv/bin/activate
}
```

## What each step becomes

The shape of a job is derived from the step's client, never configured twice.
Check it before submitting anything:

```sh
llm-annotate my-pipeline.yaml --describe-steps
```

```json
{"index": 1, "name": "write-qa", "kind": "vllm_pool", "model": "Qwen/Qwen3-8B", "servers": 4, "min_servers": 1, "gpus_per_vllm_server": 2, ...}
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
  engine:
    tensor_parallel_size: 2   # GPUs per server, at most MAX_GPUS_PER_NODE
  pool:
    servers: 4                # four such servers for this step
    min_servers: 2            # start annotating once two are ready
```

Several small server jobs schedule far sooner than one large allocation, because
each one fits on a partially used node. They also start at different times,
which is fine: the client starts once `pool.min_servers` are ready, and the
remaining servers join the run as they leave the queue.

## Serving profiles

`engine:` is the whole `vllm serve` command line, per step. The server job runs

```sh
llm-annotate <config> --serve-args <step>
```

and passes the result straight to `vllm serve`, so a value may contain spaces
and a pipeline whose steps use different models needs one submission, not one
per model:

```yaml
  engine:
    tensor_parallel_size: 2
    max_model_len: 8192
    gpu_memory_utilization: 0.90
    max_num_seqs: 256
    speculative_config:            # nested YAML, sent as JSON
      model: my-org/my-draft-model
      num_speculative_tokens: 4
    extra:                         # anything not named above
      reasoning_parser: qwen3
```

Check what a step will serve before allocating any GPU:

```sh
llm-annotate my-pipeline.yaml --serve-args write-qa
```

There is deliberately no cluster variable for anything `vllm serve` takes: a
server asks the config for its own flags, so two steps of one submission can
serve different models with different serving profiles.

## How the two jobs of a step find each other

The server array writes into `<LOG_DIR>/pool_<array-job-id>/`, one `<task>.url`
file per server containing that server's `http://<host>:<port>/v1`. A file
appears only **after** the server answers `/health`, and is removed when the job
ends, so every URL in the directory belongs to a server that is up right now.
The client polls that directory until `min_servers` of the files are there,
then passes the directory itself to the CLI as `--url-glob`, which attaches it
to that step alone — a step on another provider is left untouched. A file of
URLs would be read once; the glob is re-read while the run continues, so a
server whose file appears after the run has started still joins the pool.

Ports are `VLLM_PORT + array task id`, then probed upward for the first free one.
Two array tasks can land on the same node (a 4-GPU node fits two
`tensor_parallel_size: 2` servers), so a fixed port would collide.

The client is submitted with one `after:` dependency per element of its server
array, or-joined (`--dependency=after:1234_1?after:1234_2?...`), which releases
it as soon as the **first** server has begun. `after:<array-id>` as a whole is
satisfied only once *every* element has started, which would leave a ready
server sitting idle behind pool-mates that are still queued (a per-user GPU
quota is enough to do this: one server can occupy the whole quota, so the rest
of the pool queues behind it), burning that server's own `SERVER_TIME` before
the client ever gets to use it. Waiting in the queue rather than on a compute
node also means `CLIENT_TIME` starts counting when there is something to
annotate against: a CPU partition schedules in minutes and a GPU partition can
take days, and a client that starts first spends that gap idle and then dies on
its wall clock.

That dependency replaces the step's own `afterok:<previous client>` rather than
adding to it, since Slurm reads one separator per expression (`,` for and, `?`
for or) and the two cannot be mixed. Nothing is lost: a server cannot start
before the previous step has succeeded, so the client inherits that through the
array it waits on. A cancelled job also satisfies `after:`, so an array that
Slurm kills off releases the client instead of stranding it, and
`--kill-on-invalid-dep=yes` covers what Slurm flags as unsatisfiable outright.
The client recognises that case rather than sitting out its `POOL_WAIT`: while
it waits for URLs it asks `squeue` whether its server array still has an
element in the queue, and stops waiting when it has none left.

When a step finishes, its client `scancel`s that step's server array instead of
leaving GPU jobs idling until their time limit. Set `CANCEL_SERVERS_ON_EXIT=0`
to keep them alive.

## Run-level environment variables

These are about a single run rather than the cluster, so they stay out of the
cluster file and are set on the `submit_pipeline.sh` command line; both job types
are submitted with `--export=ALL`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ANNOTATE_CONFIG` | *the positional argument* | JSON/YAML pipeline config to run |
| `EXTRA_DEPENDENCY` | – | Slurm dependency expression (e.g. `afterok:123456`) the chain waits for. Applied to the **first** submitted step only; later steps inherit it through their predecessor, which is what lets several submissions be chained into one workflow. |
| `POOL_WAIT` | `1800` | Seconds a client waits for `min_servers` of its servers to register once it is running. Matches `READY_TIMEOUT`, so a client cannot give up on a server before the server gives up on itself. The or-joined dependency means the client only starts once a server of its pool has, so this covers a model load rather than an allocation; raise it when running `vllm_annotate.sh` by hand against a pool that is still queued. |
| `VLLM_PORT` | `8000` | Base port a server starts probing from. The array task id is added to it, then the first free port is taken. |
| `READY_TIMEOUT` | `1800` | Seconds a server waits for its own `/health` before giving up |
| `CANCEL_SERVERS_ON_EXIT` | `1` | Whether a finished client `scancel`s its step's server array. `0` leaves the GPUs running. |
| `OUTPUT_DIR`, `HUB_ID`, `OVERWRITE` | from the config | Override the config's `output_dir` / `hub_id`, or discard existing step output |
| `MAX_NUM_SAMPLES`, `SHUFFLE_SEED` | from the config | Override `dataset.max_num_samples` / `dataset.shuffle_seed` for every step of this submission. Raising the cap and resubmitting grows the run: finished rows are not annotated again. |

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
that died continues where it stopped. To resubmit only part of a pipeline, name
the steps:

```sh
./slurm/submit_pipeline.sh --steps rate-qa my-pipeline.yaml
```

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
mkdir -p logs
sbatch --array=1-4 --gres=gpu:2 --cpus-per-task=36 \
  --export=ALL,ANNOTATE_CONFIG=my-pipeline.yaml,STEP_NAME=write-qa \
  slurm/vllm_server.sh   # note the id

cat logs/pool_<id>/*.url > logs/pool_<id>/hosts.txt
```
