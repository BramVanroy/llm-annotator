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

## Overriding the config for one submission

`--set KEY=VALUE` is the same flag `llm-annotate` takes, forwarded to every step
job of the submission. A dotted key reaches a nested value, and repeating the
flag sets more than one:

```sh
./slurm/submit_pipeline.sh --set dataset.max_num_samples=50000 my-pipeline.yaml
```

Every step of one submission therefore sees the same overrides, which is what a
growing run needs: resubmitting with a higher `dataset.max_num_samples` extends
the pipeline as a whole rather than one step at a time, which is what
`docs/growing-a-run.md` describes.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SLURM_ACCOUNT` | – | Project to charge. Empty means no `--account`. |
| `GPU_PARTITION`, `CPU_PARTITION` | – | Partitions for the GPU and CPU-only jobs. Empty means the cluster's default. |
| `SERVER_TIME`, `CLIENT_TIME` | `04:00:00`, `05:00:00` | Wall time per job kind. |
| `CPUS_PER_GPU` | `8` | Cores a GPU job asks for per GPU. |
| `CLIENT_CPUS` | `8` | Cores a CPU-only annotation job asks for. |
| `MAX_GPUS_PER_NODE` | `8` | Ceiling a step's `engine.tensor_parallel_size` is checked against before anything is submitted. |
| `GPU_REQUEST`, `GPU_TYPE` | `gres`, – | How GPUs are requested: `--gres=gpu:N` or `--gpus-per-node=N`, optionally typed (`gpu:a100:N`). |
| `MAX_CONCURRENT_SERVERS` | – | Most elements of a server array that may run at once (`--array=1-N%M`), for a per-user GPU limit. Empty means no throttle. |
| `SERVER_HOST_CMD` | `hostname` | Command a server runs to get the address other nodes reach it at. Use `hostname -f`, or a command that prints one interface's address, where the short name does not resolve. |
| `MODEL_DOWNLOAD` | `0` | `1` fetches every model this submission serves in a CPU job before any GPU is allocated. |
| `DOWNLOAD_PARTITION`, `DOWNLOAD_TIME` | `$CPU_PARTITION`, `02:00:00` | Partition and wall time of those download jobs. |
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

A container works the same way, because `llm-annotate` and `vllm` may be shell
functions rather than executables. Define them at the **top level** of the
cluster file, not inside `cluster_setup_env`: every script sources the cluster
file, so a top-level function is visible in both job scripts and on the login
node, where `submit_pipeline.sh` reads the config. A function defined inside
`cluster_setup_env` reaches the job scripts only, because the login node never
calls it.

```sh
SIF=/projects/shared/images/vllm-0.29.0.sif
BINDS="/projects,/scratch-local"
llm-annotate() { apptainer exec --nv -B "$BINDS" "$SIF" llm-annotate "$@"; }
vllm()         { apptainer exec --nv -B "$BINDS" "$SIF" vllm "$@"; }
cluster_setup_env() { :; }
```

The empty `cluster_setup_env` keeps the module-and-venv handling out of the way.
Two limits: the login node has to be able to run the container runtime, and the
bind mounts have to cover the repo, the config, the output directory and the
Hugging Face cache.

## What each step becomes

The shape of a job is derived from the step's client, never configured twice.
Check it before submitting anything:

```sh
llm-annotate my-pipeline.yaml --describe-steps
```

```json
{"index": 1, "name": "write-qa", "kind": "vllm_pool", "provider": "vllm_online", "model": "Qwen/Qwen3-8B", "servers": 4, "min_servers": 2, "gpus_per_vllm_server": 2, "step_dir": "...", "batch_size": 64, "max_concurrent_batches_per_client": 4, "queue_size": 32, "max_requests_per_server": 256, "max_requests_in_flight": 1024}
{"index": 2, "name": "rate-qa", "kind": "api", "provider": "claude", "model": "claude-haiku-4-5", "servers": 1, "min_servers": 1, "gpus_per_vllm_server": 1, "step_dir": "...", "batch_size": 256, "max_concurrent_batches_per_client": null, "queue_size": null, "max_requests_per_server": null, "max_requests_in_flight": null}
```

`submit_pipeline.sh` reads the same fields as shell assignments
(`--describe-steps --format env`, one `STEP_KEY=VALUE` line per step, every
value shell-quoted) and evaluates a line per step, so a step name or a model
containing a space or a quote reaches `sbatch` intact. Both formats are
documented under "Running one step at a time" in `docs/pipeline.md`.

`max_requests_per_server` is `max_concurrent_batches_per_client` times
`batch_size`: the number of prompts one server is asked to hold at once, and so
the number its `engine.max_num_seqs` has to cover before requests start queueing
inside vLLM. `submit_pipeline.sh` prints it per step, so a pool can be sized
before any GPU is allocated.

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

`pool.min_servers` is a target with a fallback. The client
waits for that many `.url` files and gives up waiting when `squeue` reports that
the server array has no element left in the queue, or when `POOL_WAIT` is up. It
then annotates on the servers it does have, and says how many in its log
(`Annotating over 2 of 4 server(s)`). A pool that half fills therefore still
produces data, at half the throughput. With no server at all the step still
runs: a step that already finished loads its result and ends with status 0, and
any other step fails with `url_glob ... matched no files`, in which case the
`vllm-<step>_*.err` logs say why no server came up.

At a site with a per-user GPU limit, set `MAX_CONCURRENT_SERVERS` in the cluster
file. The array is then submitted as `--array=1-4%2`, so two servers run while
the other two wait. It has to be at least the largest `pool.min_servers` in the
config, otherwise the client's threshold can never be reached; the submitter
rejects that combination on the login node.

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

A server that ends before the run does (hits `SERVER_TIME`, is preempted, or
otherwise stops answering `/health`) is dropped from the pool once a batch
fails on it, and the run continues on the servers that are left. A requeued
server job that publishes its `.url` file again is picked up the same way a
late starter is, so servers of one array reaching `SERVER_TIME` at different
moments do not end the client.

The host in that URL comes from `SERVER_HOST_CMD`, which is `hostname` unless
the cluster file says otherwise. Set it to `hostname -f` where compute nodes
only resolve fully qualified names, or to a command that prints the address of
the fabric the client should use.

Ports are `VLLM_PORT + array task id`, then probed upward for the first free one.
Two array tasks can land on the same node (a 4-GPU node fits two
`tensor_parallel_size: 2` servers), so a fixed port would collide. The probe and
the bind are two separate moments, so another process can take the port in
between; vLLM then dies with `Address already in use` before `/health` ever
answers, and the server job retries on the next free port, at most
`PORT_RETRIES` times (5 by default). Any other early exit fails the task.

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
leaving GPU jobs idling until their time limit. That happens in an `EXIT` trap,
which a cgroup OOM kill, a `scancel -s KILL` or a dead node skips, so the
submitter also queues a small clean-up job per pool step:

```text
sbatch --time=00:05:00 --job-name=cancel-write-qa --cpus-per-task=1 \
  --dependency=afterany:<client> --kill-on-invalid-dep=yes --wrap="scancel <array>"
```

`afterany` means it runs however the client ended, and it cancels that one
array, so the servers of a later attempt are untouched. It uses the CPU
partition and the client's `sbatch` flags, and it is not part of the chain
between steps: the next step waits for the client, never for the clean-up. Set
`CANCEL_SERVERS_ON_EXIT=0` to keep servers alive after their step ends; the
clean-up job is then not submitted either.

## Resubmitting automatically

A step that runs out of `CLIENT_TIME` leaves a resumable run behind, but
somebody has to notice and submit it again. `--max-resubmits N` queues that
follow-up in advance:

```sh
./slurm/submit_pipeline.sh --max-resubmits 2 my-pipeline.yaml
```

Every step then gets up to `N + 1` attempts. Attempt *k* runs the same command
as attempt *k-1* and carries `--dependency=afternotok:<attempt k-1>`, so it
starts only when the attempt before it failed, timed out or was cancelled. Being
the same command is what makes it a resume: the step's progress files are read
and only the rows that are missing are sent to the model. A pool step gets a
fresh server array and a fresh clean-up job per attempt, and that attempt's
clean-up only cancels that attempt's array.

What happens to the attempts that turn out not to be needed:

- The attempt before it succeeded. `afternotok` can then never be satisfied, so
  `--kill-on-invalid-dep=yes` has Slurm remove the attempt rather than leave it
  queued. For a pool step the removed job is the server array, and Slurm counts
  a cancelled job as a satisfied `after:` dependency, so that attempt's client
  does start. It finds no server, runs the step anyway, and the library loads
  the finished step's `output/` snapshot and exits with status 0 within
  seconds. No GPU is allocated, and every later attempt ends the same way.
- The attempt before it failed, but the step had already finished in an earlier
  submission. The attempt runs, the library loads the step's `output/` snapshot
  and the job ends in seconds.

The next step waits for "any attempt of the previous step succeeded", which
Slurm writes as an or-joined dependency list:

```text
--dependency=afterok:<attempt 1>?afterok:<attempt 2>
```

A `,` between dependencies means all of them must be satisfied, a `?` means any
one of them is enough, and the two cannot be mixed in one expression. So the
next step starts as soon as any attempt of its predecessor succeeds, and the
attempts that were removed do not hold it back. If every attempt fails, the list
can never be satisfied and the next step stays in the queue with
`DependencyNeverSatisfied` until you cancel it (or, at a site that configures
`kill_invalid_depend`, Slurm removes it), which is the right outcome: nothing
runs on a half-finished input.

`--dry-run` prints the whole chain, attempts included, which is the way to check
it before you use it:

```sh
./slurm/submit_pipeline.sh --dry-run --max-resubmits 1 my-pipeline.yaml
```

## Model downloads

The servers of a pool all start at once, and on a cold cache they all download
the same weights. `MODEL_DOWNLOAD=1` in the cluster file turns that into one
small CPU job per distinct model, submitted before anything else, which the
first step then waits for with `afterok`. Every later step inherits that wait
through the chain, so a pipeline whose steps use different models downloads all
of them up front:

```text
sbatch --time=02:00:00 --job-name=download-Qwen3-8B --cpus-per-task=2 \
  --wrap="... && vllm_download_model Qwen/Qwen3-8B"
```

The job runs `hf download <model>`, the CLI that comes with `huggingface_hub`.
Only the models this submission has to serve itself are fetched (the
`vllm_pool` and `vllm_offline` steps); a hosted provider has nothing to
download, and a `model` that is an existing directory is a local checkout and is
left alone. `DOWNLOAD_PARTITION` and `DOWNLOAD_TIME` size the job.

It is off by default. The job needs a route to huggingface.co from whichever
node runs it, which not every site has, and against a warm cache it costs a
queue wait for nothing. A gated model needs a token, which the job reads the
same way any other `hf` command does: `hf auth login` once, or `HF_TOKEN` in the
environment, which `--export=ALL` carries.

Wherever you run it, `HF_HOME` decides where the weights land; it defaults to
`.cache/huggingface` in your home directory. On a cluster, point it at a project
or scratch filesystem that every compute node can read, from your shell profile
or from the cluster file. A home directory is usually too small for a few 8B
checkpoints, and a node-local path is downloaded again by every server.

## Threads of a pool client

A pool client holds one thread per request in flight, and such a thread only
waits for the network. The count is `servers` x
`max_concurrent_batches_per_client` x `batch_size`: four servers with the
defaults (four batches of 256 each) is 4096 threads. A site with a low
`ulimit -u`, or with a cgroup `pids.max` on its CPU jobs, fails the run with
`RuntimeError: can't start new thread`, so the client logs the limit it found
at the top of its log:

```console
Thread limit (ulimit -u): 4096
```

Two settings on the step's `client` block lower the count: `batch_size` (fewer
requests per batch) and `init.max_workers` (how many requests of one batch go
out at once, `None` by default, which sends all of them).

`init.timeout` belongs next to `SERVER_TIME`. It is 3600 seconds per request and
covers the wait in the server's queue as well as the generation itself. A
request that outlives its server, because the server hit `SERVER_TIME` or was
preempted, becomes an error row with an `error` and an `error_type` rather than
a failed run. A resubmission keeps those rows as they are; `llm-annotate`'s own
`--retry-errors` is what annotates them again, and the end-of-run summary lists
the error types to name:

```sh
llm-annotate my-pipeline.yaml --steps write-qa --retry-errors APITimeoutError
```

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
| `PORT_RETRIES` | `5` | How often a server job retries on the next port after `vllm serve` failed to bind the one it probed. |
| `CANCEL_SERVERS_ON_EXIT` | `1` | Whether a finished client `scancel`s its step's server array. `0` leaves the GPUs running. |
| `SBATCH_CMD` | `sbatch` | The command that queues a job, for a site whose `sbatch` is wrapped. A submit this refuses ends the run: the steps after it would otherwise depend on a job id that was never issued. |
| `OUTPUT_DIR`, `HUB_ID`, `OVERWRITE` | from the config | Override the config's `output_dir` / `hub_id`, or discard existing step output |
| `ANNOTATE_SET` | – | What `--set` fills: config overrides for this submission, one `KEY=VALUE` per line, passed to `llm-annotate --set` on every step job. Set it directly only when scripting the submitter; `--set` is the way in. |

## Resuming

Two levels, both automatic:

- **Within a step**, every annotated sample is appended to
  `<output_dir>/<NN>-<step>/annotate/<prefix>progress_backup/*.jsonl` and flushed
  immediately; a restart re-reads those files and skips the ids already present.
  A half-written final line from a killed job is detected and re-annotated.
- **Between steps**, a finished step writes `<output_dir>/<NN>-<step>/output/`,
  which a later run loads instead of recomputing.

So after a crash, a timeout or a preemption you run the same
`submit_pipeline.sh` command again. Finished steps are skipped, and the step
that died continues where it stopped. Resubmitting with a higher
`dataset.max_num_samples` extends a finished run the same way, which
`docs/growing-a-run.md` describes in full. To resubmit only part of a pipeline,
name the steps:

```sh
./slurm/submit_pipeline.sh --steps rate-qa my-pipeline.yaml
```

[`--max-resubmits`](#resubmitting-automatically) queues those follow-up
submissions up front, which is the same resume without waiting for you to
notice. It needs nothing else: the attempt runs on the same cluster, against the
same `output_dir`, so the progress files it reads are already there.

A purged scratch directory is a different case, and so is a move to another
cluster. The prepared data comes back from its Hub branch on its own; the JSONL
progress files do not. Restore a step's progress backup before you resubmit,
with that step's own directory (`<output_dir>/<NN>-<step>/annotate/`) and prefix
(`<step>_`):

```sh
python scripts/restore_progress_from_hub.py --hub-id user/my-dataset --output-dir outputs/qa/02-rate-qa/annotate --task-prefix rate-qa_
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
