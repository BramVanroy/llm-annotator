# SLURM job scripts (Snellius)

One vLLM server per SLURM job. You describe the pool with two numbers — how many
GPUs one copy of the model needs, and how many copies you want — and
`submit_pool.sh` turns that into an array of server jobs plus one client job
that annotates over all of them:

```sh
MODEL=Qwen/Qwen3.5-4B GPUS_PER_MODEL=2 NUM_SERVERS=4 \
DATASET_NAME=stanfordnlp/imdb DATASET_SPLIT=test PROMPT_FIELD=text \
MAX_NUM_SAMPLES=5000 BATCH_SIZE=64 \
  ./slurm/submit_pool.sh
```

That submits **4 jobs of 2 GPUs each**, each running one vLLM server with
`--tensor-parallel-size 2`, and a CPU-only client job that pools all four behind
a `VLLMQueueAnnotator`. Nothing about the allocation lives in an `#SBATCH`
header you have to edit.

Four small jobs schedule far sooner than one 8-GPU allocation, because each one
fits on a partially used node. They also start at different times, which is
fine: the client waits for the pool to fill up before it begins.

| File | Role |
| --- | --- |
| `submit_pool.sh` | Run on a login node. Derives `--array`, `--gres` and `--cpus-per-task` from the two knobs and submits both jobs. |
| `vllm_server.sh` | One array task = one vLLM server. Publishes its base URL once it is healthy. |
| `vllm_annotate.sh` | Client job. Waits for the pool, runs the annotation, releases the GPUs. |
| `vllm_common.sh` | Sourced by both jobs: environment setup, port selection, health polling. |

## How the two jobs find each other

The server array writes into `logs/pool_<array-job-id>/`, one `<task>.url` file
per server, containing that server's `http://<host>:<port>/v1`. A file appears
only **after** the server answers `/health`, and is removed when the job ends, so
every URL in the directory belongs to a server that is up right now. The client
polls that directory, concatenates it into `hosts.txt` and hands it to the
driver.

Ports are `VLLM_PORT + array task id`, then probed upward for the first free
one. Two array tasks can land on the same node (a 4-GPU node fits two
`GPUS_PER_MODEL=2` servers), so a fixed port would collide.

When the annotation finishes the client `scancel`s the server array, instead of
leaving four GPU jobs idling until their time limit. Set
`CANCEL_SERVERS_ON_EXIT=0` to keep them alive.

## Configuration

Everything is passed as environment variables on the `submit_pool.sh` command
line; both jobs are submitted with `--export=ALL`, so anything you set there
reaches them.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL` | *required* | Model served by every server |
| `GPUS_PER_MODEL` | `1` | GPUs per job, and `--tensor-parallel-size`. Max 4 — one server never spans nodes. |
| `NUM_SERVERS` | `2` | Number of server jobs |
| `MIN_SERVERS` | `$NUM_SERVERS` | Servers the client insists on. Lower it so one straggling job cannot hold up the run. |
| `POOL_WAIT` | `3600` | Seconds the client waits for `NUM_SERVERS` before settling for `MIN_SERVERS` |
| `GPU_PARTITION`, `SERVER_TIME` | `gpu_a100`, `04:00:00` | Server allocation |
| `CLIENT_PARTITION`, `CLIENT_TIME` | `rome`, `05:00:00` | Client allocation (no GPU needed) |
| `SLURM_ACCOUNT`, `CPUS_PER_GPU` | `tnsr72764`, `16` | Accounting and CPU share per GPU |
| `VLLM_PORT`, `MAX_MODEL_LEN`, `GPU_MEM_UTIL` | `8000`, `8192`, `0.90` | Passed to `vllm serve` |
| `VLLM_EXTRA_ARGS` | – | Extra flags appended to `vllm serve` |
| `CANCEL_SERVERS_ON_EXIT` | `1` | Cancel the server array when the client is done |
| `OUTPUT_DIR` | `outputs/vllm-server-pool` | Progress files and final dataset |
| `DATASET_NAME`, `DATASET_SPLIT`, `DATASET_CONFIG` | – | Source dataset; omit for the built-in demo texts |
| `PROMPT_FIELD`, `PROMPT_TEMPLATE` | `text`, sentiment prompt | Column and template |
| `IDX_COLUMN` | `idx` | Stable identifier column; must not exist in the source dataset |
| `MAX_NUM_SAMPLES`, `BATCH_SIZE`, `QUEUE_SIZE`, `MAX_TOKENS` | – / `64` / 2 per server / `128` | Workload sizing |
| `HUB_ID`, `TASK_PREFIX` | – | Hub backup target and column/path namespace |

`DRY_RUN=1` prints the two `sbatch` invocations instead of submitting them,
which is the quickest way to check what an allocation will actually ask for:

```sh
MODEL=x GPUS_PER_MODEL=2 NUM_SERVERS=4 DRY_RUN=1 ./slurm/submit_pool.sh
```

## Resuming

Every annotated sample is appended to
`<OUTPUT_DIR>/<TASK_PREFIX>progress_backup/*.jsonl` and flushed immediately. A
restart re-reads those files and skips the ids already present, so after a
crash, a timeout or a preemption you simply run **the same `submit_pool.sh`
command again**. A half-written final line from a killed job is detected and
re-annotated, and overlapping writers cannot duplicate a sample in the final
dataset.

## Running the client yourself

The client only speaks HTTP, so it can equally run from a login node, an
interactive session or a CPU job. Submit servers without a client by calling
`sbatch` directly, then point the driver at the pool:

```sh
sbatch --array=1-4 --gres=gpu:2 --cpus-per-task=32 \
  --export=ALL,MODEL="$MODEL",GPUS_PER_MODEL=2 slurm/vllm_server.sh   # note the id

cat logs/pool_<id>/*.url > logs/pool_<id>/hosts.txt
python examples/vllm-server-pool/vllm_server_pool.py \
    --hosts-file logs/pool_<id>/hosts.txt \
    --model "$MODEL" --wait-for-servers 900 --dataset-name stanfordnlp/imdb
```
