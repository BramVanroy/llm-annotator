# SLURM job scripts (Snellius)

Two entry points, both starting one vLLM server per GPU on every allocated node
and pooling them behind a single `VLLMQueueAnnotator`:

| Script | Default allocation | What it does |
| --- | --- | --- |
| `vllm_annotate_multinode.sh` | 2 nodes x 1 GPU | Servers **and** annotation in one job. The usual choice. |
| `vllm_server_multinode.sh` | 1 node x 1 GPU | Servers only; writes `logs/vllm_urls_<jobid>.txt` for a client elsewhere. |

Both default to a small allocation on purpose: one GPU per node fits on a
partially used node, so the job starts in minutes instead of waiting for whole
idle nodes. Scale up on the `sbatch` command line
(`--nodes=4 --gres=gpu:4 --cpus-per-task=72`); the scripts follow whatever the
allocation actually got.

`vllm_servers_lib.sh` (sourced, not submitted) launches the servers and waits for
their `/health` endpoints; `vllm_node_servers.sh` is the per-node step that
splits that node's GPU mask between its servers.

## Run an annotation job

```sh
sbatch --export=ALL,\
MODEL=Qwen/Qwen3.5-4B,\
DATASET_NAME=stanfordnlp/imdb,DATASET_SPLIT=test,PROMPT_FIELD=text,\
MAX_NUM_SAMPLES=5000,BATCH_SIZE=64 \
  slurm/vllm_annotate_multinode.sh
```

Snellius does not hand a job the submitting shell's environment, so pass
configuration through `--export=ALL,VAR=...` (or edit the defaults in the
script). Recognised variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL` | *required* | Model served by every server |
| `TP_SIZE` | `1` | GPUs per server; `4` gives one tensor-parallel server per node |
| `GPUS_PER_NODE` | `$SLURM_GPUS_ON_NODE` | GPUs to use per node; must be a multiple of `TP_SIZE` |
| `VLLM_PORT` | `8000` | Port of the first server on each node |
| `MAX_MODEL_LEN`, `GPU_MEM_UTIL` | `4096`, `0.85` | Passed to `vllm serve` |
| `OUTPUT_DIR` | `outputs/vllm-multinode` | Progress files and final dataset |
| `DATASET_NAME`, `DATASET_SPLIT`, `DATASET_CONFIG` | – | Source dataset; omit for the built-in demo texts |
| `PROMPT_FIELD`, `PROMPT_TEMPLATE` | `text`, sentiment prompt | Column and template |
| `IDX_COLUMN` | `idx` | Stable identifier column; must not exist in the source dataset |
| `MAX_NUM_SAMPLES`, `BATCH_SIZE`, `QUEUE_SIZE`, `MAX_TOKENS` | – / `64` / 2 per server / `128` | Workload sizing |
| `HUB_ID`, `TASK_PREFIX` | – | Hub backup target and column/path namespace |

Change the allocation itself (`--nodes`, `--partition`, `--time`) on the `sbatch`
command line as usual.

## Resuming

Every annotated sample is appended to
`<OUTPUT_DIR>/<TASK_PREFIX>progress_backup/*.jsonl` and flushed immediately. A
restart re-reads those files and skips the ids already present, so after a
crash, a timeout or a preemption you simply submit **the same command again**. A
half-written final line from a killed job is detected and re-annotated, and
overlapping writers cannot duplicate a sample in the final dataset.

## Pooling several server jobs

Several small server jobs pool just as well as one big allocation, and schedule
far sooner. Submit the server job as many times as you want servers, then point
one client at the union of their URL files:

```sh
sbatch --export=ALL,MODEL="$MODEL" slurm/vllm_server_multinode.sh   # note the ids
sbatch --export=ALL,MODEL="$MODEL" slurm/vllm_server_multinode.sh
cat logs/vllm_urls_<id1>.txt logs/vllm_urls_<id2>.txt > logs/pool.txt

python examples/vllm-multinode/vllm_multinode.py \
    --hosts-file logs/pool.txt \
    --model "$MODEL" --wait-for-servers 900 --dataset-name stanfordnlp/imdb
```

The client only talks HTTP, so it can run from a login node, an interactive
session, or a CPU-only job.
