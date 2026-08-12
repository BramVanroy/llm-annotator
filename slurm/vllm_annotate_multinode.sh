#!/bin/bash
#SBATCH --job-name=vllm-annotate-multinode
#SBATCH --partition=gpu_a100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=tnsr72764

# EAR energy monitoring
# see https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/62226671/Energy+Aware+Runtime+EAR
#SBATCH --ear=on
#SBATCH --ear-policy=monitoring
#SBATCH --ear-verbose=1

# One self-contained annotation job: starts a vLLM server on every GPU of every
# allocated node, then runs the queue annotator against that pool from node 0.
# The default allocation asks for one GPU on each of two nodes, which schedules
# quickly because it fits on partially used nodes; scale it up with the usual
# flags, e.g. `sbatch --nodes=4 --gres=gpu:4 --cpus-per-task=72 ...`.
#
# Usage:
#   MODEL=HuggingFaceTB/SmolLM2-135M-Instruct \
#   DATASET_NAME=stanfordnlp/imdb DATASET_SPLIT=test MAX_NUM_SAMPLES=500 \
#     sbatch slurm/vllm_annotate_multinode.sh
#
# The run is resumable: progress is written per sample to
# <OUTPUT_DIR>/progress_backup/*.jsonl and re-reading those files is all that a
# restart needs. Re-submitting the exact same command after a crash, a timeout
# or a preemption continues where the previous attempt stopped.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/bvanroy/llm-annotator}"
[[ -d "$REPO_ROOT" ]] || REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
mkdir -p logs

echo "Starting on $(date)"
echo "Host: $(hostname)"
echo "Nodes: ${SLURM_JOB_NODELIST:-<none>}"

module purge
module load 2025

# vllm-kernels ships FlashInfer's prebuilt kernels so nothing has to be
# JIT-compiled while several servers start at once; see pyproject.toml.
uv sync --frozen --extra vllm --extra vllm-kernels
# shellcheck disable=SC1091
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export REPO_ROOT
# EAR 5.2 + DCGM 4.6.0 segfault on Snellius; fall back to the NVML backend.
export EAR_GPU_DCGMI_ENABLED=0

# --- vLLM pool --------------------------------------------------------------
# shellcheck disable=SC1091
source "${REPO_ROOT}/slurm/vllm_servers_lib.sh"
vllm_env_defaults
vllm_ensure_nvcc
export MODEL TP_SIZE GPUS_PER_NODE SERVERS_PER_NODE VLLM_PORT MAX_MODEL_LEN \
  GPU_MEM_UTIL

trap vllm_stop_servers EXIT

vllm_start_servers
printf 'Server URLs: %s\n' "${SERVER_URLS[*]}"
vllm_wait_until_ready "${SERVER_URLS[@]}"

# --- Annotation -------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vllm-multinode}"
BASE_URLS=$(IFS=,; echo "${SERVER_URLS[*]}")

ANNOTATE_ARGS=(
  --base-urls "$BASE_URLS"
  --model "$MODEL"
  --output-dir "$OUTPUT_DIR"
  --batch-size "${BATCH_SIZE:-64}"
  --max-tokens "${MAX_TOKENS:-128}"
)
# Only forward the options that were actually set, so the example keeps its own
# defaults for the rest.
add_annotate_arg() {
  [[ -n "$2" ]] || return 0
  ANNOTATE_ARGS+=("$1" "$2")
}

add_annotate_arg --queue-size "${QUEUE_SIZE:-}"
add_annotate_arg --dataset-name "${DATASET_NAME:-}"
add_annotate_arg --dataset-split "${DATASET_SPLIT:-}"
add_annotate_arg --dataset-config "${DATASET_CONFIG:-}"
add_annotate_arg --prompt-field "${PROMPT_FIELD:-}"
add_annotate_arg --prompt-template "${PROMPT_TEMPLATE:-}"
add_annotate_arg --max-num-samples "${MAX_NUM_SAMPLES:-}"
add_annotate_arg --idx-column "${IDX_COLUMN:-}"
add_annotate_arg --hub-id "${HUB_ID:-}"
add_annotate_arg --task-prefix "${TASK_PREFIX:-}"

set +e
python examples/vllm-multinode/vllm_multinode.py "${ANNOTATE_ARGS[@]}"
ANNOTATE_RC=$?
set -e

echo "Finished on $(date) with status ${ANNOTATE_RC}"
exit "$ANNOTATE_RC"
