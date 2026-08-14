#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=tnsr72764

# EAR energy monitoring
# see https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/62226671/Energy+Aware+Runtime+EAR
# EAR-related envvars are set in slurm/vllm_common.sh which is sourced below
#SBATCH --ear=on
#SBATCH --ear-policy=monitoring
#SBATCH --ear-verbose=1

# One vLLM server, in its own job. Submitted as an array by
# slurm/submit_pool.sh, which derives --array, --gres and --cpus-per-task from
# NUM_SERVERS and GPUS_PER_MODEL and overrides the headers above; a bare
# `sbatch slurm/vllm_server.sh` starts a single one-GPU server with those
# header defaults.
#
# Once the server answers /health it publishes its base URL as
# <POOL_DIR>/<task>.url and removes that file again when the job ends, so the
# pool directory only ever lists servers that are actually up.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/bvanroy/llm-annotator}"
[[ -d "$REPO_ROOT" ]] || REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
mkdir -p logs

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "Starting on $(date)"
echo "Host: $(hostname), array task: ${TASK_ID}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/slurm/vllm_common.sh"
vllm_env_defaults
vllm_setup_env
vllm_ensure_nvcc

export VLLM_WORKER_MULTIPROC_METHOD=spawn

PORT=$(vllm_pick_port $(( VLLM_PORT + TASK_ID )))
URL="http://$(hostname):${PORT}/v1"
URL_FILE="${POOL_DIR}/${TASK_ID}.url"

echo "Serving ${MODEL} on ${URL} with ${GPUS_PER_MODEL} GPU(s)"

# Slurm already restricts this job to GPUS_PER_MODEL devices, so vLLM can use
# all of the ones it can see. VLLM_EXTRA_ARGS is deliberately word-split.
# shellcheck disable=SC2086
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --tensor-parallel-size "$GPUS_PER_MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  ${VLLM_EXTRA_ARGS:-} &
SERVER_PID=$!

cleanup() {
  rm -f "$URL_FILE" "${POOL_DIR}/.${TASK_ID}.tmp"
  kill "$SERVER_PID" 2> /dev/null || true
}
trap cleanup EXIT INT TERM

if ! READY_WATCH_PID="$SERVER_PID" vllm_wait_until_ready "$URL"; then
  echo "Server never became ready; giving up." >&2
  exit 1
fi

# Publish only now, and atomically: a URL in the pool directory is therefore
# always a server the client can talk to right away.
printf '%s\n' "$URL" > "${POOL_DIR}/.${TASK_ID}.tmp"
mv "${POOL_DIR}/.${TASK_ID}.tmp" "$URL_FILE"
echo "Published ${URL} to ${URL_FILE}; serving until the job ends."

status=0
wait "$SERVER_PID" || status=$?
echo "vLLM server exited with status ${status} on $(date)"
exit "$status"
