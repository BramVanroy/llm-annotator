#!/bin/bash
# Submit a vLLM server pool and the annotation client that consumes it.
#
# Run this from a login node; it is not an sbatch script itself. Two knobs
# describe the pool:
#
#   GPUS_PER_MODEL   GPUs handed to each server job, i.e. tensor-parallel size
#   NUM_SERVERS      how many such jobs to submit
#
# so GPUS_PER_MODEL=2 NUM_SERVERS=4 gives four independent jobs of two GPUs
# each, four vLLM servers, and one client annotating over all of them. Small
# independent jobs schedule far sooner than one large allocation, and a server
# that is still queued only delays the pool, it does not break it.
#
# Usage:
#   MODEL=Qwen/Qwen3.5-4B GPUS_PER_MODEL=2 NUM_SERVERS=4 \
#   DATASET_NAME=stanfordnlp/imdb DATASET_SPLIT=test MAX_NUM_SAMPLES=5000 \
#     ./slurm/submit_pool.sh
#
# Set DRY_RUN=1 to print the two submissions instead of running them.

set -euo pipefail

: "${MODEL:?Set MODEL, e.g. MODEL=Qwen/Qwen3.5-4B}"
: "${GPUS_PER_MODEL:=1}"
: "${NUM_SERVERS:=2}"
: "${MIN_SERVERS:=${NUM_SERVERS}}"
: "${GPU_PARTITION:=gpu_a100}"
: "${CLIENT_PARTITION:=rome}"
: "${SERVER_TIME:=04:00:00}"
: "${CLIENT_TIME:=05:00:00}"
: "${SLURM_ACCOUNT:=tnsr72764}"
: "${CPUS_PER_GPU:=18}"
: "${MAX_GPUS_PER_NODE:=4}"
: "${DRY_RUN:=0}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# The jobs derive their own pool directory from the array job id; a value left
# over in this shell from an earlier run must not leak in through --export=ALL.
unset POOL_DIR LOG_DIR

if (( NUM_SERVERS < 1 )); then
  echo "NUM_SERVERS (${NUM_SERVERS}) must be at least 1" >&2
  exit 1
fi

# A server's GPUs must all sit in one job on one node: vLLM's tensor
# parallelism does not span nodes here, and Snellius GPU nodes have four.
if (( GPUS_PER_MODEL < 1 || GPUS_PER_MODEL > MAX_GPUS_PER_NODE )); then
  echo "GPUS_PER_MODEL (${GPUS_PER_MODEL}) must be between 1 and" \
    "${MAX_GPUS_PER_NODE}: one server runs inside a single job on a single" \
    "node. Use a smaller model, or raise MAX_GPUS_PER_NODE if the partition" \
    "has more." >&2
  exit 1
fi

# Echo the submitted job id, or print the command under DRY_RUN. --parsable
# appends ";<cluster>" in multi-cluster setups, hence the trim.
submit() {
  local out
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'sbatch --parsable %s\n' "$*" >&2
    echo "DRYRUN"
    return 0
  fi
  out=$(sbatch --parsable "$@")
  echo "${out%%;*}"
}

# --- Server array ------------------------------------------------------------
# Flags given here override the #SBATCH headers in the job scripts, so the
# headers stay valid defaults for a bare `sbatch slurm/vllm_server.sh`.
SERVER_JOB=$(submit \
  --partition="$GPU_PARTITION" \
  --account="$SLURM_ACCOUNT" \
  --time="$SERVER_TIME" \
  --array="1-${NUM_SERVERS}" \
  --gres="gpu:${GPUS_PER_MODEL}" \
  --cpus-per-task="$(( CPUS_PER_GPU * GPUS_PER_MODEL ))" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},MODEL=${MODEL},GPUS_PER_MODEL=${GPUS_PER_MODEL}" \
  slurm/vllm_server.sh)

# Naming the pool after the array job id keeps concurrent runs apart and makes
# a stale directory obviously stale. The server jobs derive the same name from
# SLURM_ARRAY_JOB_ID, so nothing has to be told about it twice.
POOL_DIR="${REPO_ROOT}/logs/pool_${SERVER_JOB}"
[[ "$DRY_RUN" == "1" ]] || mkdir -p "$POOL_DIR"

echo "Submitted server array ${SERVER_JOB}:" \
  "${NUM_SERVERS} job(s) x ${GPUS_PER_MODEL} GPU(s) on ${GPU_PARTITION}"

# --- Client ------------------------------------------------------------------
# `after` (not `afterok`) so the client starts once the array begins running;
# it then waits for the servers to report in itself.
CLIENT_JOB=$(submit \
  --partition="$CLIENT_PARTITION" \
  --account="$SLURM_ACCOUNT" \
  --time="$CLIENT_TIME" \
  --dependency="after:${SERVER_JOB}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},MODEL=${MODEL},POOL_DIR=${POOL_DIR},NUM_SERVERS=${NUM_SERVERS},MIN_SERVERS=${MIN_SERVERS},SERVER_JOB_ID=${SERVER_JOB}" \
  slurm/vllm_annotate.sh)

echo "Submitted client job ${CLIENT_JOB} on ${CLIENT_PARTITION}," \
  "waiting for ${MIN_SERVERS}/${NUM_SERVERS} server(s)"
echo
echo "Pool directory: ${POOL_DIR}"
echo "Server logs:    logs/vllm-server_${SERVER_JOB}_*.out"
echo "Client log:     logs/vllm-annotate_${CLIENT_JOB}.out"
echo "Cancel all:     scancel ${SERVER_JOB} ${CLIENT_JOB}"
