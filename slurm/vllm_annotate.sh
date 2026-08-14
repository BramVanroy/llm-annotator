#!/bin/bash
#SBATCH --job-name=vllm-annotate
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=tnsr72764

# The client half of the pool: waits for the server jobs to publish their URLs
# in POOL_DIR, then annotates the dataset over all of them at once with a
# VLLMQueueAnnotator. It only ever speaks HTTP, so it runs on a CPU partition
# and never occupies a GPU of its own. Submitted by slurm/submit_pool.sh with a
# dependency on the server array.
#
# The run is resumable: progress is written per sample to
# <OUTPUT_DIR>/<TASK_PREFIX>progress_backup/*.jsonl and re-reading those files
# is all that a restart needs. Re-submitting the same submit_pool.sh command
# after a crash, a timeout or a preemption continues where it stopped.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/bvanroy/llm-annotator}"
[[ -d "$REPO_ROOT" ]] || REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
mkdir -p logs

: "${MODEL:?Set MODEL, e.g. MODEL=Qwen/Qwen3.5-4B}"
: "${POOL_DIR:?Set POOL_DIR; submit through slurm/submit_pool.sh}"
: "${NUM_SERVERS:=1}"
: "${MIN_SERVERS:=$NUM_SERVERS}"
: "${POOL_WAIT:=3600}"
: "${CANCEL_SERVERS_ON_EXIT:=1}"

echo "Starting on $(date)"
echo "Host: $(hostname)"
echo "Pool: ${POOL_DIR} (waiting for ${NUM_SERVERS} server(s))"

# shellcheck disable=SC1091
source "${REPO_ROOT}/slurm/vllm_common.sh"
vllm_setup_env

# Free the GPUs as soon as the annotation is done, instead of letting the
# server jobs idle until their own time limit.
release_servers() {
  if [[ "$CANCEL_SERVERS_ON_EXIT" == "1" && -n "${SERVER_JOB_ID:-}" ]]; then
    echo "Cancelling server job ${SERVER_JOB_ID}"
    scancel "$SERVER_JOB_ID" 2> /dev/null || true
  fi
}
trap release_servers EXIT

# --- Wait for the pool -------------------------------------------------------
count_urls() {
  local files=("$POOL_DIR"/*.url)
  [[ -e "${files[0]}" ]] && echo "${#files[@]}" || echo 0
}

deadline=$(( SECONDS + POOL_WAIT ))
ready=$(count_urls)
while (( ready < NUM_SERVERS )); do
  if (( SECONDS > deadline )); then
    echo "Waited ${POOL_WAIT}s for ${NUM_SERVERS} server(s), ${ready} showed up."
    break
  fi
  sleep 10
  ready=$(count_urls)
done

if (( ready < MIN_SERVERS )); then
  echo "Only ${ready} of ${NUM_SERVERS} server(s) registered in ${POOL_DIR}," \
    "need at least ${MIN_SERVERS}. See logs/vllm-server_*.err" >&2
  exit 1
fi

HOSTS_FILE="${POOL_DIR}/hosts.txt"
cat "$POOL_DIR"/*.url > "$HOSTS_FILE"
echo "Annotating over ${ready} server(s):"
cat "$HOSTS_FILE"

# --- Annotation --------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/vllm-server-pool}"

ANNOTATE_ARGS=(
  --hosts-file "$HOSTS_FILE"
  --model "$MODEL"
  --output-dir "$OUTPUT_DIR"
  --batch-size "${BATCH_SIZE:-64}"
  --max-tokens "${MAX_TOKENS:-128}"
  --wait-for-servers "${WAIT_FOR_SERVERS:-300}"
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
python examples/vllm-server-pool/vllm_server_pool.py "${ANNOTATE_ARGS[@]}"
ANNOTATE_RC=$?
set -e

echo "Finished on $(date) with status ${ANNOTATE_RC}"
exit "$ANNOTATE_RC"
