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

# Runs ONE step of a pipeline config. slurm/submit_pipeline.sh submits one of
# these per step, chained so each starts when the previous one succeeded.
#
# The step decides what this job needs, which the submitter has already worked
# out from the config with `llm-annotate --describe-steps`:
#
#   POOL_DIR set    a companion server array is starting up; wait for it to
#                   publish its URLs, then annotate over the whole pool
#   POOL_DIR unset  nothing to wait for. Either the step calls a hosted API
#                   (no accelerator at all) or it loads the model in-process,
#                   in which case the submitter asked for GPUs on this job.
#
# The run is resumable at two levels: within a step, per-sample progress is
# written to <OUTPUT_DIR>/<NN>-<step>/annotate/*/progress_backup/*.jsonl, and a
# finished step writes <NN>-<step>/output/, which a re-run loads instead of
# recomputing. Re-submitting the same submit_pipeline.sh command after a crash,
# a timeout or a preemption continues where it stopped.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/bvanroy/llm-annotator}"
[[ -d "$REPO_ROOT" ]] || REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
mkdir -p logs

: "${ANNOTATE_CONFIG:?Set ANNOTATE_CONFIG to a JSON/YAML pipeline config}"
: "${STEP_NAME:?Set STEP_NAME to the step of that config to run}"
: "${NUM_SERVERS:=1}"
: "${MIN_SERVERS:=1}"
: "${POOL_WAIT:=3600}"

echo "Starting on $(date)"
echo "Host: $(hostname)"
echo "Step: ${STEP_NAME} of ${ANNOTATE_CONFIG}"

source "${REPO_ROOT}/slurm/vllm_common.sh"
# set up needed cache dirs and environment
vllm_setup_env

# Free the GPUs as soon as this client dies or ends running
# so that the GPU server does not keep running up to timelimit
release_servers() {
  if [[ -n "${SERVER_JOB_ID:-}" ]]; then
    echo "Cancelling server job ${SERVER_JOB_ID}"
    scancel "$SERVER_JOB_ID" 2> /dev/null || true
  fi
}
trap release_servers EXIT

ANNOTATE_ARGS=(--steps "$STEP_NAME")

# When relying on a vllm pool, the client first waits until NUM_SERVERS are ready
# The server each write their own url files to pool dir
if [[ -n "${POOL_DIR:-}" ]]; then
  echo "Pool: ${POOL_DIR} (waiting for ${NUM_SERVERS} server(s))"

  count_urls() {
    local files=("$POOL_DIR"/*.url)
    # check if first entry exists, if so count; otherwise 0
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

  # Create host file from the pool dir urls
  HOSTS_FILE="${POOL_DIR}/hosts.txt"
  cat "$POOL_DIR"/*.url > "$HOSTS_FILE"
  echo "Annotating over ${ready} server(s):"
  cat "$HOSTS_FILE"

  ANNOTATE_ARGS+=(--hosts-file "$HOSTS_FILE")
fi

# Most of the needed configuration lives in the config file
# but some option can be overridden by env vars
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  ANNOTATE_ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "${HUB_ID:-}" ]]; then
  ANNOTATE_ARGS+=(--hub-id "$HUB_ID")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  ANNOTATE_ARGS+=(--overwrite)
fi

# as always, capture the python exit code and use it to exit slurm with the same code
set +e
python scripts/annotate.py "$ANNOTATE_CONFIG" "${ANNOTATE_ARGS[@]}"
ANNOTATE_RC=$?
set -e

echo "Finished step ${STEP_NAME} on $(date) with status ${ANNOTATE_RC}"
exit "$ANNOTATE_RC"
