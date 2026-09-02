#!/bin/bash
# Defaults for a manual `sbatch slurm/vllm_annotate.sh`; submit_pipeline.sh
# overrides all of them from the cluster file and the step's own config.
#SBATCH --job-name=annotate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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
# written to <output_dir>/<NN>-<step>/annotate/*/progress_backup/*.jsonl, and a
# finished step writes <NN>-<step>/output/, which a re-run loads instead of
# recomputing. Re-submitting the same submit_pipeline.sh command after a crash,
# a timeout or a preemption continues where it stopped.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"

: "${ANNOTATE_CONFIG:?Set ANNOTATE_CONFIG to a JSON/YAML pipeline config}"
: "${STEP_NAME:?Set STEP_NAME to the step of that config to run}"
: "${NUM_SERVERS:=1}"
: "${POOL_WAIT:=3600}"

echo "Starting on $(date)"
echo "Host: $(hostname)"
echo "Step: ${STEP_NAME} of ${ANNOTATE_CONFIG}"

# shellcheck source=slurm/vllm_common.sh
source "${REPO_ROOT}/slurm/vllm_common.sh"
cluster_env_load
vllm_setup_env

# Free the GPUs as soon as this client ends, so the servers do not idle until
# their own time limit.
release_servers() {
  if [[ "${CANCEL_SERVERS_ON_EXIT:-1}" != "1" ]]; then
    echo "Leaving server job ${SERVER_JOB_ID:-} running (CANCEL_SERVERS_ON_EXIT=0)"
    return
  fi
  if [[ -n "${SERVER_JOB_ID:-}" ]]; then
    echo "Cancelling server job ${SERVER_JOB_ID}"
    scancel "$SERVER_JOB_ID" 2> /dev/null || true
  fi
}
trap release_servers EXIT

ANNOTATE_ARGS=(--steps "$STEP_NAME")

# With a pool, wait for the servers to publish their URLs before starting.
if [[ -n "${POOL_DIR:-}" ]]; then
  echo "Pool: ${POOL_DIR} (waiting for ${NUM_SERVERS} server(s))"

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

  if (( ready == 0 )); then
    echo "No server registered in ${POOL_DIR}." \
      "See the vllm-${STEP_NAME}_*.err logs." >&2
    exit 1
  fi

  HOSTS_FILE="${POOL_DIR}/hosts.txt"
  cat "$POOL_DIR"/*.url > "$HOSTS_FILE"
  echo "Annotating over ${ready} server(s):"
  cat "$HOSTS_FILE"

  ANNOTATE_ARGS+=(--hosts-file "$HOSTS_FILE")
fi

# Everything else lives in the config; these three are the run-level overrides.
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  ANNOTATE_ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "${HUB_ID:-}" ]]; then
  ANNOTATE_ARGS+=(--hub-id "$HUB_ID")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  ANNOTATE_ARGS+=(--overwrite)
fi

set +e
llm-annotate "$ANNOTATE_CONFIG" "${ANNOTATE_ARGS[@]}"
ANNOTATE_RC=$?
set -e

echo "Finished step ${STEP_NAME} on $(date) with status ${ANNOTATE_RC}"
exit "$ANNOTATE_RC"
