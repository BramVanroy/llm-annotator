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
#   POOL_DIR set    a companion server array is starting up; wait for
#                   MIN_SERVERS of it to publish their URLs, then annotate over
#                   the pool, which keeps growing as the rest arrive. This job
#                   was held in the queue until one of those servers began, so
#                   what is left to wait for is a model load, not a GPU
#                   allocation
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
# How many of them have to be up before annotating starts. submit_pipeline.sh
# passes the step's own `pool.min_servers`; a manual submission that only says
# how large the pool is waits for all of it.
: "${MIN_SERVERS:=${NUM_SERVERS}}"
# Matches READY_TIMEOUT, so a client cannot give up on a server before the
# server gives up on itself. Submitted through submit_pipeline.sh this job only
# starts once a server of its own pool has, so the wait is a model load rather
# than an allocation; raise it for a manual run against a pool that is still
# queued.
: "${POOL_WAIT:=1800}"

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
  echo "Pool: ${POOL_DIR} (starting at ${MIN_SERVERS} of ${NUM_SERVERS} server(s))"

  count_urls() {
    local files=("$POOL_DIR"/*.url)
    [[ -e "${files[0]}" ]] && echo "${#files[@]}" || echo 0
  }

  # Slurm counts a cancelled job as having satisfied an `after:` dependency, so
  # a server array that was killed off (its own dependency failed, or someone
  # scancelled it) releases this client rather than holding it back. Sitting
  # out the whole POOL_WAIT for servers that are not coming only delays the
  # error, so stop as soon as the array has no element left in the queue. A
  # squeue that fails to answer says nothing about the array, so that counts
  # as alive.
  pool_alive() {
    [[ -n "${SERVER_JOB_ID:-}" ]] || return 0
    local elements
    elements=$(squeue -j "$SERVER_JOB_ID" -h -o '%T' 2> /dev/null) || return 0
    [[ -n "$elements" ]]
  }

  deadline=$(( SECONDS + POOL_WAIT ))
  ready=$(count_urls)
  while (( ready < MIN_SERVERS )); do
    if (( SECONDS > deadline )); then
      echo "Waited ${POOL_WAIT}s for ${MIN_SERVERS} server(s), ${ready} showed up."
      break
    fi
    if ! pool_alive; then
      echo "Server job ${SERVER_JOB_ID} has no element left in the queue," \
        "so ${ready} server(s) is all this step is going to get."
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

  echo "Annotating over ${ready} of ${NUM_SERVERS} server(s):"
  cat "$POOL_DIR"/*.url

  # The glob rather than a snapshot of it: the client re-reads the pool
  # directory while it runs, so the servers still queued join this step as
  # soon as they publish their URL.
  ANNOTATE_ARGS+=(--url-glob "${POOL_DIR}/*.url")
fi

# Everything else lives in the config; these are the run-level overrides.
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  ANNOTATE_ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "${HUB_ID:-}" ]]; then
  ANNOTATE_ARGS+=(--hub-id "$HUB_ID")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  ANNOTATE_ARGS+=(--overwrite)
fi
# Config overrides for this submission, one KEY=VALUE per line, as
# submit_pipeline.sh --set left them. Every step job of one submission sees the
# same ones, so a pipeline resubmitted with a larger dataset.max_num_samples
# grows as a whole rather than step by step.
if [[ -n "${ANNOTATE_SET:-}" ]]; then
  while IFS= read -r setting; do
    [[ -n "$setting" ]] && ANNOTATE_ARGS+=(--set "$setting")
  done <<< "$ANNOTATE_SET"
fi

set +e
llm-annotate "$ANNOTATE_CONFIG" "${ANNOTATE_ARGS[@]}"
ANNOTATE_RC=$?
set -e

echo "Finished step ${STEP_NAME} on $(date) with status ${ANNOTATE_RC}"
exit "$ANNOTATE_RC"
