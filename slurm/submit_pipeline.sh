#!/bin/bash
# Submit a pipeline config to SLURM, one job chain per step.
#
# Run this from a login node; it is not an sbatch script itself. It asks the
# config what each step needs (`llm-annotate --describe-steps`) and submits the
# right shape of job for it, chaining them so a step starts only once the
# previous one has succeeded:
#
#   vllm_pool       an array of GPU server jobs + a CPU client job
#   vllm_offline    one GPU job that loads the model in-process
#   api             one CPU job; a hosted provider needs no accelerator
#   vllm_online     one CPU job; the servers already exist
#
# Because each step is submitted separately, steps may use different models and
# different providers, and GPUs are only held while the step that needs them is
# running.
#
# Usage:
#   ANNOTATE_CONFIG=examples/vllm-server-pool/pipeline.yaml \
#     ./slurm/submit_pipeline.sh
#
# The model and pool size of every step come from the config; a step sizes its
# own pool with:
#
#   client:
#     pool:
#       servers: 4
#       gpus_per_vllm_server: 2
#

set -euo pipefail

: "${ANNOTATE_CONFIG:?Set ANNOTATE_CONFIG to a JSON/YAML pipeline config}"
: "${GPU_PARTITION:=gpu_a100}"
: "${CLIENT_PARTITION:=rome}"
: "${SERVER_TIME:=04:00:00}"
: "${CLIENT_TIME:=05:00:00}"
: "${SLURM_ACCOUNT:=tnsr72764}"
: "${MAX_GPUS_PER_NODE:=4}"

# CPU share per GPU depends on the partition's node shape, not something we
# want a second env var to be able to contradict.
case "$GPU_PARTITION" in
  gpu_a100) CPUS_PER_GPU=18 ;;
  gpu_h100) CPUS_PER_GPU=16 ;;
  *) CPUS_PER_GPU=16 ;;
esac

# The client is CPU-only and asks for a whole node's minimal core count, the
# smallest request Snellius accepts on each of these partitions.
case "$CLIENT_PARTITION" in
  rome) CLIENT_CPUS=16 ;;
  genoa) CLIENT_CPUS=24 ;;
  *)
    echo "CLIENT_PARTITION must be 'rome' or 'genoa', got '${CLIENT_PARTITION}'" >&2
    exit 1
    ;;
esac

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# MIN_SERVERS: how many VLLM servers should be launched before
# the client starts submitting jobs to the pool
# Defaults to 1, so the client starts as soon as a single server is online,
# but can be overridden
MIN_SERVERS_REQUESTED="${MIN_SERVERS:-}"

# A value left over in this shell from an earlier run must not leak into the
# jobs through --export=ALL; each job derives its own.
unset POOL_DIR LOG_DIR STEP_NAME SERVER_JOB_ID NUM_SERVERS MIN_SERVERS

if [[ ! -f "$ANNOTATE_CONFIG" ]]; then
  echo "ANNOTATE_CONFIG '${ANNOTATE_CONFIG}' does not exist" >&2
  exit 1
fi

# Get the cmd; either the installed script or the uv runner
if [[ -x "${REPO_ROOT}/.venv/bin/llm-annotate" ]]; then
  ANNOTATE_CMD=("${REPO_ROOT}/.venv/bin/llm-annotate")
else
  ANNOTATE_CMD=(uv run --frozen llm-annotate)
fi

# Use "describe-steps" to get a json array of the steps described in the config
# (kind of verbose to get the steps into our slurm script so we can spawn jobs)
if ! STEPS_JSON=$("${ANNOTATE_CMD[@]}" "$ANNOTATE_CONFIG" --describe-steps); then
  echo "Could not read '${ANNOTATE_CONFIG}'; see the error above." >&2
  exit 1
fi

if [[ -z "$STEPS_JSON" ]]; then
  echo "'${ANNOTATE_CONFIG}' describes no steps" >&2
  exit 1
fi

# Echo the submitted job id
submit() {
  local out
  out=$(sbatch --parsable "$@")
  echo "${out%%;*}"
}

# For a given key, get the value from a string json object
# kinda naive but works for the simple json we get from llm-annotate --describe-steps
field() {
  local value
  value=$(sed -n 's/.*"'"$2"'": *"\{0,1\}\([^,"}]*\)"\{0,1\}.*/\1/p' <<< "$1")
  [[ "$value" == "null" ]] && value=""
  printf '%s' "$value"
}

PREV_CLIENT=""
STEP_COUNT=0
ALL_JOBS=()

# loop over $STEPS_JSON, one line per step, and submit the right shape of job(s) for each
while IFS= read -r step_json; do
  [[ -n "$step_json" ]] || continue
  STEP_COUNT=$(( STEP_COUNT + 1 ))

  NAME=$(field "$step_json" name)
  KIND=$(field "$step_json" kind)
  MODEL=$(field "$step_json" model)
  SERVERS=$(field "$step_json" servers)
  GPUS_PER_VLLM_SERVER=$(field "$step_json" gpus_per_vllm_server)

  echo
  echo "Step ${STEP_COUNT} '${NAME}' (${KIND})"

  # A server's GPUs must all sit in one job on one node: vLLM's tensor
  # parallelism does not span nodes here, and Snellius GPU nodes have four.
  # As such we do not support sharding a model across multiple nodes
  if [[ "$KIND" == "vllm_pool" || "$KIND" == "vllm_offline" ]]; then
    if (( GPUS_PER_VLLM_SERVER < 1 || GPUS_PER_VLLM_SERVER > MAX_GPUS_PER_NODE )); then
      echo "  step '${NAME}' asks for ${GPUS_PER_VLLM_SERVER} GPUs per server;" \
        "must be between 1 and ${MAX_GPUS_PER_NODE}, because one server runs" \
        "inside a single job on a single node." >&2
      exit 1
    fi
  fi

  # Steps run in sequence: each one consumes the dataset the previous one
  # saved, so nothing may start until its predecessor has succeeded.
  DEP=()
  if [[ -n "$PREV_CLIENT" ]]; then
    DEP=(--dependency="afterok:${PREV_CLIENT}")
  fi

  # CPU jobs (API providers or VLLM online/pool) need a client
  # to submit requests to the server
  CLIENT_EXPORT="ALL,REPO_ROOT=${REPO_ROOT},ANNOTATE_CONFIG=${ANNOTATE_CONFIG},STEP_NAME=${NAME}"
  CLIENT_FLAGS=(
    --account="$SLURM_ACCOUNT"
    --time="$CLIENT_TIME"
    --job-name="annotate-${NAME}"
  )

  case "$KIND" in
    vllm_pool)
      # This script starts the servers, so it has to know what they serve, and
      # the config is the only place that can come from. `model` is optional
      # for provider `vllm_online` because a client can ask a running server
      # what it serves -- but nothing can ask a server that does not exist yet.
      # Catch it here rather than after four GPU allocations have been granted.
      if [[ -z "$MODEL" ]]; then
        echo "  step '${NAME}' needs vLLM servers to be started for it, but" \
          "its config names no 'model'. Set 'client.model' on the step, or" \
          "point it at servers that already exist with 'base_urls'," \
          "'hosts_file' or 'url_glob'." >&2
        exit 1
      fi

      # submit pool as an array job so we can get a single slurm ID that
      # the client depends on
      # each one running slurm/vllm_server.sh
      SERVER_JOB=$(submit \
        --partition="$GPU_PARTITION" \
        --account="$SLURM_ACCOUNT" \
        --time="$SERVER_TIME" \
        --job-name="vllm-${NAME}" \
        --array="1-${SERVERS}" \
        --gres="gpu:${GPUS_PER_VLLM_SERVER}" \
        --cpus-per-task="$(( CPUS_PER_GPU * GPUS_PER_VLLM_SERVER ))" \
        "${DEP[@]}" \
        --export="ALL,REPO_ROOT=${REPO_ROOT},MODEL=${MODEL},GPUS_PER_VLLM_SERVER=${GPUS_PER_VLLM_SERVER}" \
        slurm/vllm_server.sh)

      # Naming the pool after the array job id keeps concurrent steps and runs
      # apart; the server jobs derive the same name from SLURM_ARRAY_JOB_ID, so
      # nothing has to be told about it twice.
      POOL_DIR="${REPO_ROOT}/logs/pool_${SERVER_JOB}"
      mkdir -p "$POOL_DIR"
      echo "  servers: array ${SERVER_JOB}, ${SERVERS} x ${GPUS_PER_VLLM_SERVER} GPU(s) serving ${MODEL}"

      # `after`, not `afterok`: the client starts as soon as the array begins
      # running and waits for the servers to report in itself.
      CLIENT_JOB=$(submit \
        "${CLIENT_FLAGS[@]}" \
        --partition="$CLIENT_PARTITION" \
        --cpus-per-task="$CLIENT_CPUS" \
        --dependency="after:${SERVER_JOB}" \
        --export="${CLIENT_EXPORT},POOL_DIR=${POOL_DIR},NUM_SERVERS=${SERVERS},MIN_SERVERS=${MIN_SERVERS_REQUESTED:-1},SERVER_JOB_ID=${SERVER_JOB}" \
        slurm/vllm_annotate.sh)
      ;;

    vllm_offline)
      # The model is loaded in-process, so the annotation job is the GPU job.
      CLIENT_JOB=$(submit \
        "${CLIENT_FLAGS[@]}" \
        --partition="$GPU_PARTITION" \
        --gres="gpu:${GPUS_PER_VLLM_SERVER}" \
        --cpus-per-task="$(( CPUS_PER_GPU * GPUS_PER_VLLM_SERVER ))" \
        "${DEP[@]}" \
        --export="${CLIENT_EXPORT}" \
        slurm/vllm_annotate.sh)
      echo "  in-process on ${GPUS_PER_VLLM_SERVER} GPU(s): ${MODEL}"
      ;;

    api | vllm_online)
      # API-based job, or a simple VLLM Client one that has launched
      # the VLLM server elsewhere
      CLIENT_JOB=$(submit \
        "${CLIENT_FLAGS[@]}" \
        --partition="$CLIENT_PARTITION" \
        --cpus-per-task="$CLIENT_CPUS" \
        "${DEP[@]}" \
        --export="${CLIENT_EXPORT}" \
        slurm/vllm_annotate.sh)
      echo "  CPU only: ${MODEL:-served-default}"
      ;;

    *)
      echo "  unknown step kind '${KIND}'" >&2
      exit 1
      ;;
  esac

  echo "  client:  ${CLIENT_JOB}"
  ALL_JOBS+=("$CLIENT_JOB")
  if [[ "$KIND" == "vllm_pool" ]]; then
    ALL_JOBS+=("$SERVER_JOB")
  fi
  PREV_CLIENT="$CLIENT_JOB"
done <<< "$STEPS_JSON"

echo
echo "Submitted ${STEP_COUNT} step(s) from ${ANNOTATE_CONFIG}"
echo "Logs:       logs/annotate-*_*.out, logs/vllm-*_*.out"
echo "Cancel all: scancel ${ALL_JOBS[*]}"
