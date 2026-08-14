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
# Set DRY_RUN=1 to print the submissions instead of running them.

set -euo pipefail

: "${ANNOTATE_CONFIG:?Set ANNOTATE_CONFIG to a JSON/YAML pipeline config}"
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

# A value left over in this shell from an earlier run must not leak into the
# jobs through --export=ALL; each job derives its own.
unset POOL_DIR LOG_DIR STEP_NAME SERVER_JOB_ID NUM_SERVERS MIN_SERVERS

if [[ ! -f "$ANNOTATE_CONFIG" ]]; then
  echo "ANNOTATE_CONFIG '${ANNOTATE_CONFIG}' does not exist" >&2
  exit 1
fi

# Read the plan on the login node, so a broken config fails here in a second
# rather than after an allocation has been granted. This only needs the base
# package, not the vllm extra, so the venv is used directly if it exists.
if [[ -x "${REPO_ROOT}/.venv/bin/llm-annotate" ]]; then
  ANNOTATE_CMD=("${REPO_ROOT}/.venv/bin/llm-annotate")
else
  ANNOTATE_CMD=(uv run --frozen llm-annotate)
fi

if ! STEPS_JSON=$("${ANNOTATE_CMD[@]}" "$ANNOTATE_CONFIG" --describe-steps); then
  echo "Could not read '${ANNOTATE_CONFIG}'; see the error above." >&2
  exit 1
fi

if [[ -z "$STEPS_JSON" ]]; then
  echo "'${ANNOTATE_CONFIG}' describes no steps" >&2
  exit 1
fi

# Echo the submitted job id, or print the command under DRY_RUN. --parsable
# appends ";<cluster>" in multi-cluster setups, hence the trim.
# Callers set DRYRUN_TAG to something unique before each call. `submit` runs in
# a command substitution, so a counter incremented inside it would be lost with
# the subshell, and every fake id would come back identical -- which would hide
# exactly the dependency chain a dry run is meant to show.
DRYRUN_TAG="job"
submit() {
  local out
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'sbatch --parsable %s\n' "$*" >&2
    echo "DRY-${DRYRUN_TAG}"
    return 0
  fi
  out=$(sbatch --parsable "$@")
  echo "${out%%;*}"
}

# Pull one field out of a step's JSON line. The keys are emitted by
# --describe-steps and are all flat scalars, so this stays a simple match.
# A JSON null comes back as the empty string, so `${MODEL:-...}` defaults work
# instead of the caller having to compare against the word "null".
field() {
  local value
  value=$(sed -n 's/.*"'"$2"'": *"\{0,1\}\([^,"}]*\)"\{0,1\}.*/\1/p' <<< "$1")
  [[ "$value" == "null" ]] && value=""
  printf '%s' "$value"
}

PREV_CLIENT=""
STEP_COUNT=0
ALL_JOBS=()

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
  if [[ "$KIND" == "vllm_pool" || "$KIND" == "vllm_offline" ]]; then
    if (( GPUS_PER_VLLM_SERVER < 1 || GPUS_PER_VLLM_SERVER > MAX_GPUS_PER_NODE )); then
      echo "  step '${NAME}' asks for ${GPUS_PER_VLLM_SERVER} GPUs per server;" \
        "must be between 1 and ${MAX_GPUS_PER_NODE}, because one server runs" \
        "inside a single job on a single node. Use a smaller model, or raise" \
        "MAX_GPUS_PER_NODE if the partition has more." >&2
      exit 1
    fi
  fi

  # Steps run in sequence: each one consumes the dataset the previous one
  # saved, so nothing may start until its predecessor has succeeded.
  DEP=()
  if [[ -n "$PREV_CLIENT" ]]; then
    DEP=(--dependency="afterok:${PREV_CLIENT}")
  fi

  CLIENT_EXPORT="ALL,REPO_ROOT=${REPO_ROOT},ANNOTATE_CONFIG=${ANNOTATE_CONFIG},STEP_NAME=${NAME}"
  # The annotation job is CPU-only for every kind but `vllm_offline`: a pooled
  # step only speaks HTTP to servers that hold the GPUs in their own jobs. Each
  # case below picks its partition, so the command line never carries two
  # contradictory --partition flags for a reader to have to resolve.
  CLIENT_FLAGS=(
    --account="$SLURM_ACCOUNT"
    --time="$CLIENT_TIME"
    --job-name="annotate-${NAME}"
  )

  case "$KIND" in
    vllm_pool)
      DRYRUN_TAG="servers${STEP_COUNT}"
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
      [[ "$DRY_RUN" == "1" ]] || mkdir -p "$POOL_DIR"
      echo "  servers: array ${SERVER_JOB}, ${SERVERS} x ${GPUS_PER_VLLM_SERVER} GPU(s) serving ${MODEL}"

      # `after`, not `afterok`: the client starts as soon as the array begins
      # running and waits for the servers to report in itself.
      DRYRUN_TAG="client${STEP_COUNT}"
      CLIENT_JOB=$(submit \
        "${CLIENT_FLAGS[@]}" \
        --partition="$CLIENT_PARTITION" \
        --dependency="after:${SERVER_JOB}" \
        --export="${CLIENT_EXPORT},POOL_DIR=${POOL_DIR},NUM_SERVERS=${SERVERS},MIN_SERVERS=${MIN_SERVERS:-${SERVERS}},SERVER_JOB_ID=${SERVER_JOB}" \
        slurm/vllm_annotate.sh)
      ;;

    vllm_offline)
      # The model is loaded in-process, so the annotation job is the GPU job.
      DRYRUN_TAG="client${STEP_COUNT}"
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
      DRYRUN_TAG="client${STEP_COUNT}"
      CLIENT_JOB=$(submit \
        "${CLIENT_FLAGS[@]}" \
        --partition="$CLIENT_PARTITION" \
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
