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
#   ./slurm/submit_pipeline.sh my-pipeline.yaml
#   ./slurm/submit_pipeline.sh --dry-run my-pipeline.yaml   # print, submit nothing
#
# Site settings (partitions, accounting, cores per GPU, modules) come from
# slurm/cluster.env -- see slurm/cluster.env.example. The model, pool size and
# serving flags of every step come from the pipeline config; a step states its
# GPU count once, in engine.tensor_parallel_size, which is both what vLLM shards
# over and what the server job asks Slurm for:
#
#   client:
#     engine:
#       tensor_parallel_size: 2
#     pool:
#       servers: 4

set -euo pipefail

usage() {
  cat << 'EOF'
Usage: slurm/submit_pipeline.sh [options] <config.yaml>

Options:
  --dry-run          Print the jobs that would be submitted, submit nothing.
  --steps a,b        Submit only these steps instead of the whole pipeline.
                     Everything before them must already have finished.
  --cluster-env FILE Site settings to use (default: slurm/cluster.env).
  -h, --help         Show this message.

Common environment overrides (all optional, see slurm/README.md):
  OUTPUT_DIR, HUB_ID, OVERWRITE=1   override the config for this run
  EXTRA_DEPENDENCY=afterok:123456   hang the chain off another job
  POOL_WAIT                         how long a client waits for its servers
  CANCEL_SERVERS_ON_EXIT=0          keep servers alive after their step ends
  SERVER_TIME, CLIENT_TIME, ...     one-off overrides of the cluster file
EOF
}

DRY_RUN=0
STEP_FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --steps)
      STEP_FILTER="${2:?--steps needs a comma-separated list of step names}"
      shift
      ;;
    --cluster-env)
      CLUSTER_ENV="${2:?--cluster-env needs a file}"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option '$1'" >&2
      usage >&2
      exit 1
      ;;
    *) ANNOTATE_CONFIG="$1" ;;
  esac
  shift
done

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

# shellcheck source=slurm/vllm_common.sh
source "${REPO_ROOT}/slurm/vllm_common.sh"
cluster_env_load
[[ -n "${CLUSTER_ENV:-}" ]] && CLUSTER_ENV=$(readlink -f "$CLUSTER_ENV" 2> /dev/null || echo "$CLUSTER_ENV")

: "${ANNOTATE_CONFIG:?Give a pipeline config as an argument, or set ANNOTATE_CONFIG}"
: "${EXTRA_DEPENDENCY:=}"

if [[ ! -f "$ANNOTATE_CONFIG" ]]; then
  echo "Config '${ANNOTATE_CONFIG}' does not exist" >&2
  exit 1
fi

# A value left over in this shell from an earlier run must not leak into the
# jobs through --export=ALL; each job derives its own.
unset POOL_DIR STEP_NAME SERVER_JOB_ID NUM_SERVERS MODEL

if [[ -x "${VENV_PATH}/bin/llm-annotate" ]]; then
  ANNOTATE_CMD=("${VENV_PATH}/bin/llm-annotate")
elif command -v llm-annotate > /dev/null 2>&1; then
  ANNOTATE_CMD=(llm-annotate)
else
  ANNOTATE_CMD=(uv run --frozen llm-annotate)
fi

# One JSON object per step: what it needs to run, derived from the config so the
# submitter cannot disagree with the run about it.
if ! STEPS_JSON=$("${ANNOTATE_CMD[@]}" "$ANNOTATE_CONFIG" --describe-steps); then
  echo "Could not read '${ANNOTATE_CONFIG}'; see the error above." >&2
  exit 1
fi

if [[ -z "$STEPS_JSON" ]]; then
  echo "'${ANNOTATE_CONFIG}' describes no steps" >&2
  exit 1
fi

# Submit and echo the job id. --dry-run prints the command instead and hands
# back a placeholder, so the chain still reads as one plan.
submit() {
  if (( DRY_RUN )); then
    local arg
    printf 'sbatch' >&2
    for arg in "$@"; do
      # Quote only what a shell would misread, so the line stays copy-pastable.
      if [[ "$arg" == *[[:space:]\'\"]* ]]; then
        printf ' %q' "$arg" >&2
      else
        printf ' %s' "$arg" >&2
      fi
    done
    printf '\n' >&2
    echo "<job-id>"
    return
  fi
  local out
  out=$(sbatch --parsable "$@")
  echo "${out%%;*}"
}

# Pull one value out of a flat JSON object. Naive, but --describe-steps emits
# exactly that: one line per step, scalars only.
field() {
  local value
  value=$(sed -n 's/.*[{,] *"'"$2"'": *"\{0,1\}\([^,"}]*\)"\{0,1\}.*/\1/p' <<< "$1")
  [[ "$value" == "null" ]] && value=""
  printf '%s' "$value"
}

wants_step() {
  [[ -z "$STEP_FILTER" ]] && return 0
  [[ ",${STEP_FILTER}," == *",$1,"* ]]
}

echo "Config:  ${ANNOTATE_CONFIG}"
echo "Cluster: ${CLUSTER_ENV}$([[ -f "$CLUSTER_ENV" ]] || echo ' (not found, using defaults)')"

PREV_CLIENT=""
STEP_COUNT=0
SUBMITTED=0
ALL_JOBS=()

while IFS= read -r step_json; do
  [[ -n "$step_json" ]] || continue
  STEP_COUNT=$(( STEP_COUNT + 1 ))

  NAME=$(field "$step_json" name)
  KIND=$(field "$step_json" kind)
  MODEL=$(field "$step_json" model)
  SERVERS=$(field "$step_json" servers)
  GPUS_PER_VLLM_SERVER=$(field "$step_json" gpus_per_vllm_server)
  GPUS_PER_VLLM_SERVER="${GPUS_PER_VLLM_SERVER:-1}"

  wants_step "$NAME" || continue
  SUBMITTED=$(( SUBMITTED + 1 ))

  echo
  echo "Step ${STEP_COUNT} '${NAME}' (${KIND})"

  # A server's GPUs all sit in one job on one node: vLLM's tensor parallelism
  # does not span nodes here, so a model too large for one node is out of scope.
  if [[ "$KIND" == "vllm_pool" || "$KIND" == "vllm_offline" ]]; then
    if (( GPUS_PER_VLLM_SERVER < 1 || GPUS_PER_VLLM_SERVER > MAX_GPUS_PER_NODE )); then
      echo "  step '${NAME}' asks for ${GPUS_PER_VLLM_SERVER} GPUs per server;" \
        "must be between 1 and ${MAX_GPUS_PER_NODE} (MAX_GPUS_PER_NODE in" \
        "'${CLUSTER_ENV}'), because one server runs inside a single job on a" \
        "single node." >&2
      exit 1
    fi
  fi

  # Steps run in sequence: each one consumes the dataset the previous one
  # saved, so nothing may start until its predecessor has succeeded.
  DEP=()
  DEP_SPEC=""
  if [[ -n "$PREV_CLIENT" ]]; then
    DEP_SPEC="afterok:${PREV_CLIENT}"
  fi
  # A caller-supplied dependency gates the chain, so only the first submitted
  # step has to carry it; the rest inherit it transitively. Comma-joined, which
  # Slurm reads as "all of these must be satisfied".
  if [[ -n "$EXTRA_DEPENDENCY" && "$SUBMITTED" -eq 1 ]]; then
    DEP_SPEC="${DEP_SPEC:+${DEP_SPEC},}${EXTRA_DEPENDENCY}"
  fi
  if [[ -n "$DEP_SPEC" ]]; then
    DEP=(--dependency="$DEP_SPEC")
  fi

  ACCOUNT_FLAGS=()
  [[ -n "$SLURM_ACCOUNT" ]] && ACCOUNT_FLAGS=(--account="$SLURM_ACCOUNT")

  CLIENT_FLAGS=(
    "${ACCOUNT_FLAGS[@]}"
    --time="$CLIENT_TIME"
    --job-name="annotate-${NAME}"
    --output="${LOG_DIR}/%x_%j.out"
    --error="${LOG_DIR}/%x_%j.err"
  )

  CPU_FLAGS=(--cpus-per-task="$CLIENT_CPUS")
  [[ -n "$CPU_PARTITION" ]] && CPU_FLAGS+=(--partition="$CPU_PARTITION")

  GPU_FLAGS=(
    "$(sbatch_gpu_flag "$GPUS_PER_VLLM_SERVER")"
    --cpus-per-task="$(( CPUS_PER_GPU * GPUS_PER_VLLM_SERVER ))"
  )
  [[ -n "$GPU_PARTITION" ]] && GPU_FLAGS+=(--partition="$GPU_PARTITION")

  STEP_EXPORT="ALL,REPO_ROOT=${REPO_ROOT},CLUSTER_ENV=${CLUSTER_ENV},ANNOTATE_CONFIG=${ANNOTATE_CONFIG},STEP_NAME=${NAME}"

  case "$KIND" in
    vllm_pool)
      # This script starts the servers, so it has to know what they serve, and
      # the config is the only place that can come from. `model` is optional for
      # provider `vllm_online` because a client can ask a running server what it
      # serves -- but nothing can ask a server that does not exist yet. Catch it
      # here rather than after the GPUs have been allocated.
      if [[ -z "$MODEL" ]]; then
        echo "  step '${NAME}' needs vLLM servers to be started for it, but" \
          "its config names no 'model'. Set 'client.model' on the step, or" \
          "point it at servers that already exist with 'base_urls'," \
          "'hosts_file' or 'url_glob'." >&2
        exit 1
      fi

      # An array job, so the whole pool is one id to cancel and one pool
      # directory for the client to watch.
      SERVER_JOB=$(submit \
        "${ACCOUNT_FLAGS[@]}" \
        --time="$SERVER_TIME" \
        --job-name="vllm-${NAME}" \
        --output="${LOG_DIR}/%x_%A_%a.out" \
        --error="${LOG_DIR}/%x_%A_%a.err" \
        --array="1-${SERVERS}" \
        "${GPU_FLAGS[@]}" \
        "${SERVER_SBATCH_ARGS[@]}" \
        "${DEP[@]}" \
        --export="$STEP_EXPORT" \
        slurm/vllm_server.sh)

      # Naming the pool after the array job id keeps concurrent steps and runs
      # apart; the server jobs derive the same name from SLURM_ARRAY_JOB_ID, so
      # nothing has to be told about it twice.
      POOL_DIR="${LOG_DIR}/pool_${SERVER_JOB}"
      (( DRY_RUN )) || mkdir -p "$POOL_DIR"
      echo "  servers: array ${SERVER_JOB}, ${SERVERS} x ${GPUS_PER_VLLM_SERVER} GPU(s) serving ${MODEL}"

      # No Slurm dependency on SERVER_JOB: for a job array, `after:<jobid>` is
      # only satisfied once every element has started, not the first one, so
      # gating the client on it can leave an already-ready server sitting
      # idle behind pool-mates that are still queued (e.g. behind a per-user
      # GPU quota) -- burning that server's own time limit before the client
      # ever gets to use it. The client is submitted on the same dependency
      # as the servers instead, and waits for them itself via POOL_WAIT.
      CLIENT_JOB=$(submit \
        "${CLIENT_FLAGS[@]}" \
        "${CPU_FLAGS[@]}" \
        "${CLIENT_SBATCH_ARGS[@]}" \
        "${DEP[@]}" \
        --export="${STEP_EXPORT},POOL_DIR=${POOL_DIR},NUM_SERVERS=${SERVERS},SERVER_JOB_ID=${SERVER_JOB}" \
        slurm/vllm_annotate.sh)
      ;;

    vllm_offline)
      # The model is loaded in-process, so the annotation job is the GPU job.
      CLIENT_JOB=$(submit \
        "${CLIENT_FLAGS[@]}" \
        "${GPU_FLAGS[@]}" \
        "${SERVER_SBATCH_ARGS[@]}" \
        "${DEP[@]}" \
        --export="${STEP_EXPORT}" \
        slurm/vllm_annotate.sh)
      echo "  in-process on ${GPUS_PER_VLLM_SERVER} GPU(s): ${MODEL}"
      ;;

    api | vllm_online)
      # A hosted provider, or servers that were started somewhere else.
      CLIENT_JOB=$(submit \
        "${CLIENT_FLAGS[@]}" \
        "${CPU_FLAGS[@]}" \
        "${CLIENT_SBATCH_ARGS[@]}" \
        "${DEP[@]}" \
        --export="${STEP_EXPORT}" \
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

if (( SUBMITTED == 0 )); then
  echo "No step of '${ANNOTATE_CONFIG}' matches --steps '${STEP_FILTER}'" >&2
  exit 1
fi

echo
if (( DRY_RUN )); then
  echo "Dry run: ${SUBMITTED} step(s) of ${ANNOTATE_CONFIG} would be submitted"
  exit 0
fi
echo "Submitted ${SUBMITTED} step(s) from ${ANNOTATE_CONFIG}"
echo "Logs:       ${LOG_DIR}/annotate-*_*.out, ${LOG_DIR}/vllm-*_*.out"
echo "Cancel all: scancel ${ALL_JOBS[*]}"
