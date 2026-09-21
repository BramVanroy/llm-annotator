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
  --max-resubmits N  Chain N further attempts per step (default 0). An attempt
                     runs the same command and starts only when the attempt
                     before it did not end well, so it resumes the step where
                     that one stopped. A step counts as done when any of its
                     attempts succeeds.
  --set KEY=VALUE    Override one config key for every step of this
                     submission; repeat for more than one. Passed straight to
                     `llm-annotate --set`, so a dotted key reaches a nested
                     value: --set dataset.max_num_samples=50000.
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
WANTED_STEPS=""
MAX_RESUBMITS=0
ANNOTATE_SET="${ANNOTATE_SET:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --steps)
      WANTED_STEPS="${2:?--steps needs a comma-separated list of step names}"
      shift
      ;;
    --max-resubmits)
      MAX_RESUBMITS="${2:?--max-resubmits needs a number}"
      if [[ ! "$MAX_RESUBMITS" =~ ^[0-9]+$ ]]; then
        echo "--max-resubmits needs a number of 0 or more, got" \
          "'${MAX_RESUBMITS}'" >&2
        exit 1
      fi
      shift
      ;;
    --set)
      SETTING="${2:?--set needs KEY=VALUE}"
      if [[ "$SETTING" != *=* ]]; then
        echo "--set needs KEY=VALUE, got '${SETTING}'" >&2
        exit 1
      fi
      # One per line, because a value may contain anything a config value may
      # contain, including the comma that separates --export entries. The jobs
      # read it out of the environment instead, which --export=ALL carries.
      ANNOTATE_SET="${ANNOTATE_SET:+${ANNOTATE_SET}$'\n'}${SETTING}"
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
# The command that queues a job. A site whose sbatch is wrapped points this at
# the wrapper; the tests point it at a stub, so they never reach a scheduler.
: "${SBATCH_CMD:=sbatch}"
# A clean-up job only runs `scancel`, so this is a ceiling on the queue wait
# rather than on any work.
CLEANUP_TIME="00:05:00"

if [[ ! -f "$ANNOTATE_CONFIG" ]]; then
  echo "Config '${ANNOTATE_CONFIG}' does not exist" >&2
  exit 1
fi

# A value left over in this shell from an earlier run must not leak into the
# jobs through --export=ALL; each job derives its own.
unset POOL_DIR STEP_NAME SERVER_JOB_ID NUM_SERVERS MIN_SERVERS MODEL

# A cluster file may define `llm-annotate` as a shell function, which is how a
# container site wraps it. That wins over a venv, which a stale checkout in the
# repo would otherwise provide.
if [[ "$(type -t llm-annotate)" == "function" ]]; then
  ANNOTATE_CMD=(llm-annotate)
elif [[ -x "${VENV_PATH}/bin/llm-annotate" ]]; then
  ANNOTATE_CMD=("${VENV_PATH}/bin/llm-annotate")
elif command -v llm-annotate > /dev/null 2>&1; then
  ANNOTATE_CMD=(llm-annotate)
else
  ANNOTATE_CMD=(uv run --frozen llm-annotate)
fi

# One line of shell-quoted STEP_* assignments per step: what it needs to run,
# derived from the config so the submitter cannot disagree with the run about
# it. Every key carries the STEP_ prefix, so evaluating a line cannot overwrite
# a variable of this script or of the cluster file.
if ! STEPS_ENV=$("${ANNOTATE_CMD[@]}" "$ANNOTATE_CONFIG" \
  --describe-steps --format env); then
  echo "Could not read '${ANNOTATE_CONFIG}'; see the error above." >&2
  exit 1
fi

if [[ -z "$STEPS_ENV" ]]; then
  echo "'${ANNOTATE_CONFIG}' describes no steps" >&2
  exit 1
fi
mapfile -t STEP_LINES <<< "$STEPS_ENV"

DRY_JOB_COUNTER=""
if (( DRY_RUN )); then
  DRY_JOB_COUNTER=$(mktemp)
  trap 'rm -f "$DRY_JOB_COUNTER"' EXIT
  echo 0 > "$DRY_JOB_COUNTER"
fi

# Submit and echo the job id. --dry-run prints the command instead and hands
# back a numbered placeholder, so a chain of jobs that wait on each other still
# reads as one plan. The counter lives in a file because every call happens
# inside a command substitution, which is a subshell of its own.
submit() {
  if (( DRY_RUN )); then
    local arg count
    printf '%s' "$SBATCH_CMD" >&2
    for arg in "$@"; do
      # Quote only what a shell would misread, so the line stays copy-pastable.
      if [[ "$arg" == *[[:space:]\'\"]* ]]; then
        printf ' %q' "$arg" >&2
      else
        printf ' %s' "$arg" >&2
      fi
    done
    printf '\n' >&2
    count=$(< "$DRY_JOB_COUNTER")
    count=$(( count + 1 ))
    echo "$count" > "$DRY_JOB_COUNTER"
    echo "<job-${count}>"
    return
  fi
  local out
  out=$("$SBATCH_CMD" --parsable "$@") || return 1
  echo "${out%%;*}"
}

# Every submit is checked with this, because `set -e` does not fire for a
# command substitution that runs a function: a refused sbatch would otherwise
# leave an empty job id behind and the steps after it would depend on a job
# that was never queued.
die() {
  echo "$@" >&2
  exit 1
}

wants_step() {
  [[ -z "$WANTED_STEPS" ]] && return 0
  [[ ",${WANTED_STEPS}," == *",$1,"* ]]
}

export ANNOTATE_SET

echo "Config:  ${ANNOTATE_CONFIG}"
echo "Cluster: ${CLUSTER_ENV}$([[ -f "$CLUSTER_ENV" ]] || echo ' (not found, using defaults)')"
if [[ -n "$ANNOTATE_SET" ]]; then
  # Printed, because it travels in the environment rather than in the sbatch
  # line a --dry-run shows.
  echo "Set:     $(tr '\n' ' ' <<< "$ANNOTATE_SET")"
fi

ACCOUNT_FLAGS=()
[[ -n "$SLURM_ACCOUNT" ]] && ACCOUNT_FLAGS=(--account="$SLURM_ACCOUNT")

CPU_PARTITION_FLAGS=()
[[ -n "$CPU_PARTITION" ]] && CPU_PARTITION_FLAGS=(--partition="$CPU_PARTITION")

# Every model this submission has to serve itself, fetched once before anything
# is allocated so that a pool on a cold cache does not download the same weights
# once per server. The download jobs gate the first step and therefore, through
# the chain, every step after it. Off unless the cluster file turns it on: a
# site whose compute nodes have no route to the Hub would otherwise fail the
# submission at the first job.
DOWNLOAD_DEP=""
if [[ "$MODEL_DOWNLOAD" == "1" ]]; then
  declare -A DOWNLOAD_SEEN=()
  for step_line in "${STEP_LINES[@]}"; do
    [[ -n "$step_line" ]] || continue
    eval "$step_line"
    wants_step "$STEP_NAME" || continue
    if [[ "$STEP_KIND" != "vllm_pool" && "$STEP_KIND" != "vllm_offline" ]]; then
      continue
    fi
    [[ -n "$STEP_MODEL" ]] || continue
    # A model that is a directory on this machine is already here.
    if [[ -d "$STEP_MODEL" ]]; then
      echo "Model:   ${STEP_MODEL} is a local directory, nothing to download"
      continue
    fi
    [[ -z "${DOWNLOAD_SEEN[$STEP_MODEL]:-}" ]] || continue
    DOWNLOAD_SEEN["$STEP_MODEL"]=1

    DOWNLOAD_PARTITION_FLAGS=()
    if [[ -n "$DOWNLOAD_PARTITION" ]]; then
      DOWNLOAD_PARTITION_FLAGS=(--partition="$DOWNLOAD_PARTITION")
    fi
    printf -v MODEL_QUOTED '%q' "$STEP_MODEL"
    DOWNLOAD_JOB=$(submit \
      "${ACCOUNT_FLAGS[@]}" \
      --time="$DOWNLOAD_TIME" \
      --job-name="download-${STEP_MODEL##*/}" \
      --output="${LOG_DIR}/%x_%j.out" \
      --error="${LOG_DIR}/%x_%j.err" \
      --cpus-per-task=2 \
      "${DOWNLOAD_PARTITION_FLAGS[@]}" \
      --export="ALL,REPO_ROOT=${REPO_ROOT},CLUSTER_ENV=${CLUSTER_ENV}" \
      --wrap="cd ${REPO_ROOT} && source slurm/vllm_common.sh && cluster_env_load && vllm_setup_env && vllm_download_model ${MODEL_QUOTED}") \
      || die "Could not queue the download of '${STEP_MODEL}'."
    echo "Model:   ${STEP_MODEL} downloaded first by job ${DOWNLOAD_JOB}"
    DOWNLOAD_DEP="${DOWNLOAD_DEP:+${DOWNLOAD_DEP},}afterok:${DOWNLOAD_JOB}"
  done
fi

# What the next step waits for: an or-joined list of "this attempt of the
# previous step succeeded".
PREV_DEP=""
SEEN_STEPS=0
SUBMITTED=0
ALL_JOBS=()

for step_line in "${STEP_LINES[@]}"; do
  [[ -n "$step_line" ]] || continue
  SEEN_STEPS=$(( SEEN_STEPS + 1 ))
  eval "$step_line"
  STEP_MIN_SERVERS="${STEP_MIN_SERVERS:-1}"
  STEP_GPUS_PER_VLLM_SERVER="${STEP_GPUS_PER_VLLM_SERVER:-1}"

  wants_step "$STEP_NAME" || continue
  SUBMITTED=$(( SUBMITTED + 1 ))

  echo
  echo "Step ${SEEN_STEPS} '${STEP_NAME}' (${STEP_KIND})"

  # A pool's real concurrency, printed before any GPU is allocated: this is
  # what a server's --max-num-seqs has to cover.
  if [[ "$STEP_KIND" == "vllm_pool" || "$STEP_KIND" == "vllm_online" ]]; then
    if [[ -n "$STEP_MAX_REQUESTS_PER_SERVER" ]]; then
      echo "  up to ${STEP_MAX_REQUESTS_PER_SERVER} requests per server," \
        "${STEP_MAX_REQUESTS_IN_FLIGHT} over the pool, queue of" \
        "${STEP_QUEUE_SIZE} batches"
    fi
  fi

  GPU_FLAGS=()
  ARRAY_SPEC=""
  # A server's GPUs all sit in one job on one node: vLLM's tensor parallelism
  # does not span nodes here, so a model too large for one node is out of scope.
  if [[ "$STEP_KIND" == "vllm_pool" || "$STEP_KIND" == "vllm_offline" ]]; then
    if (( STEP_GPUS_PER_VLLM_SERVER < 1 \
      || STEP_GPUS_PER_VLLM_SERVER > MAX_GPUS_PER_NODE )); then
      echo "  step '${STEP_NAME}' asks for ${STEP_GPUS_PER_VLLM_SERVER} GPUs" \
        "per server; must be between 1 and ${MAX_GPUS_PER_NODE}" \
        "(MAX_GPUS_PER_NODE in '${CLUSTER_ENV}'), because one server runs" \
        "inside a single job on a single node." >&2
      exit 1
    fi
    # Assigned on its own line: a command substitution inside an array literal
    # keeps its failure to itself, so a rejected GPU_REQUEST would reach sbatch
    # as an empty argument.
    GPU_FLAG=$(sbatch_gpu_flag "$STEP_GPUS_PER_VLLM_SERVER") || exit 1
    GPU_FLAGS=(
      "$GPU_FLAG"
      --cpus-per-task="$(( CPUS_PER_GPU * STEP_GPUS_PER_VLLM_SERVER ))"
    )
    [[ -n "$GPU_PARTITION" ]] && GPU_FLAGS+=(--partition="$GPU_PARTITION")
  fi

  if [[ "$STEP_KIND" == "vllm_pool" ]]; then
    # This script starts the servers, so it has to know what they serve, and
    # the config is the only place that can come from. `model` is optional for
    # provider `vllm_online` because a client can ask a running server what it
    # serves -- but nothing can ask a server that does not exist yet. Catch it
    # here rather than after the GPUs have been allocated.
    if [[ -z "$STEP_MODEL" ]]; then
      echo "  step '${STEP_NAME}' needs vLLM servers to be started for it," \
        "but its config names no 'model'. Set 'client.model' on the step, or" \
        "point it at servers that already exist with 'base_urls'," \
        "'hosts_file' or 'url_glob'." >&2
      exit 1
    fi

    ARRAY_SPEC="1-${STEP_SERVERS}"
    if [[ -n "$MAX_CONCURRENT_SERVERS" ]]; then
      if [[ ! "$MAX_CONCURRENT_SERVERS" =~ ^[1-9][0-9]*$ ]]; then
        echo "  MAX_CONCURRENT_SERVERS in '${CLUSTER_ENV}' must be 1 or more," \
          "got '${MAX_CONCURRENT_SERVERS}'." >&2
        exit 1
      fi
      # The client waits for min_servers of the array to publish a URL, so a
      # throttle below that threshold holds the step for its whole POOL_WAIT
      # and then annotates on fewer servers than it asked for.
      if (( MAX_CONCURRENT_SERVERS < STEP_MIN_SERVERS )); then
        echo "  step '${STEP_NAME}' starts at ${STEP_MIN_SERVERS} ready" \
          "server(s), but MAX_CONCURRENT_SERVERS=${MAX_CONCURRENT_SERVERS} in" \
          "'${CLUSTER_ENV}' lets at most ${MAX_CONCURRENT_SERVERS} of its" \
          "array run at a time, so that threshold is never reached. Raise" \
          "MAX_CONCURRENT_SERVERS, or lower the step's pool.min_servers." >&2
        exit 1
      fi
      ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT_SERVERS}"
    fi
  fi

  CLIENT_FLAGS=(
    "${ACCOUNT_FLAGS[@]}"
    --time="$CLIENT_TIME"
    --job-name="annotate-${STEP_NAME}"
    --output="${LOG_DIR}/%x_%j.out"
    --error="${LOG_DIR}/%x_%j.err"
  )

  CPU_FLAGS=(--cpus-per-task="$CLIENT_CPUS" "${CPU_PARTITION_FLAGS[@]}")

  EXPORT_VARS="ALL,REPO_ROOT=${REPO_ROOT},CLUSTER_ENV=${CLUSTER_ENV},ANNOTATE_CONFIG=${ANNOTATE_CONFIG},STEP_NAME=${STEP_NAME}"

  # Steps run in sequence: each one consumes the dataset the previous one
  # saved, so nothing may start until its predecessor has succeeded. A
  # caller-supplied dependency and the model downloads gate the chain, so only
  # the first submitted step carries them; the rest inherit them transitively.
  # Comma-joined, which Slurm reads as "all of these must be satisfied".
  FIRST_DEP="$PREV_DEP"
  if (( SUBMITTED == 1 )); then
    FIRST_DEP="$DOWNLOAD_DEP"
    if [[ -n "$EXTRA_DEPENDENCY" ]]; then
      FIRST_DEP="${FIRST_DEP:+${FIRST_DEP},}${EXTRA_DEPENDENCY}"
    fi
  fi

  STEP_CLIENT_JOBS=()
  for (( attempt = 0; attempt <= MAX_RESUBMITS; attempt += 1 )); do
    ATTEMPT_FLAGS=()
    if (( attempt == 0 )); then
      ATTEMPT_DEP="$FIRST_DEP"
    else
      # An attempt whose predecessor succeeded can never run. Slurm removes it
      # then instead of leaving it queued forever, which unwinds the rest of
      # this step's chain with it.
      ATTEMPT_DEP="afternotok:${STEP_CLIENT_JOBS[attempt - 1]}"
      ATTEMPT_FLAGS=(--kill-on-invalid-dep=yes)
      echo "  attempt $(( attempt + 1 )) of $(( MAX_RESUBMITS + 1 )):"
    fi
    DEP_FLAGS=()
    [[ -n "$ATTEMPT_DEP" ]] && DEP_FLAGS=(--dependency="$ATTEMPT_DEP")

    case "$STEP_KIND" in
      vllm_pool)
        # An array job, so the whole pool is one id to cancel and one pool
        # directory for the client to watch.
        SERVER_JOB=$(submit \
          "${ACCOUNT_FLAGS[@]}" \
          --time="$SERVER_TIME" \
          --job-name="vllm-${STEP_NAME}" \
          --output="${LOG_DIR}/%x_%A_%a.out" \
          --error="${LOG_DIR}/%x_%A_%a.err" \
          --array="$ARRAY_SPEC" \
          "${GPU_FLAGS[@]}" \
          "${SERVER_SBATCH_ARGS[@]}" \
          "${DEP_FLAGS[@]}" \
          "${ATTEMPT_FLAGS[@]}" \
          --export="$EXPORT_VARS" \
          slurm/vllm_server.sh) \
          || die "  could not queue the servers of step '${STEP_NAME}'."

        # Naming the pool after the array job id keeps concurrent steps, runs
        # and attempts apart; the server jobs derive the same name from
        # SLURM_ARRAY_JOB_ID, so nothing has to be told about it twice.
        POOL_DIR="${LOG_DIR}/pool_${SERVER_JOB}"
        (( DRY_RUN )) || mkdir -p "$POOL_DIR"
        echo "  servers: array ${SERVER_JOB}, ${STEP_SERVERS} x" \
          "${STEP_GPUS_PER_VLLM_SERVER} GPU(s) serving ${STEP_MODEL}"
        if (( STEP_MIN_SERVERS < STEP_SERVERS )); then
          echo "  client starts at ${STEP_MIN_SERVERS} ready server(s);" \
            "the rest join the run as they arrive"
        fi

        # One `after:` per array element, or-joined, so the client is released
        # as soon as the *first* server has begun. `after:<array-id>` as a whole
        # is only satisfied once every element has started, which leaves a ready
        # server idle behind pool-mates that are still queued (a per-user GPU
        # quota is enough to do that), burning that server's own SERVER_TIME.
        # Waiting in the queue rather than on the compute node also means
        # CLIENT_TIME starts counting when there is something to annotate
        # against: a CPU partition schedules in minutes and a GPU partition can
        # take days. What is left after the dependency is the model load and the
        # rest of the pool arriving, which is the client's own POOL_WAIT.
        #
        # This replaces the step's own dependency instead of adding to it, since
        # Slurm reads one separator per expression (`,` for and, `?` for or) and
        # the two cannot be mixed. Nothing is lost: a server cannot start before
        # the previous step has succeeded, so the client inherits that through
        # the array it waits on. A cancelled job also satisfies `after:`, so an
        # array that Slurm kills off releases the client instead of stranding
        # it, and the client recognises that case rather than sitting out its
        # POOL_WAIT; --kill-on-invalid-dep covers what Slurm flags as
        # unsatisfiable outright.
        CLIENT_DEP=""
        for (( element = 1; element <= STEP_SERVERS; element += 1 )); do
          CLIENT_DEP="${CLIENT_DEP:+${CLIENT_DEP}?}after:${SERVER_JOB}_${element}"
        done

        CLIENT_JOB=$(submit \
          "${CLIENT_FLAGS[@]}" \
          "${CPU_FLAGS[@]}" \
          "${CLIENT_SBATCH_ARGS[@]}" \
          --dependency="$CLIENT_DEP" \
          --kill-on-invalid-dep=yes \
          --export="${EXPORT_VARS},POOL_DIR=${POOL_DIR},NUM_SERVERS=${STEP_SERVERS},MIN_SERVERS=${STEP_MIN_SERVERS},SERVER_JOB_ID=${SERVER_JOB}" \
          slurm/vllm_annotate.sh) \
          || die "  could not queue the client of step '${STEP_NAME}'."
        ALL_JOBS+=("$SERVER_JOB")
        ;;

      vllm_offline)
        # The model is loaded in-process, so the annotation job is the GPU job.
        CLIENT_JOB=$(submit \
          "${CLIENT_FLAGS[@]}" \
          "${GPU_FLAGS[@]}" \
          "${SERVER_SBATCH_ARGS[@]}" \
          "${DEP_FLAGS[@]}" \
          "${ATTEMPT_FLAGS[@]}" \
          --export="${EXPORT_VARS}" \
          slurm/vllm_annotate.sh) \
          || die "  could not queue step '${STEP_NAME}'."
        echo "  in-process on ${STEP_GPUS_PER_VLLM_SERVER} GPU(s):" \
          "${STEP_MODEL}"
        ;;

      api | vllm_online)
        # A hosted provider, or servers that were started somewhere else.
        CLIENT_JOB=$(submit \
          "${CLIENT_FLAGS[@]}" \
          "${CPU_FLAGS[@]}" \
          "${CLIENT_SBATCH_ARGS[@]}" \
          "${DEP_FLAGS[@]}" \
          "${ATTEMPT_FLAGS[@]}" \
          --export="${EXPORT_VARS}" \
          slurm/vllm_annotate.sh) \
          || die "  could not queue step '${STEP_NAME}'."
        echo "  CPU only: ${STEP_MODEL:-served-default}"
        ;;

      *)
        echo "  unknown step kind '${STEP_KIND}'" >&2
        exit 1
        ;;
    esac

    echo "  client:  ${CLIENT_JOB}"
    ALL_JOBS+=("$CLIENT_JOB")
    STEP_CLIENT_JOBS+=("$CLIENT_JOB")

    # The client cancels its own servers when it ends, which is faster, but a
    # cgroup OOM kill, `scancel -s KILL` or a dead node skips that trap and
    # leaves the GPUs allocated until SERVER_TIME. This job cancels the same
    # array whichever way the client ended, and only that array, so the servers
    # of a later attempt are untouched. It is removed with the rest of the
    # chain when the client it waits for is.
    if [[ "$STEP_KIND" == "vllm_pool" \
      && "${CANCEL_SERVERS_ON_EXIT:-1}" == "1" ]]; then
      CLEANUP_JOB=$(submit \
        "${ACCOUNT_FLAGS[@]}" \
        --time="$CLEANUP_TIME" \
        --job-name="cancel-${STEP_NAME}" \
        --output="${LOG_DIR}/%x_%j.out" \
        --error="${LOG_DIR}/%x_%j.err" \
        --cpus-per-task=1 \
        "${CPU_PARTITION_FLAGS[@]}" \
        "${CLIENT_SBATCH_ARGS[@]}" \
        --dependency="afterany:${CLIENT_JOB}" \
        --kill-on-invalid-dep=yes \
        --wrap="scancel ${SERVER_JOB}") \
        || die "  could not queue the clean-up of step '${STEP_NAME}'."
      echo "  cleanup: ${CLEANUP_JOB} cancels ${SERVER_JOB} whenever" \
        "${CLIENT_JOB} ends"
      ALL_JOBS+=("$CLEANUP_JOB")
    fi
  done

  # The next step needs any one attempt of this one to have succeeded, which
  # Slurm expresses with `?` as the separator of the dependency list. With no
  # resubmits this is the single `afterok:` it has always been.
  PREV_DEP=""
  for job in "${STEP_CLIENT_JOBS[@]}"; do
    PREV_DEP="${PREV_DEP:+${PREV_DEP}?}afterok:${job}"
  done
done

if (( SUBMITTED == 0 )); then
  echo "No step of '${ANNOTATE_CONFIG}' matches --steps '${WANTED_STEPS}'" >&2
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
