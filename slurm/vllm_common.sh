#!/bin/bash
# Shared helpers for the SLURM scripts. Sourced by submit_pipeline.sh on the
# login node and by both job scripts on the compute nodes.
#
# The three layers are kept apart on purpose:
#
#   pipeline config   what is annotated: models, prompts, pool sizes, serving
#                     flags. Read through the llm-annotate CLI, never parsed
#                     here.
#   cluster file      what your cluster calls things: partitions, accounting,
#                     cores per GPU, modules, how a Python environment is
#                     prepared. slurm/cluster.env by default, CLUSTER_ENV
#                     elsewhere.
#   these scripts     the job shapes, which are the same everywhere.
#
# Provides:
#   cluster_env_load             source the cluster file, then apply defaults
#   sbatch_gpu_flag <n>          the flag this cluster uses to ask for n GPUs
#   vllm_env_defaults            pool/port defaults for a job
#   vllm_setup_env               modules + Python environment inside a job
#   vllm_ensure_nvcc             make nvcc available for JIT-compiled kernels
#   vllm_serve_args <cfg> <step> the step's `vllm serve` args, one per line
#   vllm_pick_port <start>       first free TCP port at or above <start>
#   vllm_wait_until_ready <urls> block until every URL answers /health

# Scalars a cluster file may set. Anything already in the environment wins over
# the file, so `SERVER_TIME=08:00:00 ./slurm/submit_pipeline.sh cfg.yaml` is a
# one-off override and reaches the jobs through --export=ALL.
CLUSTER_ENV_KEYS=(
  SLURM_ACCOUNT
  GPU_PARTITION CPU_PARTITION
  SERVER_TIME CLIENT_TIME
  CPUS_PER_GPU CLIENT_CPUS MAX_GPUS_PER_NODE
  GPU_REQUEST GPU_TYPE
  CLUSTER_MODULES CUDA_MODULE
  VENV_PATH UV_SYNC
  LOG_DIR
)

# Read the cluster file and fill in every value the scripts rely on. Safe to
# call twice; the second call changes nothing.
cluster_env_load() {
  : "${REPO_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  : "${CLUSTER_ENV:=${REPO_ROOT}/slurm/cluster.env}"

  if [[ -f "$CLUSTER_ENV" ]]; then
    local name preset=()
    for name in "${CLUSTER_ENV_KEYS[@]}"; do
      if [[ -n "${!name+set}" ]]; then
        printf -v "__cluster_preset_${name}" '%s' "${!name}"
        preset+=("$name")
      fi
    done

    # shellcheck disable=SC1090
    source "$CLUSTER_ENV"

    for name in "${preset[@]}"; do
      local saved="__cluster_preset_${name}"
      printf -v "$name" '%s' "${!saved}"
      unset "$saved"
    done
  elif [[ -n "${CLUSTER_ENV_REQUIRED:-}" ]]; then
    echo "No cluster file at '${CLUSTER_ENV}'. Copy slurm/cluster.env.example" \
      "to slurm/cluster.env and fill it in, or point CLUSTER_ENV at your own." >&2
    return 1
  fi

  : "${SLURM_ACCOUNT:=}"
  : "${GPU_PARTITION:=}"
  : "${CPU_PARTITION:=}"
  : "${SERVER_TIME:=04:00:00}"
  : "${CLIENT_TIME:=05:00:00}"
  : "${CPUS_PER_GPU:=8}"
  : "${CLIENT_CPUS:=8}"
  : "${MAX_GPUS_PER_NODE:=8}"
  : "${GPU_REQUEST:=gres}"
  : "${GPU_TYPE:=}"
  : "${CLUSTER_MODULES:=}"
  : "${CUDA_MODULE:=}"
  : "${VENV_PATH:=${REPO_ROOT}/.venv}"
  : "${UV_SYNC:=0}"
  : "${LOG_DIR:=${REPO_ROOT}/logs}"

  # Arrays cannot travel through the environment, so these come from the
  # cluster file alone. Declare them if it did not.
  declare -p SERVER_SBATCH_ARGS > /dev/null 2>&1 || SERVER_SBATCH_ARGS=()
  declare -p CLIENT_SBATCH_ARGS > /dev/null 2>&1 || CLIENT_SBATCH_ARGS=()

  mkdir -p "$LOG_DIR"
}

# How this cluster's Slurm is asked for $1 GPUs on one node. Clusters differ in
# which of the two spellings they accept, and typed gres ("gpu:a100:2") is
# common where there is more than one kind of accelerator.
sbatch_gpu_flag() {
  local count="$1"
  case "$GPU_REQUEST" in
    gres) printf -- '--gres=gpu:%s%s' "${GPU_TYPE:+${GPU_TYPE}:}" "$count" ;;
    gpus) printf -- '--gpus-per-node=%s%s' "${GPU_TYPE:+${GPU_TYPE}:}" "$count" ;;
    *)
      echo "GPU_REQUEST must be 'gres' or 'gpus', got '${GPU_REQUEST}'" >&2
      return 1
      ;;
  esac
}

vllm_env_defaults() {
  cluster_env_load
  : "${VLLM_PORT:=8000}"
  : "${POOL_DIR:=${LOG_DIR}/pool_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}}"
  : "${READY_TIMEOUT:=1800}"

  mkdir -p "$POOL_DIR"
}

# Prepare the software environment inside a job: modules first, then a Python
# environment that has llm-annotate (and, on a server job, vllm) on PATH.
#
# A cluster file replaces this wholesale by defining cluster_setup_env, which is
# the escape hatch for clusters whose environment is not modules plus a venv --
# conda, a container runtime, a wrapper script of your own.
vllm_setup_env() {
  export PYTHONUNBUFFERED=1

  if declare -F cluster_setup_env > /dev/null; then
    cluster_setup_env
    vllm_check_env
    return
  fi

  if [[ -n "$CLUSTER_MODULES" ]] && command -v module > /dev/null 2>&1; then
    module purge
    # Unquoted on purpose: CLUSTER_MODULES is a list of module names.
    # shellcheck disable=SC2086
    module load $CLUSTER_MODULES
  fi

  if [[ "$UV_SYNC" == "1" ]]; then
    uv sync --frozen --no-install-project
  fi

  if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
  fi

  vllm_check_env
}

# The one thing every job needs, whichever way the environment was prepared.
vllm_check_env() {
  if ! command -v llm-annotate > /dev/null 2>&1; then
    echo "llm-annotate is not on PATH. Point VENV_PATH at an environment that" \
      "has it, set UV_SYNC=1 to create one, or define cluster_setup_env in" \
      "'${CLUSTER_ENV}'." >&2
    return 1
  fi
}

# Fallback for kernels that are not in the prebuilt flashinfer-jit-cache wheel.
# vLLM compiles those on first use and needs `nvcc`, dying with "Ninja build
# failed / nvcc: No such file or directory" when it is missing. Only PATH and
# CUDA_HOME are adopted from the module; LD_LIBRARY_PATH is restored so the CUDA
# runtime stays the one torch ships. Leave CUDA_MODULE empty to skip this.
vllm_ensure_nvcc() {
  if [[ -z "$CUDA_MODULE" ]] || command -v nvcc > /dev/null 2>&1; then
    return 0
  fi
  command -v module > /dev/null 2>&1 || return 0

  local ld_before="${LD_LIBRARY_PATH:-}"
  if ! module load "$CUDA_MODULE" 2> /dev/null; then
    echo "WARNING: could not load '${CUDA_MODULE}'; vLLM will fail if it needs" \
      "to JIT-compile kernels. Set CUDA_MODULE to an available toolkit." >&2
    return 0
  fi
  export LD_LIBRARY_PATH="$ld_before"

  echo "Using nvcc from ${CUDA_MODULE}: $(command -v nvcc)"
}

# Ask the config for a step's `vllm serve` arguments. Printed one per line, so
# the caller can `mapfile` them into an array and keep values that contain
# spaces intact. Run after vllm_setup_env, which puts llm-annotate on PATH.
vllm_serve_args() {
  local config="$1" step="$2"
  if ! llm-annotate "$config" --serve-args "$step"; then
    echo "Could not read serving arguments for step '${step}' of" \
      "'${config}'; see the error above." >&2
    return 1
  fi
}

# First free port at or above $1. Server jobs offset the base port by their
# array task id, because two of them can land on the same node -- an 8-GPU node
# fits four tensor_parallel_size=2 servers -- and a fixed port would collide.
vllm_pick_port() {
  local port="$1"
  local limit=$(( port + 200 ))

  while (( port < limit )); do
    # A successful connect means something is already listening there. The
    # probe runs in a subshell so the descriptor is closed again by its exit.
    if ! (exec 3<> "/dev/tcp/127.0.0.1/${port}") 2> /dev/null; then
      echo "$port"
      return 0
    fi
    port=$(( port + 1 ))
  done

  echo "No free port found between $1 and ${limit}" >&2
  return 1
}

# Block until every URL answers /health. Fails if the watched server exits
# first, or once READY_TIMEOUT is up.
vllm_wait_until_ready() {
  local urls=("$@")
  local deadline=$(( SECONDS + READY_TIMEOUT ))
  local url health

  for url in "${urls[@]}"; do
    health="${url%/v1}/health"
    until curl --silent --fail --max-time 5 "$health" > /dev/null; do
      if [[ -n "${READY_WATCH_PID:-}" ]] \
        && ! kill -0 "$READY_WATCH_PID" 2> /dev/null; then
        echo "vLLM exited before ${health} came up" >&2
        return 1
      fi
      if (( SECONDS > deadline )); then
        echo "Timed out after ${READY_TIMEOUT}s waiting for ${health}" >&2
        return 1
      fi
      sleep 10
    done
    echo "Ready: ${url}"
  done
}
