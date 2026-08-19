#!/bin/bash
# Shared helpers for the vLLM server pool. This file is source in the other scripts.
#
# The pool is one Slurm job per vLLM server: every job holds GPUS_PER_VLLM_SERVER
# GPUs and serves a single tensor-parallel server, publishing its base URL into
# POOL_DIR once it is healthy. slurm/submit_pool.sh submits the array of server
# jobs and launches the client that would read from that directory before continuing.
#
# Expects (with defaults applied by vllm_env_defaults):
#   MODEL             model id or path served by every server
#   GPUS_PER_VLLM_SERVER    GPUs per job, i.e. tensor-parallel size (default 1)
#   VLLM_PORT         base port; the array task id is added to it (default 8000)
#   MAX_MODEL_LEN     --max-model-len passed to vllm serve (default 8192)
#   GPU_MEM_UTIL      --gpu-memory-utilization (default 0.90)
#   REPO_ROOT         directory that has subdir .venv (llm-annotator root)
#   LOG_DIR           directory for slurm logs and pool directories
#   POOL_DIR          directory holding one <task>.url per server
#
# Provides:
#   vllm_env_defaults            apply the defaults above
#   vllm_setup_env               modules, uv sync, activate .venv
#   vllm_ensure_nvcc             make nvcc available for JIT-compiled kernels
#   vllm_pick_port <start>       first free TCP port at or above VLLM_PORT
#   vllm_wait_until_ready <urls> block until every URL answers /health

vllm_env_defaults() {
  # MODEL is required, if not set, exit with error message
  : "${MODEL:?Set MODEL, e.g. MODEL=Qwen/Qwen3.5-4B}"
  : "${GPUS_PER_VLLM_SERVER:=1}"
  : "${VLLM_PORT:=8000}"
  : "${MAX_MODEL_LEN:=8192}"
  : "${GPU_MEM_UTIL:=0.90}"
  : "${REPO_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
  : "${LOG_DIR:=${REPO_ROOT}/logs}"
  : "${POOL_DIR:=${LOG_DIR}/pool_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}}"
  : "${READY_TIMEOUT:=1800}"

  if (( GPUS_PER_VLLM_SERVER < 1 )); then
    echo "GPUS_PER_VLLM_SERVER ($GPUS_PER_VLLM_SERVER) must be at least 1" >&2
    return 1
  fi
  mkdir -p "$LOG_DIR" "$POOL_DIR"
}

# Load slurm modules, sync vllm and include vllm-kernels extra
# to avoid compiling kernels on the fly
vllm_setup_env() {
  module purge
  module load 2025

  uv sync --frozen --extra vllm --extra vllm-kernels
  source "${REPO_ROOT}/.venv/bin/activate"

  export PYTHONUNBUFFERED=1
  # EAR 5.2 + DCGM 4.6.0 give errors on Snellius; fall back to the NVML backend
  export EAR_GPU_DCGMI_ENABLED=0
}

# Fallback for kernels that are not in the prebuilt flashinfer-jit-cache wheel
# vLLM compiles those on first use and needs `nvcc`, which `module purge` may remove
# from loaded libs, dying with "Ninja build failed / nvcc: No such file or directory".
# Only PATH and CUDA_HOME are adopted from the CUDA module.
# LD_LIBRARY_PATH is restored so the CUDA runtime is the one from torch.
# Set CUDA_MODULE=none to skip this entirely.
vllm_ensure_nvcc() {
  # snellius' latest version is 12.8.0 atm
  # TODO: put in a request for cu130?
  : "${CUDA_MODULE:=CUDA/12.8.0}"

  if [[ "$CUDA_MODULE" == "none" ]] || command -v nvcc > /dev/null 2>&1; then
    return 0
  fi

  local ld_before="${LD_LIBRARY_PATH:-}"
  if ! module load "$CUDA_MODULE" 2> /dev/null; then
    echo "WARNING: could not load '${CUDA_MODULE}'; vLLM will fail if it needs" \
      "to JIT-compile kernels. Set CUDA_MODULE to an available toolkit." >&2
    return 0
  fi
  export LD_LIBRARY_PATH="$ld_before"

  echo "Using nvcc from ${CUDA_MODULE}: $(command -v nvcc)"
}

# Get the first free port at or above $1. Jobs offset the base port
# by their array task id to avoid them racing to use the same port when
# two jobs are assigned to the same node
# e.g. a 4-GPU node fits two GPUS_PER_VLLM_SERVER=2 servers
vllm_pick_port() {
  local port="$1"
  local limit=$(( port + 200 ))

  # brute-force it: just check 200 ports to find the one not in use
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

# Read the URLs (health endpoints of vllm servers) until
# they respond with HTTP 200 OK. If any of the servers exit before
# they are ready, or if the timeout is reached, return 1.
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
