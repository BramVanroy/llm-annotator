#!/bin/bash
# Shared helpers to start one or more vLLM servers on every node of the
# current Slurm allocation. Source this from a job script; it does not do
# anything on its own.
#
# Expects (with defaults applied by vllm_env_defaults):
#   MODEL             model id or path served by every server
#   TP_SIZE           tensor-parallel size, i.e. GPUs per server (default 1)
#   GPUS_PER_NODE     GPUs allocated per node (default 4)
#   VLLM_PORT         port of the first server on each node (default 8000)
#   MAX_MODEL_LEN     --max-model-len passed to vllm serve (default 4096)
#   GPU_MEM_UTIL      --gpu-memory-utilization (default 0.85)
#   REPO_ROOT         checkout that holds .venv
#   LOG_DIR           directory for per-server logs
#
# Provides:
#   vllm_env_defaults            apply the defaults above
#   vllm_start_servers           launch the servers, fill SERVER_URLS/SERVER_PIDS
#   vllm_wait_until_ready <urls> block until every server answers /health
#   vllm_stop_servers            terminate the launched job steps

vllm_env_defaults() {
  : "${MODEL:?Set MODEL, e.g. MODEL=Qwen/Qwen3.5-4B}"
  : "${TP_SIZE:=1}"
  # Follow whatever the allocation actually got, so changing --gres on the
  # sbatch command line is enough to change the size of the pool.
  : "${GPUS_PER_NODE:=${SLURM_GPUS_ON_NODE:-4}}"
  : "${VLLM_PORT:=8000}"
  : "${MAX_MODEL_LEN:=4096}"
  : "${GPU_MEM_UTIL:=0.85}"
  : "${REPO_ROOT:=${SLURM_SUBMIT_DIR:-$PWD}}"
  : "${LOG_DIR:=${REPO_ROOT}/logs}"
  : "${READY_TIMEOUT:=1800}"

  if (( GPUS_PER_NODE % TP_SIZE != 0 )); then
    echo "GPUS_PER_NODE ($GPUS_PER_NODE) must be a multiple of TP_SIZE ($TP_SIZE)" >&2
    return 1
  fi
  SERVERS_PER_NODE=$(( GPUS_PER_NODE / TP_SIZE ))
  mkdir -p "$LOG_DIR"
}

# Fallback for kernels that are not in the prebuilt flashinfer-jit-cache wheel
# (see the `vllm` extra in pyproject.toml): vLLM compiles those on first use and
# needs `nvcc`, which `module purge` takes away, dying with "Ninja build failed
# / nvcc: No such file or directory". Only PATH and CUDA_HOME are adopted from
# the CUDA module - LD_LIBRARY_PATH is restored so the CUDA runtime stays the
# one shipped inside the torch wheel (a different major version there would be
# a real mismatch). Set CUDA_MODULE=none to skip this entirely.
vllm_ensure_nvcc() {
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

# Start SERVERS_PER_NODE servers on every allocated node.
# Sets SERVER_URLS (array of base URLs) and SERVER_PIDS (array of srun PIDs).
vllm_start_servers() {
  local hosts host server_idx port
  readarray -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  if [[ ${#hosts[@]} -eq 0 ]]; then
    echo "No allocated hosts found in '${SLURM_JOB_NODELIST:-}'" >&2
    return 1
  fi

  SERVER_URLS=()
  SERVER_PIDS=()

  for host in "${hosts[@]}"; do
    echo "Launching ${SERVERS_PER_NODE} vLLM server(s) on ${host} (TP=${TP_SIZE})"
    # One job step per node, holding that node's GPUs. The step splits its own
    # device mask between the servers it starts, so we never point vLLM at a
    # GPU this job was not given. Job steps do not inherit the allocation's
    # GRES (Slurm >= 22.05), hence the explicit --gres here. --overlap lets the
    # step share the CPUs already held by the batch step, which otherwise
    # occupies them for the whole job and blocks step creation.
    srun --nodes=1 \
      --ntasks=1 \
      --nodelist="$host" \
      --overlap \
      --gres=gpu:"${GPUS_PER_NODE}" \
      --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
      --output="${LOG_DIR}/vllm_${host}_%j.out" \
      --error="${LOG_DIR}/vllm_${host}_%j.err" \
      bash "${REPO_ROOT}/slurm/vllm_node_servers.sh" &
    SERVER_PIDS+=("$!")

    for (( server_idx = 0; server_idx < SERVERS_PER_NODE; server_idx++ )); do
      port=$(( VLLM_PORT + server_idx ))
      SERVER_URLS+=("http://${host}:${port}/v1")
    done
  done
}

# Return non-zero as soon as one of the launched job steps has exited.
vllm_servers_alive() {
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill -0 "$pid" 2> /dev/null || return 1
  done
}

# Block until every given base URL answers /health, or until READY_TIMEOUT.
# Fails fast if a server dies while we are waiting, instead of sitting out the
# full timeout on a pool that is never going to come up.
vllm_wait_until_ready() {
  local urls=("$@")
  local deadline=$(( SECONDS + READY_TIMEOUT ))
  local url health

  for url in "${urls[@]}"; do
    health="${url%/v1}/health"
    until curl --silent --fail --max-time 5 "$health" > /dev/null; do
      if ! vllm_servers_alive; then
        echo "A vLLM job step exited before the pool was ready;" \
          "see ${LOG_DIR}/vllm_*.err" >&2
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

vllm_stop_servers() {
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill "$pid" 2> /dev/null || true
  done
  wait 2> /dev/null || true
}
