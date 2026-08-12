#!/bin/bash
# Runs inside one Slurm job step, on one node: starts SERVERS_PER_NODE vLLM
# servers, each on its own slice of the GPUs this step was given, and waits for
# them. Launched by vllm_start_servers (slurm/vllm_servers_lib.sh); not meant to
# be run by hand.

set -uo pipefail

: "${MODEL:?MODEL must be exported by the parent job}"
: "${TP_SIZE:=1}"
: "${GPUS_PER_NODE:=4}"
: "${SERVERS_PER_NODE:=$(( GPUS_PER_NODE / TP_SIZE ))}"
: "${VLLM_PORT:=8000}"
: "${MAX_MODEL_LEN:=4096}"
: "${GPU_MEM_UTIL:=0.85}"
: "${REPO_ROOT:=$PWD}"

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv/bin/activate"

export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Devices this step actually owns. Slurm normally exports them relative to the
# step's cgroup; fall back to the full node when it does not (e.g. a bare shell
# outside Slurm).
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a STEP_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
else
  STEP_DEVICES=()
  for (( device = 0; device < GPUS_PER_NODE; device++ )); do
    STEP_DEVICES+=("$device")
  done
fi

if (( ${#STEP_DEVICES[@]} < SERVERS_PER_NODE * TP_SIZE )); then
  echo "This step sees ${#STEP_DEVICES[@]} GPU(s) but needs $(( SERVERS_PER_NODE * TP_SIZE ))" >&2
  exit 1
fi

echo "$(hostname): starting ${SERVERS_PER_NODE} server(s) over GPUs ${STEP_DEVICES[*]}"

PIDS=()
for (( server_idx = 0; server_idx < SERVERS_PER_NODE; server_idx++ )); do
  devices="${STEP_DEVICES[*]:$(( server_idx * TP_SIZE )):${TP_SIZE}}"
  port=$(( VLLM_PORT + server_idx ))
  echo "$(hostname): server ${server_idx} -> port ${port}, GPU(s) ${devices// /,}"

  CUDA_VISIBLE_DEVICES="${devices// /,}" \
    vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$port" \
    --served-model-name "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    ${VLLM_EXTRA_ARGS:-} &
  PIDS+=("$!")
done

# Bring the whole step down as soon as one server dies, so the job does not sit
# there with a silently degraded pool.
trap 'kill "${PIDS[@]}" 2>/dev/null' EXIT INT TERM
wait -n "${PIDS[@]}"
status=$?
echo "$(hostname): a vLLM server exited with status ${status}" >&2
exit "$status"
