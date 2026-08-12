#!/bin/bash
#SBATCH --job-name=vllm-servers
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=tnsr72764

# EAR energy monitoring
# see https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/62226671/Energy+Aware+Runtime+EAR
#SBATCH --ear=on
#SBATCH --ear-policy=monitoring
#SBATCH --ear-verbose=1

# Serving-only job: starts a vLLM server on every GPU of every allocated node
# and keeps them alive for the duration of the job. The base URLs are written to
# logs/vllm_urls_<jobid>.txt, one per line, for clients to pick up.
#
# The default allocation is deliberately small (one node, one GPU) because it
# schedules almost immediately. Submitting it several times gives a pool that
# spans several nodes without ever waiting for them to be free at the same
# moment; point one client at the union of the URL files:
#
#   sbatch slurm/vllm_server_multinode.sh                 # note the job ids
#   sbatch slurm/vllm_server_multinode.sh
#   cat logs/vllm_urls_<id1>.txt logs/vllm_urls_<id2>.txt > logs/pool.txt
#   python examples/vllm-multinode/vllm_multinode.py \
#       --hosts-file logs/pool.txt --model "$MODEL" --wait-for-servers 900 ...
#
# Scale a single job up with the usual flags, e.g.
# `sbatch --nodes=4 --gres=gpu:4 --cpus-per-task=72 ...` for 16 servers.
# Use slurm/vllm_annotate_multinode.sh instead if you want the servers and the
# annotation run in a single job.

set -euo pipefail

: "${MODEL:?Set MODEL, e.g. MODEL=Qwen/Qwen3.5-4B}"

REPO_ROOT="${REPO_ROOT:-/home/bvanroy/llm-annotator}"
[[ -d "$REPO_ROOT" ]] || REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
mkdir -p logs

echo "Starting on $(date)"
echo "Nodes: ${SLURM_JOB_NODELIST:-<none>}"

module purge
module load 2025

# vllm-kernels ships FlashInfer's prebuilt kernels so nothing has to be
# JIT-compiled while several servers start at once; see pyproject.toml.
uv sync --frozen --extra vllm --extra vllm-kernels
# shellcheck disable=SC1091
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export REPO_ROOT
# EAR 5.2 + DCGM 4.6.0 segfault on Snellius; fall back to the NVML backend.
export EAR_GPU_DCGMI_ENABLED=0

# shellcheck disable=SC1091
source "${REPO_ROOT}/slurm/vllm_servers_lib.sh"
vllm_env_defaults
vllm_ensure_nvcc
export MODEL TP_SIZE GPUS_PER_NODE SERVERS_PER_NODE VLLM_PORT MAX_MODEL_LEN \
  GPU_MEM_UTIL

trap vllm_stop_servers EXIT

vllm_start_servers

URLS_FILE="${LOG_DIR}/vllm_urls_${SLURM_JOB_ID:-local}.txt"
printf '%s\n' "${SERVER_URLS[@]}" > "$URLS_FILE"
echo "Wrote ${#SERVER_URLS[@]} server URL(s) to ${URLS_FILE}"

vllm_wait_until_ready "${SERVER_URLS[@]}"
echo "All servers ready on $(date); serving until the job ends."

# Keep the allocation alive; the trap tears the servers down on exit.
wait
