#!/bin/bash
# Defaults for a manual `sbatch slurm/vllm_server.sh`; submit_pipeline.sh
# overrides --array, the GPU request and --cpus-per-task from the cluster file
# and the step's own config.
#SBATCH --job-name=vllm-server
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# One vLLM server in its own job. Submitted as an array by
# slurm/submit_pipeline.sh, one task per server of the step's pool.
#
# What this server serves, and how, comes from the pipeline config alone: the
# job asks `llm-annotate --serve-args` for its step's `vllm serve` arguments.
# That is why a pipeline whose steps use different models works -- each step's
# servers read their own `engine:` block -- and why no serving flag needs an
# environment variable.
#
# Once the server answers /health it publishes its base URL as
# <POOL_DIR>/<task>.url and removes that file again when the job ends, so the
# pool directory only ever lists servers that are actually up.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"

: "${ANNOTATE_CONFIG:?Set ANNOTATE_CONFIG to a JSON/YAML pipeline config}"
: "${STEP_NAME:?Set STEP_NAME to the step whose servers this job runs}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "Starting on $(date)"
echo "Host: $(hostname), array task: ${TASK_ID}"
echo "Step: ${STEP_NAME} of ${ANNOTATE_CONFIG}"

# shellcheck source=slurm/vllm_common.sh
source "${REPO_ROOT}/slurm/vllm_common.sh"
vllm_env_defaults
vllm_setup_env
vllm_ensure_nvcc

if ! command -v vllm > /dev/null 2>&1; then
  echo "vllm is not on PATH; install the 'vllm' extra in the environment this" \
    "job uses (uv sync --extra vllm --group vllm-kernels)." >&2
  exit 1
fi

export VLLM_WORKER_MULTIPROC_METHOD=spawn

URL_FILE="${POOL_DIR}/${TASK_ID}.url"
SERVE_LOG=$(mktemp)
SERVER_PID=""

cleanup() {
  rm -f "$URL_FILE" "${POOL_DIR}/.${TASK_ID}.tmp" "$SERVE_LOG"
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2> /dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# One argument per line, read into an array, so a value may contain spaces --
# --speculative-config takes a JSON object that word-splitting would destroy.
# Collected into a variable first: `set -e` does not see a command that fails
# inside the process substitution `mapfile` would otherwise read from, so a
# config that does not load would leave an empty argument list behind.
SERVE_ARGS_TEXT=$(vllm_serve_args "$ANNOTATE_CONFIG" "$STEP_NAME") || exit 1
mapfile -t SERVE_ARGS <<< "$SERVE_ARGS_TEXT"

HOST=$(vllm_server_host)
PORT_FROM=$(( VLLM_PORT + TASK_ID ))

# The port is probed here and handed to `vllm serve` afterwards, so a process
# on the same node can take it in between. That costs a port rather than the
# job: vLLM then dies with "Address already in use" before /health ever
# answers, and this retries on the next free port. Any other early exit fails
# the task.
: "${PORT_RETRIES:=5}"

for (( attempt = 0; ; attempt += 1 )); do
  PORT=$(vllm_pick_port "$PORT_FROM")
  URL="http://${HOST}:${PORT}/v1"
  : > "$SERVE_LOG"

  echo "Serving on ${URL} with: vllm serve ${SERVE_ARGS[*]}"

  # --host and --port are the only two the config cannot give. The output is
  # teed so the retry decision has something to read; tee runs beside the
  # server, so $! is still the server's own pid.
  vllm serve "${SERVE_ARGS[@]}" \
    --host 0.0.0.0 \
    --port "$PORT" > >(tee -a "$SERVE_LOG") 2>&1 &
  SERVER_PID=$!

  if READY_WATCH_PID="$SERVER_PID" vllm_wait_until_ready "$URL"; then
    break
  fi

  kill "$SERVER_PID" 2> /dev/null || true
  wait "$SERVER_PID" 2> /dev/null || true

  if (( attempt < PORT_RETRIES )) && vllm_port_bind_failed "$SERVE_LOG"; then
    echo "Port ${PORT} was taken before vLLM could bind it;" \
      "retrying on the next free port."
    PORT_FROM=$(( PORT + 1 ))
    continue
  fi

  echo "Server never became ready; giving up." >&2
  exit 1
done

# Publish only now, and atomically: a URL in the pool directory is therefore
# always a server the client can talk to right away.
printf '%s\n' "$URL" > "${POOL_DIR}/.${TASK_ID}.tmp"
mv "${POOL_DIR}/.${TASK_ID}.tmp" "$URL_FILE"
echo "Published ${URL} to ${URL_FILE}; serving until the job ends."

status=0
wait "$SERVER_PID" || status=$?
echo "vLLM server exited with status ${status} on $(date)"
exit "$status"
