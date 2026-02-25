#!/usr/bin/env bash
set -euo pipefail

# A100-80GB optimized vLLM server
# No quantization (bf16 native), 40k context

# Parse --lora flag
LORA_PATH=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --lora)
      LORA_PATH="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

MODEL="${MODEL:-Qwen/Qwen3-8B}"

# A100-80GB settings: no quantization, bf16, 40k context
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
SESSION_NAME="${SESSION_NAME:-vllm-a100}"
NOTHINK="${NOTHINK:-false}"

# Set served model name and lora name based on --lora flag
if [[ -n "$LORA_PATH" ]]; then
  LORA_NAME="${LORA_NAME:-qwen3-8b-lora}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$LORA_NAME}"
else
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b}"
fi

# Build uv run command (no bitsandbytes needed for A100)
UV_CMD=(uv run --with vllm python -m vllm.entrypoints.openai.api_server)

CMD=(
  "${UV_CMD[@]}"
  --model "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --served-model-name "$SERVED_MODEL_NAME"
  --attention-backend "$ATTENTION_BACKEND"
)

if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
  CMD+=(--trust-remote-code)
fi

if [[ -n "$LORA_PATH" ]]; then
  CMD+=(--enable-lora --lora-modules "${LORA_NAME}=${LORA_PATH}" --max-lora-rank 64)
fi

if [[ "$NOTHINK" == "true" ]]; then
  CHAT_KWARGS='{"enable_thinking":false}'
  CMD+=(--default-chat-template-kwargs "$CHAT_KWARGS")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "========================================"
echo "A100-80GB Optimized vLLM Server"
echo "========================================"
echo
echo "Config:"
echo "  MODEL:          $MODEL"
echo "  DTYPE:          $DTYPE (no quantization)"
echo "  MAX_MODEL_LEN:  $MAX_MODEL_LEN"
echo "  MAX_NUM_SEQS:   $MAX_NUM_SEQS"
echo "  GPU_MEM_UTIL:   $GPU_MEMORY_UTILIZATION"
echo "  SERVED_NAME:    $SERVED_MODEL_NAME"
if [[ -n "$LORA_PATH" ]]; then
  echo "  LORA_PATH:      $LORA_PATH"
fi
if [[ "$NOTHINK" == "true" ]]; then
  echo "  NOTHINK:        enabled (thinking disabled)"
fi
echo
echo "Launching vLLM server in tmux session: ${SESSION_NAME}"
printf '  %q' "${CMD[@]}"
echo
echo

# Start tmux server if not running
tmux start-server 2>/dev/null || true

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists. Attach with:"
  echo "  tmux attach -t ${SESSION_NAME}"
  exit 1
fi

# Write command to temp script to preserve quoting
TMPSCRIPT=$(mktemp /tmp/vllm-cmd.XXXXXX.sh)
echo '#!/bin/bash' > "$TMPSCRIPT"
printf '%q ' "${CMD[@]}" >> "$TMPSCRIPT"
chmod +x "$TMPSCRIPT"
tmux new-session -d -s "${SESSION_NAME}" -- bash -lc "$TMPSCRIPT; rm $TMPSCRIPT"
tmux set-option -t "${SESSION_NAME}" remain-on-exit on
echo "Server started. Attach with:"
echo "  tmux attach -t ${SESSION_NAME}"
