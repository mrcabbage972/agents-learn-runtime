# Qwen3-8B LoRA Training & Inference (A100-80GB)

Fine-tune **Qwen3-8B** with **Axolotl** using `.trace.json` files. Three training configurations: persistent CodeAct, forget CodeAct, and ReAct. All inference uses `/nothink` (thinking disabled).

## Setup

| Component | Configuration |
|-----------|---------------|
| **Base Model** | `Qwen/Qwen3-8B` |
| **Quantization** | 4-bit QLoRA (bitsandbytes NF4) |
| **Adapter** | LoRA (r=64, alpha=128) |
| **Training Context** | 16384 tokens |
| **Inference Context** | 40960 tokens |
| **Batch Size** | 1 (micro) x 16 (grad accum) = 16 effective |
| **Epochs** | 3 |
| **Optimizer** | AdamW torch |
| **GPU** | NVIDIA A100-80GB |
| **Attention** | FLASH_ATTN |
| **Thinking** | Disabled (`/nothink` via `enable_thinking: false`) |

## Training Configs

| Config | Agent | Output |
|--------|-------|--------|
| `axolotl_qwen3_8b_persistent_a100.yaml` | CodeAct (persistent state) | `out/qwen3-8b-persistent-a100` |
| `axolotl_qwen3_8b_forget_a100.yaml` | CodeAct (reset state) | `out/qwen3-8b-forget-a100` |
| `axolotl_qwen3_8b_react_a100.yaml` | ReAct | `out/qwen3-8b-react-a100` |

All configs share identical hyperparameters for fair comparison. Only trace data and output paths differ.

## Training

```bash
make train-persistent   # CodeAct persistent state LoRA
make train-forget       # CodeAct reset state LoRA
make train-react        # ReAct LoRA
```

The training script (`qwen3_8b_axolotl.py`) auto-converts `.trace.json` files to ShareGPT JSONL format. Use `--dry-run` to verify data conversion without training.

## Inference

```bash
make serve              # base model (bf16, 40k ctx, /nothink)
make serve-lora LORA=out/qwen3-8b-persistent-a100   # with LoRA
```

`/nothink` is always enabled via `NOTHINK=true`, which sets `--default-chat-template-kwargs {"enable_thinking": false}`.

## Benchmarks

All benchmarks require vLLM running. Configs in `train/assets/benchmark_configs/`.

### CodeAct persistent
```bash
make bench-easy-base    # easy tasks, base model
make bench-easy-lora    # easy tasks, persistent LoRA
make bench-med-base     # medium tasks, base model
make bench-med-lora     # medium tasks, persistent LoRA
```

### CodeAct forget
```bash
make bench-forget-easy-base
make bench-forget-easy-lora
make bench-forget-med-base
make bench-forget-med-lora
```

### ReAct
```bash
make bench-react-easy-base
make bench-react-easy-lora
make bench-react-med-base
make bench-react-med-lora
```

### Cross-eval (mismatched LoRA + runtime)
```bash
make bench-cross-plora-reset-easy      # persistent LoRA + reset runtime
make bench-cross-plora-reset-med
make bench-cross-flora-persist-easy    # forget LoRA + persistent runtime
make bench-cross-flora-persist-med
```

## VRAM

Training: ~50-65 GiB (QLoRA 4-bit + 16k context). Inference: ~70 GiB (bf16, 40k context).
