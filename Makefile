.PHONY: lint-all format-notebooks format-check format pyright ruff ruff-fix sort-imports check
.PHONY: serve serve-lora serve-persistent-lora serve-forget-lora serve-react-lora
.PHONY: train-persistent train-forget train-react help
.PHONY: gen-tasks-easy gen-tasks-medium
.PHONY: bench-easy-base bench-easy-lora bench-med-base bench-med-lora
.PHONY: bench-forget-easy-base bench-forget-easy-lora bench-forget-med-base bench-forget-med-lora
.PHONY: bench-react-easy-base bench-react-easy-lora bench-react-med-base bench-react-med-lora
.PHONY: bench-cross-plora-reset-easy bench-cross-plora-reset-med bench-cross-flora-persist-easy bench-cross-flora-persist-med

help:
	@echo "Pythonformer Commands (A100-80GB, /nothink enabled)"
	@echo ""
	@echo "Inference:"
	@echo "  make serve                        Base model (bf16, 40k ctx)"
	@echo "  make serve-lora LORA=<path>       With LoRA adapter"
	@echo "  make serve-persistent-lora        Persistent state LoRA"
	@echo "  make serve-forget-lora            Forget state LoRA"
	@echo "  make serve-react-lora             ReAct LoRA"
	@echo ""
	@echo "Training:"
	@echo "  make train-persistent             CodeAct persistent state LoRA"
	@echo "  make train-forget                 CodeAct reset state LoRA"
	@echo "  make train-react                  ReAct LoRA"
	@echo ""
	@echo "Task Generation:"
	@echo "  make gen-tasks-easy               Generate easy task set"
	@echo "  make gen-tasks-medium             Generate medium task set"
	@echo ""
	@echo "Benchmarks - CodeAct persistent (requires vLLM):"
	@echo "  make bench-easy-base              Easy tasks, base model"
	@echo "  make bench-easy-lora              Easy tasks, persistent LoRA"
	@echo "  make bench-med-base               Medium tasks, base model"
	@echo "  make bench-med-lora               Medium tasks, persistent LoRA"
	@echo ""
	@echo "Benchmarks - CodeAct forget:"
	@echo "  make bench-forget-easy-base       Easy tasks, base model"
	@echo "  make bench-forget-easy-lora       Easy tasks, forget LoRA"
	@echo "  make bench-forget-med-base        Medium tasks, base model"
	@echo "  make bench-forget-med-lora        Medium tasks, forget LoRA"
	@echo ""
	@echo "Benchmarks - ReAct:"
	@echo "  make bench-react-easy-base        Easy tasks, base model"
	@echo "  make bench-react-easy-lora        Easy tasks, react LoRA"
	@echo "  make bench-react-med-base         Medium tasks, base model"
	@echo "  make bench-react-med-lora         Medium tasks, react LoRA"
	@echo ""
	@echo "Benchmarks - Cross-eval:"
	@echo "  make bench-cross-plora-reset-easy     Persistent LoRA + reset runtime, easy"
	@echo "  make bench-cross-plora-reset-med      Persistent LoRA + reset runtime, medium"
	@echo "  make bench-cross-flora-persist-easy   Forget LoRA + persistent runtime, easy"
	@echo "  make bench-cross-flora-persist-med    Forget LoRA + persistent runtime, medium"
	@echo ""
	@echo "Linting:"
	@echo "  make lint-all                     Run all linters with fixes"
	@echo "  make check                        Run all checks (no fixes)"

# =============================================================================
# Inference (vLLM) - A100-80GB (bf16, 40k context, /nothink)
# =============================================================================

serve:
	NOTHINK=true ./train/inference_a100.sh

serve-lora:
	NOTHINK=true ./train/inference_a100.sh --lora $(LORA)

serve-persistent-lora:
	NOTHINK=true LORA_NAME=qwen3-8b-lora ./train/inference_a100.sh --lora out/qwen3-8b-persistent-a100

serve-forget-lora:
	NOTHINK=true LORA_NAME=qwen3-8b-forget-lora ./train/inference_a100.sh --lora out/qwen3-8b-forget-a100

serve-react-lora:
	NOTHINK=true LORA_NAME=qwen3-8b-react-lora ./train/inference_a100.sh --lora out/qwen3-8b-react-a100

# =============================================================================
# Training - A100-80GB (QLoRA, 16k context, 3 epochs)
# =============================================================================

train-persistent:
	./train/train_a100.sh \
		--base-config train/configs/axolotl_qwen3_8b_persistent_a100.yaml \
		--output-dir out/qwen3-8b-persistent-a100 \
		--session train-persistent

train-forget:
	./train/train_a100.sh \
		--base-config train/configs/axolotl_qwen3_8b_forget_a100.yaml \
		--output-dir out/qwen3-8b-forget-a100 \
		--session train-forget

train-react:
	./train/train_a100.sh \
		--trace-format react \
		--base-config train/configs/axolotl_qwen3_8b_react_a100.yaml \
		--output-dir out/qwen3-8b-react-a100 \
		--session train-react

# =============================================================================
# Task Generation
# =============================================================================

gen-tasks-easy:
	uv run python -m pythonformer.cli --config train/assets/task_configs/easy_100.json \
		--out train/assets/tasks/easy --disable-llm

gen-tasks-medium:
	uv run python -m pythonformer.cli --config train/assets/task_configs/medium_100.json \
		--out train/assets/tasks/medium --disable-llm

# =============================================================================
# Benchmarks - CodeAct persistent
# Requires: make serve (base) or make serve-lora LORA=out/qwen3-8b-persistent-a100 (lora)
# =============================================================================

bench-easy-base:
	./train/benchmark.sh --config train/assets/benchmark_configs/easy_base.yaml \
		--tasks-root train/assets/tasks/easy --session bench-easy-base

bench-easy-lora:
	./train/benchmark.sh --config train/assets/benchmark_configs/easy_lora.yaml \
		--tasks-root train/assets/tasks/easy --session bench-easy-lora

bench-med-base:
	./train/benchmark.sh --config train/assets/benchmark_configs/medium_base.yaml \
		--tasks-root train/assets/tasks/medium --session bench-med-base

bench-med-lora:
	./train/benchmark.sh --config train/assets/benchmark_configs/medium_lora.yaml \
		--tasks-root train/assets/tasks/medium --session bench-med-lora

# =============================================================================
# Benchmarks - CodeAct forget
# Requires: make serve (base) or make serve-lora LORA=out/qwen3-8b-forget-a100 (lora)
# =============================================================================

bench-forget-easy-base:
	./train/benchmark.sh --config train/assets/benchmark_configs/forget_easy_base.yaml \
		--tasks-root train/assets/tasks/easy --session bench-forget-easy-base

bench-forget-easy-lora:
	./train/benchmark.sh --config train/assets/benchmark_configs/forget_easy_lora.yaml \
		--tasks-root train/assets/tasks/easy --session bench-forget-easy-lora

bench-forget-med-base:
	./train/benchmark.sh --config train/assets/benchmark_configs/forget_medium_base.yaml \
		--tasks-root train/assets/tasks/medium --session bench-forget-med-base

bench-forget-med-lora:
	./train/benchmark.sh --config train/assets/benchmark_configs/forget_medium_lora.yaml \
		--tasks-root train/assets/tasks/medium --session bench-forget-med-lora

# =============================================================================
# Benchmarks - ReAct
# Requires: make serve (base) or make serve-lora LORA=out/qwen3-8b-react-a100 (lora)
# =============================================================================

bench-react-easy-base:
	./train/benchmark.sh --config train/assets/benchmark_configs/react_easy_base.yaml \
		--tasks-root train/assets/tasks/easy --session bench-react-easy-base

bench-react-easy-lora:
	./train/benchmark.sh --config train/assets/benchmark_configs/react_easy_lora.yaml \
		--tasks-root train/assets/tasks/easy --session bench-react-easy-lora

bench-react-med-base:
	./train/benchmark.sh --config train/assets/benchmark_configs/react_medium_base.yaml \
		--tasks-root train/assets/tasks/medium --session bench-react-med-base

bench-react-med-lora:
	./train/benchmark.sh --config train/assets/benchmark_configs/react_medium_lora.yaml \
		--tasks-root train/assets/tasks/medium --session bench-react-med-lora

# =============================================================================
# Benchmarks - Cross-eval
# bench-cross-plora-*: Requires: NOTHINK=true LORA_NAME=qwen3-8b-lora ./train/inference_a100.sh --lora out/qwen3-8b-persistent-a100
# bench-cross-flora-*: Requires: NOTHINK=true LORA_NAME=qwen3-8b-forget-lora ./train/inference_a100.sh --lora out/qwen3-8b-forget-a100
# =============================================================================

bench-cross-plora-reset-easy:
	./train/benchmark.sh --config train/assets/benchmark_configs/cross_plora_reset_easy.yaml \
		--tasks-root train/assets/tasks/easy --session bench-cross-plora-reset-easy

bench-cross-plora-reset-med:
	./train/benchmark.sh --config train/assets/benchmark_configs/cross_plora_reset_medium.yaml \
		--tasks-root train/assets/tasks/medium --session bench-cross-plora-reset-med

bench-cross-flora-persist-easy:
	./train/benchmark.sh --config train/assets/benchmark_configs/cross_flora_persistent_easy.yaml \
		--tasks-root train/assets/tasks/easy --session bench-cross-flora-persist-easy

bench-cross-flora-persist-med:
	./train/benchmark.sh --config train/assets/benchmark_configs/cross_flora_persistent_medium.yaml \
		--tasks-root train/assets/tasks/medium --session bench-cross-flora-persist-med

# =============================================================================
# Linting
# =============================================================================

lint-all: format ruff-fix sort-imports pyright

format-notebooks:
	uv run nbqa isort . && uv run nbqa ruff . --fix

format-check:
	uv run ruff format --check

format:
	uv run ruff format

pyright:
	uv run pyright

ruff:
	uv run ruff check .

ruff-fix:
	uv run ruff check . --fix

sort-imports:
	uv run ruff check --select I --fix

check: format-check ruff pyright
