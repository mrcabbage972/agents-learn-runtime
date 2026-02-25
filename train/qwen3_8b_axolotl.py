#!/usr/bin/env python3
"""Train Qwen3-8B with Axolotl (4-bit QLoRA).

This script:
1. Converts trace files to ShareGPT JSONL format
2. Runs axolotl training with the generated config
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Add train directory to path for dataloader import
sys.path.insert(0, str(Path(__file__).parent))
from dataloader import build_trace_dataset, TraceFormat


def traces_to_sharegpt_jsonl(
    trace_root: str,
    output_path: Path,
    max_samples: int = 0,
    seed: int = 3407,
    trace_format: TraceFormat = TraceFormat.REACT,
    max_seq_length: int = 2048,
) -> int:
    """Convert trace files to ShareGPT JSONL format for axolotl."""
    # Estimate max chars from max tokens (~3.5 chars per token for code)
    max_chars = int(max_seq_length * 3.5)

    dataset = build_trace_dataset(
        trace_root=trace_root,
        max_samples=max_samples if max_samples else None,
        seed=seed,
        trace_format=trace_format,
        max_chars=max_chars,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        for example in dataset:
            messages = example["messages"]
            # chat_template format uses "messages" with role/content structure
            f.write(json.dumps({"messages": messages}) + "\n")
            count += 1

    return count


def generate_axolotl_config(
    base_config: Path,
    data_path: Path,
    output_dir: str,
    max_seq_length: int,
    batch_size: int,
    grad_accum: int,
    epochs: float,
    lr: float,
    model: str,
) -> dict:
    """Generate axolotl config with overrides."""
    import yaml

    with open(base_config) as f:
        config = yaml.safe_load(f)

    # Override with CLI args
    config["base_model"] = model
    config["datasets"] = [{
        "path": str(data_path),
        "type": "chat_template",
        "field_messages": "messages",
        "message_field_role": "role",
        "message_field_content": "content",
    }]
    config["output_dir"] = output_dir
    config["sequence_len"] = max_seq_length
    config["micro_batch_size"] = batch_size
    config["gradient_accumulation_steps"] = grad_accum
    config["num_epochs"] = epochs
    config["learning_rate"] = lr

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--output-dir", default="out/qwen3-8b-a100")
    parser.add_argument("--trace-root", default="traces")
    parser.add_argument("--trace-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--base-config", default="train/configs/axolotl_qwen3_8b_persistent_a100.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Only prepare data, don't train")
    parser.add_argument("--trace-format", default="codeact", choices=["react", "codeact"],
                        help="Trace format: react or codeact (default: codeact)")
    args = parser.parse_args()

    import yaml

    # Step 1: Convert traces to JSONL
    data_dir = Path(args.output_dir) / "data"
    data_path = data_dir / "traces.jsonl"

    # Parse trace format
    trace_format = TraceFormat(args.trace_format)

    print(f"Converting traces from {args.trace_root} to {data_path} (format: {trace_format.value})...")
    count = traces_to_sharegpt_jsonl(
        trace_root=args.trace_root,
        output_path=data_path,
        max_samples=args.trace_samples,
        seed=args.seed,
        trace_format=trace_format,
        max_seq_length=args.max_seq_length,
    )
    print(f"Converted {count} traces to ShareGPT format")

    # Step 2: Generate config
    config = generate_axolotl_config(
        base_config=Path(args.base_config),
        data_path=data_path,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
        model=args.model,
    )

    # Write config
    config_path = Path(args.output_dir) / "axolotl_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Generated config at {config_path}")

    if args.dry_run:
        print("Dry run complete. Config and data prepared.")
        return

    # Step 3: Run axolotl
    print("Starting axolotl training...")

    # Disable axolotl telemetry to avoid missing whitelist.yaml bug
    import os
    os.environ["AXOLOTL_DO_NOT_TRACK"] = "1"

    cmd = [
        "accelerate", "launch", "-m", "axolotl.cli.train", str(config_path)
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env={**os.environ})


if __name__ == "__main__":
    main()
