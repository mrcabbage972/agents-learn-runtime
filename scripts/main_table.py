#!/usr/bin/env python3
"""
Generate LaTeX Table 3 (Outcome + Footprint 2x2) from evaluation summary JSONs.

What this script expects
------------------------
You provide a small "spec" JSON that tells the script where each cell's summary lives.
Each summary JSON should contain (as in your dumps):

  - average_steps (float)
  - average_total_tokens (float or int)
  - average_runtime (float seconds)
  - metrics.score.mean (float in [0,1])   OR  metrics.score.mean already in [0,100]
  - num_tasks_solved (int)
  - num_tasks_executed (int)  (optional; if absent, defaults to 100)

The script outputs a LaTeX table with Score in percent (0–100) and Solved as count or percent.

Usage
-----
1) Create a spec file (example below) and run:
   python make_table3.py table3_spec.json > table3.tex

2) Include table3.tex in your paper with \\input{table3.tex}

Example spec (table3_spec.json)
-------------------------------
{
  "caption": "Opaque Knapsack evaluation across training and runtime execution semantics. Each cell reports outcome and interaction footprint.",
  "label": "tab:main_results",
  "score_scale": "auto",              // "auto" or "fraction" or "percent"
  "solved_format": "count",           // "count" or "percent"
  "cells": [
    {"difficulty":"Easy",   "train":"Persistent", "runtime":"Persistent", "path":"easy/persist_lora_persist_rt/summary.json"},
    {"difficulty":"Easy",   "train":"Persistent", "runtime":"Reset",      "path":"easy/persist_lora_reset_rt/summary.json"},
    {"difficulty":"Easy",   "train":"Reset",      "runtime":"Persistent", "path":"easy/reset_lora_persist_rt/summary.json"},
    {"difficulty":"Easy",   "train":"Reset",      "runtime":"Reset",      "path":"easy/reset_lora_reset_rt/summary.json"},

    {"difficulty":"Medium", "train":"Persistent", "runtime":"Persistent", "path":"medium/persist_lora_persist_rt/summary.json"},
    {"difficulty":"Medium", "train":"Persistent", "runtime":"Reset",      "path":"medium/persist_lora_reset_rt/summary.json"},
    {"difficulty":"Medium", "train":"Reset",      "runtime":"Persistent", "path":"medium/reset_lora_persist_rt/summary.json"},
    {"difficulty":"Medium", "train":"Reset",      "runtime":"Reset",      "path":"medium/reset_lora_reset_rt/summary.json"}
  ]
}

Notes
-----
- If a value is missing, the script prints "?" in that table cell.
- If you want to override the score shown (e.g., use your separately-computed "mean=75.4"),
  you can add "score_override": 75.4 in the cell entry.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def bootstrap_mean_ci(xs, n_boot=5000, alpha=0.05, seed=0):
    if not xs:
        return (None, None)
    rng = random.Random(seed)
    n = len(xs)
    boot = []
    for _ in range(n_boot):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        boot.append(statistics.mean(sample))
    boot.sort()
    lo = boot[int((alpha / 2) * n_boot)]
    hi = boot[int((1 - alpha / 2) * n_boot) - 1]
    return (lo, hi)


def get_trace_files(directory: Path):
    return glob(os.path.join(str(directory), "*.trace.json"))


def parse_pair(trace_filepath: str):
    result_filepath = trace_filepath.replace(".trace.json", ".json")
    try:
        with open(result_filepath, "r", encoding="utf-8") as f:
            res_data = json.load(f)
            res = res_data.get("result", {})
            score = res.get("score", None)
            if score is None:
                return None
            return float(score)
    except Exception:
        return None


def process_directory_scores(directory: Path):
    files = get_trace_files(directory)
    scores = []
    for f in files:
        s = parse_pair(f)
        if s is not None:
            scores.append(s)
    return scores


def _get(d: Dict[str, Any], path: str) -> Optional[Any]:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _fmt_num(x: Optional[float], kind: str, is_best: bool = False) -> str:
    if x is None:
        return "?"

    val = ""
    if kind == "score":
        val = f"{x:.1f}"
    elif kind == "steps":
        val = f"{x:.2f}".rstrip("0").rstrip(".")
    elif kind == "tokens":
        val = f"{int(round(x)):,}"
    elif kind == "runtime":
        val = f"{x:.1f}"
    elif kind == "norm":
        val = f"{x:.2f}"
    else:
        val = str(x)

    return rf"\textbf{{{val}}}" if is_best else val


def _auto_score_to_percent(score: Optional[float]) -> Optional[float]:
    """
    If score looks like a fraction (<=1.5), convert to percent.
    If it already looks like a percent (e.g., 68.2), keep.
    """
    if score is None:
        return None
    if score <= 1.5:
        return score * 100.0
    return score


@dataclass
class Cell:
    difficulty: str
    train: str
    runtime: str
    path: Path
    score_override: Optional[float] = None
    metrics_dir: Optional[Path] = None


def load_cell_summary(cell: Cell, score_scale: str) -> Dict[str, Any]:
    if not cell.path.exists():
        return dict(
            score_pct=None,
            solved=None,
            executed=None,
            steps=None,
            tokens=None,
            runtime=None,
        )

    data = json.loads(cell.path.read_text())

    # score
    if cell.score_override is not None:
        score_raw = float(cell.score_override)
    else:
        score_raw = _get(data, "metrics.score.mean")
        if score_raw is None:
            # sometimes score is stored directly as top-level "score"
            score_raw = _get(data, "score.mean") or _get(data, "score")

    score_raw = float(score_raw) if score_raw is not None else None

    if score_scale == "percent":
        score_pct = score_raw
    elif score_scale == "fraction":
        score_pct = (score_raw * 100.0) if score_raw is not None else None
    else:
        score_pct = _auto_score_to_percent(score_raw)

    solved = _get(data, "num_tasks_solved")
    executed = _get(data, "num_tasks_executed")
    steps = _get(data, "average_steps")
    tokens = _get(data, "average_total_tokens")
    runtime = _get(data, "average_runtime")

    solved = int(solved) if solved is not None else None
    executed = int(executed) if executed is not None else 100
    steps = float(steps) if steps is not None else None
    tokens = float(tokens) if tokens is not None else None
    runtime = float(runtime) if runtime is not None else None

    score_ci = None
    if cell.metrics_dir is not None and cell.metrics_dir.exists():
        scores = process_directory_scores(cell.metrics_dir)

        # convert to percent consistent with table
        scores_pct = []
        for s in scores:
            if score_scale == "percent":
                scores_pct.append(s)
            elif score_scale == "fraction":
                scores_pct.append(s * 100.0)
            else:
                scores_pct.append(_auto_score_to_percent(s))

        lo, hi = bootstrap_mean_ci(scores_pct, n_boot=5000, alpha=0.05, seed=0)
        if lo is not None and hi is not None:
            score_ci = (lo, hi)

    return dict(
        score_pct=score_pct,
        score_ci=score_ci,
        solved=solved,
        executed=executed,
        steps=steps,
        tokens=tokens,
        runtime=runtime,
    )


def render_table(
    cells: List[Cell], caption: str, label: str, score_scale: str, solved_format: str
) -> str:
    difficulties = ["Easy", "Medium"]
    lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for c in cells:
        lookup[(c.difficulty, c.train, c.runtime)] = load_cell_summary(c, score_scale)

    def solved_str(solved: Optional[int], executed: Optional[int]) -> str:
        if solved is None:
            return "?"
        if solved_format == "percent":
            ex = executed or 100
            return f"{(100.0 * solved / ex):.0f}"
        return str(solved)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\setlength{\aboverulesep}{0pt}")
    lines.append(r"\setlength{\belowrulesep}{0pt}")
    lines.append(r"\setlength{\extrarowheight}{.1ex}")
    
    # ADDED: >{\columncolor{black!10}}c for the final column
    lines.append(r"\begin{tabular}{lllccccc>{\columncolor{black!10}}c}")
    lines.append(r"\toprule")
    # REORDERED: Time (s) before Norm. Score
    lines.append(
        r"\textbf{Difficulty} & \textbf{Train} & \textbf{Runtime} & "
        r"\textbf{Score (\%)} & \textbf{Solved} & \textbf{Steps} & \textbf{Tokens} & \textbf{Time (s)} & \textbf{Score / 1k Tokens} \\"
    )
    lines.append(r"\midrule")

    for diff in difficulties:
        group_metrics = [
            lookup.get((diff, t, r))
            for t in ["Persistent", "Reset"]
            for r in ["Persistent", "Reset"]
        ]
        group_metrics = [m for m in group_metrics if m and m["score_pct"] is not None]

        best_score = (
            max([m["score_pct"] for m in group_metrics]) if group_metrics else None
        )
        best_steps = min([m["steps"] for m in group_metrics]) if group_metrics else None
        best_tokens = (
            min([m["tokens"] for m in group_metrics]) if group_metrics else None
        )
        best_time = (
            min([m["runtime"] for m in group_metrics]) if group_metrics else None
        )

        # Calculate best normalized score (Score per 1k tokens)
        norm_vals = []
        for m in group_metrics:
            if (
                m["score_pct"] is not None
                and m["tokens"] is not None
                and m["tokens"] > 0
            ):
                norm_vals.append((m["score_pct"] / m["tokens"]) * 1000)
        best_norm = max(norm_vals) if norm_vals else None

        first_row_in_group = True
        for train in ["Persistent", "Reset"]:
            for runtime in ["Persistent", "Reset"]:
                m = lookup.get(
                    (diff, train, runtime),
                    dict(
                        score_pct=None,
                        solved=None,
                        executed=None,
                        steps=None,
                        tokens=None,
                        runtime=None,
                    ),
                )

                s_bold = m["score_pct"] == best_score and best_score is not None
                st_bold = m["steps"] == best_steps and best_steps is not None
                tk_bold = m["tokens"] == best_tokens and best_tokens is not None
                rt_bold = m["runtime"] == best_time and best_time is not None

                # Calculate normalized score for this cell
                score_val = m.get("score_pct")
                tok_val = m.get("tokens")
                norm_val = None
                n_bold = False
                if score_val is not None and tok_val is not None and tok_val > 0:
                    norm_val = (score_val / tok_val) * 1000
                    n_bold = best_norm is not None and abs(norm_val - best_norm) < 1e-6

                solved = solved_str(m["solved"], m["executed"])
                steps = _fmt_num(m["steps"], "steps", st_bold)
                tokens = _fmt_num(m["tokens"], "tokens", tk_bold)
                rt = _fmt_num(m["runtime"], "runtime", rt_bold)
                norm_str = _fmt_num(norm_val, "norm", n_bold)

                score = _fmt_num(m["score_pct"], "score", s_bold)
                ci = m.get("score_ci")
                if ci is not None:
                    # Convert [lo, hi] to symmetric +- margin
                    margin = (ci[1] - ci[0]) / 2.0
                    score = rf"{score} $\pm$ {margin:.1f}"

                if first_row_in_group:
                    diff_cell = (
                        rf"\multirow{{4}}{{*}}{{\rotatebox[origin=c]{{90}}{{{diff}}}}}"
                    )
                    first_row_in_group = False
                else:
                    diff_cell = ""

                # REORDERED: rt before norm_str
                lines.append(
                    f"{diff_cell} & {train} & {runtime} & {score} & {solved} & {steps} & {tokens} & {rt} & {norm_str} \\\\"
                )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=str, help="Path to table3_spec.json")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text())

    caption = spec.get(
        "caption",
        "Opaque Knapsack evaluation. Persistent training yields Trajectory Compression, achieving comparable optimality with lower token and temporal costs. \textit{Norm. Score} represents the score achieved per 1,000 generated tokens.",
    )
    label = spec.get("label", "tab:main_results")
    score_scale = spec.get("score_scale", "auto")
    solved_format = spec.get("solved_format", "count")

    cells: List[Cell] = []
    for c in spec["cells"]:
        cells.append(
            Cell(
                difficulty=c["difficulty"],
                train=c["train"],
                runtime=c["runtime"],
                path=Path(c["path"]),
                score_override=c.get("score_override"),
                metrics_dir=Path(c["metrics_path"]) if c.get("metrics_path") else None,
            )
        )

    tex = render_table(cells, caption, label, score_scale, solved_format)
    print(tex)


if __name__ == "__main__":
    main()
