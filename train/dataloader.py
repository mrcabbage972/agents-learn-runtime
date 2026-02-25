"""Trace dataset loader for training."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import List, Optional

from datasets import Dataset


class TraceFormat(str, Enum):
    """Supported trace formats for training data."""
    REACT = "react"
    CODEACT = "codeact"


def load_trace_messages(trace_path: Path) -> Optional[List[dict]]:
    """Extract multi-turn conversation from a single .trace.json file.

    Creates proper ReAct-style multi-turn format:
    [system, user_task, assistant_step0, user_obs0, assistant_step1, user_obs1, ...]
    """
    try:
        data = json.loads(trace_path.read_text())
    except Exception:
        return None

    events = data.get("events")
    if not isinstance(events, list):
        return None

    system_prompts: List[str] = []
    task_text = None
    step_events: List[dict] = []

    for event in events:
        event_type = event.get("type")
        event_data = event.get("data", {})

        if event_type == "SystemPromptEvent":
            prompts = event_data.get("prompts")
            if isinstance(prompts, list):
                system_prompts.extend([p for p in prompts if isinstance(p, str) and p.strip()])
            elif isinstance(prompts, str) and prompts.strip():
                system_prompts.append(prompts)
        elif event_type == "StartEvent":
            if task_text is None:
                task = event_data.get("task")
                if isinstance(task, str) and task.strip():
                    task_text = task
        elif event_type == "StepEvent":
            step_events.append(event_data)

    if not task_text or not step_events:
        return None

    messages: List[dict] = []

    # Add system message
    if system_prompts:
        messages.append({"role": "system", "content": "\n\n".join(system_prompts)})

    # Add initial user task
    messages.append({"role": "user", "content": task_text})

    # Add each step as assistant turn + observation as user turn
    for i, step_data in enumerate(step_events):
        assistant_text = step_data.get("assistant_text")
        if not isinstance(assistant_text, str) or not assistant_text.strip():
            continue

        # Add assistant response (thought + action)
        messages.append({"role": "assistant", "content": assistant_text})

        # Add observation as user message (except for the last step)
        # We want the model to learn to generate responses, so we include
        # observations that lead to the next assistant response
        interpreter_result = step_data.get("interpreter_result") or {}
        # Support both "results" (current format) and "tool_calls" (legacy format)
        tool_results = interpreter_result.get("results") or interpreter_result.get("tool_calls") or []

        if tool_results and i < len(step_events) - 1:
            # Format observation from tool results
            obs_parts = []
            for tc in tool_results:
                tool_name = tc.get("tool", "unknown")
                output = tc.get("output", "")
                obs_parts.append(f"Observation: [{tool_name}] {output}")

            if obs_parts:
                messages.append({"role": "user", "content": "\n".join(obs_parts)})

    # Must have at least system + user + one assistant message
    if len(messages) < 3:
        return None

    return messages


def load_codeact_messages(trace_path: Path) -> Optional[List[dict]]:
    """Extract multi-turn conversation from a CodeAct .trace.json file.

    Creates proper CodeAct-style multi-turn format:
    [system, user_task, assistant_code0, user_obs0, assistant_code1, user_obs1, ...]

    Observations come from interpreter_result.output (stdout from code execution).
    """
    try:
        data = json.loads(trace_path.read_text())
    except Exception:
        return None

    events = data.get("events")
    if not isinstance(events, list):
        return None

    system_prompts: List[str] = []
    task_text = None
    step_events: List[dict] = []

    for event in events:
        event_type = event.get("type")
        event_data = event.get("data", {})

        if event_type == "SystemPromptEvent":
            prompts = event_data.get("prompts")
            if isinstance(prompts, list):
                system_prompts.extend([p for p in prompts if isinstance(p, str) and p.strip()])
            elif isinstance(prompts, str) and prompts.strip():
                system_prompts.append(prompts)
        elif event_type == "StartEvent":
            if task_text is None:
                task = event_data.get("task")
                if isinstance(task, str) and task.strip():
                    task_text = task
        elif event_type == "StepEvent":
            step_events.append(event_data)

    if not task_text or not step_events:
        return None

    messages: List[dict] = []

    # Add system message
    if system_prompts:
        messages.append({"role": "system", "content": "\n\n".join(system_prompts)})

    # Add initial user task
    messages.append({"role": "user", "content": task_text})

    # Add each step as assistant turn + observation as user turn
    for i, step_data in enumerate(step_events):
        assistant_text = step_data.get("assistant_text")
        if not isinstance(assistant_text, str) or not assistant_text.strip():
            continue

        # Add assistant response (reasoning + code)
        messages.append({"role": "assistant", "content": assistant_text})

        # Add observation from interpreter_result (except for the last step)
        interpreter_result = step_data.get("interpreter_result") or {}

        if i < len(step_events) - 1:
            # Format observation from interpreter execution
            output = interpreter_result.get("output")
            error = interpreter_result.get("error")

            if error:
                obs_content = json.dumps({"error": error})
            elif output:
                obs_content = json.dumps({"output": output})
            else:
                # No output/error means code ran silently - still include as empty observation
                continue

            messages.append({"role": "user", "content": obs_content})

    # Must have at least system + user + one assistant message
    if len(messages) < 3:
        return None

    return messages


def truncate_messages(messages: List[dict], max_chars: int) -> List[dict]:
    """Truncate a conversation to fit within max_chars.

    Keeps system prompt and task, then as many assistant/user turns as fit.
    Truncates from the end (removes later turns first).
    """
    if not messages:
        return messages

    total_chars = sum(len(m.get("content", "")) for m in messages)
    if total_chars <= max_chars:
        return messages

    # Always keep system (if present) and first user message
    result = []
    chars_used = 0

    # Keep system message
    if messages[0]["role"] == "system":
        result.append(messages[0])
        chars_used += len(messages[0].get("content", ""))
        messages = messages[1:]

    # Keep first user message (the task)
    if messages and messages[0]["role"] == "user":
        result.append(messages[0])
        chars_used += len(messages[0].get("content", ""))
        messages = messages[1:]

    # Add as many remaining turns as fit
    for msg in messages:
        msg_chars = len(msg.get("content", ""))
        if chars_used + msg_chars > max_chars:
            break
        result.append(msg)
        chars_used += msg_chars

    # Must end with assistant message for training
    if result and result[-1]["role"] != "assistant":
        # Remove last user message if conversation ends with user
        while result and result[-1]["role"] != "assistant":
            result.pop()

    return result if len(result) >= 3 else []


def build_trace_dataset(
    trace_root: str,
    max_samples: Optional[int],
    seed: int,
    trace_format: TraceFormat = TraceFormat.REACT,
    max_chars: Optional[int] = None,
) -> Dataset:
    """Create a HuggingFace Dataset from all .trace.json files under trace_root.

    Args:
        trace_root: Directory to search for .trace.json files
        max_samples: Maximum number of samples to include (None for all)
        seed: Random seed for shuffling
        trace_format: Format of traces - REACT or CODEACT
        max_chars: Max characters per conversation (truncates if exceeded)
    """
    trace_paths = sorted(Path(trace_root).rglob("*.trace.json"))
    if not trace_paths:
        raise SystemExit(f"No .trace.json files found under {trace_root}")

    # Select loader based on format
    if trace_format == TraceFormat.CODEACT:
        loader = load_codeact_messages
    else:
        loader = load_trace_messages

    examples = []
    for trace_path in trace_paths:
        messages = loader(trace_path)
        if messages:
            # Truncate if max_chars specified
            if max_chars:
                messages = truncate_messages(messages, max_chars)
            if messages:
                examples.append({"messages": messages, "source": str(trace_path)})

    if not examples:
        raise SystemExit("No usable traces found (missing task or assistant text).")

    dataset = Dataset.from_list(examples)
    if max_samples:
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    return dataset
