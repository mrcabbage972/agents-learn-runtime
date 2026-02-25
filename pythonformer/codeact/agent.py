import json
import logging
import re
from datetime import datetime, timezone

from pythonformer.codeact.events import (
    ErrorEvent,
    EventBus,
    FinishEvent,
    ModelCallEvent,
    ModelResponseEvent,
    StartEvent,
    StepEvent,
    SystemPromptEvent,
)
from pythonformer.codeact.interpreter import AsyncInterpreter
from pythonformer.codeact.tool import Tool
from pythonformer.llm import LlmProxy, TokenUsage

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """

You are a CodeAct-style autonomous agent.

You solve tasks by alternating between:
1. Natural-language reasoning (plain text), and
2. Executable simple Python code blob (inside fenced code blocks).

Each step (output) can include at most 1 (one) code block.

Be concise in your reasoning and code.

When you are finished solving the task, ensure that you output a Python code block which calls the `finish` tool. 
Call the `finish` tool ONLY after completely solving the task, NOT on every turn.

Execution rules:
- Python code blocks are executed sequentially.
- Only expressions that are printed or explicitly returned are visible to you.
- Variable assignments alone do NOT produce observable output.
- Do not use variable names that conflict with tool names.

Output discipline:
- If a value will be needed for later reasoning or decisions, you MUST print it
  (e.g., via `print(...)`) or make it the final expression in the code block.
- Do not rely on implicit interpreter state visibility.

Tool usage:
- All tool calls must occur inside Python code blocks.
- Do not fabricate tool outputs; rely only on observed execution results.

Error handling:
- If execution fails or a needed value is missing, explain why and rerun with
  corrected code.

Completion:
- When the task is complete, provide a final plain-text answer.
- Do not emit further code after completion.

Output Structure:
You must strictly follow this format for every single turn:

1. Reflect upon the previous observation, explain your reasoning, what you want to verify, or what the next step is.
2. A single executable Python block.

You prioritize observability and correctness over brevity.
"""

FEW_SHOT_EXAMPLES = """--- EXAMPLE: PERSISTENT STATE ---
Task: Store items and sum values.

Turn 1
Assistant:
1. I will initialize the global list `items` and print its length and contents.
```python
items = [10, 20]

def foo(items):
    return items + items

print(len(items))
```
User: {"output": 2}

Turn 2 User: Python interpreter globals: ["items"] 

Assistant:

I see the variable `items` and the function `foo` in the interpreter globals. I will reuse them directly.

```python
items = foo(items)
total = sum(items)
print(f"Total: {total}")

finish()
```
User: {"output": "Total: 60"} --- EXAMPLE END --- 

--- EXAMPLE: ITERATIVE EXPLORATION ---
Task: Inspect 10 items from a list, 2 at a time.

Turn 1
Assistant:
1. I will define a helper function `batch_inspect` to handle the logic. Then I will run the first batch.
```python
import json

# Setup data
items = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
inspected = []

def batch_inspect():
    # Calculate indices based on current state
    start = len(inspected)
    # Stop if done
    if start >= len(items):
        print("Done!")
        return
        
    # Process next 2
    batch = items[start : start + 2]
    for x in batch:
        # Mock inspection logic
        inspected.append(x)
        print(f"Inspected {x}")
        
    print(f"Progress: {len(inspected)}/{len(items)}")

# Run first time
batch_inspect()
```
User: {"output": "Inspected 100\nInspected 200\nProgress: 2/10"}

Turn 2 User: ACTIVE GLOBALS: ['items', 'inspected', 'batch_inspect'] 

Assistant:

I see batch_inspect is defined. I will simply call it to process the next items.
```python
batch_inspect()
```
User: {"output": "Inspected 300\nInspected 400\nProgress: 4/10"} ---


--- EXAMPLE: BATCH PROCESSING ---
Task: Process a list of numbers 3 at a time, keeping only even numbers, and report the running sum.

Turn 1
Assistant:
1. I will define a "fat" helper function `run_step()` that handles the fetching, filtering, and reporting all in one go.
```python
# Setup
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
processed_count = 0
valid_evens = []

def run_step():
    global processed_count
    
    # 1. FETCH / ACT
    batch = numbers[processed_count : processed_count + 3]
    if not batch:
        print("Done.")
        return

    # 2. UPDATE STATE
    for n in batch:
        print(f"Saw: {n}")
        if n % 2 == 0:
            valid_evens.append(n)
    
    processed_count += len(batch)
    
    # 3. ANALYZE & REPORT (Inside the function!)
    current_sum = sum(valid_evens)
    print(f"--- Report ---")
    print(f"Progress: {processed_count}/{len(numbers)}")
    print(f"Valid Evens: {valid_evens}")
    print(f"Running Sum: {current_sum}")

# Execute first step
run_step()

"""


class FinishTool(Tool):
    name: str = "finish"
    doc: str = "Call when the task execution is finished"
    arg_doc: dict[str, str] = {}

    is_finished: bool = False

    def __init__(self):
        super().__init__()

    async def run(self) -> None:
        self.is_finished = True


def extract_code_blocks(text: str) -> list[str]:
    """
    Extract all fenced code blocks from text.

    Returns a list of code strings without the backticks or language tags.
    """
    pattern = re.compile(
        r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)```",
        re.DOTALL,
    )
    return [block.strip() for block in pattern.findall(text)]


class CodeAct:
    def __init__(
        self,
        llm_proxy: LlmProxy,
        max_num_turns: int,
        tools: list[Tool],
        bus: EventBus,
        persistent_state: bool = True,
        max_tool_calls: int | None = None,
    ):
        self.llm_proxy = llm_proxy
        self.max_num_turns = max_num_turns
        self.finish_tool = FinishTool()

        self.tools = tools + [self.finish_tool]

        self.tool_prompt = "Available tools:\n" + "\n".join(
            [x.full_python_doc() for x in self.tools]
        )
        if max_tool_calls is not None:
            self.tool_prompt += f"\n The maximum number of tool calls allowed per turn is {max_tool_calls}.\n"

        runtime_note = (
            """Runtime state: PERMANENT.

        
    
    CRITICAL RULES FOR THIS SESSION:
    1. Globals persist eternally. Once you define `x = 1` or `import math`, it is available forever.
    2. NEVER re-import libraries.
    3. NEVER paste code from previous steps. 
    4. Assume the environment is already set up with previous variables.

    SUPER IMPORTANT: FOR THE LOVE OF CODE, DO NOT DUPLICATE CODE YOU WROTE IN PREVIOUS STEPS!!!

    MORE RULES:
    1. Write reusable helper functions and then use them in subsequent steps.
    2. Planning: think of which reusable functions you may need and implement them ahead of time. Code efficiency is very important.

            """
            if persistent_state
            else "Runtime state: Python variables DO NOT persist across code blocks; each code block runs in a fresh environment."
        )

        self.message_history = [
            {"content": SYSTEM_PROMPT, "role": "system"},
            {"role": "system", "content": runtime_note},
        ]

        if persistent_state:
            self.message_history.append(
                {"role": "system", "content": FEW_SHOT_EXAMPLES}
            )

        self.message_history.append({"content": self.tool_prompt, "role": "system"})

        self.interpreter = AsyncInterpreter(
            persistent_state=persistent_state,
            max_tool_calls=max_tool_calls,
        )
        for tool in self.tools:
            self.interpreter.register_tool(tool.name, tool)
        self.bus = bus
        self.persistent_state = persistent_state

        logger.info(f"Starting agent with {len(self.tools)} tools")

    async def run(self, prompt: str):
        total_usage = TokenUsage(0, 0, 0)
        interpreter_globals = {}
        try:
            logger.info("Starting task run")

            await self.bus.emit(StartEvent(task=prompt))
            await self.bus.emit(
                SystemPromptEvent(
                    prompts=[
                        message["content"]
                        for message in self.message_history
                        if message.get("role") == "system"
                    ]
                )
            )

            self.message_history.append({"content": prompt, "role": "user"})

            for turn_idx in range(self.max_num_turns):
                if self.finish_tool.is_finished:
                    logger.info("Task execution finished")
                    await self.bus.emit(
                        FinishEvent(reason="finish_tool", token_usage=total_usage)
                    )
                    return

                message_history = list(self.message_history)
                if self.persistent_state:
                    message_history.append(
                        {
                            "role": "user",
                            "content": "Python interpreter globals: "
                            + json.dumps(list(interpreter_globals.keys())),
                        }
                    )

                call_started_at = datetime.now(timezone.utc)
                try:
                    completion = await self.llm_proxy.complete(messages=message_history)
                except Exception as llm_error:
                    # Catch LLM-specific failures (timeouts, context length errors)
                    # Log them, tell the agent, and burn a turn rather than crashing.
                    error_msg = f"LLM Call Failed: {str(llm_error)}"
                    logger.warning(error_msg)

                    await self.bus.emit(ErrorEvent(error=error_msg))

                    # Inject the error into history so the agent sees it and can retry
                    self.message_history.append({"role": "user", "content": error_msg})
                    continue

                if completion.finish_reason != "stop":
                    # Treat unexpected finish reasons as a recoverable error too
                    error_msg = f"LLM ended unexpectedly (reason: {completion.finish_reason}). Please continue."
                    await self.bus.emit(ErrorEvent(error=error_msg))
                    self.message_history.append({"role": "user", "content": error_msg})
                    continue

                response_msg = completion.text
                completion_token_usage = completion.token_usage
                total_usage = total_usage + completion_token_usage
                await self.bus.emit(
                    ModelCallEvent(
                        timestamp=call_started_at.isoformat(),
                        prompt_tokens=completion_token_usage.prompt_tokens,
                    )
                )
                await self.bus.emit(
                    ModelResponseEvent(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        completion_tokens=completion_token_usage.completion_tokens,
                        total_tokens=completion_token_usage.total_tokens,
                    )
                )

                self.message_history.append(
                    {"role": "assistant", "content": response_msg}
                )

                code_blocks = extract_code_blocks(response_msg)
                code = None
                interpreter_result = None
                if len(code_blocks) > 1:
                    error_msg = "Error: You emitted multiple code blocks. Please provide only one executable code block per turn."

                    # 1. Emit the error event
                    await self.bus.emit(ErrorEvent(error=error_msg))

                    # 2. Add error to history so the LLM sees it next turn
                    self.message_history.append({"role": "user", "content": error_msg})

                    # 3. Set a dummy result for the StepEvent logs
                    interpreter_result = {"error": error_msg}
                else:
                    if len(code_blocks) > 0:
                        code = code_blocks[0]

                        interpreter_result = await self.interpreter.run_code(
                            code_blocks[0]
                        )

                        if not interpreter_result["success"]:
                            error_msg = interpreter_result.get(
                                "error", "Unknown execution error"
                            )
                            await self.bus.emit(ErrorEvent(error=error_msg))

                        interpreter_globals = interpreter_result.pop("globals", {})
                        self.message_history.append(
                            {
                                "role": "user",
                                "content": json.dumps(interpreter_result),
                            }
                        )

                await self.bus.emit(
                    StepEvent(
                        turn=turn_idx,
                        assistant_text=response_msg,
                        code=code,
                        interpreter_result=interpreter_result,
                        token_usage=completion_token_usage,
                    )
                )
            await self.bus.emit(
                FinishEvent(reason="max_turns", token_usage=total_usage)
            )
        except Exception as e:
            await self.bus.emit(ErrorEvent(error=str(e)))
            await self.bus.emit(FinishEvent(reason="error", token_usage=total_usage))
            raise e
