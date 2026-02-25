import json
from typing import Any

# --- Import Rich Components ---
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.theme import Theme

from pythonformer.codeact.events import (
    ErrorEvent,
    Event,
    FinishEvent,
    StartEvent,
    StepEvent,
)


class CliMessageLogger:
    def __init__(self):
        # Custom theme to keep things semantic but soft on the eyes
        self.theme = Theme(
            {
                "info": "cyan",
                "warning": "yellow",
                "error": "bold red",
                "success": "bold green",
                "code.out": "dim white",
            }
        )
        self.console = Console(theme=self.theme)

    async def handle(self, event: Event) -> None:
        """Pattern matches the event type to render the appropriate UI element."""

        match event:
            case StartEvent(task=task):
                self.console.print()
                self.console.print(
                    Panel(
                        f"[bold white]{task}[/bold white]",
                        title="[success]🚀 Agent Task Started[/success]",
                        border_style="green",
                    )
                )
                self.console.print()

            case StepEvent(
                turn=turn,
                assistant_text=text,
                code=code,
                interpreter_result=result,
                token_usage=token_usage,
            ):
                # 1. Header for the turn
                header = f"[bold cyan]Step {turn}[/bold cyan]"
                if token_usage:
                    header += (
                        f" [info](tokens: prompt {token_usage.prompt_tokens}, "
                        f"completion {token_usage.completion_tokens}, "
                        f"total {token_usage.total_tokens})[/info]"
                    )
                self.console.print(Rule(header))

                # 2. The Assistant's Thought (rendered as Markdown)
                if text:
                    self.console.print(Markdown(text))
                    self.console.print()  # Spacer

                # 3. The Code Block (Syntax Highlighted)
                if False:  # code:
                    panel = Panel(
                        Syntax(code, "python", theme="monokai", line_numbers=True),
                        title="[bold blue]💻 Generated Code[/bold blue]",
                        border_style="blue",
                        expand=False,
                    )
                    self.console.print(panel)

                # 4. The Execution Result (distinct from the AI's thought)
                if result:
                    # Format dict nicely, or just print specific keys if you know them (like 'stdout')
                    formatted_result = self._format_result(result)
                    self.console.print(
                        Panel(
                            formatted_result,
                            title="[bold magenta]📉 Execution Output[/bold magenta]",
                            border_style="dim magenta",
                            expand=False,
                        )
                    )

                self.console.print()  # Bottom Spacer

            case FinishEvent(reason=reason, token_usage=token_usage):
                self.console.print(Rule(style="bold green"))
                if reason == "error":
                    self.console.print("[error]❌ Task stopped due to error.[/error]")
                else:
                    self.console.print(
                        f"[success]✅ Task Finished (Reason: {reason})[/success]"
                    )
                if token_usage:
                    self.console.print(
                        "[info]Total tokens: "
                        f"prompt {token_usage.prompt_tokens}, "
                        f"completion {token_usage.completion_tokens}, "
                        f"total {token_usage.total_tokens}[/info]"
                    )
                self.console.print()

            case ErrorEvent(error=msg):
                self.console.print(
                    Panel(
                        msg, title="[error]🔥 System Error[/error]", border_style="red"
                    )
                )

    def _format_result(self, result: dict[str, Any]) -> str:
        """Helper to make interpreter results readable."""
        # If the result has standard output, prioritize showing that cleanly
        output_str = ""
        if "stdout" in result and result["stdout"]:
            output_str += f"[code.out]{result['stdout'].strip()}[/code.out]\n"
        if "stderr" in result and result["stderr"]:
            output_str += f"[red]{result['stderr'].strip()}[/red]\n"

        # If no stdout/stderr, just pretty print the whole dict
        if not output_str:
            output_str = json.dumps(result, indent=2, default=str)

        return output_str.strip()
