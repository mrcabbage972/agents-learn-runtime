from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from pythonformer.config import (
    LLMConfig,
)
from pythonformer.families.knapsack import KnapsackEnv
from pythonformer.families.navigation import NavigationEnv
from pythonformer.families.rule_diagnosis import RuleDiagnosisEnv

FAMILY_ENVS = {
    "knapsack": KnapsackEnv,
    "navigation": NavigationEnv,
    "rule_diagnosis": RuleDiagnosisEnv,
}


class AgentConfig(BaseModel):
    name: str
    agent_type: Literal["codeact", "react"] = "codeact"
    llm: LLMConfig
    max_turns: int = Field(12, ge=1)
    timeout_s: float = Field(300.0, ge=1.0)
    persistent_state: bool = True
    max_tool_calls: int | None = None


class BenchmarkConfig(BaseModel):
    output_dir: Path
    run_name: str = "codeact"
    seed_start: int = 0
    max_concurrent_runs: int = Field(4, ge=1)
    cache_db_path: Path
    agents: list[AgentConfig]
    task_families: list[str] | None = None

    @field_validator("task_families")
    @classmethod
    def validate_task_families(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("task_families must contain at least one family.")
        invalid = sorted(set(value) - set(FAMILY_ENVS.keys()))
        if invalid:
            raise ValueError(
                "task_families must be one of: "
                + ", ".join(sorted(FAMILY_ENVS.keys()))
                + f". Invalid: {', '.join(invalid)}"
            )
        return value
