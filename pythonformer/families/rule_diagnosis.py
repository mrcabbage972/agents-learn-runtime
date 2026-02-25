import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from pythonformer.codeact.tool import Tool
from pythonformer.families.base import TaskData, TaskResult

from ..config import RuleDiagnosisConfig, RuleFamily
from .base import ToolSpec

# --- 1. Data Structures ---


class RuleDiagnosisPublic(BaseModel):
    probe_budget: int
    family_hint: Optional[str] = None
    x_domain: Dict[str, Any]
    notes: Optional[str] = None


class RuleDiagnosisPrivate(BaseModel):
    # flexible dict to handle {a, b} or {m, a, b} etc.
    params: Dict[str, Any]


class RuleDiagnosisReference(BaseModel):
    params: Dict[str, Any]


class RuleDiagnosisTaskData(
    TaskData[RuleDiagnosisPublic, RuleDiagnosisPrivate, RuleDiagnosisReference]
):
    pass


@dataclass
class RuleDiagnosisEnv:
    family: str
    params: Dict[str, Any]
    probe_budget: int

    probes_used: int = 0
    solved: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    _PROMPT_TEMPLATE = """
Goal
- Rule Diagnosis: Determine the hidden parameters of a function f(x) by probing it with inputs.

What you are given:
- Family Hint: {family}
- Probe Budget: {probe_budget} calls to test_input(x)
- Domain Info: {x_domain}
{notes}

Rules:
- You must determine the hidden parameters (e.g. 'a' and 'b' for linear functions).
- Use `test_input(x)` to get y values. 
- You have a strict limit on probes. Choose your inputs wisely.
- Once confident, use `submit(params)` to verify your answer.
"""

    @classmethod
    def from_task(cls, task_dict: dict[str, Any]) -> "RuleDiagnosisEnv":
        # We extract the family from the private params
        task = RuleDiagnosisTaskData.model_validate(task_dict)
        family = task.private.params.get("family", "unknown")

        return cls(
            family=family,
            params=task.private.params,
            probe_budget=task.public.probe_budget,
        )

    def get_goal_prompt(self) -> str:
        # If family_hint is None, we might hide the exact family,
        # but for this logic we usually expose it.
        return self._PROMPT_TEMPLATE.format(
            family=self.family,
            probe_budget=self.probe_budget,
            x_domain="Integers",  # Simplified for prompt, could read from public.x_domain
            notes="",
        )

    def get_tools(self) -> List[Tool]:
        return [TestInputTool(self), SubmitRuleTool(self)]

    def evaluate(self) -> TaskResult:
        return TaskResult(
            is_solved=self.solved,
            score=1.0 if self.solved else 0.0,
            metrics={
                "probes_used": self.probes_used,
                "probe_budget": self.probe_budget,
                "efficiency": 1.0 - (self.probes_used / self.probe_budget)
                if self.probe_budget > 0
                else 0,
                "family": self.family,
            },
        )

    async def test_input(self, x: int) -> int:
        async with self._lock:
            if self.probes_used >= self.probe_budget:
                raise RuntimeError(f"Probe budget exceeded ({self.probe_budget})")

            self.probes_used += 1

            if self.family == "linear_int":
                # y = ax + b
                a = self.params["a"]
                b = self.params["b"]
                return (a * x) + b

            elif self.family == "affine_mod":
                # y = (ax + b) % m
                m = self.params["m"]
                a = self.params["a"]
                b = self.params["b"]
                return ((a * x) + b) % m

            else:
                raise NotImplementedError(f"Unknown rule family: {self.family}")

    async def submit(self, params: dict) -> bool:
        async with self._lock:
            is_correct = True
            for k, v in self.params.items():
                if k == "family":
                    continue
                # We cast to int to ensure type safety (e.g. 1 vs 1.0)
                if k not in params or int(params[k]) != int(v):
                    is_correct = False
                    break

            self.solved = is_correct
            return is_correct


class TestInputTool(Tool):
    name: str = "test_input"
    doc: str = "Query the hidden function f(x) and return y. Consumes 1 budget."
    arg_doc: dict[str, str] = {"x": "integer input"}

    SPEC = ToolSpec(
        name="test_input",
        signature="test_input(x: int) -> int",
        description="Query the hidden function f(x) and return y.",
        args_schema={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
        returns_schema={"type": "integer"},
    )

    def __init__(self, env: RuleDiagnosisEnv):
        self._env = env
        super().__init__()

    async def run(self, x: int) -> int:
        return await self._env.test_input(x)


class SubmitRuleTool(Tool):
    name: str = "submit"
    doc: str = "Submit your hypothesized parameters. Returns True if correct."
    arg_doc: dict[str, str] = {
        "params": "a JSON string with a dictionary of parameters (e.g. `{'a': 1, 'b': 2}`)"
    }

    SPEC = ToolSpec(
        name="submit",
        signature="submit(params: object) -> {ok: bool}",
        description="Submit your hypothesized parameters. Returns true iff correct.",
        args_schema={
            "type": "object",
            "properties": {"params": {"type": "object"}},
            "required": ["params"],
        },
        returns_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
    )

    def __init__(self, env: RuleDiagnosisEnv):
        self._env = env
        super().__init__()

    async def run(self, params: str) -> bool:
        # Agent passes a JSON string, we parse it before sending to Env
        try:
            parsed_params = json.loads(params)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in params: {e}") from e

        ok = await self._env.submit(parsed_params)
        return ok


def sample_rule_diagnosis_instance(
    seed: int, cfg: RuleDiagnosisConfig
) -> RuleDiagnosisTaskData:
    rng = random.Random(seed)
    fam: RuleFamily = rng.choice(cfg.families)
    budget = rng.randint(cfg.probe_budget_range[0], cfg.probe_budget_range[1])

    params: Dict[str, Any] = {}
    public: Dict[str, Any] = {}

    if fam == "linear_int":
        a = rng.randint(cfg.linear_a_range[0], cfg.linear_a_range[1])
        if a == 0 and rng.random() < 0.7:
            a = 1
        b = rng.randint(cfg.linear_b_range[0], cfg.linear_b_range[1])
        params = {"family": fam, "a": a, "b": b}

        public = {
            "probe_budget": budget,
            "family_hint": fam,
            "x_domain": {"type": "int", "suggested_range": [-20, 20]},
        }

    elif fam == "affine_mod":
        m = rng.choice(cfg.mod_m_choices)
        a = rng.randint(cfg.mod_a_range[0], min(cfg.mod_a_range[1], m - 1))
        b = rng.randint(cfg.mod_b_range[0], min(cfg.mod_b_range[1], m - 1))
        params = {"family": fam, "m": m, "a": a, "b": b}

        public = {
            "probe_budget": budget,
            "family_hint": fam,
            "x_domain": {"type": "int", "suggested_range": [0, 2 * m]},
            "notes": "Outputs are integers; modulus may apply.",
        }

    else:
        raise ValueError(f"Unknown rule family: {fam}")

    return RuleDiagnosisTaskData(
        family="rule_diagnosis",
        seed=seed,
        difficulty={
            "rule_family": fam,
            "probe_budget": budget,
        },
        public=RuleDiagnosisPublic(**public),
        private=RuleDiagnosisPrivate(params=params),
        reference=RuleDiagnosisReference(params=params),
    )
