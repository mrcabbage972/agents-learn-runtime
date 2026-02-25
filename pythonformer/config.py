from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Tuple

from pydantic import BaseModel, Field, model_validator

FamilyName = Literal["knapsack", "navigation", "rule_diagnosis"]


class LLMConfig(BaseModel):
    """Configuration for LiteLLM calls.

    LiteLLM typically reads provider credentials from environment variables
    (e.g., OPENAI_API_KEY). This project uses `python-dotenv` to load those
    variables from a `.env` file.
    """

    model: str = "gpt-4o-mini"
    temperature: float | None = None
    max_tokens: int | None = None
    timeout_s: float = Field(60.0, ge=1.0)

    # Concurrency + resilience
    max_concurrent_requests: int = Field(8, ge=1)
    max_retries: int = Field(6, ge=0)
    backoff_base_s: float = Field(0.5, ge=0.0)
    backoff_max_s: float = Field(30.0, ge=0.0)
    jitter_s: float = Field(0.2, ge=0.0)

    # Persistent cache path (defaults under out_dir/cache/llm_cache.sqlite)
    cache_enabled: bool = True

    @model_validator(mode="after")
    def _check_backoff(self):
        if self.backoff_max_s < self.backoff_base_s:
            raise ValueError("backoff_max_s must be >= backoff_base_s")
        return self


class KnapsackConfig(BaseModel):
    num_tasks: int = Field(100, ge=0)

    n_items_range: Tuple[int, int] = (10, 30)
    weight_range: Tuple[int, int] = (1, 20)
    value_range: Tuple[int, int] = (1, 50)

    classes: List[str] = Field(default_factory=lambda: ["A", "B", "C"])
    num_allowed_classes: int = 10

    # Capacity is sampled as ratio * total_weight(allowed_class_items)
    capacity_ratio_range: Tuple[float, float] = (0.35, 0.6)

    # Re-sample instances until the optimal solution selects >= 1 item.
    force_nonempty_optimum: bool = True

    @model_validator(mode="after")
    def _validate_ranges(self):
        lo, hi = self.n_items_range
        if lo <= 0 or hi < lo:
            raise ValueError("n_items_range must be (lo>0, hi>=lo)")
        for name, rng in [
            ("weight_range", self.weight_range),
            ("value_range", self.value_range),
        ]:
            rlo, rhi = rng
            if rlo <= 0 or rhi < rlo:
                raise ValueError(f"{name} must be (lo>0, hi>=lo)")
        clo, chi = self.capacity_ratio_range
        if not (0.0 < clo <= chi <= 1.0):
            raise ValueError("capacity_ratio_range must be within (0,1]")
        if not self.classes:
            raise ValueError("classes must be non-empty")
        return self


class NavigationConfig(BaseModel):
    num_tasks: int = Field(100, ge=0)

    # K controls the shortest path length from start to goal.
    horizon_range: Tuple[int, int] = (8, 25)

    # Additional nodes to add beyond the base path nodes.
    extra_nodes_range: Tuple[int, int] = (0, 15)

    # Additional edges added as a fraction of the maximum possible extra edges.
    extra_edge_factor_range: Tuple[float, float] = (0.1, 0.35)

    # Suggested step budget exposed in `public`.
    max_steps_multiplier: float = 3.0

    @model_validator(mode="after")
    def _validate_ranges(self):
        hlo, hhi = self.horizon_range
        if hlo <= 0 or hhi < hlo:
            raise ValueError("horizon_range must be (lo>0, hi>=lo)")
        nlo, nhi = self.extra_nodes_range
        if nlo < 0 or nhi < nlo:
            raise ValueError("extra_nodes_range must be (lo>=0, hi>=lo)")
        elo, ehi = self.extra_edge_factor_range
        if not (0.0 <= elo <= ehi <= 1.0):
            raise ValueError("extra_edge_factor_range must be within [0,1]")
        return self


RuleFamily = Literal["linear_int", "affine_mod"]


class RuleDiagnosisConfig(BaseModel):
    num_tasks: int = Field(100, ge=0)

    families: List[RuleFamily] = Field(
        default_factory=lambda: ["linear_int", "affine_mod"]
    )

    # Linear: y = a*x + b
    linear_a_range: Tuple[int, int] = (-7, 7)
    linear_b_range: Tuple[int, int] = (-20, 20)

    # Mod: y = (a*x + b) mod m
    mod_m_choices: List[int] = Field(default_factory=lambda: [5, 7, 11, 13, 17, 19])
    mod_a_range: Tuple[int, int] = (1, 18)
    mod_b_range: Tuple[int, int] = (0, 18)

    # Allowed test_input calls.
    probe_budget_range: Tuple[int, int] = (3, 10)

    @model_validator(mode="after")
    def _validate_ranges(self):
        if not self.families:
            raise ValueError("families must be non-empty")

        pa, pb = self.probe_budget_range
        if pa <= 0 or pb < pa:
            raise ValueError("probe_budget_range must be (lo>0, hi>=lo)")

        if not self.mod_m_choices:
            raise ValueError("mod_m_choices must be non-empty")

        return self


class Settings(BaseModel):
    """Top-level generator settings.

    The generator is designed to be resumable: already generated tasks are skipped.
    """

    out_dir: Path = Path("out")
    run_name: str = "run"
    seed_start: int = 0

    llm_enabled: bool = False

    # Number of concurrent *task workers*.
    max_workers: int = Field(16, ge=1)

    llm: LLMConfig = Field(default_factory=LLMConfig)  # type: ignore
    knapsack: KnapsackConfig = Field(default_factory=KnapsackConfig)  # type: ignore
    navigation: NavigationConfig = Field(
        default_factory=NavigationConfig  # type: ignore
    )
    rule_diagnosis: RuleDiagnosisConfig = Field(
        default_factory=RuleDiagnosisConfig  # type: ignore
    )

    cache_db_path: Path | None = Path("cache/llm_cache.sqlite")

    @model_validator(mode="after")
    def _normalize_paths(self):
        self.out_dir = Path(self.out_dir)
        return self

    def effective_cache_path(self) -> Path:
        if self.cache_db_path is not None:
            return self.cache_db_path
        return self.out_dir / "cache" / "llm_cache.sqlite"

    @classmethod
    def load(cls, path: str | Path) -> "Settings":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
