from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List

from pythonformer.config import FamilyName, Settings
from pythonformer.families.knapsack import sample_knapsack_instance
from pythonformer.families.navigation import sample_navigation_instance
from pythonformer.families.rule_diagnosis import sample_rule_diagnosis_instance


@dataclass(frozen=True)
class TaskFamilySpec:
    name: FamilyName
    num_tasks: int
    sample: Callable[[int], Any]


def build_task_families(settings: Settings) -> List[TaskFamilySpec]:
    return [
        TaskFamilySpec(
            name="knapsack",
            num_tasks=settings.knapsack.num_tasks,
            sample=partial(sample_knapsack_instance, cfg=settings.knapsack),
        ),
        TaskFamilySpec(
            name="navigation",
            num_tasks=settings.navigation.num_tasks,
            sample=partial(sample_navigation_instance, cfg=settings.navigation),
        ),
        TaskFamilySpec(
            name="rule_diagnosis",
            num_tasks=settings.rule_diagnosis.num_tasks,
            sample=partial(sample_rule_diagnosis_instance, cfg=settings.rule_diagnosis),
        ),
    ]
