import asyncio
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel

from pythonformer.codeact.tool import Tool
from pythonformer.config import NavigationConfig
from pythonformer.families.base import TaskData, TaskResult, ToolSpec, opaque_id

# --- 1. Data Structures ---


class NavigationPublic(BaseModel):
    start_node: str
    goal_node: str
    max_steps: int
    notes: Optional[str] = None


class NavigationPrivate(BaseModel):
    # Adjacency list: node -> list of neighbors
    adjacency: Dict[str, List[str]]
    # Layer info used for generation (debugging/analysis), optional at runtime
    layer: Dict[str, int]
    one_shortest_path: List[str]


class NavigationReference(BaseModel):
    shortest_path_length: int
    one_shortest_path: List[str]


class NavigationTaskData(
    TaskData[NavigationPublic, NavigationPrivate, NavigationReference]
):
    pass


@dataclass
class NavigationEnv:
    # Static Task Data
    adj: Dict[str, List[str]]
    goal_node: str
    max_steps: int
    reference: NavigationReference

    # Dynamic State
    current_node: str
    steps_taken: int = 0
    path_history: List[str] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    _PROMPT_TEMPLATE = """
Goal
- Navigation: Traverse the graph to reach the goal node.

What you are given:
- Start Node: {start_node}
- Goal Node: {goal_node}
- Max Steps: {max_steps}

Rules:
- Use `scan()` to see connected neighbors.
- Use `move(node_id)` to travel.
- The graph is cyclic and opaque.
- You succeed when you reach the goal node within the step limit.
"""

    @classmethod
    def from_task(cls, task_dict: dict[str, Any]) -> "NavigationEnv":
        task = NavigationTaskData.model_validate(task_dict)
        return cls(
            adj=task.private.adjacency,
            goal_node=task.public.goal_node,
            max_steps=task.public.max_steps,
            reference=task.reference,
            current_node=task.public.start_node,
            path_history=[task.public.start_node],
        )

    def get_goal_prompt(self) -> str:
        return self._PROMPT_TEMPLATE.format(
            start_node=self.path_history[0],
            goal_node=self.goal_node,
            max_steps=self.max_steps,
        )

    def get_tools(self) -> List[Tool]:
        return [ScanTool(self), MoveTool(self)]

    def evaluate(self) -> TaskResult:
        is_solved = self.current_node == self.goal_node

        # Optimal steps = edges = nodes - 1
        optimal_len = self.reference.shortest_path_length
        actual_len = len(self.path_history) - 1  # steps taken

        if is_solved:
            # Score: 1.0 if optimal, decreasing if inefficient
            # Avoid division by zero if optimal is 0 (start==goal)
            if optimal_len == 0:
                score = 1.0
            else:
                score = max(0.0, min(1.0, optimal_len / max(1, actual_len)))
        else:
            score = 0.0

        return TaskResult(
            is_solved=is_solved,
            score=score,
            metrics={
                "steps_taken": actual_len,
                "optimal_steps": optimal_len,
                "visited_unique": len(set(self.path_history)),
                "step_limit": self.max_steps,
            },
        )

    async def scan(self) -> str:
        async with self._lock:
            # Return neighbors sorted for determinism
            return ",".join(sorted(self.adj.get(self.current_node, [])))

    async def move(self, node_id: str) -> None:
        async with self._lock:
            if self.steps_taken >= self.max_steps:
                raise RuntimeError(f"Max steps ({self.max_steps}) exceeded.")

            neighbors = self.adj.get(self.current_node, [])
            if node_id not in neighbors:
                raise ValueError(
                    f"Node {node_id} is not adjacent to {self.current_node}."
                )

            self.current_node = node_id
            self.steps_taken += 1
            self.path_history.append(node_id)


class ScanTool(Tool):
    name: str = "scan"
    doc: str = (
        "Return a comma separated list of neighboring node IDs from the current node."
    )
    arg_doc: dict[str, str] = {}

    SPEC = ToolSpec(
        name="scan",
        signature="scan() -> [str]",
        description="Return a list of neighboring node IDs from the current node.",
        args_schema={"type": "object", "properties": {}},
        returns_schema={"type": "array", "items": {"type": "string"}},
    )

    def __init__(self, env: NavigationEnv):
        self._env = env
        super().__init__()

    async def run(self) -> str:
        return await self._env.scan()


class MoveTool(Tool):
    name: str = "move"
    doc: str = "Move to an adjacent node."
    arg_doc: dict[str, str] = {"node_id": "the target node id"}

    SPEC = ToolSpec(
        name="move",
        signature="move(node_id: str) -> void",
        description="Move to an adjacent node.",
        args_schema={
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
        returns_schema={"type": "null"},
    )

    def __init__(self, env: NavigationEnv):
        self._env = env
        super().__init__()

    async def run(self, node_id: str) -> None:
        await self._env.move(node_id)


def _add_edge(adj: Dict[str, Set[str]], u: str, v: str) -> None:
    adj.setdefault(u, set()).add(v)
    adj.setdefault(v, set()).add(u)


def sample_navigation_instance(seed: int, cfg: NavigationConfig) -> NavigationTaskData:
    """Generate a cyclic navigation instance with guaranteed shortest path length K."""
    rng = random.Random(seed)

    K = rng.randint(cfg.horizon_range[0], cfg.horizon_range[1])
    extra_nodes = rng.randint(cfg.extra_nodes_range[0], cfg.extra_nodes_range[1])
    extra_edge_factor = rng.uniform(
        cfg.extra_edge_factor_range[0], cfg.extra_edge_factor_range[1]
    )

    # Base path
    base_nodes = [opaque_id(rng, prefix="node") for _ in range(K + 1)]
    start = base_nodes[0]
    goal = base_nodes[-1]

    layer: Dict[str, int] = {nid: i for i, nid in enumerate(base_nodes)}
    adj: Dict[str, Set[str]] = {nid: set() for nid in base_nodes}

    for i in range(K):
        _add_edge(adj, base_nodes[i], base_nodes[i + 1])

    # Extra nodes
    all_nodes = list(base_nodes)
    for _ in range(extra_nodes):
        nid = opaque_id(rng, prefix="node")
        # Place in any layer except the goal layer to keep a "destination" feel.
        lvl = rng.randint(0, max(0, K - 1))
        layer[nid] = lvl
        adj[nid] = set()
        all_nodes.append(nid)

        # Connect to a random existing node in same/adjacent layer.
        candidates = [x for x in all_nodes if x != nid and abs(layer[x] - lvl) <= 1]
        if not candidates:
            candidates = [start]
        parent = rng.choice(candidates)
        _add_edge(adj, nid, parent)

        # Optionally add a second connection to create cycles.
        if rng.random() < 0.35:
            candidates2 = [x for x in candidates if x != parent]
            if candidates2:
                _add_edge(adj, nid, rng.choice(candidates2))

    # Extra edges
    possible_pairs: List[Tuple[str, str]] = []
    for i in range(len(all_nodes)):
        u = all_nodes[i]
        for j in range(i + 1, len(all_nodes)):
            v = all_nodes[j]
            if v in adj[u]:
                continue
            if abs(layer[u] - layer[v]) <= 1:
                possible_pairs.append((u, v))

    rng.shuffle(possible_pairs)
    m = int(extra_edge_factor * len(possible_pairs))
    for u, v in possible_pairs[:m]:
        _add_edge(adj, u, v)

    max_steps = int(math.ceil(cfg.max_steps_multiplier * K))

    # Convert Sets to Lists for JSON serialization
    adj_json = {k: sorted(list(v)) for k, v in adj.items()}

    return NavigationTaskData(
        family="navigation",
        seed=seed,
        difficulty={
            "horizon_K": K,
            "n_nodes": len(all_nodes),
            "extra_nodes": extra_nodes,
            "extra_edge_factor": extra_edge_factor,
        },
        public=NavigationPublic(
            start_node=start,
            goal_node=goal,
            max_steps=max_steps,
            notes="Graph is cyclic; environment does not track visited history.",
        ),
        private=NavigationPrivate(
            adjacency=adj_json,
            layer=layer,
            one_shortest_path=base_nodes,
        ),
        reference=NavigationReference(
            shortest_path_length=K,
            one_shortest_path=base_nodes,
        ),
    )
