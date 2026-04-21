"""Search strategies for planning."""

from __future__ import annotations

import heapq
from collections import deque
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from pydpocl.core.plan import Plan

if TYPE_CHECKING:
    from pydpocl.core.interfaces import PlanningProblem
    from pydpocl.planning.llm_policy import LLMPolicy

T = TypeVar("T")


class SearchStrategy(Protocol, Generic[T]):
    """Protocol for search strategies."""

    def add_plan(self, plan: T) -> None:
        """Add a plan to the frontier."""
        ...

    def get_next_plan(self) -> T | None:
        """Get the next plan to expand."""
        ...

    def is_empty(self) -> bool:
        """Check if the frontier is empty."""
        ...

    def size(self) -> int:
        """Get the size of the frontier."""
        ...


class BestFirstSearch:
    """Best-first search using a priority queue.

    Heap entries are (priority, counter, plan).  The monotonic counter
    guarantees a unique second element so Plan objects are never compared.
    """

    def __init__(self) -> None:
        self._frontier: list[tuple[float, int, Plan]] = []
        self._counter = 0

    def add_plan(self, plan: Plan, priority: float | None = None) -> None:
        """Add a plan to the frontier with an explicit priority (lower = better)."""
        p = priority if priority is not None else plan.cost
        heapq.heappush(self._frontier, (p, self._counter, plan))
        self._counter += 1

    def get_next_plan(self) -> Plan | None:
        """Get the next plan to expand."""
        if not self._frontier:
            return None
        _, _, plan = heapq.heappop(self._frontier)
        return plan

    def is_empty(self) -> bool:
        """Check if the frontier is empty."""
        return len(self._frontier) == 0

    def size(self) -> int:
        """Get the size of the frontier."""
        return len(self._frontier)


class BreadthFirstSearch:
    """Breadth-first search using a queue."""

    def __init__(self) -> None:
        self._frontier: deque[Plan] = deque()

    def add_plan(self, plan: Plan, priority: float | None = None) -> None:
        """Add a plan to the frontier (priority is ignored for FIFO)."""
        self._frontier.append(plan)

    def get_next_plan(self) -> Plan | None:
        """Get the next plan to expand."""
        if not self._frontier:
            return None
        return self._frontier.popleft()

    def is_empty(self) -> bool:
        """Check if the frontier is empty."""
        return len(self._frontier) == 0

    def size(self) -> int:
        """Get the size of the frontier."""
        return len(self._frontier)


class DepthFirstSearch:
    """Depth-first search using a stack."""

    def __init__(self) -> None:
        self._frontier: list[Plan] = []

    def add_plan(self, plan: Plan, priority: float | None = None) -> None:
        """Add a plan to the frontier (priority is ignored for LIFO)."""
        self._frontier.append(plan)

    def get_next_plan(self) -> Plan | None:
        """Get the next plan to expand."""
        if not self._frontier:
            return None
        return self._frontier.pop()

    def is_empty(self) -> bool:
        """Check if the frontier is empty."""
        return len(self._frontier) == 0

    def size(self) -> int:
        """Get the size of the frontier."""
        return len(self._frontier)


class LLMFrontier:
    """Best-first heap whose pop step is delegated to an LLM policy.

    The frontier keeps a standard ``(priority, counter, plan)`` heap so that
    insertions and bookkeeping match :class:`BestFirstSearch`.  When the
    planner asks for the next plan we peek the top-K entries by priority,
    serialize them, and ask the configured :class:`LLMPolicy` to pick one.
    The chosen plan is removed (and the others remain in the frontier).
    """

    def __init__(
        self,
        policy: "LLMPolicy",
        problem: "PlanningProblem",
        top_k: int = 10,
    ) -> None:
        self._policy = policy
        self._problem = problem
        self._top_k = max(1, int(top_k))
        self._frontier: list[tuple[float, int, Plan]] = []
        self._counter = 0

    def add_plan(self, plan: Plan, priority: float | None = None) -> None:
        """Add a plan to the frontier with an explicit priority (lower = better)."""
        p = priority if priority is not None else plan.cost
        heapq.heappush(self._frontier, (p, self._counter, plan))
        self._counter += 1

    def get_next_plan(self) -> Plan | None:
        if not self._frontier:
            return None
        if len(self._frontier) == 1 or self._top_k == 1:
            _, _, plan = heapq.heappop(self._frontier)
            return plan

        # Pop top-K (cheap with heappop) then re-push the losers.
        k = min(self._top_k, len(self._frontier))
        popped: list[tuple[float, int, Plan]] = [
            heapq.heappop(self._frontier) for _ in range(k)
        ]
        candidates = [(prio, plan) for prio, _, plan in popped]

        chosen = self._policy.select_node(candidates, self._problem)

        chosen_id = chosen.id
        for entry in popped:
            if entry[2].id != chosen_id:
                heapq.heappush(self._frontier, entry)
        return chosen

    def is_empty(self) -> bool:
        return len(self._frontier) == 0

    def size(self) -> int:
        return len(self._frontier)


def create_search_strategy(
    strategy_name: str,
    *,
    policy: "LLMPolicy | None" = None,
    problem: "PlanningProblem | None" = None,
    top_k: int = 10,
) -> SearchStrategy[Plan]:
    """Factory function for creating search strategies.

    The ``llm`` strategy additionally requires ``policy`` and ``problem``
    so the LLM can see the planning context when picking a frontier node.
    """
    if strategy_name == "llm":
        if policy is None or problem is None:
            raise ValueError(
                "LLM search strategy requires both policy and problem"
            )
        return LLMFrontier(policy=policy, problem=problem, top_k=top_k)

    strategies: dict[str, Any] = {
        "best_first": BestFirstSearch,
        "breadth_first": BreadthFirstSearch,
        "depth_first": DepthFirstSearch,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown search strategy: {strategy_name}")

    return strategies[strategy_name]()
