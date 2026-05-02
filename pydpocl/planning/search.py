"""Search strategies for POCL planning."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from pydpocl.core.plan import Plan

if TYPE_CHECKING:
    from pydpocl.core.interfaces import PlanningProblem
    from pydpocl.planning.llm_policy import LLMPolicy

T = TypeVar("T")


class SearchStrategy(Protocol, Generic[T]):
    """Protocol for search strategies."""

    def add_plan(self, plan: T, priority: float) -> None:
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
    """POCL best-first frontier backed by a min-heap.

    Heap entries are ``(priority, counter, plan)`` where the monotonic
    counter breaks ties without comparing Plan objects directly.

    Priority is supplied by the caller and represents ``f(n) = g(n) + h(n)``
    where ``g(n)`` counts non-dummy plan steps and ``h(n)`` is the chosen
    POCL heuristic estimate.
    """

    def __init__(self) -> None:
        self._frontier: list[tuple[float, int, Plan]] = []
        self._counter = 0

    def add_plan(self, plan: Plan, priority: float = 0.0) -> None:
        """Add a plan to the frontier."""
        heapq.heappush(self._frontier, (priority, self._counter, plan))
        self._counter += 1

    def get_next_plan(self) -> Plan | None:
        """Pop and return the plan with the lowest priority value."""
        if not self._frontier:
            return None
        _, _, plan = heapq.heappop(self._frontier)
        return plan

    def is_empty(self) -> bool:
        return len(self._frontier) == 0

    def size(self) -> int:
        return len(self._frontier)


class LLMSearch:
    """Best-first heap whose pop step is delegated to an LLM policy.

    The frontier keeps a standard ``(priority, counter, plan)`` heap so that
    insertions and bookkeeping match :class:`BestFirstSearch`.  When the
    planner asks for the next plan, the entire frontier is presented to the
    LLM policy to pick from; the chosen plan is removed and the rest remain.
    """

    def __init__(
        self,
        policy: "LLMPolicy",
        problem: "PlanningProblem",
    ) -> None:
        self._policy = policy
        self._problem = problem
        self._frontier: list[tuple[float, int, Plan]] = []
        self._counter = 0

    def add_plan(self, plan: Plan, priority: float = 0.0) -> None:
        """Add a plan to the frontier."""
        heapq.heappush(self._frontier, (priority, self._counter, plan))
        self._counter += 1

    def get_next_plan(self) -> Plan | None:
        if not self._frontier:
            return None
        if len(self._frontier) == 1:
            _, _, plan = heapq.heappop(self._frontier)
            return plan

        all_entries: list[tuple[float, int, Plan]] = []
        while self._frontier:
            all_entries.append(heapq.heappop(self._frontier))

        candidates = [(prio, plan) for prio, _, plan in all_entries]
        chosen = self._policy.select_node(candidates, self._problem)

        chosen_id = chosen.id
        for entry in all_entries:
            if entry[2].id != chosen_id:
                heapq.heappush(self._frontier, entry)
        return chosen

    def is_empty(self) -> bool:
        return len(self._frontier) == 0

    def size(self) -> int:
        return len(self._frontier)


# Backward-compatible alias kept for any code that imported LLMFrontier directly.
LLMFrontier = LLMSearch


def create_search_strategy(
    strategy_name: str,
    *,
    policy: "LLMPolicy | None" = None,
    problem: "PlanningProblem | None" = None,
) -> Any:
    """Return a search strategy instance by name.

    Supported strategies
    --------------------
    best_first - POCL best-first with explicit f(n) priority (default)
    llm        - LLM-delegated node selection; requires ``policy`` and ``problem``
    """
    if strategy_name == "llm":
        if policy is None or problem is None:
            raise ValueError(
                "LLM search strategy requires both policy and problem"
            )
        return LLMSearch(policy=policy, problem=problem)

    if strategy_name == "best_first":
        return BestFirstSearch()

    raise ValueError(
        f"Unknown search strategy: {strategy_name!r}. Valid options: best_first, llm"
    )
