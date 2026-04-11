"""Search strategies for planning."""

from __future__ import annotations

import heapq
from collections import deque
from typing import Generic, Protocol, TypeVar

from pydpocl.core.plan import Plan

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


def create_search_strategy(strategy_name: str) -> SearchStrategy[Plan]:
    """Factory function for creating search strategies."""
    strategies = {
        "best_first": BestFirstSearch,
        "breadth_first": BreadthFirstSearch,
        "depth_first": DepthFirstSearch,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown search strategy: {strategy_name}")

    return strategies[strategy_name]()
