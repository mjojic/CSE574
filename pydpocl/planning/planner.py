"""Main DPOCL planner implementation."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from pydpocl.core.interfaces import BasePlanner, PlanningProblem
from pydpocl.core.plan import Plan, create_initial_plan
from pydpocl.planning.heuristic import create_heuristic
from pydpocl.planning.pocl_expansion import (
    expand_open_condition,
    find_unresolved_threats,
    resolve_threat,
)
from pydpocl.planning.search import create_search_strategy


@dataclass
class PlanningStatistics:
    """Statistics for a planning run."""

    nodes_expanded: int = 0
    nodes_visited: int = 0
    solutions_found: int = 0
    time_elapsed: float = 0.0
    peak_frontier_size: int = 0
    timeout_reached: bool = False


class DPOCLPlanner(BasePlanner[Plan, Plan]):
    """POCL planner with systematic frontier-based search.

    Each node in the search space is an immutable partial plan.  At each
    iteration the planner pops the best plan, checks for unresolved threats
    (resolved by promotion/demotion branching), then resolves one open
    condition (by reusing an existing step or instantiating a new operator).
    All alternatives are pushed onto the frontier, giving complete search.
    """

    def __init__(
        self,
        search_strategy: str = "best_first",
        heuristic: str = "zero",
        verbose: bool = False,
    ):
        self.search_strategy = search_strategy
        self.heuristic = heuristic
        self.verbose = verbose
        self.statistics = PlanningStatistics()

    def solve(
        self,
        problem: PlanningProblem,
        max_solutions: int = 1,
        timeout: float | None = None,
    ) -> Iterator[Plan]:
        """Solve the planning problem and yield solutions.

        Args:
            problem: The planning problem to solve
            max_solutions: Maximum number of solutions to find
            timeout: Timeout in seconds (None for no timeout)

        Yields:
            Complete plans that solve the problem
        """
        start_time = time.time()
        self.statistics = PlanningStatistics()

        if self.verbose:
            print(f"Starting DPOCL planning with {self.search_strategy} search")
            print(f"Using {self.heuristic} heuristic")
            print(f"Looking for up to {max_solutions} solutions")
            if timeout:
                print(f"Timeout: {timeout} seconds")

        heuristic_fn = create_heuristic(self.heuristic)
        frontier = create_search_strategy(self.search_strategy)

        initial_plan = create_initial_plan(problem.initial_state, problem.goal_state)
        priority = initial_plan.cost + heuristic_fn.estimate(initial_plan)
        frontier.add_plan(initial_plan, priority=priority)

        while not frontier.is_empty():
            # --- bookkeeping ---
            if timeout and (time.time() - start_time) > timeout:
                self.statistics.timeout_reached = True
                break
            if self.statistics.solutions_found >= max_solutions:
                break

            plan = frontier.get_next_plan()
            if plan is None:
                break
            self.statistics.nodes_visited += 1

            # --- threat resolution (priority over open conditions) ---
            threats = find_unresolved_threats(plan)
            if threats:
                link, threat_step = threats[0]
                successors = resolve_threat(plan, link, threat_step)
                self.statistics.nodes_expanded += 1
                for s in successors:
                    p = s.cost + heuristic_fn.estimate(s)
                    frontier.add_plan(s, priority=p)
                    self._track_frontier(frontier)
                continue

            # --- solution check (no flaws and no threats) ---
            if plan.is_complete:
                self.statistics.solutions_found += 1
                if self.verbose:
                    print(f"  Solution #{self.statistics.solutions_found} found "
                          f"({len(plan.steps)} steps, depth {plan.depth})")
                yield plan
                continue

            # --- open-condition expansion ---
            flaw = plan.select_flaw()
            if flaw is None:
                continue

            successors = expand_open_condition(plan, flaw, problem)
            self.statistics.nodes_expanded += 1
            for s in successors:
                p = s.cost + heuristic_fn.estimate(s)
                frontier.add_plan(s, priority=p)
                self._track_frontier(frontier)

        self.statistics.time_elapsed = time.time() - start_time

    # ------------------------------------------------------------------

    def _track_frontier(self, frontier) -> None:
        size = frontier.size()
        if size > self.statistics.peak_frontier_size:
            self.statistics.peak_frontier_size = size

    def get_statistics(self) -> dict[str, Any]:
        """Return planning statistics."""
        return {
            "nodes_expanded": self.statistics.nodes_expanded,
            "nodes_visited": self.statistics.nodes_visited,
            "solutions_found": self.statistics.solutions_found,
            "time_elapsed": self.statistics.time_elapsed,
            "peak_frontier_size": self.statistics.peak_frontier_size,
            "timeout_reached": self.statistics.timeout_reached,
        }
