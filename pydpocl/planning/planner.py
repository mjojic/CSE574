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
    select_flaw_lcfr,
    select_flaw_zlifo,
)
from pydpocl.planning.plan_fingerprint import structural_fingerprint
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
    duplicates_pruned: int = 0


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
        dedupe_structural: bool = True,
        flaw_order: str = "lcfr",
    ):
        self.search_strategy = search_strategy
        self.heuristic = heuristic
        self.verbose = verbose
        self.dedupe_structural = dedupe_structural
        # Backward-compatible alias: MRV previously referred to this LCFR selector.
        if flaw_order == "mrv":
            flaw_order = "lcfr"
        if flaw_order not in ("lcfr", "zlifo", "priority"):
            raise ValueError("flaw_order must be 'lcfr', 'zlifo', or 'priority'")
        self.flaw_order = flaw_order
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
        # Structural fingerprints already enqueued or expanded — avoids revisiting
        # the same partial plan reached via different expansion orders.
        seen: set[tuple] = set()

        initial_plan = create_initial_plan(problem.initial_state, problem.goal_state)
        self._try_push(initial_plan, frontier, heuristic_fn, seen)

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
                    self._try_push(s, frontier, heuristic_fn, seen)
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
            if self.flaw_order == "lcfr":
                flaw = select_flaw_lcfr(plan, problem)
            elif self.flaw_order == "zlifo":
                flaw = select_flaw_zlifo(plan, problem)
            else:
                flaw = plan.select_flaw()
            if flaw is None:
                continue

            successors = expand_open_condition(plan, flaw, problem)
            self.statistics.nodes_expanded += 1
            for s in successors:
                self._try_push(s, frontier, heuristic_fn, seen)

        self.statistics.time_elapsed = time.time() - start_time

    # ------------------------------------------------------------------

    def _try_push(
        self,
        plan: Plan,
        frontier,
        heuristic_fn,
        seen: set[tuple],
    ) -> None:
        if self.dedupe_structural:
            fp = structural_fingerprint(plan)
            if fp in seen:
                self.statistics.duplicates_pruned += 1
                return
            seen.add(fp)
        p = plan.cost + heuristic_fn.estimate(plan)
        frontier.add_plan(plan, priority=p)
        self._track_frontier(frontier)

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
            "duplicates_pruned": self.statistics.duplicates_pruned,
        }
