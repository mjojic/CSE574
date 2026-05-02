"""Main POCL planner implementation."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from pydpocl.core.flaw_simple import OpenConditionFlaw
from pydpocl.core.interfaces import BasePlanner, PlanningProblem
from pydpocl.core.plan import Plan, create_initial_plan
from pydpocl.planning.heuristic import OpenConditionHeuristic, create_heuristic
from pydpocl.planning.llm_policy import LLMConfig, LLMPolicy, LLMPolicyError
from pydpocl.planning.plan_fingerprint import structural_fingerprint
from pydpocl.planning.pocl_expansion import (
    apply_resolver,
    enumerate_resolvers,
    expand_open_condition,
    find_unresolved_threats,
    resolve_threat,
    select_flaw_fifo,
    select_flaw_lcfr,
    select_flaw_lifo,
    select_flaw_random,
    select_flaw_zlifo,
)
from pydpocl.planning.search import BestFirstSearch, LLMSearch

_VALID_HEURISTICS = ("oc", "fc", "tc", "ps", "add", "max", "ff", "llm")
_VALID_FLAW_SELECTION_STRATS = ("lcfr", "zlifo", "lifo", "fifo", "random", "llm")
_VALID_RESOLVER_STRATS = ("enumerate", "llm")


def _g_value(plan: Plan) -> int:
    """Number of non-dummy steps added to the plan so far (plan-length proxy)."""
    return sum(1 for s in plan.steps if not s.name.startswith("__"))


@dataclass
class PendingExpansion:
    """A deferred OR-decision that the planner can revisit after a dead end.

    When the LLM picks a single resolver for an open-condition flaw, the
    planner enqueues only that resolver's child and stashes a
    :class:`PendingExpansion` describing the remaining alternatives.  If the
    main frontier later runs dry (no descendants of the chosen branch led to
    a goal) the planner pops this record, re-prompts the LLM with the
    rejected ids excluded, and pushes the next chosen successor.
    """

    plan: Plan
    flaw: OpenConditionFlaw
    excluded: set[str] = field(default_factory=set)


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
    llm_calls: int = 0
    llm_retries: int = 0


class DPOCLPlanner(BasePlanner[Plan, Plan]):
    """POCL planner with best-first frontier-based search.

    Each node in the search space is an immutable partial plan.  At each
    iteration the planner pops the lowest-priority plan, resolves any
    unresolved threats (promotion/demotion branching), then resolves one
    open condition (by reusing an existing step or instantiating a new
    operator).  All successors are pushed with priority f(n) = g(n) + h(n)
    where g(n) is the number of non-dummy steps already in the plan and h(n)
    is the chosen POCL heuristic estimate.

    Parameters
    ----------
    heuristic:
        POCL heuristic: ``oc`` (open-condition count), ``add`` (h_add),
        ``max`` (h_max), ``ff`` (h_FF), or ``llm`` (delegates node selection
        to the LLM — the LLM picks from the full frontier instead of using
        a numeric h-value).
    flaw_selection_strat:
        Open-condition flaw ordering: ``lcfr`` (least-constrained-first /
        fail-first), ``zlifo``, or ``llm`` (delegates flaw choice to the LLM).
    resolver_strat:
        How resolvers are expanded: ``enumerate`` (push every consistent
        successor) or ``llm`` (pick one and reprompt on dead ends).
    """

    def __init__(
        self,
        heuristic: str = "add",
        verbose: bool = False,
        dedupe_structural: bool = True,
        flaw_selection_strat: str = "lcfr",
        resolver_strat: str = "enumerate",
        llm_config: LLMConfig | None = None,
    ):
        if heuristic not in _VALID_HEURISTICS:
            raise ValueError(
                f"heuristic must be one of {_VALID_HEURISTICS}"
            )
        if flaw_selection_strat not in _VALID_FLAW_SELECTION_STRATS:
            raise ValueError(
                f"flaw_selection_strat must be one of {_VALID_FLAW_SELECTION_STRATS}"
            )
        if resolver_strat not in _VALID_RESOLVER_STRATS:
            raise ValueError(
                f"resolver_strat must be one of {_VALID_RESOLVER_STRATS}"
            )

        self.heuristic = heuristic
        self.verbose = verbose
        self.dedupe_structural = dedupe_structural
        self.flaw_selection_strat = flaw_selection_strat
        self.resolver_strat = resolver_strat
        self.llm_config = llm_config or LLMConfig()
        self.statistics = PlanningStatistics()
        self._pending_expansions: list[PendingExpansion] = []

    @property
    def _llm_search(self) -> bool:
        """True when the LLM drives node selection (heuristic == 'llm')."""
        return self.heuristic == "llm"

    def _needs_llm(self) -> bool:
        return (
            self._llm_search
            or self.flaw_selection_strat == "llm"
            or self.resolver_strat == "llm"
        )

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
        self._pending_expansions = []

        commit_resolver = self.resolver_strat == "llm"

        if self.verbose:
            print(
                f"Starting POCL planning"
                f" | heuristic={self.heuristic}"
                f" | flaw_selection={self.flaw_selection_strat}"
                f" | resolver={self.resolver_strat}"
            )
            if self._needs_llm():
                print(f"LLM control enabled (model={self.llm_config.model})")
            print(f"Looking for up to {max_solutions} solutions")
            if timeout:
                print(f"Timeout: {timeout} seconds")

        llm_policy: LLMPolicy | None = None
        if self._needs_llm():
            llm_policy = LLMPolicy(self.llm_config)

        if self._llm_search:
            if llm_policy is None:
                raise ValueError("heuristic='llm' requires a configured LLM policy")
            frontier = LLMSearch(policy=llm_policy, problem=problem)
            # Priority is not used for LLM node selection; use OC as a cheap dummy.
            heuristic_fn = OpenConditionHeuristic()
        else:
            frontier = BestFirstSearch()
            heuristic_fn = create_heuristic(self.heuristic)
            heuristic_fn.prepare(problem)

        seen: set[tuple] = set()

        initial_plan = create_initial_plan(problem.initial_state, problem.goal_state)
        self._try_push(initial_plan, frontier, heuristic_fn, seen)

        try:
            while True:
                if timeout and (time.time() - start_time) > timeout:
                    self.statistics.timeout_reached = True
                    break
                if self.statistics.solutions_found >= max_solutions:
                    break

                if frontier.is_empty():
                    if not commit_resolver or not self._pending_expansions:
                        break
                    pe = self._pending_expansions[-1]
                    progressed = self._reprompt_resolver(
                        pe,
                        problem=problem,
                        llm_policy=llm_policy,  # type: ignore[arg-type]
                        heuristic_fn=heuristic_fn,
                        frontier=frontier,
                        seen=seen,
                    )
                    if not progressed:
                        self._pending_expansions.pop()
                    continue

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
                        print(
                            f"  Solution #{self.statistics.solutions_found} found "
                            f"({len(plan.steps)} steps, depth {plan.depth})"
                        )
                    yield plan
                    continue

                # --- flaw selection ---
                if self.flaw_selection_strat == "llm" and llm_policy is not None:
                    flaw = llm_policy.select_flaw(plan, problem)
                elif self.flaw_selection_strat == "lcfr":
                    flaw = select_flaw_lcfr(plan, problem)
                elif self.flaw_selection_strat == "zlifo":
                    flaw = select_flaw_zlifo(plan, problem)
                elif self.flaw_selection_strat == "lifo":
                    flaw = select_flaw_lifo(plan, problem)
                elif self.flaw_selection_strat == "fifo":
                    flaw = select_flaw_fifo(plan, problem)
                else:  # random
                    flaw = select_flaw_random(plan, problem)
                if flaw is None:
                    continue

                # --- open-condition expansion ---
                if commit_resolver and llm_policy is not None:
                    self._expand_single_resolver(
                        plan,
                        flaw,
                        problem=problem,
                        llm_policy=llm_policy,
                        heuristic_fn=heuristic_fn,
                        frontier=frontier,
                        seen=seen,
                    )
                else:
                    successors = expand_open_condition(plan, flaw, problem)
                    self.statistics.nodes_expanded += 1
                    for s in successors:
                        self._try_push(s, frontier, heuristic_fn, seen)
        finally:
            self.statistics.time_elapsed = time.time() - start_time
            if llm_policy is not None:
                self.statistics.llm_calls = llm_policy.calls
                self.statistics.llm_retries = llm_policy.retries

    # ------------------------------------------------------------------

    def _try_push(
        self,
        plan: Plan,
        frontier,
        heuristic_fn,
        seen: set[tuple],
    ) -> bool:
        """Push *plan* onto *frontier* unless its structural fingerprint was already seen.

        Returns True if the plan was pushed.
        """
        if self.dedupe_structural:
            fp = structural_fingerprint(plan)
            if fp in seen:
                self.statistics.duplicates_pruned += 1
                return False
            seen.add(fp)
        g = _g_value(plan)
        h = heuristic_fn.estimate(plan)
        frontier.add_plan(plan, priority=float(g + h))
        self._track_frontier(frontier)
        return True

    def _expand_single_resolver(
        self,
        plan: Plan,
        flaw: OpenConditionFlaw,
        *,
        problem: PlanningProblem,
        llm_policy: LLMPolicy,
        heuristic_fn,
        frontier,
        seen: set[tuple],
    ) -> None:
        """LLM-driven expansion that commits to one resolver and stashes the rest."""
        candidates = enumerate_resolvers(plan, flaw, problem)
        if not candidates:
            return

        self.statistics.nodes_expanded += 1
        chosen = llm_policy.select_resolver(plan, flaw, candidates, problem)
        successor = apply_resolver(plan, flaw, chosen)
        excluded: set[str] = {chosen.id}
        if successor is not None:
            self._try_push(successor, frontier, heuristic_fn, seen)
        if len(candidates) > 1:
            self._pending_expansions.append(
                PendingExpansion(plan=plan, flaw=flaw, excluded=excluded)
            )

    def _reprompt_resolver(
        self,
        pe: PendingExpansion,
        *,
        problem: PlanningProblem,
        llm_policy: LLMPolicy,
        heuristic_fn,
        frontier,
        seen: set[tuple],
    ) -> bool:
        """Try to push one more sibling resolver for the deferred decision *pe*.

        Returns True if a new successor was selected and False when every
        candidate has been exhausted or the LLM failed.
        """
        candidates = enumerate_resolvers(pe.plan, pe.flaw, problem)
        remaining = [c for c in candidates if c.id not in pe.excluded]
        if not remaining:
            return False
        try:
            chosen = llm_policy.select_resolver(
                pe.plan,
                pe.flaw,
                remaining,
                problem,
                excluded=pe.excluded,
            )
        except LLMPolicyError as exc:
            if self.verbose:
                print(f"  Reprompt failed: {exc}")
            return False

        pe.excluded.add(chosen.id)
        successor = apply_resolver(pe.plan, pe.flaw, chosen)
        if successor is not None:
            self._try_push(successor, frontier, heuristic_fn, seen)
            self.statistics.nodes_expanded += 1
        return True

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
            "llm_calls": self.statistics.llm_calls,
            "llm_retries": self.statistics.llm_retries,
        }
