"""POCL heuristic functions.

Two families are provided:

Plan-space structural heuristics
    Work directly on the partial-plan structure (step count, flaw sets,
    causal-link threats).  No delete-relaxed simulation is needed.
    Keys: ``oc``, ``fc``, ``tc``, ``ps``

State-based repair-cost heuristics
    Lift classical delete-relaxed state-space estimates into the plan space
    by evaluating them over the set of open conditions.  Require a one-time
    ``prepare(problem)`` call to build the reachability table.
    Keys: ``add``, ``max``, ``ff``
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pydpocl.core.interfaces import PlanningProblem
    from pydpocl.core.plan import Plan


class Heuristic(Protocol):
    """Protocol for POCL heuristic functions."""

    def prepare(self, problem: PlanningProblem) -> None:
        """Pre-compute data from the ground problem.  Called once per solve."""
        ...

    def estimate(self, plan: Plan) -> float:
        """Return h(plan) — estimated cost remaining to fix all open conditions."""
        ...


# ===========================================================================
# Plan-space structural heuristics
# ===========================================================================
#
# These heuristics count structural features of the partial plan directly.
# They require no relaxed-planning computation and have O(|flaws|) or
# O(|steps| × |links|) runtime per call.


class OpenConditionHeuristic:
    """h = |open conditions|  (``oc``).

    Each unsatisfied precondition in the partial plan counts as one unit of
    remaining work.  No preparation needed.  Equivalent to the open-condition
    count used in early UCPOP / SNLP evaluations.
    """

    def prepare(self, problem: PlanningProblem) -> None:
        pass

    def estimate(self, plan: Plan) -> float:
        return float(len(plan.flaws))


class FlawCountHeuristic:
    """h = |open conditions| + |unresolved threats|  (``fc``).

    Counts every flaw type the planner must resolve before declaring a
    solution: open preconditions *and* causal-link threats.  Slightly
    stronger signal than OC alone when threats are common.
    """

    def prepare(self, problem: PlanningProblem) -> None:
        pass

    def estimate(self, plan: Plan) -> float:
        from pydpocl.planning.pocl_expansion import find_unresolved_threats
        return float(len(plan.flaws) + len(find_unresolved_threats(plan)))


class ThreatCountHeuristic:
    """h = |unresolved causal-link threats|  (``tc``).

    Counts only the threatened causal links that have not yet been resolved
    by promotion or demotion.  Useful as a stand-alone signal when threat
    density is the dominant bottleneck (e.g. highly concurrent domains).
    """

    def prepare(self, problem: PlanningProblem) -> None:
        pass

    def estimate(self, plan: Plan) -> float:
        from pydpocl.planning.pocl_expansion import find_unresolved_threats
        return float(len(find_unresolved_threats(plan)))


class PlanSizePenaltyHeuristic:
    """h = number of non-dummy steps already in the plan  (``ps``).

    Penalises large partial plans by adding the current plan size to g(n),
    which makes f(n) = 2 × #steps + 0 for complete plans.  This steers
    best-first search away from partial plans that have accumulated many
    steps without closing open conditions, favouring parsimonious solutions.

    Note: this is a quality-preference penalty, not a remaining-work
    estimate — it is inadmissible and should be used as a tiebreaker
    or combined with a repair-cost heuristic.
    """

    def prepare(self, problem: PlanningProblem) -> None:
        pass

    def estimate(self, plan: Plan) -> float:
        return float(sum(1 for s in plan.steps if not s.name.startswith("__")))


# ===========================================================================
# State-based repair-cost heuristics
# ===========================================================================
#
# These heuristics lift classical delete-relaxed state-space estimates into
# the plan space by treating the open conditions of the partial plan as the
# sub-goal set and solving a delete-relaxed planning sub-problem to bound
# the remaining repair cost.
#
# Cf. Bercher, Geier & Biundo — "A Planning-Based Approach to Automatic
# Assistance for Plan Repair" and related work on heuristic guidance for
# plan-space and hybrid planning — for the formal grounding of state-based
# estimates in partial-order plan spaces and empirical evidence that they
# dominate purely structural count-based estimates on most benchmarks.


def _build_relaxed_costs(
    initial_state: set,
    operators: list,
    mode: str,
) -> dict:
    """Delete-relaxed Bellman-Ford fixed point shared by h_add and h_max.

    Parameters
    ----------
    initial_state:
        Set of Literal objects true in the initial state.
    operators:
        Ground operators (GroundStep instances).
    mode:
        ``"add"`` — additive aggregation (h_add);
        ``"max"`` — max aggregation (h_max).

    Returns
    -------
    dict mapping Literal -> float cost.
    Positive initial-state literals have cost 0.0; unreachable literals are
    absent (callers treat absence as ``math.inf``).
    """
    costs: dict = {lit: 0.0 for lit in initial_state if lit.positive}

    changed = True
    while changed:
        changed = False
        for op in operators:
            pre_costs = [costs.get(pre, math.inf) for pre in op.preconditions]

            if mode == "add":
                op_cost = 1.0 + sum(pre_costs) if pre_costs else 1.0
            else:
                op_cost = 1.0 + max(pre_costs) if pre_costs else 1.0

            if math.isinf(op_cost):
                continue

            for eff in op.effects:
                if not eff.positive:  # ignore delete effects
                    continue
                if op_cost < costs.get(eff, math.inf):
                    costs[eff] = op_cost
                    changed = True

    return costs


class HAddHeuristic:
    """h_add: sum of additive relaxed costs over all open conditions  (``add``).

    ``h_add(p) = 0`` if ``p`` holds in the initial state; otherwise
    ``min over operators o achieving p: 1 + Σ h_add(pre)`` for each
    precondition of o.  ``estimate(plan) = Σ h_add(f.condition)``.

    Inadmissible (counts shared sub-goals multiple times) but highly
    informative; drives search toward plans that close open conditions cheaply.
    """

    def __init__(self) -> None:
        self._costs: dict = {}

    def prepare(self, problem: PlanningProblem) -> None:
        self._costs = _build_relaxed_costs(
            problem.initial_state, list(problem.operators), "add"
        )

    def estimate(self, plan: Plan) -> float:
        if not plan.flaws:
            return 0.0
        return sum(self._costs.get(f.condition, math.inf) for f in plan.flaws)


class HMaxHeuristic:
    """h_max: max relaxed cost over all open conditions  (``max``).

    Same fixed point as h_add but with ``1 + max(...)`` aggregation.
    ``estimate(plan) = max h_max(f.condition)``.

    Admissible in the delete-relaxed sense; provides a theoretically sound
    lower bound.  Tends to be less informative than h_add in practice.
    """

    def __init__(self) -> None:
        self._costs: dict = {}

    def prepare(self, problem: PlanningProblem) -> None:
        self._costs = _build_relaxed_costs(
            problem.initial_state, list(problem.operators), "max"
        )

    def estimate(self, plan: Plan) -> float:
        if not plan.flaws:
            return 0.0
        return max(self._costs.get(f.condition, math.inf) for f in plan.flaws)


class HFFHeuristic:
    """h_FF: size of a delete-relaxed plan over the open conditions  (``ff``).

    Follows Hoffmann & Nebel (2001): for each unachieved open condition,
    greedily select the operator with the lowest h_add precondition cost
    (best supporter), add its preconditions to the pending set, and count
    the distinct operators chosen.

    Inadmissible but typically the strongest of the three state-based
    heuristics; excellent coverage on blocksworld-style instances.
    """

    def __init__(self) -> None:
        self._costs: dict = {}
        self._operators: list = []
        self._initial: set = set()

    def prepare(self, problem: PlanningProblem) -> None:
        self._operators = list(problem.operators)
        self._initial = {lit for lit in problem.initial_state if lit.positive}
        self._costs = _build_relaxed_costs(
            problem.initial_state, self._operators, "add"
        )

    def estimate(self, plan: Plan) -> float:
        if not plan.flaws:
            return 0.0
        goals = {f.condition for f in plan.flaws}
        return float(self._extract_relaxed_plan(goals))

    def _extract_relaxed_plan(self, goals: set) -> int:
        """Backwards greedy relaxed-plan extraction; returns operator count."""
        achieved = set(self._initial)
        rplan: set[str] = set()
        pending = goals - achieved

        while pending:
            g = min(pending, key=lambda lit: self._costs.get(lit, math.inf))
            pending.discard(g)

            if g in achieved:
                continue
            if self._costs.get(g, math.inf) == math.inf:
                return 10**7  # unreachable

            best_op = None
            best_cost = math.inf
            for op in self._operators:
                if not any(eff == g and eff.positive for eff in op.effects):
                    continue
                pre_costs = [self._costs.get(pre, math.inf) for pre in op.preconditions]
                op_cost = sum(pre_costs) if pre_costs else 0.0
                if op_cost < best_cost:
                    best_cost = op_cost
                    best_op = op

            if best_op is None:
                return 10**7

            rplan.add(best_op.signature)
            achieved.add(g)
            for pre in best_op.preconditions:
                if pre not in achieved:
                    pending.add(pre)

        return len(rplan)


# ===========================================================================
# Factory
# ===========================================================================

def create_heuristic(name: str) -> Heuristic:
    """Return a heuristic instance by name.

    Plan-space structural heuristics
    ---------------------------------
    oc  - open-condition count
    fc  - flaw count  (open conditions + causal-link threats)
    tc  - threat count (causal-link threats only)
    ps  - plan-size penalty (current non-dummy step count)

    State-based repair-cost heuristics
    ------------------------------------
    add - h_add  (additive delete-relaxed fixed point, inadmissible)
    max - h_max  (max delete-relaxed fixed point, admissible)
    ff  - h_FF   (Hoffmann relaxed-plan size, inadmissible, strongest)
    """
    heuristics: dict[str, type] = {
        # plan-space structural
        "oc":  OpenConditionHeuristic,
        "fc":  FlawCountHeuristic,
        "tc":  ThreatCountHeuristic,
        "ps":  PlanSizePenaltyHeuristic,
        # state-based repair-cost
        "add": HAddHeuristic,
        "max": HMaxHeuristic,
        "ff":  HFFHeuristic,
    }

    if name not in heuristics:
        raise ValueError(
            f"Unknown heuristic: {name!r}. "
            f"Valid options: {', '.join(heuristics)}"
        )

    return heuristics[name]()
