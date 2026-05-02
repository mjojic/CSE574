"""POCL expansion: threat detection, open-condition resolution, operator instantiation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from pydpocl.core.flaw_simple import CausalLink, OpenConditionFlaw
from pydpocl.core.interfaces import PlanningProblem
from pydpocl.core.plan import Plan
from pydpocl.core.step import GroundStep, Step
from pydpocl.core.types import StepId


def find_unresolved_threats(plan: Plan) -> list[tuple[CausalLink, Step]]:
    """Return (link, threatening_step) pairs for threats not yet resolved by orderings."""
    threats = []
    for link in plan.causal_links:
        for step in plan.steps:
            if step.id == link.source.id or step.id == link.target.id:
                continue
            if not link.is_threatened_by(step):
                continue
            # Already resolved if step is constrained before the producer
            # or after the consumer in the ordering graph.
            if plan.has_path(step.step_id, link.source.step_id):
                continue
            if plan.has_path(link.target.step_id, step.step_id):
                continue
            threats.append((link, step))
    return threats


def resolve_threat(plan: Plan, link: CausalLink, threat: Step) -> list[Plan]:
    """Branch into promotion (threat < producer) and demotion (consumer < threat)."""
    successors: list[Plan] = []

    promoted = plan.add_ordering(threat.step_id, link.source.step_id)
    if promoted.is_consistent:
        successors.append(promoted)

    demoted = plan.add_ordering(link.target.step_id, threat.step_id)
    if demoted.is_consistent:
        successors.append(demoted)

    return successors


def instantiate_operator(template: GroundStep) -> GroundStep:
    """Create a fresh copy of an operator template with a new UUID."""
    return GroundStep(
        name=template.name,
        parameters=template.parameters,
        _preconditions=template._preconditions,
        _effects=template._effects,
        height=template.height,
        depth=template.depth,
        step_number=template.step_number,
        instantiable=template.instantiable,
    )


def count_resolution_options(
    plan: Plan,
    flaw: OpenConditionFlaw,
    problem: PlanningProblem,
) -> int:
    """How many distinct successors expand_open_condition would generate for *flaw*."""
    consumer = flaw.step
    needed = flaw.condition
    n = 0
    for step in plan.steps:
        if step.id == consumer.id:
            continue
        if step.supports(needed):
            n += 1
    for op in problem.operators:
        if op.supports(needed):
            n += 1
    return n


def select_flaw_lcfr(
    plan: Plan,
    problem: PlanningProblem,
) -> OpenConditionFlaw | None:
    """LCFR: pick the flaw with the fewest possible resolvers (fail-first)."""
    if not plan.flaws:
        return None
    scored: list[tuple[int, str, str, OpenConditionFlaw]] = []
    for f in plan.flaws:
        c = count_resolution_options(plan, f, problem)
        scored.append((c, f.step.signature, f.condition.signature, f))
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    return scored[0][3]


def select_flaw_zlifo(
    plan: Plan,
    problem: PlanningProblem,
) -> OpenConditionFlaw | None:
    """ZLIFO: forced open conditions first (LIFO tiebreak), then unforced LIFO.

    Priority cascade (ground-STRIPS simplification of Pollack et al. 1997):
      1. Forced flaws (repair cost <= 1) -- cost-0 dead-ends win, else LIFO.
      2. Unforced flaws (repair cost >= 2) -- LIFO (highest age = most recent).

    Threats are handled before this function is called, so tier-1 of the
    original ZLIFO (nonseparable-threat preference) is already satisfied by
    the planner loop.  Separable threats do not exist in a ground STRIPS model,
    so that tier is also omitted.
    """
    if not plan.flaws:
        return None

    forced: list[tuple[int, OpenConditionFlaw]] = []
    unforced: list[OpenConditionFlaw] = []

    for f in plan.flaws:
        cost = count_resolution_options(plan, f, problem)
        if cost <= 1:
            forced.append((cost, f))
        else:
            unforced.append(f)

    if forced:
        # Cost-0 (dead-end) flaws first; within same cost, pick most recent.
        forced.sort(key=lambda x: (x[0], -x[1].age))
        return forced[0][1]

    # All unforced: pick LIFO (highest age = most recently introduced).
    return max(unforced, key=lambda f: f.age)


def select_flaw_lifo(
    plan: Plan,
    problem: PlanningProblem,  # noqa: ARG001 – kept for uniform signature
) -> OpenConditionFlaw | None:
    """LIFO: repair the most recently introduced flaw (highest age).

    Simple stack-like baseline – no resolver counting required.
    """
    if not plan.flaws:
        return None
    return max(plan.flaws, key=lambda f: (f.age, f.step.signature, f.condition.signature))


def select_flaw_fifo(
    plan: Plan,
    problem: PlanningProblem,  # noqa: ARG001
) -> OpenConditionFlaw | None:
    """FIFO: repair the oldest introduced flaw (lowest age).

    Simple queue-like baseline – no resolver counting required.
    """
    if not plan.flaws:
        return None
    return min(plan.flaws, key=lambda f: (f.age, f.step.signature, f.condition.signature))


def select_flaw_random(
    plan: Plan,
    problem: PlanningProblem,  # noqa: ARG001
) -> OpenConditionFlaw | None:
    """Random: choose a uniformly random flaw.

    Lower-bound baseline for comparing informed strategies.
    """
    if not plan.flaws:
        return None
    return random.choice(list(plan.flaws))


@dataclass(frozen=True, slots=True)
class ResolverCandidate:
    """A single way to resolve an open-condition flaw without building the successor.

    ``kind`` is ``"reuse"`` when the producer is an existing step in the plan,
    or ``"new"`` when a fresh operator instance must be added.  Successor plans
    are created lazily via :func:`apply_resolver` so callers (e.g. the LLM
    policy) can enumerate options cheaply and only materialize the chosen one.
    """

    id: str
    kind: str  # "reuse" | "new"
    description: str
    producer_signature: str
    source_step_id: StepId | None = None
    operator_template: GroundStep | None = field(default=None, compare=False)


def enumerate_resolvers(
    plan: Plan,
    flaw: OpenConditionFlaw,
    problem: PlanningProblem,
) -> list[ResolverCandidate]:
    """Enumerate all candidate resolvers for ``flaw`` without building successors.

    Each candidate has a stable ``id`` of the form ``reuse:<step-short-id>`` or
    ``new:<operator-signature>`` (with a numeric suffix if collisions arise).
    """
    consumer = flaw.step
    needed = flaw.condition
    candidates: list[ResolverCandidate] = []
    used_ids: set[str] = set()

    def _claim(base: str) -> str:
        if base not in used_ids:
            used_ids.add(base)
            return base
        n = 1
        while f"{base}#{n}" in used_ids:
            n += 1
        cid = f"{base}#{n}"
        used_ids.add(cid)
        return cid

    # Branch A: reuse an existing step that already produces the needed literal.
    for step in plan.steps:
        if step.id == consumer.id:
            continue
        if not step.supports(needed):
            continue
        rid = _claim(f"reuse:{step.id.hex[:8]}")
        candidates.append(
            ResolverCandidate(
                id=rid,
                kind="reuse",
                description=f"Reuse existing step {step.signature} as producer",
                producer_signature=step.signature,
                source_step_id=step.step_id,
            )
        )

    # Branch B: instantiate a fresh operator from the problem definition.
    for template in problem.operators:
        if not template.supports(needed):
            continue
        rid = _claim(f"new:{template.signature}")
        candidates.append(
            ResolverCandidate(
                id=rid,
                kind="new",
                description=f"Add a new step {template.signature} as producer",
                producer_signature=template.signature,
                operator_template=template,
            )
        )

    return candidates


def apply_resolver(
    plan: Plan,
    flaw: OpenConditionFlaw,
    resolver: ResolverCandidate,
) -> Plan | None:
    """Materialize the successor plan for ``resolver`` or return ``None`` if inconsistent."""
    consumer = flaw.step
    needed = flaw.condition

    if resolver.kind == "reuse":
        if resolver.source_step_id is None:
            return None
        source = plan.get_step(resolver.source_step_id)
        if source is None:
            return None
        new_plan = plan.add_causal_link(source, consumer, needed)
    elif resolver.kind == "new":
        if resolver.operator_template is None:
            return None
        new_op = instantiate_operator(resolver.operator_template)
        orderings: list[tuple[StepId, StepId]] = []
        initial_step = plan.initial_step
        goal_step = plan.goal_step
        if initial_step:
            orderings.append((initial_step.step_id, new_op.step_id))
        if goal_step:
            orderings.append((new_op.step_id, goal_step.step_id))
        plan_with_step = plan.add_step(new_op, orderings=orderings)
        new_plan = plan_with_step.add_causal_link(new_op, consumer, needed)
    else:
        return None

    return new_plan if new_plan.is_consistent else None


def expand_open_condition(
    plan: Plan,
    flaw: OpenConditionFlaw,
    problem: PlanningProblem,
) -> list[Plan]:
    """Generate all successor plans that resolve *flaw* (an unsatisfied precondition).

    Branch A: reuse an existing step in the plan that already produces the
    needed literal (includes __INITIAL__ for literals in the initial state).
    Branch B: instantiate a fresh operator from the problem's operator list.
    """
    successors: list[Plan] = []
    for cand in enumerate_resolvers(plan, flaw, problem):
        succ = apply_resolver(plan, flaw, cand)
        if succ is not None:
            successors.append(succ)
    return successors
