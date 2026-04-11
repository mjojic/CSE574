"""POCL expansion: threat detection, open-condition resolution, operator instantiation."""

from __future__ import annotations

from pydpocl.core.flaw_simple import CausalLink, OpenConditionFlaw
from pydpocl.core.interfaces import PlanningProblem
from pydpocl.core.plan import Plan
from pydpocl.core.step import GroundStep, Step


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
    consumer = flaw.step
    needed = flaw.condition
    successors: list[Plan] = []

    initial_step = plan.initial_step
    goal_step = plan.goal_step

    # --- Branch A: link from an existing step ---
    for step in plan.steps:
        if step.id == consumer.id:
            continue
        if not step.supports(needed):
            continue
        new_plan = plan.add_causal_link(step, consumer, needed)
        if new_plan.is_consistent:
            successors.append(new_plan)

    # --- Branch B: add a new operator instance ---
    for template in problem.operators:
        if not template.supports(needed):
            continue

        new_op = instantiate_operator(template)

        orderings = []
        if initial_step:
            orderings.append((initial_step.step_id, new_op.step_id))
        if goal_step:
            orderings.append((new_op.step_id, goal_step.step_id))

        plan_with_step = plan.add_step(new_op, orderings=orderings)
        new_plan = plan_with_step.add_causal_link(new_op, consumer, needed)
        if new_plan.is_consistent:
            successors.append(new_plan)

    return successors
