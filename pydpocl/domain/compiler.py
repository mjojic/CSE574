"""PDDL domain and problem compilation."""

from __future__ import annotations

from pathlib import Path

from pydpocl.domain.strips_pddl import GroundPlanningProblem, compile_strips_pddl


def compile_domain_and_problem(
    domain_file: Path,
    problem_file: Path,
) -> GroundPlanningProblem:
    """Compile a PDDL domain and problem into a grounded planning problem.

    Parses STRIPS PDDL (with optional :typing), grounds all operators against
    the problem objects with type-compatible substitutions, and returns a
    GroundPlanningProblem containing the initial state, goal state, and the
    full set of ground operators.

    Args:
        domain_file: Path to the PDDL domain file
        problem_file: Path to the PDDL problem file

    Returns:
        GroundPlanningProblem with initial_state, goal_state, and operators
    """
    return compile_strips_pddl(domain_file, problem_file)
