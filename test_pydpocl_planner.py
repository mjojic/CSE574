#!/usr/bin/env python3
"""Smoke-test for the pydpocl POCL planner on the socks-and-shoes problem.

This script constructs a minimal planning problem using pydpocl types
and runs DPOCLPlanner.solve(). Because solve() is currently a stub that
yields no plans, assertions only check the contract (imports, iterator
protocol, stats dict). Pass --expect-solution to additionally assert
that at least one plan is returned (useful once the planner is wired up).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

from pydpocl.core.literal import Literal, create_literal
from pydpocl.core.step import GroundStep
from pydpocl.planning.llm_policy import LLMConfig
from pydpocl.planning.planner import DPOCLPlanner


@dataclass
class GroundPlanningProblem:
    """Concrete PlanningProblem satisfying the pydpocl Protocol."""

    initial_state: set[Literal] = field(default_factory=set)
    goal_state: set[Literal] = field(default_factory=set)
    operators: list[GroundStep] = field(default_factory=list)

    def is_goal_satisfied(self, state: set[Literal]) -> bool:
        return self.goal_state <= state


def socks_and_shoes_problem() -> GroundPlanningProblem:
    """Build the classic socks-and-shoes problem using pydpocl types."""

    right_sock_on = create_literal("RightSockOn")
    right_shoe_on = create_literal("RightShoeOn")
    left_sock_on = create_literal("LeftSockOn")
    left_shoe_on = create_literal("LeftShoeOn")

    operators = [
        GroundStep(
            name="RightSock",
            parameters=(),
            _preconditions=frozenset(),
            _effects=frozenset([right_sock_on]),
            step_number=0,
        ),
        GroundStep(
            name="RightShoe",
            parameters=(),
            _preconditions=frozenset([right_sock_on]),
            _effects=frozenset([right_shoe_on]),
            step_number=1,
        ),
        GroundStep(
            name="LeftSock",
            parameters=(),
            _preconditions=frozenset(),
            _effects=frozenset([left_sock_on]),
            step_number=2,
        ),
        GroundStep(
            name="LeftShoe",
            parameters=(),
            _preconditions=frozenset([left_sock_on]),
            _effects=frozenset([left_shoe_on]),
            step_number=3,
        ),
    ]

    return GroundPlanningProblem(
        initial_state=set(),
        goal_state={right_shoe_on, left_shoe_on},
        operators=operators,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-solution",
        action="store_true",
        help="Assert that the planner returns at least one solution (fails on stub).",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help=(
            "Use the LLM as a search heuristic (node, flaw, and single-resolver"
            " selection). Requires either $OPENAI_API_KEY or an OpenAI-compatible"
            " server (e.g. vLLM) reachable via --llm-base-url / $OPENAI_BASE_URL."
        ),
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model id (defaults to $OPENAI_MODEL / $LLM_MODEL or the LLMConfig default).",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="OpenAI-compatible base URL (default: $OPENAI_BASE_URL / $LLM_BASE_URL).",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="Override $OPENAI_API_KEY (placeholder is sent automatically for local servers).",
    )
    parser.add_argument(
        "--llm-response-format",
        choices=("json_schema", "json_object", "none"),
        default="json_schema",
        help="LLM output constraint mode (default: json_schema).",
    )
    parser.add_argument(
        "--llm-top-k",
        type=int,
        default=10,
        help="Number of frontier candidates to expose to the LLM per node selection.",
    )
    args = parser.parse_args()

    problem = socks_and_shoes_problem()
    print(f"Initial state : {len(problem.initial_state)} literals")
    print(f"Goal state    : {', '.join(l.signature for l in problem.goal_state)}")
    print(f"Operators     : {', '.join(op.signature for op in problem.operators)}")
    print()

    llm_config = None
    if args.llm:
        llm_config = LLMConfig(
            top_k=args.llm_top_k,
            response_format=args.llm_response_format,
        )
        if args.llm_model:
            llm_config.model = args.llm_model
        if args.llm_base_url:
            llm_config.base_url = args.llm_base_url
        if args.llm_api_key:
            llm_config.api_key = args.llm_api_key
        print(
            f"[INFO] LLM mode ON (model={llm_config.model},"
            f" base_url={llm_config.base_url or 'default'},"
            f" top_k={llm_config.top_k})"
        )

    planner = DPOCLPlanner(
        search_strategy="best_first",
        heuristic="zero",
        verbose=True,
        use_llm=args.llm,
        llm_config=llm_config,
    )

    solutions = list(planner.solve(problem, max_solutions=1, timeout=10.0))
    stats = planner.get_statistics()

    print()
    print(f"Solutions found: {len(solutions)}")
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    for i, plan in enumerate(solutions, 1):
        print(f"\nSolution {i}:")
        for step in plan.to_execution_sequence():
            print(f"  - {step.signature}")

    # --- smoke assertions ---
    assert isinstance(stats, dict), "get_statistics() must return a dict"
    assert "time_elapsed" in stats, "stats must include time_elapsed"

    if args.expect_solution:
        assert len(solutions) > 0, "Expected at least one solution"
        print("\n[PASS] Solution assertion passed.")
    elif len(solutions) == 0:
        print("\n[INFO] No solutions (planner stub). Pass --expect-solution once solve() is implemented.")

    print("[PASS] Smoke test passed.")


if __name__ == "__main__":
    main()
