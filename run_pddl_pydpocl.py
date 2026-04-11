#!/usr/bin/env python3
"""Load STRIPS PDDL domain + problem and run pydpocl DPOCLPlanner.

Example:
  python run_pddl_pydpocl.py \\
    instances/blocksworld/generated_domain.pddl \\
    instances/blocksworld/generated/instance-1.pddl

Requires STRIPS-style PDDL (see pydpocl.domain.strips_pddl for supported subset).
Large domains may need a high --timeout; search is complete but can be slow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydpocl.domain.strips_pddl import compile_strips_pddl
from pydpocl.planning.planner import DPOCLPlanner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "domain",
        type=Path,
        help="Path to the PDDL domain file",
    )
    parser.add_argument(
        "problem",
        type=Path,
        help="Path to the PDDL problem file",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=120.0,
        help="Search timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--max-solutions",
        "-k",
        type=int,
        default=1,
        help="Stop after this many plans (default: 1)",
    )
    parser.add_argument(
        "--strategy",
        default="best_first",
        choices=["best_first", "breadth_first", "depth_first"],
        help="Frontier strategy",
    )
    parser.add_argument(
        "--heuristic",
        default="goal_count",
        choices=["zero", "goal_count", "relaxed_plan"],
        help="Heuristic for best_first (ignored ordering-wise for BFS/DFS)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Planner verbose output",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable structural duplicate pruning (slower, for debugging)",
    )
    parser.add_argument(
        "--flaw-order",
        choices=["mrv", "priority"],
        default="mrv",
        help="Open-condition selection: mrv = fewest resolvers first (default, faster), "
        "priority = use built-in flaw priority",
    )
    args = parser.parse_args()

    if not args.domain.is_file():
        print(f"Domain not found: {args.domain}", file=sys.stderr)
        sys.exit(1)
    if not args.problem.is_file():
        print(f"Problem not found: {args.problem}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading:\n  domain  {args.domain}\n  problem {args.problem}\n")
    problem = compile_strips_pddl(args.domain, args.problem)
    print(
        f"Compiled: {len(problem.initial_state)} initial literals, "
        f"{len(problem.goal_state)} goal literals, "
        f"{len(problem.operators)} ground operators\n"
    )

    planner = DPOCLPlanner(
        search_strategy=args.strategy,
        heuristic=args.heuristic,
        verbose=args.verbose,
        dedupe_structural=not args.no_dedupe,
        flaw_order=args.flaw_order,
    )

    solutions = list(
        planner.solve(
            problem,
            max_solutions=args.max_solutions,
            timeout=args.timeout,
        )
    )
    stats = planner.get_statistics()

    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    if not solutions:
        print("\nNo solution within the time limit (or unsatisfiable).")
        sys.exit(2)

    for i, plan in enumerate(solutions, 1):
        print(f"\nSolution {i} ({len(plan.to_execution_sequence())} primitive steps):")
        for j, step in enumerate(plan.to_execution_sequence(), 1):
            print(f"  {j}. {step.signature}")


if __name__ == "__main__":
    main()
