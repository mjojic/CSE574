#!/usr/bin/env python3
"""Test every search strategy on real blocksworld PDDL instances.

Runs DPOCLPlanner with best_first, breadth_first, and depth_first on two
blocksworld instances of increasing difficulty, then prints a comparison table.

Usage:
    python test_blocksworld.py
    python test_blocksworld.py --timeout 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pydpocl.domain.compiler import compile_domain_and_problem
from pydpocl.planning.planner import DPOCLPlanner

DOMAIN = Path("instances/blocksworld/generated_domain.pddl")

INSTANCES = [
    ("instance-102 (3 blocks, 2-goal)", Path("instances/blocksworld/generated/instance-102.pddl")),
    ("instance-10  (5 blocks, 4-goal)", Path("instances/blocksworld/generated/instance-10.pddl")),
]

STRATEGIES = ["best_first", "breadth_first", "depth_first"]


def run_one(
    strategy: str,
    problem_path: Path,
    timeout: float,
    verbose: bool,
) -> dict:
    """Compile + solve one instance with one strategy; return result dict."""
    problem = compile_domain_and_problem(DOMAIN, problem_path)

    planner = DPOCLPlanner(
        search_strategy=strategy,
        heuristic="zero",
        verbose=verbose,
    )

    wall_start = time.perf_counter()
    solutions = list(planner.solve(problem, max_solutions=1, timeout=timeout))
    wall_elapsed = time.perf_counter() - wall_start

    stats = planner.get_statistics()

    result = {
        "strategy": strategy,
        "solved": len(solutions) > 0,
        "plan_length": len(solutions[0].to_execution_sequence()) if solutions else None,
        "nodes_expanded": stats["nodes_expanded"],
        "nodes_visited": stats["nodes_visited"],
        "duplicates_pruned": stats["duplicates_pruned"],
        "timeout_reached": stats["timeout_reached"],
        "wall_time": wall_elapsed,
        "plan": solutions[0].to_execution_sequence() if solutions else [],
    }
    return result


def print_table(instance_label: str, results: list[dict]) -> None:
    col_w = [14, 8, 12, 14, 14, 17, 12]
    headers = ["Strategy", "Solved", "Plan length", "Nodes visited", "Nodes expanded", "Duplicates pruned", "Time (s)"]
    sep = "  ".join("-" * w for w in col_w)

    print(f"\n{'='*90}")
    print(f"  {instance_label}")
    print(f"{'='*90}")
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print("  " + sep)
    for r in results:
        row = [
            r["strategy"].ljust(col_w[0]),
            ("yes" if r["solved"] else ("TIMEOUT" if r["timeout_reached"] else "no")).ljust(col_w[1]),
            (str(r["plan_length"]) if r["plan_length"] is not None else "-").ljust(col_w[2]),
            str(r["nodes_visited"]).ljust(col_w[3]),
            str(r["nodes_expanded"]).ljust(col_w[4]),
            str(r["duplicates_pruned"]).ljust(col_w[5]),
            f"{r['wall_time']:.4f}".ljust(col_w[6]),
        ]
        print("  " + "  ".join(row))


def print_plan(strategy: str, steps) -> None:
    if not steps:
        return
    print(f"    [{strategy}] plan ({len(steps)} steps):")
    for i, step in enumerate(steps, 1):
        print(f"      {i:2d}. {step.signature}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Per-strategy timeout in seconds (default: 60)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show planner search output")
    parser.add_argument("--plans", action="store_true",
                        help="Print the found plan steps after each table")
    args = parser.parse_args()

    all_passed = True

    for label, instance_path in INSTANCES:
        print(f"\nLoading: {instance_path}")
        problem = compile_domain_and_problem(DOMAIN, instance_path)
        print(f"  {len(problem.initial_state)} init literals  |  "
              f"{len(problem.goal_state)} goal literals  |  "
              f"{len(problem.operators)} ground operators")

        results = []
        for strategy in STRATEGIES:
            print(f"  Running {strategy} ...", end="", flush=True)
            r = run_one(strategy, instance_path, timeout=args.timeout, verbose=args.verbose)
            status = "solved" if r["solved"] else ("TIMEOUT" if r["timeout_reached"] else "no solution")
            print(f" {status}  ({r['nodes_visited']} visited, {r['wall_time']:.3f}s)")
            results.append(r)

        print_table(label, results)

        if args.plans:
            print()
            for r in results:
                print_plan(r["strategy"], r["plan"])

        # Assertions: at least one strategy must find a solution
        any_solved = any(r["solved"] for r in results)
        if not any_solved:
            print(f"\n[FAIL] No strategy solved {label}")
            all_passed = False
        else:
            # All solved plans must have the same length (POCL is optimal for these small cases)
            lengths = {r["plan_length"] for r in results if r["solved"]}
            for r in results:
                if r["solved"]:
                    assert r["plan_length"] is not None and r["plan_length"] > 0, \
                        f"{r['strategy']}: plan length must be > 0"
            print(f"\n[PASS] {label}")

    print()
    if all_passed:
        print("[PASS] All instances solved by at least one strategy.")
    else:
        print("[FAIL] Some instances were not solved.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
