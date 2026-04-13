#!/usr/bin/env python3
"""Benchmark planner strategy/heuristic combos on blocksworld instances.

Auto-discovers problem sets under instances/blocksworld and runs configurable
strategy/heuristic combinations.

Usage:
    python test_blocksworld.py
    python test_blocksworld.py --timeout 30
    python test_blocksworld.py --strategies best_first,depth_first --heuristics zero,goal_count
    python test_blocksworld.py --problem-types generated_basic --max-per-type 20
    python test_blocksworld.py --problem-types mystery --per-type-limits mystery:all
    python test_blocksworld.py --strategies best_first --max-per-type 10
    python test_blocksworld.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydpocl.domain.compiler import compile_domain_and_problem
from pydpocl.domain.strips_pddl import GroundPlanningProblem
from pydpocl.planning.planner import DPOCLPlanner

BLOCKSWORLD_ROOT = Path("instances/blocksworld")
DEFAULT_DOMAIN = BLOCKSWORLD_ROOT / "generated_domain.pddl"
DOMAIN_FILE_NAME = "generated_domain.pddl"
INSTANCE_RE = re.compile(r"instance-(\d+)\.pddl$")
RESULTS_DIR = Path("results")

STRATEGIES = ("best_first", "breadth_first", "depth_first")
HEURISTICS = ("zero", "goal_count", "relaxed_plan")
FLAW_ORDERS = ("mrv", "priority")


@dataclass(frozen=True, slots=True)
class ProblemCase:
    """A discovered planning problem with its dataset/type metadata."""

    problem_type: str
    label: str
    problem_path: Path
    domain_path: Path


def parse_csv_choices(raw: str | None, *, valid: tuple[str, ...], label: str) -> list[str]:
    """Parse comma-separated values and validate against allowed choices."""
    if raw is None:
        return list(valid)

    parsed = [part.strip() for part in raw.split(",") if part.strip()]
    if not parsed:
        raise ValueError(f"{label} list is empty; valid values: {', '.join(valid)}")

    invalid = [item for item in parsed if item not in valid]
    if invalid:
        raise ValueError(
            f"Unknown {label}: {', '.join(invalid)}. Valid values: {', '.join(valid)}"
        )

    # Keep user-specified order but drop accidental duplicates.
    return list(dict.fromkeys(parsed))


def parse_per_type_limits(
    raw: str | None,
    *,
    valid_types: set[str],
) -> dict[str, int | None]:
    """Parse per-type limits like 'generated_basic:20,mystery:all'."""
    if raw is None:
        return {}

    limits: dict[str, int | None] = {}
    entries = [part.strip() for part in raw.split(",") if part.strip()]
    if not entries:
        raise ValueError("per-type-limits is empty")

    for entry in entries:
        if ":" not in entry:
            raise ValueError(
                f"Invalid per-type-limits entry '{entry}'. "
                "Expected format <type>:<count|all>"
            )
        problem_type, limit_text = (part.strip() for part in entry.split(":", 1))
        if problem_type not in valid_types:
            raise ValueError(
                f"Unknown problem type '{problem_type}' in per-type-limits. "
                f"Valid values: {', '.join(sorted(valid_types))}"
            )

        if limit_text.lower() == "all":
            limits[problem_type] = None
            continue

        try:
            limit_value = int(limit_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid limit '{limit_text}' for '{problem_type}'. Use a positive integer or 'all'."
            ) from exc
        if limit_value <= 0:
            raise ValueError(f"Limit for '{problem_type}' must be > 0")
        limits[problem_type] = limit_value

    return limits


def _instance_sort_key(path: Path, root: Path) -> tuple[int, str]:
    """Stable sort key that prioritizes numeric instance id."""
    match = INSTANCE_RE.match(path.name)
    idx = int(match.group(1)) if match else 10**9
    rel = path.relative_to(root).as_posix()
    return (idx, rel)


def find_domain_for_problem(problem_path: Path, root: Path) -> Path:
    """Find the closest generated_domain.pddl walking up from problem path."""
    for parent in [problem_path.parent, *problem_path.parents]:
        candidate = parent / DOMAIN_FILE_NAME
        if candidate.exists():
            return candidate
        if parent == root:
            break
    return DEFAULT_DOMAIN


def discover_problem_sets(root: Path) -> dict[str, list[ProblemCase]]:
    """Discover all instance-*.pddl files grouped by top-level problem type."""
    by_type: dict[str, list[ProblemCase]] = defaultdict(list)
    all_instances = sorted(root.rglob("instance-*.pddl"), key=lambda p: _instance_sort_key(p, root))

    for problem_path in all_instances:
        rel = problem_path.relative_to(root)
        if not rel.parts:
            continue
        problem_type = rel.parts[0]
        domain_path = find_domain_for_problem(problem_path, root)
        case = ProblemCase(
            problem_type=problem_type,
            label=rel.as_posix(),
            problem_path=problem_path,
            domain_path=domain_path,
        )
        by_type[problem_type].append(case)

    return dict(by_type)


def select_problem_cases(
    discovered: dict[str, list[ProblemCase]],
    selected_types: list[str],
    max_per_type: int | None,
    per_type_limits: dict[str, int | None],
) -> list[ProblemCase]:
    """Select instances from each type based on global/per-type limits."""
    selected: list[ProblemCase] = []
    for problem_type in selected_types:
        cases = discovered[problem_type]
        limit = per_type_limits.get(problem_type, max_per_type)
        selected.extend(cases if limit is None else cases[:limit])
    return selected


def run_one(
    strategy: str,
    heuristic: str,
    flaw_order: str,
    problem: GroundPlanningProblem,
    timeout: float,
    verbose: bool,
) -> dict:
    """Solve one pre-compiled instance with one strategy/heuristic; return result dict."""
    planner = DPOCLPlanner(
        search_strategy=strategy,
        heuristic=heuristic,
        flaw_order=flaw_order,
        verbose=verbose,
    )

    wall_start = time.perf_counter()
    solutions = list(planner.solve(problem, max_solutions=1, timeout=timeout))
    wall_elapsed = time.perf_counter() - wall_start

    stats = planner.get_statistics()

    result = {
        "strategy": strategy,
        "heuristic": heuristic,
        "flaw_order": flaw_order,
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


# Module-level worker: required for pickling on Windows (spawn start method).
def _run_one_packed(args: tuple) -> dict:
    """Unpack args and call run_one(); return result tagged with case_index."""
    case_index, strategy, heuristic, flaw_order, problem, timeout = args
    try:
        result = run_one(
            strategy=strategy,
            heuristic=heuristic,
            flaw_order=flaw_order,
            problem=problem,
            timeout=timeout,
            verbose=False,  # suppress verbose in workers; interleaved stdout is unreadable
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "strategy": strategy,
            "heuristic": heuristic,
            "flaw_order": flaw_order,
            "solved": False,
            "plan_length": None,
            "nodes_expanded": 0,
            "nodes_visited": 0,
            "duplicates_pruned": 0,
            "timeout_reached": False,
            "wall_time": 0.0,
            "plan": [],
            "error": str(exc),
        }
    result["_case_index"] = case_index
    return result


def print_table(instance_label: str, results: list[dict]) -> None:
    col_w = [14, 14, 10, 8, 12, 14, 14, 17, 12]
    headers = [
        "Strategy",
        "Heuristic",
        "Flaw order",
        "Solved",
        "Plan length",
        "Nodes visited",
        "Nodes expanded",
        "Duplicates pruned",
        "Time (s)",
    ]
    sep = "  ".join("-" * w for w in col_w)

    print(f"\n{'='*120}")
    print(f"  {instance_label}")
    print(f"{'='*120}")
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print("  " + sep)
    for r in results:
        row = [
            r["strategy"].ljust(col_w[0]),
            r["heuristic"].ljust(col_w[1]),
            r["flaw_order"].ljust(col_w[2]),
            ("yes" if r["solved"] else ("TIMEOUT" if r["timeout_reached"] else "no")).ljust(col_w[3]),
            (str(r["plan_length"]) if r["plan_length"] is not None else "-").ljust(col_w[4]),
            str(r["nodes_visited"]).ljust(col_w[5]),
            str(r["nodes_expanded"]).ljust(col_w[6]),
            str(r["duplicates_pruned"]).ljust(col_w[7]),
            f"{r['wall_time']:.4f}".ljust(col_w[8]),
        ]
        print("  " + "  ".join(row))


def print_summary(case_reports: list[dict], total_cases: int) -> list[dict]:
    """Print a per-combo aggregate summary and return the summary rows as dicts."""
    # Collect all runs across every case, grouped by (strategy, heuristic, flaw_order).
    from collections import defaultdict as _dd

    combo_buckets: dict[tuple, list[dict]] = _dd(list)
    for case in case_reports:
        for run in case["runs"]:
            key = (run["strategy"], run["heuristic"], run["flaw_order"])
            combo_buckets[key].append(run)

    rows: list[dict] = []
    for key in combo_buckets:
        strategy, heuristic, flaw_order = key
        runs = combo_buckets[key]
        n = len(runs)
        solved_runs = [r for r in runs if r["solved"]]
        timeout_runs = [r for r in runs if r["timeout_reached"]]
        n_solved = len(solved_runs)
        n_timeout = len(timeout_runs)
        solve_rate = 100.0 * n_solved / n if n else 0.0

        avg_time_solved = (
            sum(r["wall_time_seconds"] for r in solved_runs) / n_solved
            if n_solved else None
        )
        avg_nodes_solved = (
            sum(r["nodes_visited"] for r in solved_runs) / n_solved
            if n_solved else None
        )
        avg_plan_len = (
            sum(r["plan_length"] for r in solved_runs) / n_solved
            if n_solved else None
        )

        rows.append({
            "strategy": strategy,
            "heuristic": heuristic,
            "flaw_order": flaw_order,
            "total_instances": n,
            "solved": n_solved,
            "timeouts": n_timeout,
            "solve_rate_pct": round(solve_rate, 1),
            "avg_time_solved_s": round(avg_time_solved, 4) if avg_time_solved is not None else None,
            "avg_nodes_visited_solved": round(avg_nodes_solved, 1) if avg_nodes_solved is not None else None,
            "avg_plan_length_solved": round(avg_plan_len, 2) if avg_plan_len is not None else None,
        })

    # Sort: solve_rate desc, then avg_time_solved asc (None sorts last).
    rows.sort(key=lambda r: (
        -r["solve_rate_pct"],
        r["avg_time_solved_s"] if r["avg_time_solved_s"] is not None else float("inf"),
    ))

    col_w = [14, 14, 10, 9, 8, 9, 11, 15, 20, 17]
    headers = [
        "Strategy",
        "Heuristic",
        "Flaw order",
        "Instances",
        "Solved",
        "Timeouts",
        "Solve rate",
        "Avg time (s)",
        "Avg nodes visited",
        "Avg plan length",
    ]
    sep = "  ".join("-" * w for w in col_w)

    print(f"\n{'#'*120}")
    print(f"  SUMMARY  ({total_cases} instance(s))")
    print(f"{'#'*120}")
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print("  " + sep)
    for r in rows:
        avg_t = f"{r['avg_time_solved_s']:.4f}" if r["avg_time_solved_s"] is not None else "-"
        avg_n = f"{r['avg_nodes_visited_solved']:.0f}" if r["avg_nodes_visited_solved"] is not None else "-"
        avg_p = f"{r['avg_plan_length_solved']:.1f}" if r["avg_plan_length_solved"] is not None else "-"
        row = [
            r["strategy"].ljust(col_w[0]),
            r["heuristic"].ljust(col_w[1]),
            r["flaw_order"].ljust(col_w[2]),
            str(r["total_instances"]).ljust(col_w[3]),
            str(r["solved"]).ljust(col_w[4]),
            str(r["timeouts"]).ljust(col_w[5]),
            f"{r['solve_rate_pct']:.1f}%".ljust(col_w[6]),
            avg_t.ljust(col_w[7]),
            avg_n.ljust(col_w[8]),
            avg_p.ljust(col_w[9]),
        ]
        print("  " + "  ".join(row))

    return rows


def print_plan(strategy: str, heuristic: str, flaw_order: str, steps) -> None:
    if not steps:
        return
    print(f"    [{strategy} | {heuristic} | {flaw_order}] plan ({len(steps)} steps):")
    for i, step in enumerate(steps, 1):
        print(f"      {i:2d}. {step.signature}")


def save_run_results(
    *,
    selected_strategies: list[str],
    selected_heuristics: list[str],
    selected_flaw_orders: list[str],
    selected_types: list[str],
    max_per_type: int | None,
    per_type_limits: dict[str, int | None],
    timeout: float,
    total_cases: int,
    case_reports: list[dict],
    summary: list[dict],
    all_passed: bool,
    output_dir: Path = RESULTS_DIR,
) -> Path:
    """Persist one benchmark run to a uniquely named JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S-%f")
    filename = (
        f"blocksworld-{timestamp}"
        f"-s{len(selected_strategies)}"
        f"-h{len(selected_heuristics)}"
        f"-f{len(selected_flaw_orders)}"
        f"-t{len(selected_types)}"
        f"-n{total_cases}.json"
    )
    out_path = output_dir / filename

    payload = {
        "run_timestamp_local": now.isoformat(timespec="seconds"),
        "all_passed": all_passed,
        "configuration": {
            "timeout_seconds": timeout,
            "strategies": selected_strategies,
            "heuristics": selected_heuristics,
            "flaw_orders": selected_flaw_orders,
            "problem_types": selected_types,
            "max_per_type": max_per_type,
            "per_type_limits": {
                k: ("all" if v is None else v) for k, v in sorted(per_type_limits.items())
            },
            "total_cases": total_cases,
        },
        "summary": summary,
        "cases": case_reports,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-run timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help=f"Comma-separated subset of strategies (default: all: {', '.join(STRATEGIES)})",
    )
    parser.add_argument(
        "--heuristics",
        type=str,
        default=None,
        help=f"Comma-separated subset of heuristics (default: all: {', '.join(HEURISTICS)})",
    )
    parser.add_argument(
        "--flaw-orders",
        type=str,
        default=None,
        help=f"Comma-separated subset of flaw selection strategies (default: all: {', '.join(FLAW_ORDERS)})",
    )
    parser.add_argument(
        "--problem-types",
        type=str,
        default=None,
        help="Comma-separated top-level blocksworld problem types (default: all discovered)",
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=None,
        help="Run only first N instances from each selected type (default: all)",
    )
    parser.add_argument(
        "--per-type-limits",
        type=str,
        default=None,
        help="Override limits per type: type:count or type:all (e.g., generated_basic:20,mystery:all)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show planner search output (sequential only)")
    parser.add_argument(
        "--plans",
        action="store_true",
        help="Print the found plan steps after each table",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help=(
            "Number of parallel worker processes (default: number of CPU cores). "
            "Use 1 for sequential execution."
        ),
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("[ERROR] --workers must be >= 1")

    discovered = discover_problem_sets(BLOCKSWORLD_ROOT)
    if not discovered:
        raise SystemExit(f"[ERROR] No blocksworld instances found under {BLOCKSWORLD_ROOT}")

    available_types = tuple(sorted(discovered.keys()))

    if args.max_per_type is not None and args.max_per_type <= 0:
        raise SystemExit("[ERROR] --max-per-type must be > 0")

    try:
        selected_strategies = parse_csv_choices(
            args.strategies, valid=STRATEGIES, label="strategies"
        )
        selected_heuristics = parse_csv_choices(
            args.heuristics, valid=HEURISTICS, label="heuristics"
        )
        selected_flaw_orders = parse_csv_choices(
            args.flaw_orders, valid=FLAW_ORDERS, label="flaw orders"
        )
        selected_types = parse_csv_choices(
            args.problem_types, valid=available_types, label="problem types"
        )
        per_type_limits = parse_per_type_limits(
            args.per_type_limits,
            valid_types=set(available_types),
        )
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    selected_cases = select_problem_cases(
        discovered,
        selected_types=selected_types,
        max_per_type=args.max_per_type,
        per_type_limits=per_type_limits,
    )
    if not selected_cases:
        raise SystemExit("[ERROR] No instances selected. Adjust type filters/limits.")

    workers = args.workers
    parallel = workers > 1

    print("Configuration:")
    print(f"  timeout     : {args.timeout}s")
    print(f"  strategies  : {', '.join(selected_strategies)}")
    print(f"  heuristics  : {', '.join(selected_heuristics)}")
    print(f"  flaw orders : {', '.join(selected_flaw_orders)}")
    print(f"  types       : {', '.join(selected_types)}")
    print(f"  max/type    : {'all' if args.max_per_type is None else args.max_per_type}")
    print(
        "  per-type    : "
        + (
            ", ".join(
                f"{k}:{'all' if v is None else v}"
                for k, v in sorted(per_type_limits.items())
            )
            if per_type_limits
            else "(none)"
        )
    )
    print(f"  total cases : {len(selected_cases)}")
    print(f"  workers     : {workers} ({'parallel' if parallel else 'sequential'})")

    # -----------------------------------------------------------------------
    # Compile each problem exactly once.
    # -----------------------------------------------------------------------
    print("\nCompiling problems...")
    compiled: list[GroundPlanningProblem] = []
    for case in selected_cases:
        problem = compile_domain_and_problem(case.domain_path, case.problem_path)
        compiled.append(problem)
        print(
            f"  {case.label}  |  "
            f"{len(problem.initial_state)} init  "
            f"{len(problem.goal_state)} goal  "
            f"{len(problem.operators)} ops"
        )

    # -----------------------------------------------------------------------
    # Build flat work-item list: (case_index, strategy, heuristic, flaw_order,
    #                              problem, timeout)
    # -----------------------------------------------------------------------
    combos_per_case = len(selected_strategies) * len(selected_heuristics) * len(selected_flaw_orders)
    total_runs = len(selected_cases) * combos_per_case

    work_items: list[tuple] = [
        (case_idx, strategy, heuristic, flaw_order, compiled[case_idx], args.timeout)
        for case_idx, _case in enumerate(selected_cases)
        for strategy in selected_strategies
        for heuristic in selected_heuristics
        for flaw_order in selected_flaw_orders
    ]

    # -----------------------------------------------------------------------
    # Execute: parallel or sequential.
    # -----------------------------------------------------------------------
    print(f"\nRunning {total_runs} benchmark(s) with {workers} worker(s)...")

    # Accumulate results grouped by case index; preserve insertion order.
    case_results: dict[int, list[dict]] = {i: [] for i in range(len(selected_cases))}
    completed = 0

    def _report(r: dict) -> None:
        nonlocal completed
        completed += 1
        case_idx = r["_case_index"]
        status = "solved" if r["solved"] else ("TIMEOUT" if r["timeout_reached"] else "no solution")
        error_tag = f"  ERROR: {r['error']}" if "error" in r else ""
        print(
            f"  [{completed}/{total_runs}] "
            f"{r['strategy']} + {r['heuristic']} + {r['flaw_order']} "
            f"on {selected_cases[case_idx].label} ... "
            f"{status} ({r['nodes_visited']} visited, {r['wall_time']:.3f}s)"
            f"{error_tag}"
        )
        case_results[case_idx].append(r)

    if parallel:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one_packed, item): item for item in work_items}
            for future in as_completed(futures):
                _report(future.result())
    else:
        for item in work_items:
            _report(_run_sequential(item, args.verbose))

    # -----------------------------------------------------------------------
    # Post-process: restore original combination order within each case,
    # print tables, evaluate PASS/FAIL, build JSON report.
    # -----------------------------------------------------------------------
    # Results may arrive out of order when parallel; sort to match work_items order.
    combo_order = [
        (s, h, f)
        for s in selected_strategies
        for h in selected_heuristics
        for f in selected_flaw_orders
    ]

    def _combo_key(r: dict) -> int:
        try:
            return combo_order.index((r["strategy"], r["heuristic"], r["flaw_order"]))
        except ValueError:
            return len(combo_order)

    all_passed = True
    case_reports: list[dict] = []

    for case_idx, case in enumerate(selected_cases):
        results = sorted(case_results[case_idx], key=_combo_key)

        print_table(case.label, results)

        if args.plans:
            print()
            for r in results:
                print_plan(r["strategy"], r["heuristic"], r["flaw_order"], r["plan"])

        any_solved = any(r["solved"] for r in results)
        if not any_solved:
            print(f"\n[FAIL] No selected strategy/heuristic combination solved {case.label}")
            all_passed = False
        else:
            for r in results:
                if r["solved"]:
                    assert r["plan_length"] is not None and r["plan_length"] > 0, \
                        f"{r['strategy']} + {r['heuristic']} + {r['flaw_order']}: plan length must be > 0"
            print(f"\n[PASS] {case.label}")

        case_reports.append(
            {
                "case_label": case.label,
                "problem_type": case.problem_type,
                "problem_path": str(case.problem_path),
                "domain_path": str(case.domain_path),
                "any_solved": any_solved,
                "runs": [
                    {
                        "strategy": r["strategy"],
                        "heuristic": r["heuristic"],
                        "flaw_order": r["flaw_order"],
                        "solved": r["solved"],
                        "timeout_reached": r["timeout_reached"],
                        "plan_length": r["plan_length"],
                        "nodes_visited": r["nodes_visited"],
                        "nodes_expanded": r["nodes_expanded"],
                        "duplicates_pruned": r["duplicates_pruned"],
                        "wall_time_seconds": r["wall_time"],
                        "plan_signatures": [step.signature for step in r["plan"]],
                    }
                    for r in results
                ],
            }
        )

    summary = print_summary(case_reports, total_cases=len(selected_cases))

    saved_path = save_run_results(
        selected_strategies=selected_strategies,
        selected_heuristics=selected_heuristics,
        selected_flaw_orders=selected_flaw_orders,
        selected_types=selected_types,
        max_per_type=args.max_per_type,
        per_type_limits=per_type_limits,
        timeout=args.timeout,
        total_cases=len(selected_cases),
        case_reports=case_reports,
        summary=summary,
        all_passed=all_passed,
    )

    print()
    print(f"Results saved: {saved_path}")
    if all_passed:
        print("[PASS] All instances solved by at least one strategy.")
    else:
        print("[FAIL] Some instances were not solved.")
        raise SystemExit(1)


def _run_sequential(item: tuple, verbose: bool) -> dict:
    """Run a single work item sequentially with the verbose flag respected."""
    case_index, strategy, heuristic, flaw_order, problem, timeout = item
    try:
        result = run_one(
            strategy=strategy,
            heuristic=heuristic,
            flaw_order=flaw_order,
            problem=problem,
            timeout=timeout,
            verbose=verbose,
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "strategy": strategy,
            "heuristic": heuristic,
            "flaw_order": flaw_order,
            "solved": False,
            "plan_length": None,
            "nodes_expanded": 0,
            "nodes_visited": 0,
            "duplicates_pruned": 0,
            "timeout_reached": False,
            "wall_time": 0.0,
            "plan": [],
            "error": str(exc),
        }
    result["_case_index"] = case_index
    return result


if __name__ == "__main__":
    main()
