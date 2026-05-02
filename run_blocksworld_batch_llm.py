#!/usr/bin/env python3
"""Batch-run Blocksworld instances through the LLM-driven planner.

Iterates over numeric instance ids in an **inclusive** range
``[--instance-start, --instance-end]`` under
``<blocksworld-root>/<problem-type>/instance-{N}.pddl`` (default problem type
``generated_basic``), runs the same LLM planner code path as
``run_blocksworld_one_llm.py``, and writes a single JSON file containing
per-instance planner statistics plus aggregate means/medians.

Per the spec, instances that hit the per-run timeout are recorded but
**excluded from the aggregate averages**; LLM/compile errors are treated the
same way and listed separately.

Pointing at a local vLLM endpoint serving Qwen3-32B-FP8 (port must match
``scripts/start_vllm_qwen3_32b_fp8.sh``, default ``PORT=8877``)::

    export OPENAI_BASE_URL=http://localhost:8877/v1
    python run_blocksworld_batch_llm.py \\
        --instance-start 1 --instance-end 50 \\
        --output-json results/llm_batch_1_50.json \\
        --llm-model Qwen/Qwen3-32B-FP8 \\
        --llm-response-format json_object
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from pydpocl.domain.compiler import compile_domain_and_problem
from pydpocl.planning.llm_policy import DEFAULT_MODEL, LLMConfig
from pydpocl.planning.planner import DPOCLPlanner

from test_blocksworld import HEURISTICS

INSTANCE_RE = re.compile(r"^instance-(\d+)\.pddl$")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BLOCKSWORLD_ROOT = SCRIPT_DIR / "instances" / "blocksworld"
DOMAIN_FILE_NAME = "generated_domain.pddl"

# Numeric stat keys aggregated across non-timeout / non-error runs.
AGGREGATE_KEYS: tuple[str, ...] = (
    "nodes_visited",
    "nodes_expanded",
    "duplicates_pruned",
    "peak_frontier_size",
    "llm_calls",
    "llm_retries",
    "wall_time_s",
    "plan_length",
)


def find_domain_for_problem(problem_path: Path, root: Path) -> Path:
    """Walk up from a problem file to the closest ``generated_domain.pddl``."""
    for parent in [problem_path.parent, *problem_path.parents]:
        candidate = parent / DOMAIN_FILE_NAME
        if candidate.exists():
            return candidate
        if parent == root:
            break
    return root / DOMAIN_FILE_NAME


def build_instance_paths(
    blocksworld_root: Path,
    problem_type: str,
    instance_start: int,
    instance_end: int,
) -> tuple[dict[int, Path], list[int]]:
    """Return (existing instance_id -> path, missing ids) for the inclusive range."""
    type_dir = blocksworld_root / problem_type
    if not type_dir.is_dir():
        raise SystemExit(
            f"[ERROR] Problem-type directory not found: {type_dir}"
        )

    available: dict[int, Path] = {}
    for entry in type_dir.iterdir():
        if not entry.is_file():
            continue
        match = INSTANCE_RE.match(entry.name)
        if match:
            available[int(match.group(1))] = entry

    existing: dict[int, Path] = {}
    missing: list[int] = []
    for idx in range(instance_start, instance_end + 1):
        if idx in available:
            existing[idx] = available[idx]
        else:
            missing.append(idx)
    return existing, missing


def run_instance(
    instance_id: int,
    problem_path: Path,
    domain_path: Path,
    *,
    heuristic: str,
    timeout: float,
    llm_config: LLMConfig,
    verbose: bool,
) -> dict[str, Any]:
    """Run one instance and return a stats dict (no exceptions propagate)."""
    record: dict[str, Any] = {
        "instance_id": instance_id,
        "problem_path": str(problem_path),
        "domain_path": str(domain_path),
        "status": "unknown",
        "solved": False,
        "timeout_reached": False,
        "wall_time_s": 0.0,
        "plan_length": None,
        "nodes_visited": 0,
        "nodes_expanded": 0,
        "duplicates_pruned": 0,
        "peak_frontier_size": 0,
        "llm_calls": 0,
        "llm_retries": 0,
        "error": None,
    }

    try:
        problem = compile_domain_and_problem(domain_path, problem_path)
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = f"compile_failed: {exc}"
        if verbose:
            traceback.print_exc()
        return record

    planner = DPOCLPlanner(
        search_strategy="best_first",  # ignored when use_llm=True
        heuristic=heuristic,
        flaw_order="lcfr",  # ignored when use_llm=True
        verbose=verbose,
        use_llm=True,
        llm_config=llm_config,
    )

    wall_start = time.perf_counter()
    try:
        solutions = list(planner.solve(problem, max_solutions=1, timeout=timeout))
    except Exception as exc:  # noqa: BLE001
        record["wall_time_s"] = time.perf_counter() - wall_start
        record["status"] = "error"
        record["error"] = f"solve_failed: {exc}"
        if verbose:
            traceback.print_exc()
        # Still try to read whatever stats the planner accumulated.
        try:
            stats = planner.get_statistics()
            record["nodes_visited"] = stats["nodes_visited"]
            record["nodes_expanded"] = stats["nodes_expanded"]
            record["duplicates_pruned"] = stats["duplicates_pruned"]
            record["peak_frontier_size"] = stats["peak_frontier_size"]
            record["llm_calls"] = stats["llm_calls"]
            record["llm_retries"] = stats["llm_retries"]
            record["timeout_reached"] = stats["timeout_reached"]
        except Exception:  # noqa: BLE001
            pass
        return record

    record["wall_time_s"] = time.perf_counter() - wall_start
    stats = planner.get_statistics()
    record["nodes_visited"] = stats["nodes_visited"]
    record["nodes_expanded"] = stats["nodes_expanded"]
    record["duplicates_pruned"] = stats["duplicates_pruned"]
    record["peak_frontier_size"] = stats["peak_frontier_size"]
    record["llm_calls"] = stats["llm_calls"]
    record["llm_retries"] = stats["llm_retries"]
    record["timeout_reached"] = bool(stats["timeout_reached"])

    if solutions:
        record["solved"] = True
        record["plan_length"] = len(solutions[0].to_execution_sequence())
        record["status"] = "solved"
    elif record["timeout_reached"]:
        record["status"] = "timeout"
    else:
        record["status"] = "unsolved"

    return record


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean/median over non-timeout, non-error rows."""
    eligible = [r for r in runs if r["status"] in ("solved", "unsolved")]

    means: dict[str, float | None] = {}
    medians: dict[str, float | None] = {}
    for key in AGGREGATE_KEYS:
        values = [r[key] for r in eligible if r.get(key) is not None]
        if values:
            means[key] = float(statistics.fmean(values))
            medians[key] = float(statistics.median(values))
        else:
            means[key] = None
            medians[key] = None

    return {
        "n_used": len(eligible),
        "n_solved": sum(1 for r in eligible if r["status"] == "solved"),
        "n_unsolved": sum(1 for r in eligible if r["status"] == "unsolved"),
        "mean": means,
        "median": medians,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--problem-type",
        default="generated_basic",
        help="Problem-type directory under <blocksworld-root> (default: generated_basic).",
    )
    parser.add_argument(
        "--instance-start",
        type=int,
        required=True,
        help="Inclusive start of the instance-id range (matches instance-N.pddl).",
    )
    parser.add_argument(
        "--instance-end",
        type=int,
        required=True,
        help="Inclusive end of the instance-id range (matches instance-N.pddl).",
    )
    parser.add_argument(
        "--blocksworld-root",
        type=Path,
        default=DEFAULT_BLOCKSWORLD_ROOT,
        help=(
            "Root directory containing problem-type subfolders "
            f"(default: {DEFAULT_BLOCKSWORLD_ROOT})."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to write the aggregated JSON results.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-run timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--heuristic",
        default="zero",
        choices=HEURISTICS,
        help="Heuristic used for the f = g + h ranking exposed to the LLM (default: zero).",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help=(
            "Model identifier (default: $OPENAI_MODEL / $LLM_MODEL or"
            f" {DEFAULT_MODEL}). For vLLM, pass the served model id, e.g."
            " 'Qwen/Qwen3-32B-FP8'."
        ),
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help=(
            "OpenAI-compatible base URL (default: $OPENAI_BASE_URL /"
            " $LLM_BASE_URL). For local vLLM match start_vllm_qwen3_32b_fp8.sh"
            " (default port 8877), e.g. 'http://localhost:8877/v1'."
        ),
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="Override $OPENAI_API_KEY (local servers usually ignore this).",
    )
    parser.add_argument(
        "--llm-response-format",
        choices=("json_schema", "json_object", "none"),
        default="json_schema",
        help="How to constrain LLM outputs (default: json_schema).",
    )
    parser.add_argument(
        "--llm-single-resolver",
        dest="llm_single_resolver",
        action="store_true",
        default=True,
        help="Pick exactly one resolver per expansion (default).",
    )
    parser.add_argument(
        "--llm-expand-all",
        dest="llm_single_resolver",
        action="store_false",
        help="Legacy mode: enqueue every consistent successor for the chosen flaw.",
    )
    parser.add_argument(
        "--llm-top-k",
        type=int,
        default=10,
        help="Frontier candidates exposed to the LLM per node selection (default: 10).",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=3,
        help="Structured-output retry budget per LLM call (default: 3).",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the LLM (default: 0.0).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress planner verbose output.",
    )
    args = parser.parse_args()

    if args.instance_end < args.instance_start:
        raise SystemExit(
            f"[ERROR] --instance-end ({args.instance_end}) is less than"
            f" --instance-start ({args.instance_start})."
        )

    blocksworld_root: Path = args.blocksworld_root.resolve()
    existing, missing = build_instance_paths(
        blocksworld_root,
        args.problem_type,
        args.instance_start,
        args.instance_end,
    )

    llm_config = LLMConfig(
        top_k=args.llm_top_k,
        max_retries=args.llm_max_retries,
        temperature=args.llm_temperature,
        response_format=args.llm_response_format,
        single_resolver=args.llm_single_resolver,
    )
    if args.llm_model:
        llm_config.model = args.llm_model
    if args.llm_base_url:
        llm_config.base_url = args.llm_base_url
    if args.llm_api_key:
        llm_config.api_key = args.llm_api_key

    print(
        f"[batch] problem_type={args.problem_type} "
        f"range=[{args.instance_start}..{args.instance_end}] "
        f"found={len(existing)} missing={len(missing)} "
        f"timeout={args.timeout}s "
        f"llm_model={llm_config.model} "
        f"llm_base_url={llm_config.base_url or 'default'}",
        file=sys.stderr,
    )

    runs: list[dict[str, Any]] = []
    sorted_ids = sorted(existing.keys())
    for position, instance_id in enumerate(sorted_ids, start=1):
        problem_path = existing[instance_id]
        domain_path = find_domain_for_problem(problem_path, blocksworld_root)
        print(
            f"[batch] ({position}/{len(sorted_ids)}) instance-{instance_id}"
            f" -> {problem_path.relative_to(blocksworld_root)}",
            file=sys.stderr,
            flush=True,
        )
        record = run_instance(
            instance_id=instance_id,
            problem_path=problem_path,
            domain_path=domain_path,
            heuristic=args.heuristic,
            timeout=args.timeout,
            llm_config=llm_config,
            verbose=not args.quiet,
        )
        runs.append(record)
        print(
            f"[batch]   status={record['status']} solved={record['solved']}"
            f" wall={record['wall_time_s']:.2f}s"
            f" visited={record['nodes_visited']}"
            f" expanded={record['nodes_expanded']}"
            f" llm_calls={record['llm_calls']}",
            file=sys.stderr,
            flush=True,
        )

    aggregate = aggregate_runs(runs)
    output: dict[str, Any] = {
        "meta": {
            "problem_type": args.problem_type,
            "instance_start": args.instance_start,
            "instance_end": args.instance_end,
            "blocksworld_root": str(blocksworld_root),
            "timeout_s": args.timeout,
            "heuristic": args.heuristic,
            "llm_model": llm_config.model,
            "llm_base_url": llm_config.base_url,
            "llm_response_format": llm_config.response_format,
            "llm_single_resolver": llm_config.single_resolver,
            "llm_top_k": llm_config.top_k,
            "llm_max_retries": llm_config.max_retries,
            "llm_temperature": llm_config.temperature,
        },
        "counts": {
            "requested": args.instance_end - args.instance_start + 1,
            "found": len(existing),
            "missing": len(missing),
            "timeout": sum(1 for r in runs if r["status"] == "timeout"),
            "error": sum(1 for r in runs if r["status"] == "error"),
            "solved": sum(1 for r in runs if r["status"] == "solved"),
            "unsolved": sum(1 for r in runs if r["status"] == "unsolved"),
        },
        "missing_instance_ids": missing,
        "timeout_instance_ids": [r["instance_id"] for r in runs if r["status"] == "timeout"],
        "error_instance_ids": [r["instance_id"] for r in runs if r["status"] == "error"],
        "aggregate": aggregate,
        "runs": runs,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as fh:
        json.dump(output, fh, indent=2)

    print(
        f"[batch] wrote {args.output_json}"
        f"  used={aggregate['n_used']}/{len(runs)}"
        f" timeouts={output['counts']['timeout']}"
        f" errors={output['counts']['error']}"
        f" missing={output['counts']['missing']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
