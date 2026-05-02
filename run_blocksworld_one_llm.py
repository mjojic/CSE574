#!/usr/bin/env python3
"""Run a single Blocksworld instance through the LLM-driven planner.

Reuses the discovery/compilation helpers in ``test_blocksworld.py`` so the
problem layout stays consistent. The planner is configured with ``use_llm=True``,
which delegates next-node selection, flaw selection, and (by default) one
single-resolver expansion per step to the policy in
``pydpocl.planning.llm_policy``. The same code path supports the official
OpenAI API and any OpenAI-compatible server (vLLM, llama.cpp, etc.).

Pointing at a local vLLM endpoint serving Qwen3-32B-FP8::

    export OPENAI_BASE_URL=http://localhost:8000/v1
    python run_blocksworld_one_llm.py \\
        --problem-type generated_basic --instance 1 \\
        --llm-model Qwen/Qwen3-32B-FP8 \\
        --llm-response-format json_object

Or pass the URL directly::

    python run_blocksworld_one_llm.py \\
        --problem-type generated_basic --instance 1 \\
        --llm-base-url http://localhost:8000/v1 \\
        --llm-model Qwen/Qwen3-32B-FP8

Original OpenAI usage still works unchanged (just set ``OPENAI_API_KEY``)::

    python run_blocksworld_one_llm.py --problem-type mystery --instance 3 --timeout 120
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pydpocl.domain.compiler import compile_domain_and_problem
from pydpocl.planning.llm_policy import DEFAULT_MODEL, LLMConfig
from pydpocl.planning.planner import DPOCLPlanner

from test_blocksworld import (
    BLOCKSWORLD_ROOT,
    HEURISTICS,
    INSTANCE_RE,
    ProblemCase,
    discover_problem_sets,
)


def find_instance(problem_type: str, instance_id: int) -> ProblemCase:
    """Look up a single ProblemCase by problem_type and numeric instance id."""
    discovered = discover_problem_sets(BLOCKSWORLD_ROOT)
    if problem_type not in discovered:
        valid = ", ".join(sorted(discovered)) or "(none)"
        raise SystemExit(
            f"[ERROR] Unknown --problem-type {problem_type!r}. Available: {valid}"
        )

    for case in discovered[problem_type]:
        match = INSTANCE_RE.search(case.problem_path.name)
        if match and int(match.group(1)) == instance_id:
            return case

    raise SystemExit(
        f"[ERROR] No instance #{instance_id} under {BLOCKSWORLD_ROOT / problem_type}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem-type",
        required=True,
        help="Top-level blocksworld problem type, e.g. generated_basic / mystery",
    )
    parser.add_argument(
        "--instance",
        type=int,
        required=True,
        help="Numeric instance id (matches instance-N.pddl)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-run timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--heuristic",
        default="zero",
        choices=HEURISTICS,
        help="Heuristic used for the f = g + h ranking exposed to the LLM (default: zero)",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help=(
            "Model identifier (default: $OPENAI_MODEL / $LLM_MODEL or"
            f" {DEFAULT_MODEL}). For vLLM, pass the model id served by the"
            " endpoint, e.g. 'Qwen/Qwen3-32B-FP8'."
        ),
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help=(
            "OpenAI-compatible base URL (default: $OPENAI_BASE_URL /"
            " $LLM_BASE_URL). For vLLM use e.g. 'http://localhost:8000/v1'."
        ),
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help=(
            "Override $OPENAI_API_KEY. Local servers usually ignore this; a"
            " placeholder is sent when a base URL is set without a key."
        ),
    )
    parser.add_argument(
        "--llm-response-format",
        choices=("json_schema", "json_object", "none"),
        default="json_schema",
        help=(
            "How to constrain LLM outputs (default: json_schema). Use"
            " json_object for vLLM builds that lack strict structured-output"
            " support; the policy auto-falls back if the server rejects"
            " json_schema."
        ),
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
        help=(
            "Legacy mode: enqueue every consistent successor for the chosen"
            " flaw rather than relying on LLM single-resolver selection."
        ),
    )
    parser.add_argument(
        "--llm-top-k",
        type=int,
        default=10,
        help="Frontier candidates exposed to the LLM per node selection (default: 10)",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=3,
        help="Structured-output retry budget per LLM call (default: 3)",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the LLM (default: 0.0)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress planner verbose output (still prints the final summary).",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the executable plan steps after the run.",
    )
    args = parser.parse_args()

    case = find_instance(args.problem_type, args.instance)

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

    expansion_label = (
        "single-resolver+reprompt" if llm_config.single_resolver else "expand-all"
    )

    print("Running one Blocksworld instance in LLM mode")
    print(f"  problem      : {case.problem_path}")
    print(f"  domain       : {case.domain_path}")
    print(f"  heuristic    : {args.heuristic}")
    print(f"  timeout      : {args.timeout}s")
    print(f"  llm model    : {llm_config.model}")
    print(f"  llm base url : {llm_config.base_url or 'default'}")
    print(f"  llm top_k    : {llm_config.top_k}")
    print(f"  llm format   : {llm_config.response_format}")
    print(f"  llm expansion: {expansion_label}")

    print("\nCompiling problem...")
    problem = compile_domain_and_problem(case.domain_path, case.problem_path)
    print(
        f"  init={len(problem.initial_state)}  "
        f"goal={len(problem.goal_state)}  "
        f"ops={len(problem.operators)}"
    )

    planner = DPOCLPlanner(
        search_strategy="best_first",  # ignored when use_llm=True (LLMFrontier is used)
        heuristic=args.heuristic,
        flaw_order="lcfr",  # ignored when use_llm=True
        verbose=not args.quiet,
        use_llm=True,
        llm_config=llm_config,
    )

    print("\nSolving with LLM-driven planner...")
    wall_start = time.perf_counter()
    solutions = list(planner.solve(problem, max_solutions=1, timeout=args.timeout))
    wall_elapsed = time.perf_counter() - wall_start
    stats = planner.get_statistics()

    print("\nResult:")
    print(f"  solved          : {'yes' if solutions else ('TIMEOUT' if stats['timeout_reached'] else 'no')}")
    print(f"  wall time       : {wall_elapsed:.3f}s")
    print(f"  nodes visited   : {stats['nodes_visited']}")
    print(f"  nodes expanded  : {stats['nodes_expanded']}")
    print(f"  duplicates pruned: {stats['duplicates_pruned']}")
    print(f"  llm calls       : {stats['llm_calls']}")
    print(f"  llm retries     : {stats['llm_retries']}")

    if solutions:
        plan = solutions[0]
        steps = plan.to_execution_sequence()
        print(f"  plan length     : {len(steps)}")
        if args.print_plan:
            print("\nPlan:")
            for i, step in enumerate(steps, 1):
                print(f"  {i:2d}. {step.signature}")
    else:
        print("  plan length     : -")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
