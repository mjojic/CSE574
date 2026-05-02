#!/usr/bin/env python3
"""Master benchmark runner for the PyDPOCL planner.

Replaces ``run_blocksworld_one_llm.py``, ``run_blocksworld_batch_llm.py``, and
``test_blocksworld.py`` with a single domain-agnostic driver.

Highlights
----------
* Auto-discovers domains/problem-sets under ``instances/``.
* Sweeps any subset of ``(heuristic, flaw-selection, resolver)`` combos.
  Classical-only by default; LLM combos opt-in per-axis or via shorthand
  (``--include-llm`` / ``--all-combos``).
* Bounded parallel pool. Hybrid executor: ``ProcessPoolExecutor`` when no
  combo touches the LLM, ``ThreadPoolExecutor`` otherwise (override with
  ``--executor``).
* Per-instance/combo result rows + per-combo aggregates (means/medians)
  written to a single JSON in ``results/`` and a pretty stdout table.
* Optional ``--trials K`` repeats the whole grid in the same pool.

Examples
--------
Default classical sweep on blocksworld/generated_basic, instances 1-50::

    python run.py --domain blocksworld --problem-set generated_basic \
        --instance-start 1 --instance-end 50 --max-concurrency 8

Full 48-combo grid against a local vLLM endpoint::

    set OPENAI_BASE_URL=http://localhost:8877/v1
    python run.py --domain blocksworld --problem-set generated_basic \
        --instances 1-20 --all-combos \
        --llm-model Qwen/Qwen3-32B-FP8 --llm-response-format json_object \
        --executor threads --max-concurrency 16

Targeted: only LLM-as-resolver with add+ff heuristics::

    python run.py --domain depots --problem-set generated_basic \
        --instances 1-30 \
        --heuristics add,ff --flaw-selection-strats lcfr --resolver-strats llm \
        --max-concurrency 4 --trials 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydpocl.domain.compiler import compile_domain_and_problem
from pydpocl.planning.llm_policy import DEFAULT_MODEL, LLMConfig
from pydpocl.planning.planner import DPOCLPlanner

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
INSTANCES_ROOT = SCRIPT_DIR / "instances"
RESULTS_DIR = SCRIPT_DIR / "results"
DOMAIN_FILE_NAME = "generated_domain.pddl"
INSTANCE_RE = re.compile(r"^instance-(\d+)\.pddl$")

HEURISTICS: tuple[str, ...] = ("oc", "fc", "tc", "ps", "add", "max", "ff", "llm")
FLAW_SELECTION_STRATS: tuple[str, ...] = ("lcfr", "zlifo", "lifo", "fifo", "random", "llm")
RESOLVER_STRATS: tuple[str, ...] = ("enumerate", "llm")

CLASSICAL_HEURISTICS: tuple[str, ...] = tuple(h for h in HEURISTICS if h != "llm")
CLASSICAL_FLAW_SELECTION_STRATS: tuple[str, ...] = tuple(
    f for f in FLAW_SELECTION_STRATS if f != "llm"
)
CLASSICAL_RESOLVER_STRATS: tuple[str, ...] = tuple(
    r for r in RESOLVER_STRATS if r != "llm"
)

# Numeric stat keys aggregated across non-timeout / non-error rows.
AGGREGATE_KEYS: tuple[str, ...] = (
    "wall_time_s",
    "nodes_visited",
    "nodes_expanded",
    "duplicates_pruned",
    "peak_frontier_size",
    "llm_calls",
    "llm_retries",
    "plan_length",
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _has_instance_files(directory: Path) -> bool:
    """Return True if `directory` contains at least one ``instance-N.pddl``."""
    if not directory.is_dir():
        return False
    for entry in directory.rglob("instance-*.pddl"):
        if INSTANCE_RE.match(entry.name):
            return True
    return False


def discover_domains(instances_root: Path) -> list[str]:
    """List immediate subdirs of ``instances/`` that contain instance files."""
    if not instances_root.is_dir():
        return []
    domains: list[str] = []
    for entry in sorted(instances_root.iterdir()):
        if entry.is_dir() and _has_instance_files(entry):
            domains.append(entry.name)
    return domains


def discover_problem_sets(domain_root: Path) -> list[str]:
    """List immediate subdirs of a domain that contain instance files."""
    if not domain_root.is_dir():
        return []
    sets: list[str] = []
    for entry in sorted(domain_root.iterdir()):
        if entry.is_dir() and _has_instance_files(entry):
            sets.append(entry.name)
    return sets


def find_domain_pddl(problem_path: Path, domain_root: Path) -> Path:
    """Walk up from ``problem_path`` to the closest ``generated_domain.pddl``."""
    for parent in [problem_path.parent, *problem_path.parents]:
        candidate = parent / DOMAIN_FILE_NAME
        if candidate.exists():
            return candidate
        if parent == domain_root:
            break
    fallback = domain_root / DOMAIN_FILE_NAME
    if not fallback.exists():
        raise FileNotFoundError(
            f"No {DOMAIN_FILE_NAME} found for {problem_path} under {domain_root}"
        )
    return fallback


def discover_instances(problem_set_dir: Path) -> dict[int, Path]:
    """Return ``{instance_id: path}`` for every ``instance-N.pddl`` directly under dir."""
    available: dict[int, Path] = {}
    if not problem_set_dir.is_dir():
        return available
    for entry in problem_set_dir.iterdir():
        if not entry.is_file():
            continue
        match = INSTANCE_RE.match(entry.name)
        if match:
            available[int(match.group(1))] = entry
    return available


# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------


def parse_csv_choices(
    raw: str | None,
    *,
    valid: tuple[str, ...],
    default: tuple[str, ...],
    label: str,
) -> list[str]:
    """Parse a CSV list of choices.

    ``None`` yields the default. The literal ``all`` expands to ``valid``.
    Duplicates are dropped while preserving user order.
    """
    if raw is None:
        return list(default)
    if raw.strip().lower() == "all":
        return list(valid)
    parsed = [part.strip() for part in raw.split(",") if part.strip()]
    if not parsed:
        raise ValueError(
            f"{label} list is empty; valid values: {', '.join(valid)} (or 'all')"
        )
    invalid = [item for item in parsed if item not in valid]
    if invalid:
        raise ValueError(
            f"Unknown {label}: {', '.join(invalid)}."
            f" Valid values: {', '.join(valid)} (or 'all')."
        )
    return list(dict.fromkeys(parsed))


def parse_instance_spec(spec: str) -> list[int]:
    """Parse a CSV mix of ids and ranges, e.g. ``1,2,5-10,42``."""
    out: set[int] = set()
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError as exc:
                raise ValueError(f"Bad instance range '{token}'") from exc
            if hi < lo:
                raise ValueError(f"Range '{token}' has end < start")
            out.update(range(lo, hi + 1))
        else:
            try:
                out.add(int(token))
            except ValueError as exc:
                raise ValueError(f"Bad instance id '{token}'") from exc
    return sorted(out)


def select_instances(
    available: dict[int, Path],
    *,
    instance: int | None,
    instances_csv: str | None,
    instance_start: int | None,
    instance_end: int | None,
    max_per_set: int | None,
) -> tuple[list[int], list[int]]:
    """Resolve the requested instance ids; return ``(found_sorted, missing)``."""
    requested: list[int] | None = None
    if instance is not None:
        requested = [instance]
    elif instances_csv is not None:
        requested = parse_instance_spec(instances_csv)
    elif instance_start is not None or instance_end is not None:
        if instance_start is None or instance_end is None:
            raise ValueError(
                "--instance-start and --instance-end must be used together"
            )
        if instance_end < instance_start:
            raise ValueError(
                f"--instance-end ({instance_end}) < --instance-start ({instance_start})"
            )
        requested = list(range(instance_start, instance_end + 1))

    if requested is None:
        ids = sorted(available.keys())
        if max_per_set is not None:
            ids = ids[:max_per_set]
        return ids, []

    found = [i for i in requested if i in available]
    missing = [i for i in requested if i not in available]
    if max_per_set is not None:
        found = found[:max_per_set]
    return found, missing


# ---------------------------------------------------------------------------
# Combo grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Combo:
    heuristic: str
    flaw_selection_strat: str
    resolver_strat: str

    @property
    def needs_llm(self) -> bool:
        return (
            self.heuristic == "llm"
            or self.flaw_selection_strat == "llm"
            or self.resolver_strat == "llm"
        )

    @property
    def label(self) -> str:
        return f"h={self.heuristic}|f={self.flaw_selection_strat}|r={self.resolver_strat}"


def build_combo_grid(
    *,
    heuristics: list[str],
    flaw_selection_strats: list[str],
    resolver_strats: list[str],
) -> list[Combo]:
    return [
        Combo(h, f, r)
        for h in heuristics
        for f in flaw_selection_strats
        for r in resolver_strats
    ]


# ---------------------------------------------------------------------------
# LLM config
# ---------------------------------------------------------------------------


def build_llm_config(args: argparse.Namespace) -> LLMConfig:
    cfg = LLMConfig(
        max_retries=args.llm_max_retries,
        temperature=args.llm_temperature,
        response_format=args.llm_response_format,
    )
    if args.llm_model:
        cfg.model = args.llm_model
    if args.llm_base_url:
        cfg.base_url = args.llm_base_url
    if args.llm_api_key:
        cfg.api_key = args.llm_api_key
    return cfg


def validate_llm_config(cfg: LLMConfig) -> None:
    """Fail fast when LLM combos are requested without a reachable endpoint."""
    has_base_url = bool(cfg.base_url) or bool(os.environ.get("OPENAI_BASE_URL")) \
        or bool(os.environ.get("LLM_BASE_URL"))
    has_api_key = bool(cfg.api_key) or bool(os.environ.get("OPENAI_API_KEY"))
    if not has_base_url and not has_api_key:
        raise SystemExit(
            "[ERROR] LLM combos requested but no endpoint configured.\n"
            "        Set --llm-base-url (e.g. http://localhost:8877/v1) for local\n"
            "        vLLM, or OPENAI_API_KEY for the OpenAI API."
        )


# ---------------------------------------------------------------------------
# Job model + worker
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JobSpec:
    job_id: int
    trial: int
    instance_id: int
    instance_label: str
    problem_path: str
    domain_path: str
    heuristic: str
    flaw_selection_strat: str
    resolver_strat: str
    timeout: float
    llm_config: LLMConfig | None
    capture_plan: bool


def _empty_stats_record(spec: JobSpec) -> dict[str, Any]:
    return {
        "job_id": spec.job_id,
        "trial": spec.trial,
        "instance_id": spec.instance_id,
        "instance_label": spec.instance_label,
        "problem_path": spec.problem_path,
        "domain_path": spec.domain_path,
        "heuristic": spec.heuristic,
        "flaw_selection_strat": spec.flaw_selection_strat,
        "resolver_strat": spec.resolver_strat,
        "needs_llm": (
            spec.heuristic == "llm"
            or spec.flaw_selection_strat == "llm"
            or spec.resolver_strat == "llm"
        ),
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
        "plan_signatures": None,
    }


def _run_job(spec: JobSpec) -> dict[str, Any]:
    """Compile + solve one job. Returns a stats dict; never raises."""
    record = _empty_stats_record(spec)

    try:
        problem = compile_domain_and_problem(
            Path(spec.domain_path), Path(spec.problem_path)
        )
    except Exception as exc:  # noqa: BLE001
        record["status"] = "compile_error"
        record["error"] = f"compile_failed: {exc}"
        record["traceback"] = traceback.format_exc()
        return record

    planner = DPOCLPlanner(
        heuristic=spec.heuristic,
        flaw_selection_strat=spec.flaw_selection_strat,
        resolver_strat=spec.resolver_strat,
        verbose=False,
        llm_config=spec.llm_config,
    )

    wall_start = time.perf_counter()
    try:
        solutions = list(
            planner.solve(problem, max_solutions=1, timeout=spec.timeout)
        )
    except Exception as exc:  # noqa: BLE001
        record["wall_time_s"] = time.perf_counter() - wall_start
        record["status"] = "error"
        record["error"] = f"solve_failed: {exc}"
        record["traceback"] = traceback.format_exc()
        try:
            stats = planner.get_statistics()
            record["nodes_visited"] = stats["nodes_visited"]
            record["nodes_expanded"] = stats["nodes_expanded"]
            record["duplicates_pruned"] = stats["duplicates_pruned"]
            record["peak_frontier_size"] = stats["peak_frontier_size"]
            record["llm_calls"] = stats["llm_calls"]
            record["llm_retries"] = stats["llm_retries"]
            record["timeout_reached"] = bool(stats["timeout_reached"])
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
        plan = solutions[0]
        steps = plan.to_execution_sequence()
        record["solved"] = True
        record["plan_length"] = len(steps)
        record["status"] = "solved"
        if spec.capture_plan:
            record["plan_signatures"] = [step.signature for step in steps]
    elif record["timeout_reached"]:
        record["status"] = "timeout"
    else:
        record["status"] = "unsolved"

    return record


# ---------------------------------------------------------------------------
# Executor selection + dispatch
# ---------------------------------------------------------------------------


def decide_executor(
    *, any_llm: bool, override: str
) -> tuple[str, int]:
    """Return ``(executor_kind, default_max_concurrency)``."""
    cpu = os.cpu_count() or 4
    if override == "threads":
        kind = "threads"
    elif override == "processes":
        kind = "processes"
    else:  # auto / hybrid
        kind = "threads" if any_llm else "processes"
    default = min(32, cpu * 4) if kind == "threads" else cpu
    return kind, default


def _open_executor(kind: str, max_concurrency: int):
    if kind == "threads":
        return ThreadPoolExecutor(max_workers=max_concurrency)
    return ProcessPoolExecutor(max_workers=max_concurrency)


def dispatch_grid(
    jobs: list[JobSpec],
    *,
    executor_kind: str,
    max_concurrency: int,
    partial_jsonl_path: Path,
    quiet: bool,
) -> list[dict[str, Any]]:
    """Run all jobs through a bounded pool, streaming results to JSONL."""
    if not jobs:
        return []

    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    completed = 0
    total = len(jobs)
    started_at = time.perf_counter()

    partial_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    partial_fh = partial_jsonl_path.open("a", encoding="utf-8")
    try:
        with _open_executor(executor_kind, max_concurrency) as executor:
            future_to_spec: dict[Future, JobSpec] = {
                executor.submit(_run_job, spec): spec for spec in jobs
            }
            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                try:
                    record = future.result()
                except Exception as exc:  # noqa: BLE001
                    record = _empty_stats_record(spec)
                    record["status"] = "error"
                    record["error"] = f"worker_crashed: {exc}"
                    record["traceback"] = traceback.format_exc()

                with lock:
                    completed += 1
                    results.append(record)
                    partial_fh.write(json.dumps(record) + "\n")
                    partial_fh.flush()
                    if not quiet:
                        elapsed = time.perf_counter() - started_at
                        status_tag = record["status"]
                        extra = ""
                        if status_tag == "solved":
                            extra = f" plan={record['plan_length']}"
                        elif record["error"]:
                            err = str(record["error"]).splitlines()[0][:80]
                            extra = f"  err={err}"
                        print(
                            f"  [{completed}/{total}] "
                            f"{record['instance_label']} | "
                            f"h={record['heuristic']} "
                            f"f={record['flaw_selection_strat']} "
                            f"r={record['resolver_strat']} | "
                            f"{status_tag} t={record['wall_time_s']:.2f}s "
                            f"visited={record['nodes_visited']} "
                            f"expanded={record['nodes_expanded']} "
                            f"llm={record['llm_calls']}"
                            f"{extra}  "
                            f"[+{elapsed:.1f}s wall]",
                            flush=True,
                        )
    finally:
        partial_fh.close()
    return results


# ---------------------------------------------------------------------------
# Aggregation + tables
# ---------------------------------------------------------------------------


def aggregate_by_combo(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-(heuristic, flaw, resolver) aggregate; means/medians over solved+unsolved only."""
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        key = (r["heuristic"], r["flaw_selection_strat"], r["resolver_strat"])
        buckets[key].append(r)

    rows: list[dict[str, Any]] = []
    for key, runs in buckets.items():
        h, f, r_strat = key
        n_total = len(runs)
        n_solved = sum(1 for x in runs if x["status"] == "solved")
        n_unsolved = sum(1 for x in runs if x["status"] == "unsolved")
        n_timeout = sum(1 for x in runs if x["status"] == "timeout")
        n_error = sum(
            1 for x in runs if x["status"] in ("error", "compile_error")
        )

        eligible = [
            x for x in runs if x["status"] in ("solved", "unsolved")
        ]
        means: dict[str, float | None] = {}
        medians: dict[str, float | None] = {}
        for k in AGGREGATE_KEYS:
            vals = [x[k] for x in eligible if x.get(k) is not None]
            if vals:
                means[k] = float(statistics.fmean(vals))
                medians[k] = float(statistics.median(vals))
            else:
                means[k] = None
                medians[k] = None

        rows.append(
            {
                "heuristic": h,
                "flaw_selection_strat": f,
                "resolver_strat": r_strat,
                "n_total": n_total,
                "n_solved": n_solved,
                "n_unsolved": n_unsolved,
                "n_timeout": n_timeout,
                "n_error": n_error,
                "n_eligible_for_aggregate": len(eligible),
                "solve_rate_pct": (
                    round(100.0 * n_solved / n_total, 1) if n_total else 0.0
                ),
                "mean": means,
                "median": medians,
            }
        )

    rows.sort(
        key=lambda row: (
            -row["solve_rate_pct"],
            row["mean"]["wall_time_s"]
            if row["mean"]["wall_time_s"] is not None
            else float("inf"),
        )
    )
    return rows


def print_combo_table(rows: list[dict[str, Any]], *, total_instances: int) -> None:
    col_w = [6, 16, 11, 9, 8, 8, 8, 7, 11, 12, 14, 12]
    headers = [
        "h",
        "Flaw sel.",
        "Resolver",
        "Total",
        "Solved",
        "Timeout",
        "Error",
        "Rate",
        "Mean t(s)",
        "Mean nodes",
        "Mean expanded",
        "Mean plan",
    ]
    sep = "  ".join("-" * w for w in col_w)
    print(f"\n{'#' * 120}")
    print(f"  PER-COMBO SUMMARY  ({total_instances} instance(s))")
    print(f"{'#' * 120}")
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print("  " + sep)

    def _fmt(value: float | None, *, fmt: str) -> str:
        return format(value, fmt) if value is not None else "-"

    for row in rows:
        cells = [
            row["heuristic"].ljust(col_w[0]),
            row["flaw_selection_strat"].ljust(col_w[1]),
            row["resolver_strat"].ljust(col_w[2]),
            str(row["n_total"]).ljust(col_w[3]),
            str(row["n_solved"]).ljust(col_w[4]),
            str(row["n_timeout"]).ljust(col_w[5]),
            str(row["n_error"]).ljust(col_w[6]),
            f"{row['solve_rate_pct']:.1f}%".ljust(col_w[7]),
            _fmt(row["mean"]["wall_time_s"], fmt=".3f").ljust(col_w[8]),
            _fmt(row["mean"]["nodes_visited"], fmt=".1f").ljust(col_w[9]),
            _fmt(row["mean"]["nodes_expanded"], fmt=".1f").ljust(col_w[10]),
            _fmt(row["mean"]["plan_length"], fmt=".1f").ljust(col_w[11]),
        ]
        print("  " + "  ".join(cells))


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--domain",
        required=True,
        help=(
            "Domain folder name under instances/, e.g. blocksworld, depots,"
            " logistics, mystery_blocksworld."
        ),
    )
    parser.add_argument(
        "--problem-set",
        required=True,
        help=(
            "Problem-set sub-folder under instances/<domain>/, e.g."
            " generated, generated_basic, generated_basic_3."
        ),
    )

    sel = parser.add_argument_group("instance selection")
    sel.add_argument("--instance", type=int, default=None,
                     help="Single instance id to run.")
    sel.add_argument("--instances", type=str, default=None,
                     help="CSV mix of ids/ranges, e.g. '1,2,5-10,42'.")
    sel.add_argument("--instance-start", type=int, default=None,
                     help="Inclusive start of an instance range (use with --instance-end).")
    sel.add_argument("--instance-end", type=int, default=None,
                     help="Inclusive end of an instance range (use with --instance-start).")
    sel.add_argument("--max-per-set", type=int, default=None,
                     help="Cap the number of instances when no explicit range/list is given (or after filtering).")

    grid = parser.add_argument_group("combo grid")
    grid.add_argument(
        "--heuristics",
        type=str,
        default=None,
        help=(
            "CSV subset of heuristics, or 'all'. Default: all classical "
            f"({', '.join(CLASSICAL_HEURISTICS)})."
        ),
    )
    grid.add_argument(
        "--flaw-selection-strats",
        type=str,
        default=None,
        help=(
            "CSV subset of flaw-selection strategies, or 'all'. Default: all"
            f" classical ({', '.join(CLASSICAL_FLAW_SELECTION_STRATS)})."
        ),
    )
    grid.add_argument(
        "--resolver-strats",
        type=str,
        default=None,
        help=(
            "CSV subset of resolver strategies, or 'all'. Default: all"
            f" classical ({', '.join(CLASSICAL_RESOLVER_STRATS)})."
        ),
    )
    grid.add_argument(
        "--include-llm",
        action="store_true",
        help="Append 'llm' to every axis selection (alongside whatever else was chosen).",
    )
    grid.add_argument(
        "--all-combos",
        action="store_true",
        help="Use the full 8x3x2 grid (all heuristics x all flaw strats x all resolver strats).",
    )

    runp = parser.add_argument_group("runtime")
    runp.add_argument("--timeout", type=float, default=120.0,
                      help="Per-job timeout in seconds (default: 120).")
    runp.add_argument("--trials", type=int, default=1,
                      help="Repeat the full grid this many times (default: 1).")
    runp.add_argument("--max-concurrency", type=int, default=None,
                      help="Bounded worker-pool size (default: cpu_count() for processes, min(32, 4*cpu) for threads).")
    runp.add_argument("--executor", choices=("auto", "threads", "processes"), default="auto",
                      help="Executor type. 'auto' = ProcessPool when no LLM combo is present, ThreadPool otherwise.")

    llm = parser.add_argument_group("llm (used only when any combo selects 'llm')")
    llm.add_argument("--llm-model", default=None,
                     help=f"Model id (default: $OPENAI_MODEL/$LLM_MODEL or {DEFAULT_MODEL}).")
    llm.add_argument("--llm-base-url", default=None,
                     help="OpenAI-compatible base URL (default: $OPENAI_BASE_URL/$LLM_BASE_URL). Use http://localhost:8877/v1 for the local vLLM script.")
    llm.add_argument("--llm-api-key", default=None,
                     help="Override $OPENAI_API_KEY (most local servers ignore this).")
    llm.add_argument("--llm-response-format", choices=("json_schema", "json_object", "none"),
                     default="json_schema",
                     help="Structured-output mode for the LLM (default: json_schema).")
    llm.add_argument("--llm-max-retries", type=int, default=3,
                     help="Structured-output retry budget per LLM call (default: 3).")
    llm.add_argument("--llm-temperature", type=float, default=0.0,
                     help="Sampling temperature (default: 0.0).")

    out = parser.add_argument_group("output")
    out.add_argument("--output", type=Path, default=None,
                     help="Path to consolidated JSON (default: timestamped name under results/).")
    out.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                     help=f"Directory for default-named outputs (default: {RESULTS_DIR}).")
    out.add_argument("--quiet", action="store_true",
                     help="Suppress per-job progress lines.")
    out.add_argument("--verbose", action="store_true",
                     help="(Reserved) emit extra config debug at startup.")
    out.add_argument("--print-plans", action="store_true",
                     help="Capture and store plan step signatures in the JSON output.")

    return parser


def _resolve_combo_axes(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    """Resolve heuristic/flaw/resolver axes with shorthand handling."""
    if args.all_combos:
        h_default = HEURISTICS
        f_default = FLAW_SELECTION_STRATS
        r_default = RESOLVER_STRATS
    else:
        h_default = CLASSICAL_HEURISTICS
        f_default = CLASSICAL_FLAW_SELECTION_STRATS
        r_default = CLASSICAL_RESOLVER_STRATS

    selected_h = parse_csv_choices(
        args.heuristics, valid=HEURISTICS, default=h_default, label="heuristic"
    )
    selected_f = parse_csv_choices(
        args.flaw_selection_strats,
        valid=FLAW_SELECTION_STRATS,
        default=f_default,
        label="flaw-selection strat",
    )
    selected_r = parse_csv_choices(
        args.resolver_strats,
        valid=RESOLVER_STRATS,
        default=r_default,
        label="resolver strat",
    )

    if args.include_llm:
        if "llm" not in selected_h:
            selected_h.append("llm")
        if "llm" not in selected_f:
            selected_f.append("llm")
        if "llm" not in selected_r:
            selected_r.append("llm")

    return selected_h, selected_f, selected_r


def _llm_config_dict(cfg: LLMConfig) -> dict[str, Any]:
    d = asdict(cfg)
    if d.get("api_key"):
        d["api_key"] = "***redacted***"
    return d


def _default_output_path(
    args: argparse.Namespace, *, n_h: int, n_f: int, n_r: int, n_inst: int
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = (
        f"run-{timestamp}-{args.domain}-{args.problem_set}"
        f"-h{n_h}f{n_f}r{n_r}-n{n_inst}-T{args.trials}.json"
    )
    return args.results_dir / name


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.trials < 1:
        raise SystemExit("[ERROR] --trials must be >= 1")
    if args.max_per_set is not None and args.max_per_set <= 0:
        raise SystemExit("[ERROR] --max-per-set must be > 0")

    domains = discover_domains(INSTANCES_ROOT)
    if args.domain not in domains:
        raise SystemExit(
            f"[ERROR] Unknown --domain {args.domain!r}. "
            f"Available: {', '.join(domains) or '(none)'}."
        )
    domain_root = INSTANCES_ROOT / args.domain
    sets = discover_problem_sets(domain_root)
    if args.problem_set not in sets:
        raise SystemExit(
            f"[ERROR] Unknown --problem-set {args.problem_set!r} in domain"
            f" {args.domain!r}. Available: {', '.join(sets) or '(none)'}."
        )
    problem_set_dir = domain_root / args.problem_set

    available = discover_instances(problem_set_dir)
    if not available:
        raise SystemExit(
            f"[ERROR] No instance-N.pddl files found in {problem_set_dir}"
        )

    try:
        selected_ids, missing_ids = select_instances(
            available,
            instance=args.instance,
            instances_csv=args.instances,
            instance_start=args.instance_start,
            instance_end=args.instance_end,
            max_per_set=args.max_per_set,
        )
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    if not selected_ids:
        raise SystemExit("[ERROR] No instances selected. Adjust the selection flags.")

    try:
        sel_h, sel_f, sel_r = _resolve_combo_axes(args)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    combos = build_combo_grid(
        heuristics=sel_h,
        flaw_selection_strats=sel_f,
        resolver_strats=sel_r,
    )
    if not combos:
        raise SystemExit("[ERROR] Empty combo grid.")

    any_llm = any(c.needs_llm for c in combos)

    llm_config: LLMConfig | None = None
    if any_llm:
        llm_config = build_llm_config(args)
        validate_llm_config(llm_config)

    executor_kind, default_concurrency = decide_executor(
        any_llm=any_llm, override=args.executor
    )
    max_concurrency = args.max_concurrency or default_concurrency
    if max_concurrency < 1:
        raise SystemExit("[ERROR] --max-concurrency must be >= 1")

    output_path: Path = args.output or _default_output_path(
        args,
        n_h=len(sel_h),
        n_f=len(sel_f),
        n_r=len(sel_r),
        n_inst=len(selected_ids),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_jsonl_path = output_path.with_suffix(output_path.suffix + ".partial.jsonl")

    print("Configuration:")
    print(f"  domain                : {args.domain}")
    print(f"  problem-set           : {args.problem_set}")
    print(f"  instances             : {len(selected_ids)} found"
          f" (missing: {len(missing_ids)})")
    print(f"  heuristics            : {', '.join(sel_h)}")
    print(f"  flaw selection strats : {', '.join(sel_f)}")
    print(f"  resolver strats       : {', '.join(sel_r)}")
    print(f"  combos                : {len(combos)}"
          f"  ({'incl. LLM' if any_llm else 'classical only'})")
    print(f"  trials                : {args.trials}")
    print(f"  timeout               : {args.timeout}s")
    print(f"  executor              : {executor_kind}"
          f"  (max-concurrency={max_concurrency})")
    if llm_config is not None:
        print(f"  llm model             : {llm_config.model}")
        print(f"  llm base url          : {llm_config.base_url or 'default (OpenAI)'}")
        print(f"  llm response format   : {llm_config.response_format}")
    print(f"  output                : {output_path}")
    if missing_ids and not args.quiet:
        sample = ", ".join(str(i) for i in missing_ids[:10])
        more = f" (+{len(missing_ids) - 10} more)" if len(missing_ids) > 10 else ""
        print(f"  missing ids           : {sample}{more}")

    job_specs: list[JobSpec] = []
    job_id = 0
    for trial in range(args.trials):
        for instance_id in selected_ids:
            problem_path = available[instance_id]
            domain_path = find_domain_pddl(problem_path, domain_root)
            label = f"{args.domain}/{args.problem_set}/instance-{instance_id}"
            for combo in combos:
                job_specs.append(
                    JobSpec(
                        job_id=job_id,
                        trial=trial,
                        instance_id=instance_id,
                        instance_label=label,
                        problem_path=str(problem_path),
                        domain_path=str(domain_path),
                        heuristic=combo.heuristic,
                        flaw_selection_strat=combo.flaw_selection_strat,
                        resolver_strat=combo.resolver_strat,
                        timeout=args.timeout,
                        llm_config=llm_config if combo.needs_llm else None,
                        capture_plan=args.print_plans,
                    )
                )
                job_id += 1

    print(f"\nDispatching {len(job_specs)} job(s) "
          f"({len(combos)} combos x {len(selected_ids)} instances "
          f"x {args.trials} trial(s))...")
    print(f"Streaming partial results to {partial_jsonl_path}\n")

    if partial_jsonl_path.exists():
        partial_jsonl_path.unlink()

    started_wall = time.perf_counter()
    runs = dispatch_grid(
        job_specs,
        executor_kind=executor_kind,
        max_concurrency=max_concurrency,
        partial_jsonl_path=partial_jsonl_path,
        quiet=args.quiet,
    )
    total_wall = time.perf_counter() - started_wall

    runs.sort(key=lambda r: r["job_id"])

    by_combo = aggregate_by_combo(runs)
    by_combo_per_trial: list[dict[str, Any]] | None = None
    if args.trials > 1:
        per_trial: list[dict[str, Any]] = []
        runs_by_trial: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in runs:
            runs_by_trial[r["trial"]].append(r)
        for trial in sorted(runs_by_trial.keys()):
            per_trial.append(
                {
                    "trial": trial,
                    "by_combo": aggregate_by_combo(runs_by_trial[trial]),
                }
            )
        by_combo_per_trial = per_trial

    counts = {
        "requested_unique_instances": len(selected_ids),
        "missing_instances": len(missing_ids),
        "total_jobs": len(runs),
        "solved": sum(1 for r in runs if r["status"] == "solved"),
        "unsolved": sum(1 for r in runs if r["status"] == "unsolved"),
        "timeout": sum(1 for r in runs if r["status"] == "timeout"),
        "error": sum(1 for r in runs if r["status"] in ("error", "compile_error")),
    }

    payload: dict[str, Any] = {
        "meta": {
            "timestamp_local": datetime.now().isoformat(timespec="seconds"),
            "domain": args.domain,
            "problem_set": args.problem_set,
            "instances_root": str(INSTANCES_ROOT),
            "domain_root": str(domain_root),
            "problem_set_dir": str(problem_set_dir),
            "wall_time_total_s": total_wall,
        },
        "configuration": {
            "timeout_s": args.timeout,
            "trials": args.trials,
            "executor": executor_kind,
            "max_concurrency": max_concurrency,
            "heuristics": sel_h,
            "flaw_selection_strats": sel_f,
            "resolver_strats": sel_r,
            "combos": [c.label for c in combos],
            "include_llm_shorthand": bool(args.include_llm),
            "all_combos_shorthand": bool(args.all_combos),
            "any_llm": any_llm,
            "llm_config": _llm_config_dict(llm_config) if llm_config else None,
            "selected_instance_ids": selected_ids,
        },
        "counts": counts,
        "missing_instance_ids": missing_ids,
        "by_combo": by_combo,
        "by_combo_per_trial": by_combo_per_trial,
        "runs": runs,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if partial_jsonl_path.exists():
        try:
            partial_jsonl_path.unlink()
        except OSError:
            pass

    print_combo_table(by_combo, total_instances=len(selected_ids))

    print(
        f"\nWrote {output_path}"
        f"  jobs={counts['total_jobs']}"
        f"  solved={counts['solved']}"
        f"  timeout={counts['timeout']}"
        f"  error={counts['error']}"
        f"  wall={total_wall:.2f}s",
    )

    return 0 if counts["error"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
