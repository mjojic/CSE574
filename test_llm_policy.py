#!/usr/bin/env python3
"""Mocked unit tests for the LLM-driven POCL search policy.

These tests do not contact any LLM; they verify the structural behavior of
``serialize_plan_full``, ``enumerate_resolvers``, and the planner's
single-resolver expansion + reprompt-on-dead-end agenda by stubbing
``LLMPolicy.select_*`` methods.

Run::

    python test_llm_policy.py             # plain script mode (fail-fast)
    pytest test_llm_policy.py             # works under pytest too
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from pydpocl.core.literal import Literal, create_literal
from pydpocl.core.step import GroundStep
from pydpocl.planning.llm_policy import (
    LLMConfig,
    serialize_plan_full,
    serialize_problem,
    serialize_resolver_candidates,
)
from pydpocl.planning.planner import DPOCLPlanner, PendingExpansion
from pydpocl.planning.pocl_expansion import (
    apply_resolver,
    enumerate_resolvers,
)


# ---------------------------------------------------------------------------
# Minimal planning problem used by every test (socks-and-shoes).
# ---------------------------------------------------------------------------


@dataclass
class _Problem:
    initial_state: set[Literal] = field(default_factory=set)
    goal_state: set[Literal] = field(default_factory=set)
    operators: list[GroundStep] = field(default_factory=list)

    def is_goal_satisfied(self, state: set[Literal]) -> bool:
        return self.goal_state <= state


def _socks_and_shoes() -> _Problem:
    rs_on = create_literal("RightSockOn")
    rsh_on = create_literal("RightShoeOn")
    ls_on = create_literal("LeftSockOn")
    lsh_on = create_literal("LeftShoeOn")
    ops = [
        GroundStep(
            name="RightSock",
            parameters=(),
            _preconditions=frozenset(),
            _effects=frozenset([rs_on]),
            step_number=0,
        ),
        GroundStep(
            name="RightShoe",
            parameters=(),
            _preconditions=frozenset([rs_on]),
            _effects=frozenset([rsh_on]),
            step_number=1,
        ),
        GroundStep(
            name="LeftSock",
            parameters=(),
            _preconditions=frozenset(),
            _effects=frozenset([ls_on]),
            step_number=2,
        ),
        GroundStep(
            name="LeftShoe",
            parameters=(),
            _preconditions=frozenset([ls_on]),
            _effects=frozenset([lsh_on]),
            step_number=3,
        ),
    ]
    return _Problem(
        initial_state=set(),
        goal_state={rsh_on, lsh_on},
        operators=ops,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_serialize_plan_full_has_all_pocl_state() -> None:
    """The full snapshot must expose every POCL component the LLM needs."""
    from pydpocl.core.plan import create_initial_plan

    problem = _socks_and_shoes()
    plan = create_initial_plan(problem.initial_state, problem.goal_state)
    payload = serialize_plan_full(plan, problem)

    expected_keys = {
        "id",
        "depth",
        "cost",
        "steps",
        "orderings",
        "causal_links",
        "threats",
        "flaws",
        "goal",
    }
    missing = expected_keys - payload.keys()
    assert not missing, f"serialize_plan_full missing keys: {missing}"

    # Initial plan: __INITIAL__ -> __GOAL__, plus two open-condition flaws
    # (one per goal literal). No causal links, no threats yet.
    step_kinds = sorted(s["kind"] for s in payload["steps"])
    assert step_kinds == ["goal", "init"], f"unexpected step kinds: {step_kinds}"
    assert payload["causal_links"] == []
    assert payload["threats"] == []
    assert len(payload["flaws"]) == 2
    assert set(payload["goal"]) == {"RightShoeOn()", "LeftShoeOn()"}
    # Every flaw should advertise its consumer step's signature plus a count.
    for flaw in payload["flaws"]:
        assert flaw["step_signature"].startswith("__GOAL__")
        assert isinstance(flaw["resolver_count"], int)


def test_enumerate_resolvers_and_apply_match_legacy_expansion() -> None:
    """Every (enumerate -> apply) round trip must reproduce expand_open_condition."""
    from pydpocl.core.plan import create_initial_plan
    from pydpocl.planning.pocl_expansion import expand_open_condition

    problem = _socks_and_shoes()
    plan = create_initial_plan(problem.initial_state, problem.goal_state)
    flaw = next(iter(plan.flaws))

    legacy = expand_open_condition(plan, flaw, problem)
    candidates = enumerate_resolvers(plan, flaw, problem)

    materialized = [s for s in (apply_resolver(plan, flaw, c) for c in candidates) if s]

    assert len(legacy) == len(materialized), (
        f"legacy={len(legacy)} vs materialized={len(materialized)}"
    )
    # Resolver ids must be unique within one prompt.
    ids = [c.id for c in candidates]
    assert len(ids) == len(set(ids)), f"duplicate resolver ids: {ids}"

    # Candidate descriptors should describe valid kinds only.
    kinds = {c.kind for c in candidates}
    assert kinds <= {"reuse", "new"}, f"unexpected kinds: {kinds}"


def test_serialize_resolver_candidates_includes_operator_details() -> None:
    """Resolver payloads should expose enough info for the LLM to reason."""
    from pydpocl.core.plan import create_initial_plan

    problem = _socks_and_shoes()
    plan = create_initial_plan(problem.initial_state, problem.goal_state)
    flaw = next(iter(plan.flaws))
    candidates = enumerate_resolvers(plan, flaw, problem)
    payload = serialize_resolver_candidates(plan, flaw, candidates)

    assert payload, "no resolver payload generated"
    for entry in payload:
        assert {"resolver_id", "kind", "producer", "description"} <= entry.keys()
        if entry["kind"] == "new":
            assert "operator_preconditions" in entry
            assert "operator_effects" in entry
        elif entry["kind"] == "reuse":
            assert "producer_step_id" in entry


def test_serialize_problem_dedupes_operators() -> None:
    problem = _socks_and_shoes()
    payload = serialize_problem(problem)
    names = [op["name"] for op in payload["operators"]]
    assert sorted(names) == sorted(set(names))


# ---------------------------------------------------------------------------
# Planner integration tests with a stubbed LLMPolicy.
# ---------------------------------------------------------------------------


class _StubPolicy:
    """Stand-in for :class:`LLMPolicy` used by the planner under test.

    Records every decision and returns deterministic answers so we can
    exercise the single-resolver + reprompt agenda end-to-end without an
    HTTP roundtrip.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.retries = 0
        self.flaw_calls = 0
        self.resolver_prompts: list[tuple[int, list[str], set[str]]] = []

    # The planner only needs select_flaw and select_resolver in
    # single-resolver mode; select_node is consulted by LLMFrontier when
    # there's more than one frontier candidate.
    def select_flaw(self, plan, problem):  # noqa: ANN001
        self.flaw_calls += 1
        self.calls += 1
        # Pick the flaw with the lexicographically smallest condition
        # signature for determinism.
        flaws = list(plan.flaws)
        flaws.sort(key=lambda f: f.condition.signature)
        return flaws[0]

    def select_resolver(self, plan, flaw, candidates, problem, excluded=None):  # noqa: ANN001
        self.calls += 1
        excluded_set = set(excluded or set())
        ids = [c.id for c in candidates]
        self.resolver_prompts.append((len(self.resolver_prompts), ids, excluded_set))
        # Deterministic: take the first candidate.
        return candidates[0]

    def select_node(self, candidates, problem):  # noqa: ANN001
        self.calls += 1
        # Deterministic: pick the lowest-priority (best) candidate.
        return min(candidates, key=lambda pair: pair[0])[1]


def _run_planner_with_stub(stub: _StubPolicy):
    """Drive DPOCLPlanner.solve with stub injected as the LLMPolicy."""
    import pydpocl.planning.planner as planner_mod

    original = planner_mod.LLMPolicy
    planner_mod.LLMPolicy = lambda config=None: stub  # type: ignore[assignment]
    try:
        problem = _socks_and_shoes()
        planner = DPOCLPlanner(
            search_strategy="best_first",
            heuristic="zero",
            verbose=False,
            use_llm=True,
            llm_config=LLMConfig(top_k=4, single_resolver=True),
        )
        solutions = list(planner.solve(problem, max_solutions=1, timeout=10.0))
        return solutions, planner
    finally:
        planner_mod.LLMPolicy = original


def test_single_resolver_planner_finds_a_plan_via_stub() -> None:
    """End-to-end: stubbed LLM should still drive the planner to a goal."""
    stub = _StubPolicy()
    solutions, planner = _run_planner_with_stub(stub)
    assert solutions, (
        "stubbed LLM planner failed to find a solution; "
        f"stats={planner.get_statistics()}"
    )
    stats = planner.get_statistics()
    assert stats["llm_calls"] >= 1, "expected at least one stub LLM call"
    # Resolver selection should have been consulted at least once.
    assert stub.resolver_prompts, "select_resolver was never called"


def _two_ways_problem() -> _Problem:
    """Tiny problem where the goal literal has two distinct producer operators.

    Used to guarantee that ``enumerate_resolvers`` returns >= 2 candidates so
    the reprompt-with-exclusion path can be exercised.
    """
    light_on = create_literal("LightOn")
    return _Problem(
        initial_state=set(),
        goal_state={light_on},
        operators=[
            GroundStep(
                name="FlipSwitch",
                parameters=(),
                _preconditions=frozenset(),
                _effects=frozenset([light_on]),
                step_number=0,
            ),
            GroundStep(
                name="ClapHands",
                parameters=(),
                _preconditions=frozenset(),
                _effects=frozenset([light_on]),
                step_number=1,
            ),
        ],
    )


def test_reprompt_excludes_previous_choice() -> None:
    """When the planner reprompts after a dead end, the rejected id is excluded."""
    from pydpocl.core.plan import create_initial_plan
    from pydpocl.planning.heuristic import ZeroHeuristic
    from pydpocl.planning.search import BestFirstSearch

    problem = _two_ways_problem()
    plan = create_initial_plan(problem.initial_state, problem.goal_state)
    flaw = next(iter(plan.flaws))
    candidates = enumerate_resolvers(plan, flaw, problem)
    assert len(candidates) >= 2, (
        f"test requires multiple resolvers, got {[c.id for c in candidates]}"
    )

    stub = _StubPolicy()
    planner = DPOCLPlanner(
        search_strategy="best_first",
        heuristic="zero",
        verbose=False,
        use_llm=True,
        llm_config=LLMConfig(top_k=4, single_resolver=True),
    )
    pe = PendingExpansion(
        plan=plan,
        flaw=flaw,
        excluded={candidates[0].id},
    )

    frontier = BestFirstSearch()
    progressed = planner._reprompt_resolver(
        pe,
        problem=problem,
        llm_policy=stub,  # type: ignore[arg-type]
        heuristic_fn=ZeroHeuristic(),
        frontier=frontier,
        seen=set(),
    )
    assert progressed, "expected reprompt to advance the agenda"

    # The stub recorded what it saw; the candidate list passed to it should
    # NOT include the excluded id, and the explicit excluded payload should.
    _, ids_seen, excluded_seen = stub.resolver_prompts[-1]
    assert candidates[0].id not in ids_seen
    assert candidates[0].id in excluded_seen
    # After the reprompt, the chosen id is also added to excluded for next time.
    assert len(pe.excluded) == 2


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


_TESTS = [
    test_serialize_plan_full_has_all_pocl_state,
    test_enumerate_resolvers_and_apply_match_legacy_expansion,
    test_serialize_resolver_candidates_includes_operator_details,
    test_serialize_problem_dedupes_operators,
    test_single_resolver_planner_finds_a_plan_via_stub,
    test_reprompt_excludes_previous_choice,
]


def main() -> int:
    failures: list[tuple[str, BaseException]] = []
    for fn in _TESTS:
        name = fn.__name__
        try:
            fn()
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"[FAIL] {name}: {exc}")
        except Exception as exc:  # pragma: no cover - debug aid
            failures.append((name, exc))
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"[ OK ] {name}")
    if failures:
        print(f"\n{len(failures)} of {len(_TESTS)} tests failed.")
        return 1
    print(f"\nAll {len(_TESTS)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
