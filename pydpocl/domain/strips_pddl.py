"""Parse STRIPS PDDL (limited subset) and ground operators into a pydpocl problem."""

from __future__ import annotations

import itertools
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from pydpocl.core.literal import Literal, create_literal
from pydpocl.core.step import GroundStep


@dataclass
class GroundPlanningProblem:
    """PlanningProblem-compatible bundle produced from STRIPS PDDL."""

    initial_state: set[Literal] = field(default_factory=set)
    goal_state: set[Literal] = field(default_factory=set)
    operators: list[GroundStep] = field(default_factory=list)

    def is_goal_satisfied(self, state: set[Literal]) -> bool:
        return self.goal_state <= state


def _strip_comments(text: str) -> str:
    return re.sub(r";[^\n]*", "", text)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[()]|[^\s()]+", _strip_comments(text))


def _parse_s_expr(tokens: deque[str]) -> str | list:
    t = tokens.popleft()
    if t != "(":
        return t
    lst: list = []
    while tokens[0] != ")":
        lst.append(_parse_s_expr(tokens))
    tokens.popleft()
    return lst


def parse_pddl(text: str) -> list:
    """Parse one top-level PDDL s-expression."""
    toks = deque(_tokenize(text))
    if not toks:
        raise ValueError("Empty PDDL")
    return _parse_s_expr(toks)


def _get_blocks(expr: list, tag: str) -> list | None:
    for item in expr:
        if isinstance(item, list) and item and item[0] == tag:
            return item
    return None


def _flatten_and(expr: list | str) -> list:
    if isinstance(expr, str):
        return [expr]
    if not expr:
        return []
    if expr[0] == "and":
        out: list = []
        for part in expr[1:]:
            out.extend(_flatten_and(part))
        return out
    return [expr]


def _sexp_to_literals(
    sexp: list | str,
    subst: dict[str, str],
    positive: bool = True,
) -> set[Literal]:
    if isinstance(sexp, str):
        return {create_literal(sexp, positive=positive)}
    if sexp[0] == "not":
        inner = sexp[1]
        return _sexp_to_literals(inner, subst, positive=not positive)
    pred = sexp[0]
    resolved: list[str] = []
    for a in sexp[1:]:
        if isinstance(a, str):
            resolved.append(subst.get(a, a))
        else:
            raise ValueError(f"Nested formula not supported in atom: {a}")
    return {create_literal(pred, *resolved, positive=positive)}


def _formula_to_literals(sexp: list | str, subst: dict[str, str]) -> set[Literal]:
    """Convert a (possibly negated) conjunction into a set of Literals."""
    out: set[Literal] = set()
    for conj in _flatten_and(sexp):
        out |= _sexp_to_literals(conj, subst)
    return out


def _parse_objects_block(block: list) -> list[str]:
    # (:objects a b c) or typed lists — we only support flat untyped names
    names: list[str] = []
    for x in block[1:]:
        if isinstance(x, str) and not x.startswith(":"):
            names.append(x)
        elif isinstance(x, list):
            # (:objects a b - type c d - type) — take only names before '-'
            for y in x:
                if y == "-":
                    break
                if isinstance(y, str) and not y.startswith(":"):
                    names.append(y)
    return names


def _parse_action(block: list) -> tuple[str, list[str], list, list]:
    """Parse (:action name :parameters ... :precondition ... :effect ...).

    PDDL often flattens keywords into one list (not nested sublists per keyword).
    """
    if not block or block[0] != ":action":
        raise ValueError("Not an :action block")
    name = block[1]
    params: list[str] = []
    pre: list | None = None
    eff: list | None = None
    i = 2
    while i < len(block):
        tag = block[i]
        if tag == ":parameters":
            i += 1
            plist = block[i]
            plist = plist if isinstance(plist, list) else [plist]
            params = [p for p in plist if isinstance(p, str) and p.startswith("?")]
            i += 1
        elif tag == ":precondition":
            i += 1
            pre = block[i]
            i += 1
        elif tag == ":effect":
            i += 1
            eff = block[i]
            i += 1
        else:
            i += 1
    if pre is None or eff is None:
        raise ValueError(f"Missing precondition/effect in action {name}")
    return name, params, pre, eff


def _groundings_for_params(
    params: list[str],
    objects: list[str],
) -> list[dict[str, str]]:
    if not params:
        return [{}]
    if len(params) == 1:
        return [{params[0]: o} for o in objects]
    # binary: all ordered pairs with distinct objects (standard for stack/unstack)
    pairs = []
    for a, b in itertools.product(objects, repeat=2):
        if a != b:
            pairs.append({params[0]: a, params[1]: b})
    return pairs


def compile_strips_pddl(domain_path: Path, problem_path: Path) -> GroundPlanningProblem:
    """Parse STRIPS domain + problem and return a fully grounded planning problem.

    Supported: :strips, flat :objects, :init as conjunction of positive atoms,
    :goal as (and ...). Actions use :parameters, :precondition, :effect with (and ...).
    """
    domain_expr = parse_pddl(domain_path.read_text())
    problem_expr = parse_pddl(problem_path.read_text())

    if not isinstance(domain_expr, list) or domain_expr[0] != "define":
        raise ValueError("Domain must start with (define ...)")
    if not isinstance(problem_expr, list) or problem_expr[0] != "define":
        raise ValueError("Problem must start with (define ...)")

    actions_raw: list[tuple[str, list[str], list, list]] = []
    for item in domain_expr[2:]:
        if isinstance(item, list) and item and item[0] == ":action":
            actions_raw.append(_parse_action(item))

    pob = _get_blocks(problem_expr, ":objects")
    if pob is None:
        raise ValueError("Problem has no :objects")
    objects = _parse_objects_block(pob)

    init_b = _get_blocks(problem_expr, ":init")
    goal_b = _get_blocks(problem_expr, ":goal")
    if init_b is None or goal_b is None:
        raise ValueError("Problem needs :init and :goal")
    # Init may be (:init (and ...)) or (:init atom atom ...) without top-level and
    init_rest = init_b[1:]
    if (
        len(init_rest) == 1
        and isinstance(init_rest[0], list)
        and init_rest[0]
        and init_rest[0][0] == "and"
    ):
        init_expr = init_rest[0]
    else:
        init_expr = ["and", *init_rest]
    goal_expr = goal_b[1]

    initial_state: set[Literal] = set()
    for lit in _flatten_and(init_expr):
        if isinstance(lit, list) and lit and lit[0] != "not":
            initial_state |= _sexp_to_literals(lit, {})

    goal_state = _formula_to_literals(goal_expr, {})

    operators: list[GroundStep] = []
    step_no = 0
    for name, params, pre, eff in actions_raw:
        for subst in _groundings_for_params(params, objects):
            try:
                pre_lits = _formula_to_literals(pre, subst)
                eff_lits = _formula_to_literals(eff, subst)
            except (ValueError, IndexError, KeyError) as e:
                continue
            operators.append(
                GroundStep(
                    name=name,
                    parameters=tuple(subst[p] for p in params),
                    _preconditions=frozenset(pre_lits),
                    _effects=frozenset(eff_lits),
                    step_number=step_no,
                )
            )
            step_no += 1

    return GroundPlanningProblem(
        initial_state=initial_state,
        goal_state=goal_state,
        operators=operators,
    )
