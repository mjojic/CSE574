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


# ---------------------------------------------------------------------------
# Tokeniser and s-expression parser
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Typed-list parsing (shared by :types, :objects, and :parameters)
# ---------------------------------------------------------------------------

def _parse_typed_list(tokens: list[str]) -> list[tuple[str, str]]:
    """Parse a flat PDDL typed list into (name, type) pairs.

    For example ``["a", "b", "-", "type1", "c", "-", "type2"]``
    becomes ``[("a", "type1"), ("b", "type1"), ("c", "type2")]``.
    Items that appear before any ``-`` separator default to type ``"object"``.
    Type names are lowercased for case-insensitive matching.
    """
    result: list[tuple[str, str]] = []
    pending: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-":
            i += 1
            typ = tokens[i].lower() if i < len(tokens) else "object"
            for name in pending:
                result.append((name, typ))
            pending = []
        else:
            pending.append(tok)
        i += 1
    for name in pending:
        result.append((name, "object"))
    return result


# ---------------------------------------------------------------------------
# Type hierarchy
# ---------------------------------------------------------------------------

def _parse_types_block(block: list) -> dict[str, str]:
    """Parse ``(:types ...)`` and return a child -> parent type mapping.

    All user-defined types ultimately descend from ``"object"``.
    """
    tokens = [x for x in block[1:] if isinstance(x, str)]
    typed = _parse_typed_list(tokens)
    hierarchy: dict[str, str] = {"object": "object"}
    for child, parent in typed:
        hierarchy[child.lower()] = parent.lower()
    return hierarchy


def _is_subtype(child_type: str, required_type: str, hierarchy: dict[str, str]) -> bool:
    """Return True if *child_type* is equal to or a subtype of *required_type*."""
    child_type = child_type.lower()
    required_type = required_type.lower()
    if required_type == "object":
        return True
    current = child_type
    visited: set[str] = set()
    while current not in visited:
        if current == required_type:
            return True
        visited.add(current)
        current = hierarchy.get(current, "object")
    return False


# ---------------------------------------------------------------------------
# Formula helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Block parsers
# ---------------------------------------------------------------------------

def _parse_objects_block(block: list) -> list[tuple[str, str]]:
    """Parse ``(:objects ...)`` and return ``(name, type)`` pairs.

    Handles both typed (``name - Type``) and untyped object lists.
    Untyped objects default to type ``"object"``.
    """
    tokens = [x for x in block[1:] if isinstance(x, str)]
    return _parse_typed_list(tokens)


def _parse_action(
    block: list,
) -> tuple[str, list[tuple[str, str]], list, list]:
    """Parse ``(:action name :parameters ... :precondition ... :effect ...)``.

    Returns ``(name, typed_params, precondition_sexp, effect_sexp)`` where
    *typed_params* is a list of ``(varname, type)`` pairs.  Untyped parameters
    get type ``"object"``.
    """
    if not block or block[0] != ":action":
        raise ValueError("Not an :action block")
    name = block[1]
    params_typed: list[tuple[str, str]] = []
    pre: list | None = None
    eff: list | None = None
    i = 2
    while i < len(block):
        tag = block[i]
        if tag == ":parameters":
            i += 1
            plist = block[i]
            plist = plist if isinstance(plist, list) else [plist]
            plist_strs = [p for p in plist if isinstance(p, str)]
            all_typed = _parse_typed_list(plist_strs)
            # Keep only entries whose name is a PDDL variable (starts with ?)
            params_typed = [(n, t) for n, t in all_typed if n.startswith("?")]
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
    return name, params_typed, pre, eff


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------

def _groundings_for_params(
    params: list[tuple[str, str]],
    typed_objects: list[tuple[str, str]],
    type_hierarchy: dict[str, str],
) -> list[dict[str, str]]:
    """Generate all type-compatible ground substitutions for *params*.

    For each parameter only objects whose type is a subtype of (or equal to)
    the parameter's declared type are considered.  For untyped domains all
    objects default to type ``"object"`` and therefore match every parameter.

    Returns a list of substitution dicts ``{varname: object_name}``.
    """
    if not params:
        return [{}]

    candidates: list[list[str]] = []
    for _varname, param_type in params:
        compatible = [
            obj_name
            for obj_name, obj_type in typed_objects
            if _is_subtype(obj_type, param_type, type_hierarchy)
        ]
        if not compatible:
            return []
        candidates.append(compatible)

    result: list[dict[str, str]] = []
    for combo in itertools.product(*candidates):
        subst = {varname: obj for (varname, _), obj in zip(params, combo)}
        result.append(subst)
    return result


# ---------------------------------------------------------------------------
# Public compilation entry point
# ---------------------------------------------------------------------------

def compile_strips_pddl(domain_path: Path, problem_path: Path) -> GroundPlanningProblem:
    """Parse STRIPS domain + problem and return a fully grounded planning problem.

    Supported PDDL features:
    - ``(:requirements :strips)`` and ``(:requirements :strips :typing)``
    - ``(:types ...)`` type hierarchy (child -> parent chain up to ``object``)
    - Typed or untyped ``(:objects ...)``
    - ``(:init ...)`` as a flat list of positive ground atoms
    - ``(:goal (and ...))``
    - Actions with typed or untyped ``(:parameters ...)``, ``(:precondition ...)``,
      and ``(:effect ...)`` using conjunctions and ``(not ...)`` literals
    """
    domain_expr = parse_pddl(domain_path.read_text())
    problem_expr = parse_pddl(problem_path.read_text())

    if not isinstance(domain_expr, list) or domain_expr[0] != "define":
        raise ValueError("Domain must start with (define ...)")
    if not isinstance(problem_expr, list) or problem_expr[0] != "define":
        raise ValueError("Problem must start with (define ...)")

    # Build type hierarchy from domain (trivial single-level for untyped domains)
    types_b = _get_blocks(domain_expr, ":types")
    type_hierarchy = _parse_types_block(types_b) if types_b else {"object": "object"}

    # Extract action schemata
    actions_raw: list[tuple[str, list[tuple[str, str]], list, list]] = []
    for item in domain_expr[2:]:
        if isinstance(item, list) and item and item[0] == ":action":
            actions_raw.append(_parse_action(item))

    # Typed objects from problem
    pob = _get_blocks(problem_expr, ":objects")
    if pob is None:
        raise ValueError("Problem has no :objects")
    typed_objects = _parse_objects_block(pob)

    init_b = _get_blocks(problem_expr, ":init")
    goal_b = _get_blocks(problem_expr, ":goal")
    if init_b is None or goal_b is None:
        raise ValueError("Problem needs :init and :goal")

    # Init may be (:init (and ...)) or (:init atom atom ...) without a top-level and
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
    for name, params_typed, pre, eff in actions_raw:
        param_vars = [v for v, _ in params_typed]
        for subst in _groundings_for_params(params_typed, typed_objects, type_hierarchy):
            try:
                pre_lits = _formula_to_literals(pre, subst)
                eff_lits = _formula_to_literals(eff, subst)
            except (ValueError, IndexError, KeyError):
                continue
            operators.append(
                GroundStep(
                    name=name,
                    parameters=tuple(subst[v] for v in param_vars),
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
