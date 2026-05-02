"""LLM-driven decision policy for the DPOCL planner.

When ``--llm`` is enabled the planner outsources three control decisions to a
chat model exposed through any OpenAI-compatible endpoint (the official
OpenAI API, a self-hosted vLLM server, etc.):

  1. **Next-node selection** -- which partial plan from the search frontier
     should be expanded next (when ``LLMFrontier`` is in use).
  2. **Flaw selection** -- which open-condition flaw of the popped plan
     should be resolved next.
  3. **Resolver selection** -- which single resolver (existing-step reuse or
     new-operator instantiation) should be applied to the chosen flaw, with
     re-prompting after a previously chosen resolver has been abandoned.

All calls go through :class:`LLMPolicy`, which prefers OpenAI structured
outputs (JSON schema, ``strict=True``) and gracefully falls back to plain
``json_object`` mode when the backing server does not support strict schemas
(common with many vLLM builds).  Returned payloads are validated with
pydantic; on malformed output or out-of-set ids the policy retries up to
``LLMConfig.max_retries`` times before raising :class:`LLMPolicyError`.

Configuration sources (in order of precedence) for the underlying client:

* explicit ``LLMConfig`` fields (``model``, ``base_url``, ``api_key``)
* environment variables ``OPENAI_MODEL`` / ``LLM_MODEL`` for the model id
  and ``OPENAI_BASE_URL`` / ``LLM_BASE_URL`` for the endpoint
* the OpenAI SDK's own defaults (``OPENAI_API_KEY`` etc.)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError

from pydpocl.core.flaw_simple import OpenConditionFlaw
from pydpocl.core.plan import Plan
from pydpocl.planning.pocl_expansion import (
    ResolverCandidate,
    count_resolution_options,
    find_unresolved_threats,
)

if TYPE_CHECKING:
    from pydpocl.core.interfaces import PlanningProblem


# ---------------------------------------------------------------------------
# Public configuration / errors
# ---------------------------------------------------------------------------


DEFAULT_MODEL = "gpt-5.4-mini"


class LLMPolicyError(RuntimeError):
    """Raised when the LLM keeps returning invalid responses after retries."""


def _env_model() -> str:
    return (
        os.environ.get("OPENAI_MODEL")
        or os.environ.get("LLM_MODEL")
        or DEFAULT_MODEL
    )


def _env_base_url() -> str | None:
    return os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")


@dataclass
class LLMConfig:
    """Configuration for :class:`LLMPolicy`."""

    model: str = field(default_factory=_env_model)
    top_k: int = 10
    max_retries: int = 3
    temperature: float = 0.0
    # Optional explicit api key; otherwise the OpenAI SDK reads OPENAI_API_KEY.
    # When ``base_url`` points at a local server (vLLM) and no key is set, a
    # placeholder is used because most local servers ignore auth.
    api_key: str | None = None
    # Optional explicit OpenAI-compatible base URL (e.g. "http://localhost:8000/v1").
    base_url: str | None = field(default_factory=_env_base_url)
    # "json_schema" (strict OpenAI-style structured outputs),
    # "json_object" (looser; ask the model for JSON via response_format), or
    # "none" (rely entirely on prompt instructions). vLLM works best with
    # "json_object" or "none" depending on the version.
    response_format: str = "json_schema"
    # If True (default) the planner asks the LLM to pick exactly one resolver
    # per open-condition expansion; failed branches are re-prompted with the
    # rejected resolver excluded.  Set False to keep the legacy "expand all
    # successors at once" behavior for benchmarking.
    single_resolver: bool = True


# ---------------------------------------------------------------------------
# Pydantic schemas for structured outputs
# ---------------------------------------------------------------------------


class NodeChoice(BaseModel):
    """LLM response schema for next-node selection."""

    choice_id: str = Field(description="The candidate plan id to expand next.")
    reason: str = Field(
        default="",
        max_length=400,
        description="Brief justification (<= 400 chars).",
    )


class FlawChoice(BaseModel):
    """LLM response schema for flaw selection."""

    flaw_id: str = Field(description="The open-condition flaw id to resolve next.")
    reason: str = Field(
        default="",
        max_length=400,
        description="Brief justification (<= 400 chars).",
    )


class ResolverChoice(BaseModel):
    """LLM response schema for single-resolver selection."""

    resolver_id: str = Field(
        description="The resolver id to apply to the current open-condition flaw."
    )
    reason: str = Field(
        default="",
        max_length=400,
        description="Brief justification (<= 400 chars).",
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _short_id(plan: Plan) -> str:
    """A short stable id derived from the plan's UUID (first 8 hex chars)."""
    return plan.id.hex[:8]


def _step_short_id(step: Any) -> str:
    """Short stable id for a step within one prompt payload."""
    return step.id.hex[:8]


def serialize_problem(problem: "PlanningProblem") -> dict[str, Any]:
    """Compact JSON-friendly representation of a planning problem."""
    initial = sorted(lit.signature for lit in problem.initial_state)
    goal = sorted(lit.signature for lit in problem.goal_state)

    operators: list[dict[str, Any]] = []
    seen_ops: set[str] = set()
    for op in problem.operators:
        sig = op.signature
        if sig in seen_ops:
            continue
        seen_ops.add(sig)
        operators.append(
            {
                "name": sig,
                "preconditions": sorted(p.signature for p in op.preconditions),
                "effects": sorted(e.signature for e in op.effects),
            }
        )
    return {"initial": initial, "goal": goal, "operators": operators}


def serialize_plan_full(
    plan: Plan, problem: "PlanningProblem"
) -> dict[str, Any]:
    """Full JSON-friendly snapshot of a partial plan.

    Includes steps with preconditions/effects, ordering constraints, causal
    links, open-condition flaws (with resolver counts), unresolved threats,
    and the goal literals.  Used as the body of every LLM decision prompt so
    the model sees the same uniform POCL state regardless of which kind of
    decision (node / flaw / resolver) it is being asked to make.
    """
    sl = plan.step_lookup

    steps: list[dict[str, Any]] = []
    for step in plan.topological_order():
        is_initial = step.name == "__INITIAL__"
        is_goal = step.name == "__GOAL__"
        kind = "init" if is_initial else "goal" if is_goal else "action"
        steps.append(
            {
                "id": _step_short_id(step),
                "kind": kind,
                "name": str(step.name),
                "signature": step.signature,
                "preconditions": sorted(p.signature for p in step.preconditions),
                "effects": sorted(e.signature for e in step.effects),
            }
        )

    orderings = sorted(
        (_step_short_id(sl[before]), _step_short_id(sl[after]))
        for before, after in plan.orderings
        if before in sl and after in sl
    )

    causal_links = sorted(
        (
            {
                "source": _step_short_id(link.source),
                "source_signature": link.source.signature,
                "target": _step_short_id(link.target),
                "target_signature": link.target.signature,
                "condition": link.condition.signature,
            }
            for link in plan.causal_links
        ),
        key=lambda d: (d["source"], d["target"], d["condition"]),
    )

    threats = sorted(
        (
            {
                "link_source": _step_short_id(link.source),
                "link_target": _step_short_id(link.target),
                "link_condition": link.condition.signature,
                "threat_step": _step_short_id(threat),
                "threat_signature": threat.signature,
            }
            for link, threat in find_unresolved_threats(plan)
        ),
        key=lambda d: (d["link_source"], d["link_target"], d["threat_step"]),
    )

    flaws = []
    for f in plan.flaws:
        flaws.append(
            {
                "id": f.id.hex[:8],
                "step": _step_short_id(f.step),
                "step_signature": f.step.signature,
                "condition": f.condition.signature,
                "resolver_count": count_resolution_options(plan, f, problem),
            }
        )
    flaws.sort(key=lambda d: (d["resolver_count"], d["step_signature"], d["condition"]))

    return {
        "id": _short_id(plan),
        "depth": plan.depth,
        "cost": plan.cost,
        "steps": steps,
        "orderings": orderings,
        "causal_links": causal_links,
        "threats": threats,
        "flaws": flaws,
        "goal": sorted(g.signature for g in problem.goal_state),
    }


def serialize_plan_summary(plan: Plan) -> dict[str, Any]:
    """Compact summary of a plan for next-node selection."""
    flaws_summary = [
        {"step": f.step.signature, "cond": f.condition.signature}
        for f in plan.flaws
    ]
    threats = find_unresolved_threats(plan)
    return {
        "id": _short_id(plan),
        "steps_topo": [
            step.signature
            for step in plan.topological_order()
            if not step.name.startswith("__")
        ],
        "open_flaws": flaws_summary,
        "open_flaws_count": len(plan.flaws),
        "causal_links_count": len(plan.causal_links),
        "threats_count": len(threats),
        "depth": plan.depth,
        "cost": plan.cost,
    }


def serialize_resolver_candidates(
    plan: Plan,
    flaw: OpenConditionFlaw,
    candidates: list[ResolverCandidate],
) -> list[dict[str, Any]]:
    """Render resolver candidates for the LLM prompt.

    Each entry is grounded in the partial plan: ``reuse`` resolvers cite the
    short id of the producing step (so the LLM can cross-reference the
    serialized plan), and ``new`` resolvers spell out the operator template's
    preconditions/effects so the LLM can reason about cascade flaws.
    """
    payload: list[dict[str, Any]] = []
    for cand in candidates:
        entry: dict[str, Any] = {
            "resolver_id": cand.id,
            "kind": cand.kind,
            "producer": cand.producer_signature,
            "description": cand.description,
        }
        if cand.kind == "reuse" and cand.source_step_id is not None:
            source_step = plan.get_step(cand.source_step_id)
            if source_step is not None:
                entry["producer_step_id"] = _step_short_id(source_step)
        elif cand.kind == "new" and cand.operator_template is not None:
            tmpl = cand.operator_template
            entry["operator_preconditions"] = sorted(
                p.signature for p in tmpl.preconditions
            )
            entry["operator_effects"] = sorted(e.signature for e in tmpl.effects)
        payload.append(entry)
    return payload


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = (
    "You are a search-control policy embedded in a Partial-Order Causal-Link"
    " (POCL) planner. POCL maintains partial plans with steps, ordering"
    " constraints, causal links (producer --[condition]--> consumer), open"
    " conditions (preconditions not yet supported by a causal link, which we"
    " call FLAWS), and threats (a step whose effect could clobber a causal"
    " link's condition). The planner expands one partial plan at a time:"
    " first resolves any threat by promotion/demotion, then if the plan has"
    " no flaws it is a solution, otherwise it picks one open-condition flaw"
    " and resolves it by either reusing an existing step or instantiating a"
    " new operator that produces the needed condition.\n\n"
    "Your job is to make ONE control decision per call. The user message"
    " gives you the problem and either (a) a list of candidate plans on the"
    " frontier, (b) a list of open-condition flaws of the current plan, or"
    " (c) a list of resolvers for one specific flaw. Pick the option most"
    " likely to lead to a complete, low-cost plan as quickly as possible."
    " Prefer narrow choices (few resolvers) and choices that unblock"
    " progress toward the goal; avoid plans that look likely to loop or"
    " grow without converging.\n\n"
    "Reply with ONLY a JSON object that matches the provided schema. Use the"
    " exact id strings shown in the input. Do not invent ids."
)


def _build_node_user_prompt(
    problem_payload: dict[str, Any],
    candidates_payload: list[dict[str, Any]],
) -> str:
    return (
        "Decision: NEXT_NODE_SELECTION\n"
        "Pick exactly one candidate plan to expand next.\n\n"
        "Problem:\n"
        + json.dumps(problem_payload, indent=2)
        + "\n\nCandidates (top-K of the frontier, ranked best-first by f = g + h):\n"
        + json.dumps(candidates_payload, indent=2)
        + "\n\nReturn JSON {\"choice_id\": <one of the ids above>,"
        " \"reason\": <short string>}."
    )


def _build_flaw_user_prompt(
    problem_payload: dict[str, Any],
    plan_payload: dict[str, Any],
) -> str:
    return (
        "Decision: FLAW_SELECTION\n"
        "Pick exactly one open-condition flaw of the current plan to resolve"
        " next. Lower resolver_count usually means a more constrained, less"
        " branchy choice (fail-first heuristic).\n\n"
        "Problem:\n"
        + json.dumps(problem_payload, indent=2)
        + "\n\nCurrent plan:\n"
        + json.dumps(plan_payload, indent=2)
        + "\n\nReturn JSON {\"flaw_id\": <one of the flaw ids above>,"
        " \"reason\": <short string>}."
    )


def _build_resolver_user_prompt(
    problem_payload: dict[str, Any],
    plan_payload: dict[str, Any],
    flaw_payload: dict[str, Any],
    candidates_payload: list[dict[str, Any]],
    excluded_payload: list[str],
) -> str:
    excluded_block = ""
    if excluded_payload:
        excluded_block = (
            "\n\nPreviously rejected resolvers (their search subtrees were"
            " abandoned without finding a goal). Do NOT pick any of these:\n"
            + json.dumps(sorted(excluded_payload), indent=2)
        )
    return (
        "Decision: RESOLVER_SELECTION\n"
        "Pick exactly ONE resolver to apply to the chosen open-condition"
        " flaw. Only this resolver's successor plan will be enqueued; if"
        " its subtree later dead-ends you will be re-prompted with this"
        " resolver added to the rejection list.\n\n"
        "Problem:\n"
        + json.dumps(problem_payload, indent=2)
        + "\n\nCurrent plan:\n"
        + json.dumps(plan_payload, indent=2)
        + "\n\nFlaw to resolve:\n"
        + json.dumps(flaw_payload, indent=2)
        + "\n\nResolver candidates:\n"
        + json.dumps(candidates_payload, indent=2)
        + excluded_block
        + "\n\nReturn JSON {\"resolver_id\": <one of the resolver ids above>,"
        " \"reason\": <short string>}."
    )


# ---------------------------------------------------------------------------
# JSON schemas (kept in sync with the pydantic models above)
# ---------------------------------------------------------------------------


_NODE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["choice_id", "reason"],
    "properties": {
        "choice_id": {"type": "string"},
        "reason": {"type": "string"},
    },
}

_FLAW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["flaw_id", "reason"],
    "properties": {
        "flaw_id": {"type": "string"},
        "reason": {"type": "string"},
    },
}

_RESOLVER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["resolver_id", "reason"],
    "properties": {
        "resolver_id": {"type": "string"},
        "reason": {"type": "string"},
    },
}


# ---------------------------------------------------------------------------
# LLMPolicy
# ---------------------------------------------------------------------------


class LLMPolicy:
    """OpenAI-compatible decision policy for the planner.

    Instances are cheap: the underlying client is created lazily on first
    use so unit tests that never trigger an LLM call do not require an API
    key.  The same code path is used for the official OpenAI API and for
    self-hosted vLLM servers; only ``LLMConfig.base_url`` (and optionally
    ``response_format``) needs to change.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._client: Any = None  # openai.OpenAI, created lazily
        self.calls: int = 0
        self.retries: int = 0

    # -- client management ------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from dotenv import load_dotenv
        except ImportError:  # pragma: no cover - dotenv is in requirements
            load_dotenv = None  # type: ignore[assignment]
        if load_dotenv is not None:
            load_dotenv()

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMPolicyError(
                "openai package is required for --llm mode"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        elif self.config.base_url and not os.environ.get("OPENAI_API_KEY"):
            # Local servers (vLLM, llama.cpp, etc.) typically ignore the key
            # but the SDK still requires a non-empty value.
            kwargs["api_key"] = "EMPTY"
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    # -- public decision API ----------------------------------------------

    def select_node(
        self,
        candidates: list[tuple[float, Plan]],
        problem: "PlanningProblem",
    ) -> Plan:
        """Pick one plan from ``candidates``.

        ``candidates`` is a list of ``(priority, plan)`` pairs, already
        truncated to the top-K by the caller and sorted best-first.
        """
        if not candidates:
            raise ValueError("select_node called with empty candidates")
        if len(candidates) == 1:
            return candidates[0][1]

        # id -> plan map (short ids derived from UUIDs are unique within K).
        id_map: dict[str, Plan] = {}
        candidates_payload: list[dict[str, Any]] = []
        for rank, (priority, plan) in enumerate(candidates):
            cid = _short_id(plan)
            # If two plans happen to share the prefix, disambiguate.
            while cid in id_map:
                cid = cid + "_"
            id_map[cid] = plan
            summary = serialize_plan_summary(plan)
            summary["id"] = cid
            summary["rank"] = rank
            summary["priority_f"] = priority
            candidates_payload.append(summary)

        problem_payload = serialize_problem(problem)
        user_prompt = _build_node_user_prompt(problem_payload, candidates_payload)

        allowed_ids = set(id_map.keys())
        choice = self._call_structured(
            user_prompt=user_prompt,
            schema_name="node_choice",
            schema=_NODE_SCHEMA,
            model_cls=NodeChoice,
            allowed_field="choice_id",
            allowed_values=allowed_ids,
        )
        return id_map[choice.choice_id]

    def select_flaw(
        self,
        plan: Plan,
        problem: "PlanningProblem",
    ) -> OpenConditionFlaw:
        """Pick one open-condition flaw of ``plan`` to resolve next."""
        flaws = list(plan.flaws)
        if not flaws:
            raise ValueError("select_flaw called on a flawless plan")
        if len(flaws) == 1:
            return flaws[0]

        id_map: dict[str, OpenConditionFlaw] = {}
        for f in flaws:
            fid = f.id.hex[:8]
            while fid in id_map:
                fid = fid + "_"
            id_map[fid] = f

        plan_payload = serialize_plan_full(plan, problem)
        # Re-stamp flaw ids in the payload so they match id_map exactly.
        plan_payload["flaws"] = [
            {
                "id": fid,
                "step": _step_short_id(f.step),
                "step_signature": f.step.signature,
                "condition": f.condition.signature,
                "resolver_count": count_resolution_options(plan, f, problem),
            }
            for fid, f in id_map.items()
        ]

        problem_payload = serialize_problem(problem)
        user_prompt = _build_flaw_user_prompt(problem_payload, plan_payload)

        allowed_ids = set(id_map.keys())
        choice = self._call_structured(
            user_prompt=user_prompt,
            schema_name="flaw_choice",
            schema=_FLAW_SCHEMA,
            model_cls=FlawChoice,
            allowed_field="flaw_id",
            allowed_values=allowed_ids,
        )
        return id_map[choice.flaw_id]

    def select_resolver(
        self,
        plan: Plan,
        flaw: OpenConditionFlaw,
        candidates: list[ResolverCandidate],
        problem: "PlanningProblem",
        excluded: set[str] | None = None,
    ) -> ResolverCandidate:
        """Pick exactly one resolver from ``candidates`` for ``flaw``.

        ``excluded`` lists resolver ids whose subtrees have already been
        abandoned by the planner; the LLM is shown them so it can avoid
        repeating an obviously failed choice and is forbidden from selecting
        them via the ``allowed_values`` whitelist.
        """
        if not candidates:
            raise ValueError("select_resolver called with no candidates")
        if len(candidates) == 1:
            return candidates[0]

        id_map = {c.id: c for c in candidates}
        problem_payload = serialize_problem(problem)
        plan_payload = serialize_plan_full(plan, problem)
        flaw_payload = {
            "id": flaw.id.hex[:8],
            "step": _step_short_id(flaw.step),
            "step_signature": flaw.step.signature,
            "condition": flaw.condition.signature,
        }
        candidates_payload = serialize_resolver_candidates(plan, flaw, candidates)
        excluded_payload = sorted(excluded or set())

        user_prompt = _build_resolver_user_prompt(
            problem_payload=problem_payload,
            plan_payload=plan_payload,
            flaw_payload=flaw_payload,
            candidates_payload=candidates_payload,
            excluded_payload=excluded_payload,
        )

        choice = self._call_structured(
            user_prompt=user_prompt,
            schema_name="resolver_choice",
            schema=_RESOLVER_SCHEMA,
            model_cls=ResolverChoice,
            allowed_field="resolver_id",
            allowed_values=set(id_map.keys()),
        )
        return id_map[choice.resolver_id]

    # -- structured-output call with retry --------------------------------

    def _build_response_format(
        self,
        schema_name: str,
        schema: dict[str, Any],
        mode: str,
    ) -> dict[str, Any] | None:
        if mode == "none":
            return None
        if mode == "json_object":
            return {"type": "json_object"}
        # Default: strict JSON schema (OpenAI-style structured outputs).
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": {**schema, "additionalProperties": False},
                "strict": True,
            },
        }

    def _call_structured(
        self,
        *,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        model_cls: type[BaseModel],
        allowed_field: str,
        allowed_values: set[str],
    ) -> Any:
        client = self._ensure_client()

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        # Track the active response_format mode; we may downgrade on server
        # errors that indicate the strict schema path is not supported.
        active_mode = self.config.response_format

        last_error: str | None = None
        attempts = max(1, self.config.max_retries)
        for attempt in range(attempts):
            self.calls += 1
            response_format = self._build_response_format(
                schema_name, schema, active_mode
            )
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            try:
                completion = client.chat.completions.create(**kwargs)
                raw = completion.choices[0].message.content or ""
                payload = json.loads(raw)
                parsed = model_cls.model_validate(payload)
                value = getattr(parsed, allowed_field)
                if value not in allowed_values:
                    raise ValueError(
                        f"{allowed_field}={value!r} is not in allowed set"
                    )
                return parsed
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
            except Exception as exc:  # network / API errors
                last_error = f"API error: {exc!r}"
                # Heuristic fallback: if the server rejects the strict
                # json_schema path, drop down to json_object on the next try.
                msg = repr(exc).lower()
                if active_mode == "json_schema" and (
                    "json_schema" in msg
                    or "response_format" in msg
                    or "unsupported" in msg
                    or "not supported" in msg
                    or "invalid" in msg
                ):
                    active_mode = "json_object"

            self.retries += 1
            if attempt < attempts - 1:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous reply was invalid: {last_error}."
                            " Reply ONLY with a JSON object matching the"
                            " schema, using one of the ids shown above."
                        ),
                    }
                )

        raise LLMPolicyError(
            f"LLM failed to return a valid {schema_name} after"
            f" {attempts} attempts: {last_error}"
        )
