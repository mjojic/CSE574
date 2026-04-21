"""LLM-driven decision policy for the DPOCL planner.

When ``--llm`` is enabled the planner outsources two decisions to an OpenAI
chat model:

  1. **Next-node selection** -- which partial plan from the search frontier
     should be expanded next.
  2. **Flaw selection** -- which open-condition flaw of the popped plan
     should be resolved next.

Both calls go through :class:`LLMPolicy`, which uses the OpenAI structured
outputs feature (JSON schema, ``strict=True``) and validates the returned
JSON with pydantic.  If the model returns a malformed payload or an id
that is not in the allowed set, the policy retries up to
``LLMConfig.max_retries`` times before raising :class:`LLMPolicyError`.
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


@dataclass
class LLMConfig:
    """Configuration for :class:`LLMPolicy`."""

    model: str = field(
        default_factory=lambda: os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    )
    top_k: int = 10
    max_retries: int = 3
    temperature: float = 0.0
    # Optional explicit api key; otherwise the OpenAI SDK reads OPENAI_API_KEY.
    api_key: str | None = None


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


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


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


def _short_id(plan: Plan) -> str:
    """A short stable id derived from the plan's UUID (first 8 hex chars)."""
    return plan.id.hex[:8]


def _topo_signatures(plan: Plan) -> list[str]:
    """Return step signatures in topological order, excluding dummy steps."""
    sigs: list[str] = []
    for step in plan.topological_order():
        if step.name.startswith("__"):
            continue
        sigs.append(step.signature)
    return sigs


def serialize_plan_summary(plan: Plan) -> dict[str, Any]:
    """Compact summary of a plan for next-node selection."""
    flaws_summary = [
        {"step": f.step.signature, "cond": f.condition.signature}
        for f in plan.flaws
    ]
    threats = find_unresolved_threats(plan)
    return {
        "id": _short_id(plan),
        "steps_topo": _topo_signatures(plan),
        "open_flaws": flaws_summary,
        "open_flaws_count": len(plan.flaws),
        "causal_links_count": len(plan.causal_links),
        "threats_count": len(threats),
        "depth": plan.depth,
        "cost": plan.cost,
    }


def serialize_plan_detail(
    plan: Plan, problem: "PlanningProblem"
) -> dict[str, Any]:
    """Detailed plan view for flaw selection (steps + flaws with resolver counts)."""
    causal_links = [
        {
            "source": link.source.signature,
            "target": link.target.signature,
            "condition": link.condition.signature,
        }
        for link in plan.causal_links
    ]
    flaws = []
    for f in plan.flaws:
        flaws.append(
            {
                "id": f.id.hex[:8],
                "step": f.step.signature,
                "condition": f.condition.signature,
                "resolver_count": count_resolution_options(plan, f, problem),
            }
        )
    return {
        "id": _short_id(plan),
        "steps_topo": _topo_signatures(plan),
        "causal_links": causal_links,
        "flaws": flaws,
        "depth": plan.depth,
        "cost": plan.cost,
    }


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
    " and branches over every existing step or new operator that could"
    " support it.\n\n"
    "Your job is to make ONE control decision per call. The user message"
    " gives you the problem and either (a) a list of candidate plans on the"
    " frontier or (b) a list of open-condition flaws of the current plan."
    " Pick the option most likely to lead to a complete, low-cost plan as"
    " quickly as possible. Prefer narrow choices (few resolvers) and choices"
    " that unblock progress toward the goal; avoid plans that look likely to"
    " loop or grow without converging.\n\n"
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


# ---------------------------------------------------------------------------
# LLMPolicy
# ---------------------------------------------------------------------------


class LLMPolicy:
    """OpenAI-backed decision policy for the planner.

    Instances are cheap: the underlying OpenAI client is created lazily on
    first use so unit tests that never trigger an LLM call do not require an
    API key.
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

        plan_payload = serialize_plan_detail(plan, problem)
        # Re-stamp flaw ids in the payload so they match id_map exactly.
        plan_payload["flaws"] = [
            {
                "id": fid,
                "step": f.step.signature,
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

    # -- structured-output call with retry --------------------------------

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
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": {**schema, "additionalProperties": False},
                "strict": True,
            },
        }

        last_error: str | None = None
        attempts = max(1, self.config.max_retries)
        for attempt in range(attempts):
            self.calls += 1
            try:
                completion = client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    response_format=response_format,
                    temperature=self.config.temperature,
                )
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
