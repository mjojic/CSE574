"""Structural fingerprint for partial plans (ignore Step UUID identity)."""

from __future__ import annotations

from pydpocl.core.plan import Plan
from pydpocl.core.step import Step


def structural_fingerprint(plan: Plan) -> tuple:
    """Return a hashable summary of plan structure for duplicate detection.

    Two plans that differ only in freshly allocated Step UUIDs but have the
    same action signatures, causal links, orderings, and open flaws map to
    the same fingerprint.
    """
    sl = plan.step_lookup

    def step_key(s: Step) -> tuple:
        pre = frozenset(x.signature for x in sorted(s.preconditions, key=lambda lit: lit.signature))
        eff = frozenset(x.signature for x in sorted(s.effects, key=lambda lit: lit.signature))
        return (str(s.name), s.parameters, pre, eff)

    steps_part = tuple(sorted(step_key(s) for s in plan.steps))

    links_part = tuple(
        sorted(
            (link.source.signature, link.condition.signature, link.target.signature)
            for link in plan.causal_links
        )
    )

    ord_part = tuple(
        sorted(
            (sl[before].signature, sl[after].signature)
            for before, after in plan.orderings
            if before in sl and after in sl
        )
    )

    flaws_part = tuple(
        sorted(
            (flaw.step.signature, flaw.condition.signature) for flaw in plan.flaws
        )
    )

    return (steps_part, links_part, ord_part, flaws_part)
