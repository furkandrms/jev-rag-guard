"""Stage 3.5: the post-grounding caveat check.

A grounded answer can still be misleading on its own: a headline number or
yes/no that's technically supported by the context but only applies under a
condition, exception, extra fee, or variant-specific limitation the answer
doesn't mention. This runs only after grounding has already passed -- an
answer that failed grounding is flagged for trust reasons already and isn't
also checked for caveats.
"""

from __future__ import annotations

from .decision_model import DecisionModel
from .types import CaveatResult, Chunk

CAVEAT_QUESTION = (
    "Does this answer depend on a condition, exception, extra fee, or "
    "variant/plan-specific limitation mentioned in the context that isn't "
    "reflected in the answer, such that showing the answer alone could "
    "mislead the user?"
)


def check_caveat(
    answer: str,
    chunks: list[Chunk],
    model: DecisionModel,
    threshold: float = 0.5,
) -> CaveatResult:
    """Judge whether `answer` needs a caveat given the context it was grounded against."""
    combined_context = "\n\n---\n\n".join(chunk.text for chunk in chunks)
    state = f"Answer: {answer}\n\nRetrieved context:\n{combined_context}"
    probability = model.noul(state, CAVEAT_QUESTION)
    has_caveat = probability >= threshold

    return CaveatResult(
        has_caveat=has_caveat,
        probability=probability,
        reason="Answer depends on a condition/exception worth surfacing." if has_caveat else None,
    )
