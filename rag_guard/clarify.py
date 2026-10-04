"""Stage 1.5: the post-relevance ambiguity check.

A question can retrieve plenty of on-topic, well-supported context and
still be unanswerable responsibly -- if the context's answer depends on
something the user never specified (vehicle type, plan tier, mileage) and
the retrieved chunks disagree depending on that missing detail, answering
anyway means silently guessing which branch applies. This asks that
question directly, once, on the chunks that survived relevance filtering.
"""

from __future__ import annotations

from .decision_model import DecisionModel
from .types import Chunk, ClarifyResult

AMBIGUITY_QUESTION = (
    "Does answering this question correctly depend on information the user "
    "did not provide (such as product/vehicle variant, plan tier, or "
    "mileage), where the retrieved context gives different, conflicting "
    "answers depending on that missing information? Answer yes only if the "
    "context actually contains multiple branch-specific answers that "
    "conflict without it -- not merely because the question is short."
)


def check_ambiguity(
    query: str,
    chunks: list[Chunk],
    model: DecisionModel,
    threshold: float = 0.6,
) -> ClarifyResult:
    """Judge whether `query` needs clarification given the kept `chunks`.

    Call this on the chunks that survived relevance filtering, same as
    `check_sufficiency` -- an empty list can't be ambiguous, it's simply
    insufficient, so this always returns `needs_clarification=False` for it.
    """
    if not chunks:
        return ClarifyResult(needs_clarification=False, probability=0.0, reason=None)

    combined_context = "\n\n---\n\n".join(chunk.text for chunk in chunks)
    state = f"Question: {query}\n\nRetrieved context:\n{combined_context}"
    probability = model.noul(state, AMBIGUITY_QUESTION)
    needs_clarification = probability >= threshold

    return ClarifyResult(
        needs_clarification=needs_clarification,
        probability=probability,
        reason=(
            "Question is ambiguous: the context's answer depends on details "
            "the user didn't provide."
            if needs_clarification
            else None
        ),
    )
