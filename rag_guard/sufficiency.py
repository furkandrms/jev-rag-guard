"""Stage 2: the pre-generation sufficiency gate.

This is the highest-leverage check in the library. Most RAG hallucination
does not come from the generator lying -- it comes from the generator being
handed a context that does not actually contain the answer and doing its
best anyway. Asking "is this enough to answer confidently and completely"
*before* generating lets you short-circuit straight to an honest "I don't
have enough information" instead of paying for a generation you can't trust.
"""

from __future__ import annotations

from .decision_model import DecisionModel
from .types import Chunk, SufficiencyResult

SUFFICIENCY_QUESTION = (
    "Does the retrieved context below contain enough information to answer "
    "the question confidently and completely? Answer no if the context is "
    "empty, off-topic, only partially relevant, or would require guessing "
    "to fill gaps."
)


def check_sufficiency(
    query: str,
    chunks: list[Chunk],
    model: DecisionModel,
    threshold: float = 0.6,
) -> SufficiencyResult:
    """Judge whether `chunks` are enough to answer `query` without guessing.

    Call this on the chunks that *survived* relevance filtering, not the raw
    retrieval results -- irrelevant chunks should already be gone by this
    point (see `rag_guard.relevance`).
    """
    if not chunks:
        return SufficiencyResult(
            sufficient=False, probability=0.0, reason="No chunks were retrieved or kept."
        )

    combined_context = "\n\n---\n\n".join(chunk.text for chunk in chunks)
    state = f"Question: {query}\n\nRetrieved context:\n{combined_context}"
    probability = model.noul(state, SUFFICIENCY_QUESTION)

    return SufficiencyResult(
        sufficient=probability >= threshold,
        probability=probability,
        reason=None if probability >= threshold else "Context judged insufficient by decision model.",
    )
