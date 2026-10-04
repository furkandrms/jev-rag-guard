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

# A deliberately different question from SUFFICIENCY_QUESTION, not just a
# lower bar on the same one. SUFFICIENCY_QUESTION asks for "confidently AND
# COMPLETELY" -- a context that nails half of a two-part question and says
# nothing about the other half fails that exactly as hard as a context with
# no relevant content at all, so both land near probability 0 and a single
# probability threshold can't tell them apart (confirmed empirically: a
# context fully covering one sub-question and silent on another one scored
# ~0.1 here, indistinguishable from true irrelevance). This question asks
# the thing that actually matters for deciding whether a *partial* answer
# is worth attempting at all.
PARTIAL_QUESTION = (
    "Does the retrieved context below provide a confident, well-supported "
    "answer to at least one distinct part of the question, even though it "
    "may not cover everything the question asks? Answer no if the context "
    "is empty, entirely off-topic, or doesn't clearly support an answer to "
    "any part of the question on its own."
)


def check_sufficiency(
    query: str,
    chunks: list[Chunk],
    model: DecisionModel,
    threshold: float = 0.6,
    partial_threshold: float | None = None,
) -> SufficiencyResult:
    """Judge whether `chunks` are enough to answer `query` without guessing.

    Call this on the chunks that *survived* relevance filtering, not the raw
    retrieval results -- irrelevant chunks should already be gone by this
    point (see `rag_guard.relevance`).

    `partial_threshold`, if given, enables a second decision call
    (`PARTIAL_QUESTION`) whenever the main check comes back insufficient --
    deliberately a different question, not a lower bar on the same
    probability (see the comment above `PARTIAL_QUESTION` for why that
    doesn't work). Costs one extra decision call, but only on the
    already-failing path. Leave `partial_threshold` as None (the default)
    to skip it entirely -- anything below `threshold` is then always a
    plain `sufficient=False, partial=False`, matching the old behavior.
    """
    if not chunks:
        return SufficiencyResult(
            sufficient=False, probability=0.0, reason="No chunks were retrieved or kept."
        )

    combined_context = "\n\n---\n\n".join(chunk.text for chunk in chunks)
    state = f"Question: {query}\n\nRetrieved context:\n{combined_context}"
    probability = model.noul(state, SUFFICIENCY_QUESTION)
    sufficient = probability >= threshold

    partial = False
    if not sufficient and partial_threshold is not None:
        partial_probability = model.noul(state, PARTIAL_QUESTION)
        partial = partial_probability >= partial_threshold

    if sufficient:
        reason = None
    elif partial:
        reason = "Context only partially covers the question (decision model)."
    else:
        reason = "Context judged insufficient by decision model."

    return SufficiencyResult(
        sufficient=sufficient, probability=probability, reason=reason, partial=partial
    )
