"""Stage 3: post-generation grounding / citation verification.

Passing the sufficiency gate does not guarantee the generator actually used
the context correctly -- it can still drift, over-generalize, or add detail
that sounds plausible but isn't in any retrieved passage. This stage splits
the generated answer into claims and checks each one against the retrieved
context individually, so a single unsupported sentence doesn't get lost
inside an otherwise-good answer.
"""

from __future__ import annotations

import re

from .decision_model import DecisionModel
from .types import Chunk, GroundingClaim, GroundingResult

GROUNDING_QUESTION = (
    "Is this claim directly supported by the context above? Answer no if "
    "the claim adds specifics, numbers, or conclusions that are not present "
    "in the context, even if they sound plausible."
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_claims(answer: str) -> list[str]:
    """Naive sentence-level split of a generated answer into checkable claims.

    Good enough as a first pass; swap in a proper claim-decomposition step
    (e.g. an LLM call that extracts atomic claims) if your answers are long
    or multi-clause and sentence boundaries are too coarse.
    """
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer.strip()) if s.strip()]


def check_grounding(
    answer: str,
    chunks: list[Chunk],
    model: DecisionModel,
    threshold: float = 0.5,
    min_coverage: float = 0.8,
) -> GroundingResult:
    """Check each claim in `answer` against `chunks` and roll up a coverage score.

    `min_coverage` is the fraction of claims that must be judged supported
    for the whole answer to count as grounded. A single ungrounded aside in
    an otherwise well-supported answer may still pass; a mostly-invented
    answer will not.
    """
    claim_texts = split_claims(answer)
    if not claim_texts:
        return GroundingResult(grounded=False, coverage=0.0, claims=[])

    context = "\n\n---\n\n".join(f"[{chunk.id}] {chunk.text}" for chunk in chunks)

    claims: list[GroundingClaim] = []
    for claim_text in claim_texts:
        state = f"Context:\n{context}\n\nClaim: {claim_text}"
        probability = model.noul(state, GROUNDING_QUESTION)
        supported = probability >= threshold
        # Weak hint only (substring match), not the basis for the supported
        # verdict -- useful for surfacing *which* chunk likely backs a claim
        # when it's shown to a human, not for deciding pass/fail.
        supporting_ids = [
            chunk.id for chunk in chunks if claim_text.lower()[:24] in chunk.text.lower()
        ]
        claims.append(
            GroundingClaim(
                claim=claim_text,
                supported=supported,
                probability=probability,
                supporting_chunk_ids=supporting_ids,
            )
        )

    coverage = sum(1 for c in claims if c.supported) / len(claims)
    return GroundingResult(grounded=coverage >= min_coverage, coverage=coverage, claims=claims)
