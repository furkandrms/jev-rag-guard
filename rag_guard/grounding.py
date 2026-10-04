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
    "Is this claim supported by the context above? The claim may combine or "
    "paraphrase facts drawn from multiple separate passages in the context "
    "-- that is still supported, as long as every specific fact, number, or "
    "conclusion in the claim can be traced to something actually stated "
    "somewhere in the context. Answer no only if the claim adds a specific, "
    "number, or conclusion that is not present anywhere in the context, "
    "even if it sounds plausible -- not merely because no single passage "
    "states the whole claim on its own."
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

# A numbered-list marker ("1.", "2)", "(3).") at the start of a line, the
# kind a generated answer uses for a multi-item list. Matched so it can be
# shielded from the sentence splitter below -- without this, the splitter
# (which breaks right after ". " before a capital/digit) fires on the
# marker itself: "...as follows:\n\n1. NSL-KDD was used" becomes two
# claims, "...as follows:\n\n1." and "NSL-KDD was used", and that first
# claim is a bare list number with no checkable content in it, which the
# decision model correctly judges unsupported -- dragging an otherwise
# fully-grounded, itemized answer's coverage below threshold for a reason
# that has nothing to do with whether the answer is actually accurate.
_LIST_MARKER_RE = re.compile(r"(^|\n)([ \t]*\(?\d{1,3}[.)])(?=[ \t])")

# Trailing words after which a "." is almost never a sentence boundary --
# without this, "Dr. Smith confirmed it." or "e.g. the Q3 report." gets cut
# into two bogus claims, and grounding ends up judging fragments like "Dr."
# on their own.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "approx",
    "fig", "eq", "no", "e.g", "i.e", "u.s", "u.k", "e.u", "a.m", "p.m",
}


def split_claims(answer: str) -> list[str]:
    """Sentence-level split of a generated answer into checkable claims.

    Slightly abbreviation-aware (see `_ABBREVIATIONS`) so common titles and
    Latin abbreviations don't get cut mid-sentence; still a heuristic, not a
    parser -- swap in a proper claim-decomposition step (e.g. an LLM call
    that extracts atomic claims) if your answers are long or multi-clause
    and sentence boundaries are too coarse.
    """
    text = answer.strip()
    if not text:
        return []

    # Shield list markers from the splitter: insert a NUL right after the
    # marker so the splitter's "whitespace then capital/digit" lookahead
    # no longer matches there, then strip the NUL back out below.
    protected = _LIST_MARKER_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}\x00", text)

    claims: list[str] = []
    for part in (p.strip().replace("\x00", "") for p in _SENTENCE_SPLIT_RE.split(protected)):
        if not part:
            continue
        if claims:
            last_word = re.split(r"\s+", claims[-1].rstrip(".!?\"'"))[-1].lower()
            if last_word in _ABBREVIATIONS:
                claims[-1] = f"{claims[-1]} {part}"
                continue
        claims.append(part)
    return claims


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
