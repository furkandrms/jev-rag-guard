"""Stage 1: per-chunk relevance filtering.

Vector search returns nearest neighbours by embedding distance. That is not
the same question as "does this passage help answer the query" -- a chunk
can be topically close while being useless (wrong entity, wrong time period,
answers a different sub-question). This stage asks that question directly,
per chunk, as a single typed Noul call, and lets deterministic code apply
the threshold.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .decision_model import DecisionModel
from .types import Chunk, RelevanceResult

RELEVANCE_QUESTION = (
    "Does this passage contain information that helps answer the question?"
)

# Each chunk's judgment is an independent, I/O-bound `model.noul()` call
# (an LLM API round trip for every real backend) -- run sequentially, a
# dozen-chunk retrieval spends a dozen round trips back to back before
# sufficiency even starts. Capped rather than unbounded so a large retrieval
# (TOP_K raised well past this) doesn't burst past a provider's concurrent
# rate limit.
_MAX_WORKERS = 8


def filter_relevant_chunks(
    query: str,
    chunks: list[Chunk],
    model: DecisionModel,
    threshold: float = 0.5,
) -> list[RelevanceResult]:
    """Judge each chunk's relevance to `query` and mark which ones to keep.

    Returns one `RelevanceResult` per input chunk, in the same order, so
    callers can inspect what was dropped and why -- this is a filter, not a
    reranker: order is preserved, nothing is reshuffled by score. Chunks are
    judged concurrently (see `_MAX_WORKERS`); `ThreadPoolExecutor.map`
    returns results in input order regardless of which worker finishes
    first, so the parallelism is invisible to callers.
    """
    if not chunks:
        return []

    def judge(chunk: Chunk) -> RelevanceResult:
        state = f"Question: {query}\n\nPassage:\n{chunk.text}"
        probability = model.noul(state, RELEVANCE_QUESTION)
        return RelevanceResult(chunk=chunk, probability=probability, kept=probability >= threshold)

    with ThreadPoolExecutor(max_workers=min(len(chunks), _MAX_WORKERS)) as pool:
        return list(pool.map(judge, chunks))


def rerank_kept_chunks(results: list[RelevanceResult]) -> list[Chunk]:
    """Order kept chunks by relevance probability, most relevant first.

    `filter_relevant_chunks` deliberately preserves retrieval order (see
    module docstring) so the audit trail stays aligned with the original
    chunk list. Call this separately when you want the chunks actually
    handed to the generator ordered by relevance instead -- LLMs attend
    more reliably to the start of a long context, so a strong chunk buried
    behind several weak ones is a real, avoidable source of hallucination
    that a pure filter doesn't fix.
    """
    kept = [r for r in results if r.kept]
    return [r.chunk for r in sorted(kept, key=lambda r: r.probability, reverse=True)]
