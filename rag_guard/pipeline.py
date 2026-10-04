"""RagGuard: orchestrates the three checks around your existing generation step.

rag-guard does not do retrieval or generation itself -- it wraps them. You
bring your own vector store and your own `generate_fn` (however you already
call your LLM); RagGuard decides, at each stage, whether it's worth
proceeding, and gives you a full trace of why.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .decision_model import DecisionModel
from .grounding import check_grounding
from .relevance import filter_relevant_chunks, rerank_kept_chunks
from .sufficiency import check_sufficiency
from .types import Chunk, GuardReport

GenerateFn = Callable[[str, list[Chunk]], str]


@dataclass
class RagGuard:
    """Wraps a RAG pipeline's retrieval -> generation step with typed decision checks.

    Parameters
    ----------
    model:
        The `DecisionModel` backing every check (see `rag_guard.decision_model`).
    relevance_threshold:
        Minimum P(relevant) for a chunk to survive stage 1. Chunks below this
        are dropped before the sufficiency check and never reach generation.
    sufficiency_threshold:
        Minimum P(sufficient) for the pipeline to proceed to generation at all.
        Below this, RagGuard returns `insufficient_context_message` instead of
        calling `generate_fn`, saving the generation cost entirely.
    grounding_threshold:
        Minimum P(supported) for a single claim to count as grounded.
    grounding_min_coverage:
        Minimum fraction of claims that must be grounded for the whole answer
        to pass. Below this, the answer is still returned (callers decide
        what to do -- show it flagged, retry generation, escalate to a
        stronger model, etc.) but `action` is set to "ungrounded_answer_flagged".
    check_grounding_enabled:
        Set False to skip stage 3 entirely (e.g. if `generate_fn` already has
        its own citation mechanism you trust).
    rerank_by_relevance:
        If True, chunks are handed to `generate_fn` ordered by relevance
        probability (most relevant first) instead of retrieval order.
        `GuardReport.kept_chunks` still reflects retrieval order for audit
        purposes -- this only reorders what the generator actually sees.
    """

    model: DecisionModel
    relevance_threshold: float = 0.5
    sufficiency_threshold: float = 0.6
    grounding_threshold: float = 0.5
    grounding_min_coverage: float = 0.8
    check_grounding_enabled: bool = True
    rerank_by_relevance: bool = False
    insufficient_context_message: str = (
        "I don't have enough information in the retrieved context to answer "
        "this confidently."
    )

    def run(self, query: str, chunks: list[Chunk], generate_fn: GenerateFn) -> GuardReport:
        """Run the full guarded pipeline for one query.

        Stops before calling `generate_fn` if the sufficiency gate fails.
        Otherwise calls `generate_fn(query, kept_chunks)` exactly once and,
        if grounding checks are enabled, verifies the result.
        """
        relevance = filter_relevant_chunks(
            query, chunks, self.model, threshold=self.relevance_threshold
        )
        kept_chunks = (
            rerank_kept_chunks(relevance)
            if self.rerank_by_relevance
            else [r.chunk for r in relevance if r.kept]
        )

        sufficiency = check_sufficiency(
            query, kept_chunks, self.model, threshold=self.sufficiency_threshold
        )

        if not sufficiency.sufficient:
            return GuardReport(
                query=query,
                relevance=relevance,
                sufficiency=sufficiency,
                answer=self.insufficient_context_message,
                grounding=None,
                action="insufficient_context",
            )

        answer = generate_fn(query, kept_chunks)

        if not self.check_grounding_enabled:
            return GuardReport(
                query=query,
                relevance=relevance,
                sufficiency=sufficiency,
                answer=answer,
                grounding=None,
                action="answered",
            )

        grounding = check_grounding(
            answer,
            kept_chunks,
            self.model,
            threshold=self.grounding_threshold,
            min_coverage=self.grounding_min_coverage,
        )

        action = "answered" if grounding.grounded else "ungrounded_answer_flagged"
        return GuardReport(
            query=query,
            relevance=relevance,
            sufficiency=sufficiency,
            answer=answer,
            grounding=grounding,
            action=action,
        )
