"""rag-guard: a typed-decision accuracy layer for RAG pipelines.

Vector search finds passages that are semantically *close* to a query.
It does not tell you whether those passages actually *answer* the query,
whether you have *enough* of them to answer confidently, or whether the
model's generated answer is actually *supported* by what was retrieved.

rag-guard fills that gap with six small, cheap, typed checks run around
your existing retrieval + generation pipeline:

  0. intent      -- pre-retrieval: "does this query need to be escalated to
                    a human, or refused outright, before we even retrieve?"
  1. relevance   -- per-chunk: "does this passage help answer the query?"
  1.5. clarify   -- post-relevance: "is the question missing information
                    the context answers differently depending on?"
  2. sufficiency -- pre-generation: "is the kept context enough to answer
                    confidently and completely?"
  3. grounding   -- post-generation: "is every claim in the answer actually
                    supported by the retrieved context?"
  3.5. caveat    -- post-grounding: "does the answer depend on a condition
                    or exception that must be surfaced to the user?"

Each check is a typed, probability-bearing decision (a "Noul"/"Choice"
question in TypeSafe/Jev terms) rather than a free-form model call, which
keeps it fast, cheap, and auditable. The decision model itself is
pluggable -- see `rag_guard.decision_model`.
"""

from .caveat import check_caveat
from .clarify import check_ambiguity
from .decision_model import DecisionModel, HeuristicDecisionModel
from .intent import check_intent
from .pipeline import RagGuard
from .relevance import rerank_kept_chunks
from .types import (
    CaveatResult,
    Chunk,
    ClarifyResult,
    GroundingClaim,
    GroundingResult,
    GuardReport,
    IntentResult,
    RelevanceResult,
    SufficiencyResult,
)

__all__ = [
    "Chunk",
    "RelevanceResult",
    "SufficiencyResult",
    "IntentResult",
    "ClarifyResult",
    "CaveatResult",
    "GroundingClaim",
    "GroundingResult",
    "GuardReport",
    "DecisionModel",
    "HeuristicDecisionModel",
    "RagGuard",
    "rerank_kept_chunks",
    "check_intent",
    "check_ambiguity",
    "check_caveat",
]

__version__ = "0.2.0"

