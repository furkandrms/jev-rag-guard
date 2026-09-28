"""rag-guard: a typed-decision accuracy layer for RAG pipelines.

Vector search finds passages that are semantically *close* to a query.
It does not tell you whether those passages actually *answer* the query,
whether you have *enough* of them to answer confidently, or whether the
model's generated answer is actually *supported* by what was retrieved.

rag-guard fills that gap with three small, cheap, typed checks run around
your existing retrieval + generation pipeline:

  1. relevance   -- per-chunk: "does this passage help answer the query?"
  2. sufficiency -- pre-generation: "is the kept context enough to answer
                    confidently and completely?"
  3. grounding   -- post-generation: "is every claim in the answer actually
                    supported by the retrieved context?"

Each check is a typed, probability-bearing decision (a "Noul" question in
TypeSafe/Jev terms: yes/no with a calibrated probability) rather than a
free-form model call, which keeps it fast, cheap, and auditable. The
decision model itself is pluggable -- see `rag_guard.decision_model`.
"""

from .decision_model import DecisionModel, HeuristicDecisionModel
from .pipeline import RagGuard
from .types import (
    Chunk,
    GroundingClaim,
    GroundingResult,
    GuardReport,
    RelevanceResult,
    SufficiencyResult,
)

__all__ = [
    "Chunk",
    "RelevanceResult",
    "SufficiencyResult",
    "GroundingClaim",
    "GroundingResult",
    "GuardReport",
    "DecisionModel",
    "HeuristicDecisionModel",
    "RagGuard",
]

__version__ = "0.1.0"
