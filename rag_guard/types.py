"""Plain data types shared across rag-guard's three check stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Chunk:
    """A single retrieved passage, as it comes out of your vector store."""

    id: str
    text: str
    source: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelevanceResult:
    """Outcome of the per-chunk relevance check."""

    chunk: Chunk
    probability: float
    kept: bool


@dataclass(frozen=True)
class SufficiencyResult:
    """Outcome of the pre-generation sufficiency gate."""

    sufficient: bool
    probability: float
    reason: Optional[str] = None


@dataclass(frozen=True)
class GroundingClaim:
    """One claim extracted from the generated answer and its support check."""

    claim: str
    supported: bool
    probability: float
    supporting_chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GroundingResult:
    """Outcome of the post-generation grounding check."""

    grounded: bool
    coverage: float  # fraction of claims judged supported
    claims: list[GroundingClaim] = field(default_factory=list)


@dataclass(frozen=True)
class GuardReport:
    """Full trace of what RagGuard decided at each stage, plus the final action.

    `action` is one of:
      - "insufficient_context"    -- stopped before generation, context too weak
      - "ungrounded_answer_flagged" -- answer generated but failed the grounding
                                        check (caller decides whether to show,
                                        retry, or escalate it)
      - "answered"                -- passed all checks
    """

    query: str
    relevance: list[RelevanceResult]
    sufficiency: SufficiencyResult
    answer: Optional[str] = None
    grounding: Optional[GroundingResult] = None
    action: str = "unknown"

    @property
    def kept_chunks(self) -> list[Chunk]:
        return [r.chunk for r in self.relevance if r.kept]
