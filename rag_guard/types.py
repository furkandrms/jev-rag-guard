"""Plain data types shared across rag-guard's three check stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A single retrieved passage, as it comes out of your vector store."""

    id: str
    text: str
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RelevanceResult:
    """Outcome of the per-chunk relevance check."""

    chunk: Chunk
    probability: float
    kept: bool


@dataclass(frozen=True)
class SufficiencyResult:
    """Outcome of the pre-generation sufficiency gate.

    `partial` is True when the context clearly doesn't clear the full
    `sufficient` bar but also isn't empty/off-topic -- i.e. a mixed question
    where some of it is answerable and some isn't. `sufficient` and
    `partial` are mutually exclusive; both False means the context gave no
    usable basis to answer anything.
    """

    sufficient: bool
    probability: float
    reason: str | None = None
    partial: bool = False


@dataclass(frozen=True)
class IntentResult:
    """Outcome of the pre-retrieval intent gate (stage 0).

    Classifies the raw query itself, before any retrieval happens, into one
    of three buckets: "proceed" (handle normally), "escalate" (safety/legal/
    major-failure situation -- hand off to a human, don't answer from docs),
    or "refuse" (asks for secrets, credentials, or another person's private
    data -- must not be answered regardless of what's in the corpus).
    """

    action: str  # "proceed" | "escalate" | "refuse"
    probability: float
    reason: str | None = None


@dataclass(frozen=True)
class ClarifyResult:
    """Outcome of the post-relevance ambiguity check.

    True when the question is missing information (e.g. vehicle type,
    mileage) that the retrieved context answers differently depending on,
    so answering without asking would mean silently guessing which branch
    applies.
    """

    needs_clarification: bool
    probability: float
    reason: str | None = None


@dataclass(frozen=True)
class CaveatResult:
    """Outcome of the post-grounding caveat check.

    True when the generated answer depends on conditions, exceptions, extra
    fees, or vehicle/plan-specific limitations present in the context that
    must be surfaced to the user, not just the headline answer.
    """

    has_caveat: bool
    probability: float
    reason: str | None = None


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
      - "refuse"                  -- stopped at the intent gate, before
                                      retrieval: request asks for secrets or
                                      another person's private data
      - "escalate"                -- stopped at the intent gate, before
                                      retrieval: safety-critical, legal, or
                                      major-failure situation needing a human
      - "clarify"                 -- stopped after relevance filtering:
                                      question is missing information the
                                      context answers differently depending on
      - "insufficient_context"    -- stopped before generation, context too weak
      - "ungrounded_answer_flagged" -- answer generated but failed the grounding
                                        check (caller decides whether to show,
                                        retry, or escalate it)
      - "answered_with_caveat"    -- passed all checks, but the answer depends
                                      on conditions/exceptions worth surfacing
      - "answered"                -- passed all checks, no caveats found
    """

    query: str
    relevance: list[RelevanceResult]
    sufficiency: SufficiencyResult
    answer: str | None = None
    grounding: GroundingResult | None = None
    action: str = "unknown"
    intent: IntentResult | None = None
    clarify: ClarifyResult | None = None
    caveat: CaveatResult | None = None

    @property
    def kept_chunks(self) -> list[Chunk]:
        return [r.chunk for r in self.relevance if r.kept]

