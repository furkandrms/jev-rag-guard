"""Stage 0: the pre-retrieval intent gate.

Some requests should never reach retrieval or generation at all, regardless
of what the corpus contains: a user describing a safety-critical or legal
situation needs a human, not a document lookup; a request for secrets or
another person's private data must be refused outright. Both are properties
of the *query itself*, so this check runs first, before any chunks are
fetched.
"""

from __future__ import annotations

from .decision_model import DecisionModel
from .types import IntentResult

INTENT_OPTIONS = {
    "proceed": (
        "A normal question the system should try to answer using retrieved "
        "document context."
    ),
    "escalate": (
        "The user describes a safety-critical situation (injury, fire, smoke, "
        "exposed high-voltage wiring, brake failure), a major equipment "
        "failure, or a legal threat (e.g. intends to sue) -- this needs an "
        "immediate human or crisis-team handoff, not a document-based answer."
    ),
    "refuse": (
        "The request asks the system to disclose confidential internal "
        "material (source code, license keys, diagnostic-tool credentials) "
        "or another identifiable person's private data (contact details, "
        "invoices, records tied to them) -- this must not be answered "
        "regardless of what the corpus contains."
    ),
}

INTENT_QUESTION = (
    "Classify this user message into exactly one of the given categories, "
    "based on the message alone."
)


def check_intent(
    query: str,
    model: DecisionModel,
    threshold: float = 0.5,
) -> IntentResult:
    """Judge whether `query` should be escalated or refused before retrieval.

    Defaults to "proceed" unless "escalate" or "refuse" is both the model's
    top choice and clears `threshold` -- an ambiguous/low-confidence call
    falls through to the normal pipeline rather than silently blocking it.
    """
    chosen, distribution = model.choice(f"Message: {query}", INTENT_QUESTION, INTENT_OPTIONS)
    probability = distribution.get(chosen, 0.0)
    action = chosen if chosen != "proceed" and probability >= threshold else "proceed"
    reason = (
        None
        if action == "proceed"
        else f"Query classified as {action!r} by the intent gate (p={probability:.2f})."
    )
    return IntentResult(action=action, probability=probability, reason=reason)
