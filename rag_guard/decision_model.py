"""The typed-decision interface rag-guard's checks are built on.

This mirrors the shape used by typed decision models such as TypeSafe's Jev
(and its System One primitives Choice / Score / Noul): you hand the model
*state* plus a *bounded question*, and get back a *constrained* answer with a
calibrated probability -- never free-form text. That shape is what makes the
checks in this library cheap, fast, and auditable: each one is a single
yes/no or closed-set judgment, not a generation.

`DecisionModel` is the seam. Swap in whatever backs it:

  - `HeuristicDecisionModel` -- zero-dependency, deterministic, lexical-overlap
    based. Good for tests, demos, and offline development. It is NOT an
    accurate relevance/grounding judge -- replace it before trusting results.
  - `OpenAIDecisionModel` / `AnthropicDecisionModel` -- ask a real LLM for a
    structured yes/no + probability. Slower and pricier than a native typed-
    decision model, but work today with widely available APIs.
  - A real Jev client, if you have TypeSafe API access -- implement
    `DecisionModel` around `typesafe-sdk-python` and every check in this
    library gets faster and cheaper for free.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Mapping


class DecisionModel(ABC):
    """Abstract typed-decision backend: state + typed question -> probability."""

    @abstractmethod
    def noul(self, state: str, question: str) -> float:
        """Return P(yes) in [0, 1] for a yes/no `question` about `state`."""
        raise NotImplementedError

    @abstractmethod
    def choice(
        self, state: str, question: str, options: Mapping[str, str]
    ) -> tuple[str, dict[str, float]]:
        """Return (chosen_key, probability_distribution_over_option_keys)."""
        raise NotImplementedError

    def score(self, state: str, question: str, levels: list[str]) -> tuple[str, float]:
        """Ordered-rubric judgment. Default implementation: Choice over the levels."""
        options = {str(i): level for i, level in enumerate(levels)}
        chosen_key, distribution = self.choice(state, question, options)
        return levels[int(chosen_key)], distribution[chosen_key]


_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "and", "or", "but", "with", "this",
    "that", "these", "those", "it", "its", "as", "at", "by", "from", "does",
    "do", "did", "has", "have", "had", "not", "no", "can", "will", "would",
    "should", "could", "what", "which", "who", "whom", "how", "why", "when",
    "where",
    # Boilerplate words that show up in rag-guard's own question templates
    # (relevance.py, sufficiency.py, grounding.py). Without these, every
    # state/question pair scores artificially high overlap just from the
    # shared template wording ("Question:", "Passage:", "contain
    # information", etc.) regardless of actual topical relevance.
    "question", "passage", "answer", "answers", "help", "helps", "contain",
    "contains", "information", "context", "claim", "claims", "supported",
    "confidently", "completely", "directly", "enough", "retrieved", "above",
    "given", "only", "below",
}


def _tokenize(text: str) -> set[str]:
    return {
        w.lower()
        for w in _WORD_RE.findall(text)
        if w.lower() not in _STOPWORDS and len(w) > 1
    }


_LABEL_RE = re.compile(
    r"(?:^|\n)(Question|Passage|Retrieved context|Context|Claim):\s*",
    re.IGNORECASE,
)


def _split_labeled_state(state: str) -> dict[str, str]:
    """Split rag-guard's conventional 'Label: text' state strings into a dict.

    rag-guard's own checks always build state as labeled sections, e.g.
    ``"Question: ...\\n\\nPassage:\\n..."`` or ``"Context:\\n...\\n\\nClaim: ..."``.
    Recovering those sections lets the heuristic compare the two things that
    actually carry signal (question vs. passage, or context vs. claim)
    instead of diluting the comparison with instruction boilerplate.
    """
    parts = _LABEL_RE.split(state)
    segments: dict[str, str] = {}
    # re.split with one capturing group returns [pre, label, text, label, text, ...]
    for i in range(1, len(parts) - 1, 2):
        segments[parts[i].strip().lower()] = parts[i + 1].strip()
    return segments


class HeuristicDecisionModel(DecisionModel):
    """Deterministic, zero-dependency stand-in for a real decision model.

    Scores `noul` as normalized token overlap between the two meaningful
    halves of the state (question vs. passage, or context vs. claim -- see
    `_split_labeled_state`), mapped into a probability with a soft curve so
    partial overlap doesn't read as near-certainty. It has no real language
    understanding -- it exists so the rest of this library is testable and
    demoable without an API key, and so you have a clear seam to swap in
    something accurate before relying on results.
    """

    def __init__(self, midpoint: float = 0.18, steepness: float = 12.0) -> None:
        self.midpoint = midpoint
        self.steepness = steepness

    def _overlap_score(self, state: str, question: str) -> float:
        segments = _split_labeled_state(state)
        if "question" in segments and "passage" in segments:
            left, right = segments["question"], segments["passage"]
        elif "question" in segments and "retrieved context" in segments:
            left, right = segments["question"], segments["retrieved context"]
        elif "context" in segments and "claim" in segments:
            left, right = segments["claim"], segments["context"]
        else:
            left, right = question, state

        left_tokens = _tokenize(left)
        right_tokens = _tokenize(right)
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = left_tokens & right_tokens
        ratio = len(overlap) / len(left_tokens)
        # logistic squash around `midpoint` so scores spread across (0, 1)
        # instead of clustering near the extremes.
        x = self.steepness * (ratio - self.midpoint)
        return 1.0 / (1.0 + pow(2.718281828, -x))

    def noul(self, state: str, question: str) -> float:
        return round(self._overlap_score(state, question), 4)

    def choice(
        self, state: str, question: str, options: Mapping[str, str]
    ) -> tuple[str, dict[str, float]]:
        scores = {
            key: self._overlap_score(state, f"{question} {label}")
            for key, label in options.items()
        }
        total = sum(scores.values()) or 1.0
        distribution = {k: round(v / total, 4) for k, v in scores.items()}
        chosen = max(distribution, key=distribution.get)
        return chosen, distribution


class _StructuredLLMDecisionModel(DecisionModel):
    """Shared plumbing for LLM-backed decision models that return structured JSON.

    Subclasses implement `_complete_json`, which sends a prompt asking for a
    JSON object matching a small schema and returns the parsed dict. This
    class turns that into the typed Noul/Choice interface.
    """

    NOUL_INSTRUCTIONS = (
        "You are a typed decision function, not a chat assistant. "
        "Answer strictly as JSON: {{\"probability\": <float 0 to 1>}}. "
        "`probability` is your calibrated probability that the answer to the "
        "question is yes, given only the state below. No other text.\n\n"
        "State:\n{state}\n\nQuestion: {question}"
    )

    CHOICE_INSTRUCTIONS = (
        "You are a typed decision function, not a chat assistant. "
        "Choose exactly one option key and give a probability distribution "
        "over all option keys that sums to 1. Answer strictly as JSON: "
        "{{\"probabilities\": {{\"<key>\": <float>, ...}}}}. "
        "No other text.\n\n"
        "State:\n{state}\n\nQuestion: {question}\n\nOptions:\n{options}"
    )

    def _complete_json(self, prompt: str) -> dict:
        raise NotImplementedError

    def noul(self, state: str, question: str) -> float:
        prompt = self.NOUL_INSTRUCTIONS.format(state=state, question=question)
        result = self._complete_json(prompt)
        prob = float(result.get("probability", 0.0))
        return max(0.0, min(1.0, prob))

    def choice(
        self, state: str, question: str, options: Mapping[str, str]
    ) -> tuple[str, dict[str, float]]:
        options_text = "\n".join(f"- {k}: {v}" for k, v in options.items())
        prompt = self.CHOICE_INSTRUCTIONS.format(
            state=state, question=question, options=options_text
        )
        result = self._complete_json(prompt)
        raw = result.get("probabilities", {})
        distribution = {k: float(raw.get(k, 0.0)) for k in options}
        total = sum(distribution.values()) or 1.0
        distribution = {k: v / total for k, v in distribution.items()}
        chosen = max(distribution, key=distribution.get)
        return chosen, distribution


class OpenAIDecisionModel(_StructuredLLMDecisionModel):
    """Decision model backed by the OpenAI API's structured JSON output.

    Requires the `openai` package and an API key. This is a compatibility
    adapter, not a native typed-decision model: it costs and latencies like
    an LLM call, not like Jev's ~200-450ms typed-decision path. Use it to
    get rag-guard working today; swap in a native typed-decision client
    later for the speed/cost win.
    """

    def __init__(self, model: str = "gpt-4o-mini", client: object | None = None) -> None:
        self.model = model
        if client is not None:
            self._client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "OpenAIDecisionModel requires the 'openai' package: "
                    "pip install openai"
                ) from exc
            self._client = OpenAI()

    def _complete_json(self, prompt: str) -> dict:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        return json.loads(content)


class AnthropicDecisionModel(_StructuredLLMDecisionModel):
    """Decision model backed by the Anthropic API.

    Same compatibility-adapter caveat as `OpenAIDecisionModel`: this is an
    LLM call shaped to look like a typed decision, not a native one.
    """

    def __init__(self, model: str = "claude-haiku-4-5", client: object | None = None) -> None:
        self.model = model
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "AnthropicDecisionModel requires the 'anthropic' package: "
                    "pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic()

    def _complete_json(self, prompt: str) -> dict:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            temperature=0,
            messages=[{"role": "user", "content": prompt + "\n\nRespond with JSON only."}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        # Models sometimes wrap JSON in prose or fences despite instructions;
        # pull out the first {...} block rather than failing on strict parse.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in model response: {text!r}")
        return json.loads(match.group(0))
