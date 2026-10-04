"""Shared test fixtures: a fully-scripted decision model for exact assertions.

HeuristicDecisionModel is deterministic but its scores depend on lexical
overlap, which makes it awkward to assert exact pipeline branches against.
FakeDecisionModel instead returns pre-configured answers keyed by the
question text, so pipeline-branch tests can assert precisely: "given a
sufficiency probability of 0.2, RagGuard must not call generate_fn."
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from rag_guard.decision_model import DecisionModel


class FakeDecisionModel(DecisionModel):
    """Returns canned probabilities keyed by (question substring)."""

    def __init__(self, default: float = 0.9) -> None:
        self.default = default
        # list of (substring, probability) checked in order; first match wins
        self.noul_rules: list[tuple[str, float]] = []
        # list of (substring, chosen_key, distribution) checked in order
        self.choice_rules: list[tuple[str, str, dict[str, float]]] = []
        self.calls: list[tuple[str, str]] = []

    def when(self, question_substring: str, probability: float) -> FakeDecisionModel:
        self.noul_rules.append((question_substring, probability))
        return self

    def when_choice(
        self, question_substring: str, chosen_key: str, distribution: dict[str, float]
    ) -> FakeDecisionModel:
        self.choice_rules.append((question_substring, chosen_key, distribution))
        return self

    def noul(self, state: str, question: str) -> float:
        self.calls.append((state, question))
        for substring, probability in self.noul_rules:
            if substring in state or substring in question:
                return probability
        return self.default

    def choice(
        self, state: str, question: str, options: Mapping[str, str]
    ) -> tuple[str, dict[str, float]]:
        for substring, chosen_key, distribution in self.choice_rules:
            if substring in state or substring in question:
                return chosen_key, distribution
        keys = list(options)
        distribution = {k: (1.0 if i == 0 else 0.0) for i, k in enumerate(keys)}
        return keys[0], distribution


@pytest.fixture
def fake_model() -> FakeDecisionModel:
    return FakeDecisionModel()
