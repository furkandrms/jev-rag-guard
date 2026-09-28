import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rag_guard.decision_model import (
    AnthropicDecisionModel,
    HeuristicDecisionModel,
    JevDecisionModel,
    OpenAIDecisionModel,
)


def test_heuristic_noul_high_for_strong_overlap():
    model = HeuristicDecisionModel()
    state = "Question: What is the capital of France?\n\nPassage:\nParis is the capital of France."
    prob = model.noul(state, "Does this passage contain information that helps answer the question?")
    assert prob > 0.5


def test_heuristic_noul_low_for_no_overlap():
    model = HeuristicDecisionModel()
    state = "Question: What is the capital of France?\n\nPassage:\nBananas are a good source of potassium."
    prob = model.noul(state, "Does this passage contain information that helps answer the question?")
    assert prob < 0.5


def test_heuristic_noul_bounded():
    model = HeuristicDecisionModel()
    prob = model.noul("some state text here", "some question here")
    assert 0.0 <= prob <= 1.0


def test_heuristic_choice_distribution_sums_to_one():
    model = HeuristicDecisionModel()
    _, distribution = model.choice(
        "The invoice was for the billing department.",
        "Which team should handle this?",
        {"billing": "Charges and refunds", "technical": "Bugs", "account": "Login"},
    )
    assert abs(sum(distribution.values()) - 1.0) < 1e-3  # rounded to 4dp for readability
    assert set(distribution) == {"billing", "technical", "account"}


# --- OpenAIDecisionModel: mocked client, no network access ------------------


def _openai_client_returning(content: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    return client


def test_openai_noul_builds_prompt_and_parses_probability():
    client = _openai_client_returning(json.dumps({"probability": 0.87}))
    model = OpenAIDecisionModel(client=client)

    prob = model.noul("Question: X?\n\nPassage:\nY.", "Does this help?")

    assert prob == 0.87
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["temperature"] == 0
    prompt = kwargs["messages"][0]["content"]
    assert "Question: X?" in prompt
    assert "Passage:\nY." in prompt
    assert "Does this help?" in prompt


def test_openai_noul_clamps_out_of_range_probability():
    client = _openai_client_returning(json.dumps({"probability": 1.5}))
    model = OpenAIDecisionModel(client=client)
    assert model.noul("state", "question") == 1.0

    client = _openai_client_returning(json.dumps({"probability": -0.3}))
    model = OpenAIDecisionModel(client=client)
    assert model.noul("state", "question") == 0.0


def test_openai_choice_builds_options_and_normalizes_distribution():
    client = _openai_client_returning(
        json.dumps({"probabilities": {"a": 0.2, "b": 0.6}})
    )
    model = OpenAIDecisionModel(client=client)

    chosen, distribution = model.choice("state", "question", {"a": "Option A", "b": "Option B"})

    assert chosen == "b"
    assert abs(sum(distribution.values()) - 1.0) < 1e-9
    assert distribution["b"] > distribution["a"]
    prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "- a: Option A" in prompt
    assert "- b: Option B" in prompt


def test_openai_choice_missing_key_defaults_to_zero():
    client = _openai_client_returning(json.dumps({"probabilities": {"a": 1.0}}))
    model = OpenAIDecisionModel(client=client)

    chosen, distribution = model.choice("state", "question", {"a": "A", "b": "B"})

    assert chosen == "a"
    assert distribution["b"] == 0.0


# --- AnthropicDecisionModel: mocked client, no network access ---------------


def _anthropic_client_returning(text: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)]
    )
    return client


def test_anthropic_noul_builds_prompt_and_parses_probability():
    client = _anthropic_client_returning(json.dumps({"probability": 0.42}))
    model = AnthropicDecisionModel(client=client)

    prob = model.noul("Question: X?\n\nPassage:\nY.", "Does this help?")

    assert prob == 0.42
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["max_tokens"] == 256
    assert "temperature" not in kwargs  # unsupported by the current Anthropic API
    prompt = kwargs["messages"][0]["content"]
    assert "Question: X?" in prompt
    assert "Respond with JSON only." in prompt


def test_anthropic_malformed_json_recovery_strips_surrounding_prose():
    # Real models sometimes wrap JSON in prose/fences despite instructions;
    # _complete_json must pull out the first {...} block instead of failing.
    client = _anthropic_client_returning(
        'Sure, here is my answer:\n```json\n{"probability": 0.73}\n```\nHope that helps!'
    )
    model = AnthropicDecisionModel(client=client)

    prob = model.noul("state", "question")

    assert prob == 0.73


def test_anthropic_no_json_object_raises_value_error():
    client = _anthropic_client_returning("I cannot answer that.")
    model = AnthropicDecisionModel(client=client)

    with pytest.raises(ValueError, match="No JSON object found"):
        model.noul("state", "question")


def test_anthropic_choice_normalizes_distribution():
    client = _anthropic_client_returning(json.dumps({"probabilities": {"a": 0.3, "b": 0.3}}))
    model = AnthropicDecisionModel(client=client)

    chosen, distribution = model.choice("state", "question", {"a": "A", "b": "B"})

    assert chosen == "a"  # first key wins ties (max() takes first max)
    assert abs(distribution["a"] - 0.5) < 1e-9
    assert abs(distribution["b"] - 0.5) < 1e-9


# --- Opt-in live tests: real API calls, skipped unless a key is present -----
#
# These are the only tests that prove the adapters work against the live
# APIs rather than just parsing JSON correctly against a mock. They cost a
# real API call each and are never run by default CI (see .github/workflows/ci.yml).


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="requires OPENAI_API_KEY"
)
def test_openai_live_noul_easy_case():
    model = OpenAIDecisionModel()
    state = (
        "Question: What is the capital of France?\n\n"
        "Passage:\nParis is the capital and most populous city of France."
    )
    prob = model.noul(
        state, "Does this passage contain information that helps answer the question?"
    )
    assert prob > 0.5


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="requires ANTHROPIC_API_KEY"
)
def test_anthropic_live_noul_easy_case():
    model = AnthropicDecisionModel()
    state = (
        "Question: What is the capital of France?\n\n"
        "Passage:\nParis is the capital and most populous city of France."
    )
    prob = model.noul(
        state, "Does this passage contain information that helps answer the question?"
    )
    assert prob > 0.5


# --- JevDecisionModel: mocked client, no network access ---------------------
#
# Unlike the OpenAI/Anthropic adapters, Jev's System One API returns typed
# Noul/Choice answers directly -- there's no prompt string or JSON blob to
# inspect, so these tests assert on the request rag_guard builds (state,
# questions passed to system_one) and the typed response it reads back.


def _jev_client_returning(**answers: SimpleNamespace) -> MagicMock:
    client = MagicMock()
    client.system_one.return_value = SimpleNamespace(answers=answers)
    return client


def test_jev_noul_builds_question_and_reads_probability():
    client = _jev_client_returning(result=SimpleNamespace(type="noul", noul=0.87))
    model = JevDecisionModel(client=client)

    prob = model.noul("Question: X?\n\nPassage:\nY.", "Does this help?")

    assert prob == 0.87
    kwargs = client.system_one.call_args.kwargs
    assert kwargs["state"] == "Question: X?\n\nPassage:\nY."
    assert kwargs["questions"]["result"].instructions == "Does this help?"


def test_jev_choice_passes_options_as_criteria_and_reads_distribution():
    client = _jev_client_returning(
        result=SimpleNamespace(
            choice="b", probabilities={"a": 0.2, "b": 0.8}, confidence=0.6
        )
    )
    model = JevDecisionModel(client=client)

    chosen, distribution = model.choice(
        "state", "Which team?", {"a": "Option A", "b": "Option B"}
    )

    assert chosen == "b"
    assert distribution == {"a": 0.2, "b": 0.8}
    kwargs = client.system_one.call_args.kwargs
    assert kwargs["questions"]["result"].criteria == {"a": "Option A", "b": "Option B"}


def test_jev_model_override_is_passed_through():
    client = _jev_client_returning(result=SimpleNamespace(type="noul", noul=0.5))
    model = JevDecisionModel(model="jev-1.13.0", client=client)

    model.noul("state", "question")

    assert client.system_one.call_args.kwargs["model"] == "jev-1.13.0"


@pytest.mark.skipif(
    not os.environ.get("TYPESAFE_API_KEY"), reason="requires TYPESAFE_API_KEY"
)
def test_jev_live_noul_easy_case():
    model = JevDecisionModel()
    state = (
        "Question: What is the capital of France?\n\n"
        "Passage:\nParis is the capital and most populous city of France."
    )
    prob = model.noul(
        state, "Does this passage contain information that helps answer the question?"
    )
    assert prob > 0.5
