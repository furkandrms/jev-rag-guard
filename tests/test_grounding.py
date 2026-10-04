from rag_guard.grounding import check_grounding, split_claims
from rag_guard.types import Chunk

from .conftest import FakeDecisionModel


def test_split_claims_splits_on_sentence_boundaries():
    claims = split_claims("Paris is the capital of France. It has about 2 million residents.")
    assert claims == [
        "Paris is the capital of France.",
        "It has about 2 million residents.",
    ]


def test_split_claims_handles_single_sentence():
    assert split_claims("Just one claim here.") == ["Just one claim here."]


def test_split_claims_empty_string():
    assert split_claims("") == []


def test_split_claims_does_not_split_on_abbreviations():
    claims = split_claims("Dr. Smith confirmed the results. The study ran for 3 months.")
    assert claims == [
        "Dr. Smith confirmed the results.",
        "The study ran for 3 months.",
    ]


def test_split_claims_does_not_split_on_latin_abbreviations():
    claims = split_claims("Nimbus supports several languages, e.g. Python and SQL.")
    assert claims == ["Nimbus supports several languages, e.g. Python and SQL."]


def test_fully_grounded_answer_passes():
    model = FakeDecisionModel(default=0.9)
    chunks = [Chunk(id="a", text="Paris is the capital of France.")]

    result = check_grounding(
        "Paris is the capital of France.", chunks, model, threshold=0.5, min_coverage=0.8
    )

    assert result.grounded is True
    assert result.coverage == 1.0
    assert result.claims[0].supported is True


def test_fully_ungrounded_answer_fails():
    model = FakeDecisionModel(default=0.1)
    chunks = [Chunk(id="a", text="Paris is the capital of France.")]

    result = check_grounding(
        "The Eiffel Tower was built in 1889 by an alien civilization.",
        chunks,
        model,
        threshold=0.5,
        min_coverage=0.8,
    )

    assert result.grounded is False
    assert result.coverage == 0.0


def test_partial_grounding_respects_min_coverage():
    # Two claims: one the fake model marks supported, one it doesn't.
    model = FakeDecisionModel(default=0.9)
    model.when("second claim", 0.1)
    chunks = [Chunk(id="a", text="supporting text")]

    answer = "This is the first claim. This is the second claim here."
    result = check_grounding(answer, chunks, model, threshold=0.5, min_coverage=0.8)

    assert result.coverage == 0.5
    assert result.grounded is False  # 0.5 < 0.8 min_coverage

    # Same data, looser coverage requirement should pass.
    result_loose = check_grounding(answer, chunks, model, threshold=0.5, min_coverage=0.4)
    assert result_loose.grounded is True


def test_empty_answer_is_not_grounded():
    model = FakeDecisionModel(default=0.9)
    result = check_grounding("", [Chunk(id="a", text="text")], model)
    assert result.grounded is False
    assert result.claims == []
