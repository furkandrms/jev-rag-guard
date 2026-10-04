from rag_guard.sufficiency import check_sufficiency
from rag_guard.types import Chunk

from .conftest import FakeDecisionModel


def test_no_chunks_is_always_insufficient():
    model = FakeDecisionModel(default=0.99)  # even a model that would say "yes" doesn't get asked
    result = check_sufficiency("query", [], model)

    assert result.sufficient is False
    assert result.probability == 0.0
    assert result.reason is not None
    assert model.calls == []  # short-circuited before any decision call


def test_sufficient_above_threshold():
    model = FakeDecisionModel(default=0.8)
    chunks = [Chunk(id="a", text="Paris is the capital of France.")]

    result = check_sufficiency("What is the capital of France?", chunks, model, threshold=0.6)

    assert result.sufficient is True
    assert result.probability == 0.8


def test_insufficient_below_threshold():
    model = FakeDecisionModel(default=0.4)
    chunks = [Chunk(id="a", text="unrelated passage")]

    result = check_sufficiency("some question", chunks, model, threshold=0.6)

    assert result.sufficient is False
    assert result.reason is not None


def test_boundary_is_inclusive():
    model = FakeDecisionModel(default=0.6)
    chunks = [Chunk(id="a", text="text")]

    result = check_sufficiency("question", chunks, model, threshold=0.6)

    assert result.sufficient is True


def test_partial_question_fires_a_second_call_when_insufficient():
    # SUFFICIENCY_QUESTION fails (0.1 -- well below threshold); PARTIAL_QUESTION
    # is a genuinely different question, matched on its own distinct wording,
    # not a lower bar on the same probability (see sufficiency.py's comment
    # on why that doesn't distinguish "half-answerable" from "irrelevant").
    model = FakeDecisionModel(default=0.1)
    model.when("at least one distinct part", 0.8)  # PARTIAL_QUESTION fires
    chunks = [Chunk(id="a", text="covers half the question")]

    result = check_sufficiency(
        "two-part question", chunks, model, threshold=0.6, partial_threshold=0.5
    )

    assert result.sufficient is False
    assert result.partial is True
    assert result.reason is not None
    assert len(model.calls) == 2  # both the sufficiency and the partial question were asked


def test_partial_question_not_asked_when_sufficient():
    model = FakeDecisionModel(default=0.9)
    model.when("at least one distinct part", 0.8)  # would fire if asked, but shouldn't be

    result = check_sufficiency(
        "question", [Chunk(id="a", text="text")], model, threshold=0.6, partial_threshold=0.5
    )

    assert result.sufficient is True
    assert result.partial is False
    assert len(model.calls) == 1  # only SUFFICIENCY_QUESTION -- short-circuited, already sufficient


def test_below_partial_threshold_is_not_partial():
    model = FakeDecisionModel(default=0.1)
    model.when("at least one distinct part", 0.2)  # PARTIAL_QUESTION also fails

    result = check_sufficiency(
        "question", [Chunk(id="a", text="unrelated passage")], model, threshold=0.6, partial_threshold=0.5
    )

    assert result.sufficient is False
    assert result.partial is False


def test_partial_disabled_by_default():
    model = FakeDecisionModel(default=0.1)
    model.when("at least one distinct part", 0.9)  # would fire if partial_threshold were set

    result = check_sufficiency(
        "mixed question", [Chunk(id="a", text="covers half the question")], model, threshold=0.6
    )

    assert result.sufficient is False
    assert result.partial is False  # no partial_threshold given -- old behavior exactly
    assert len(model.calls) == 1  # PARTIAL_QUESTION never asked
