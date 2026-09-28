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
