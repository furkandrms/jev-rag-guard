from rag_guard.relevance import filter_relevant_chunks, rerank_kept_chunks
from rag_guard.types import Chunk, RelevanceResult

from .conftest import FakeDecisionModel


def test_keeps_chunks_at_or_above_threshold():
    model = FakeDecisionModel(default=0.5)
    chunks = [Chunk(id="a", text="text a"), Chunk(id="b", text="text b")]

    results = filter_relevant_chunks("query", chunks, model, threshold=0.5)

    assert all(r.kept for r in results)


def test_drops_chunks_below_threshold():
    model = FakeDecisionModel(default=0.3)
    chunks = [Chunk(id="a", text="text a")]

    results = filter_relevant_chunks("query", chunks, model, threshold=0.5)

    assert results[0].kept is False
    assert results[0].probability == 0.3


def test_preserves_order_and_count():
    model = FakeDecisionModel(default=0.9)
    chunks = [Chunk(id=str(i), text=f"text {i}") for i in range(5)]

    results = filter_relevant_chunks("query", chunks, model, threshold=0.5)

    assert [r.chunk.id for r in results] == [c.id for c in chunks]


def test_empty_input_returns_empty_output():
    model = FakeDecisionModel()
    assert filter_relevant_chunks("query", [], model) == []


def test_rerank_orders_kept_chunks_by_probability_descending():
    a, b, c = Chunk(id="a", text="a"), Chunk(id="b", text="b"), Chunk(id="c", text="c")
    results = [
        RelevanceResult(chunk=a, probability=0.6, kept=True),
        RelevanceResult(chunk=b, probability=0.95, kept=True),
        RelevanceResult(chunk=c, probability=0.3, kept=False),
    ]

    assert rerank_kept_chunks(results) == [b, a]


def test_rerank_kept_chunks_empty_when_none_kept():
    results = [RelevanceResult(chunk=Chunk(id="a", text="a"), probability=0.1, kept=False)]
    assert rerank_kept_chunks(results) == []
