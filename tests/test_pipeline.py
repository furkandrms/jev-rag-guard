from rag_guard.pipeline import RagGuard
from rag_guard.types import Chunk

from .conftest import FakeDecisionModel


def _chunks(n: int) -> list[Chunk]:
    return [Chunk(id=str(i), text=f"passage {i} about the topic") for i in range(n)]


def test_stops_before_generation_when_insufficient():
    model = FakeDecisionModel(default=0.9)
    model.when("enough information", 0.1)  # sufficiency question fails
    guard = RagGuard(model=model, sufficiency_threshold=0.6)

    generate_calls = []

    def generate_fn(query, chunks):
        generate_calls.append((query, chunks))
        return "should never be called"

    report = guard.run("some question", _chunks(3), generate_fn)

    assert report.action == "insufficient_context"
    assert report.answer == guard.insufficient_context_message
    assert generate_calls == []  # generation was skipped entirely


def test_full_success_path():
    model = FakeDecisionModel(default=0.95)
    guard = RagGuard(model=model)

    def generate_fn(query, chunks):
        return "This is a well grounded answer."

    report = guard.run("some question", _chunks(3), generate_fn)

    assert report.action == "answered"
    assert report.answer == "This is a well grounded answer."
    assert report.grounding is not None
    assert report.grounding.grounded is True


def test_ungrounded_answer_is_flagged_not_hidden():
    model = FakeDecisionModel(default=0.95)
    model.when("directly supported", 0.05)  # grounding question fails for every claim
    guard = RagGuard(model=model)

    def generate_fn(query, chunks):
        return "This claim is entirely made up."

    report = guard.run("some question", _chunks(2), generate_fn)

    assert report.action == "ungrounded_answer_flagged"
    # The answer is still returned -- rag-guard flags, it doesn't silently swallow.
    assert report.answer == "This claim is entirely made up."
    assert report.grounding.grounded is False


def test_grounding_can_be_disabled():
    model = FakeDecisionModel(default=0.95)
    guard = RagGuard(model=model, check_grounding_enabled=False)

    report = guard.run("q", _chunks(1), lambda q, c: "answer")

    assert report.action == "answered"
    assert report.grounding is None


def test_relevance_filtering_narrows_chunks_before_generation():
    model = FakeDecisionModel(default=0.9)
    guard = RagGuard(model=model, relevance_threshold=0.5)

    seen_chunks = []

    def generate_fn(query, chunks):
        seen_chunks.extend(chunks)
        return "answer"

    all_chunks = _chunks(4)
    report = guard.run("q", all_chunks, generate_fn)

    assert len(seen_chunks) == 4  # all kept, since default probability is above threshold
    assert report.kept_chunks == seen_chunks


def test_report_kept_chunks_property_matches_relevance():
    model = FakeDecisionModel(default=0.9)
    model.when("passage 0", 0.1)  # drop the first chunk specifically
    guard = RagGuard(model=model, relevance_threshold=0.5)

    report = guard.run("q", _chunks(3), lambda q, c: "answer")

    kept_ids = [c.id for c in report.kept_chunks]
    assert "0" not in kept_ids
    assert set(kept_ids) == {"1", "2"}
