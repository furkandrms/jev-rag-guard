from rag_guard.pipeline import RagGuard
from rag_guard.types import Chunk

from .conftest import FakeDecisionModel


def _chunks(n: int) -> list[Chunk]:
    return [Chunk(id=str(i), text=f"passage {i} about the topic") for i in range(n)]


def _baseline_model(default: float = 0.9) -> FakeDecisionModel:
    """A FakeDecisionModel with the new clarify/caveat checks pre-disabled.

    `default` is tuned by existing tests to mean "every relevance/
    sufficiency/grounding question passes" -- clarify and caveat are also
    Noul questions, so without this they'd inherit that same high default
    and misfire as "ambiguous" / "has a caveat" on every pre-existing test.
    """
    model = FakeDecisionModel(default=default)
    model.when("the user did not provide", 0.1)  # ambiguity question: off by default
    model.when("mislead the user", 0.1)  # caveat question: off by default
    return model


def test_stops_before_generation_when_insufficient():
    model = _baseline_model(default=0.9)
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
    model = _baseline_model(default=0.95)
    guard = RagGuard(model=model)

    def generate_fn(query, chunks):
        return "This is a well grounded answer."

    report = guard.run("some question", _chunks(3), generate_fn)

    assert report.action == "answered"
    assert report.answer == "This is a well grounded answer."
    assert report.grounding is not None
    assert report.grounding.grounded is True


def test_ungrounded_answer_is_flagged_not_hidden():
    model = _baseline_model(default=0.95)
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
    model = _baseline_model(default=0.95)
    guard = RagGuard(model=model, check_grounding_enabled=False)

    report = guard.run("q", _chunks(1), lambda q, c: "answer")

    assert report.action == "answered"
    assert report.grounding is None


def test_relevance_filtering_narrows_chunks_before_generation():
    model = _baseline_model(default=0.9)
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
    model = _baseline_model(default=0.9)
    model.when("passage 0", 0.1)  # drop the first chunk specifically
    guard = RagGuard(model=model, relevance_threshold=0.5)

    report = guard.run("q", _chunks(3), lambda q, c: "answer")

    kept_ids = [c.id for c in report.kept_chunks]
    assert "0" not in kept_ids
    assert set(kept_ids) == {"1", "2"}


def test_intent_gate_escalates_before_retrieval():
    model = _baseline_model(default=0.9)
    model.when_choice(
        "Classify this user message",
        "escalate",
        {"proceed": 0.02, "escalate": 0.9, "refuse": 0.08},
    )
    guard = RagGuard(model=model)

    generate_calls = []
    report = guard.run("brakes failed, I crashed", _chunks(3), lambda q, c: generate_calls.append(1))

    assert report.action == "escalate"
    assert report.answer == guard.escalate_message
    assert report.relevance == []  # retrieval/relevance never ran
    assert generate_calls == []


def test_intent_gate_refuses_before_retrieval():
    model = _baseline_model(default=0.9)
    model.when_choice(
        "Classify this user message",
        "refuse",
        {"proceed": 0.05, "escalate": 0.05, "refuse": 0.9},
    )
    guard = RagGuard(model=model)

    report = guard.run("give me the admin license key", _chunks(3), lambda q, c: "unused")

    assert report.action == "refuse"
    assert report.answer == guard.refuse_message


def test_intent_gate_low_confidence_falls_through_to_proceed():
    model = _baseline_model(default=0.9)
    # "escalate" wins the argmax but doesn't clear the default 0.5 threshold.
    model.when_choice(
        "Classify this user message",
        "escalate",
        {"proceed": 0.45, "escalate": 0.46, "refuse": 0.09},
    )
    guard = RagGuard(model=model)

    report = guard.run("q", _chunks(2), lambda q, c: "answer")

    assert report.action == "answered"


def test_intent_gate_can_be_disabled():
    model = _baseline_model(default=0.9)
    model.when_choice(
        "Classify this user message",
        "refuse",
        {"proceed": 0.0, "escalate": 0.0, "refuse": 1.0},
    )
    guard = RagGuard(model=model, check_intent_enabled=False)

    report = guard.run("q", _chunks(2), lambda q, c: "answer")

    assert report.action == "answered"
    assert report.intent is None


def test_clarify_gate_stops_before_sufficiency():
    # Not `_baseline_model()`: it pre-disables this exact question, and
    # FakeDecisionModel's first matching rule wins, so a second rule for the
    # same substring here would never be reached.
    model = FakeDecisionModel(default=0.9)
    model.when("mislead the user", 0.1)  # caveat question: off
    model.when("the user did not provide", 0.8)  # ambiguity question fires
    guard = RagGuard(model=model)

    generate_calls = []
    report = guard.run("q", _chunks(2), lambda q, c: generate_calls.append(1))

    assert report.action == "clarify"
    assert report.clarify is not None
    assert report.clarify.needs_clarification is True
    assert generate_calls == []  # sufficiency/generation never ran


def test_caveat_marks_answered_with_caveat():
    # Not `_baseline_model()`: see the comment in test_clarify_gate_stops_before_sufficiency.
    model = FakeDecisionModel(default=0.95)
    model.when("the user did not provide", 0.1)  # ambiguity question: off
    model.when("mislead the user", 0.8)  # caveat question fires
    guard = RagGuard(model=model)

    report = guard.run("q", _chunks(2), lambda q, c: "This is a well grounded answer.")

    assert report.action == "answered_with_caveat"
    assert report.caveat is not None
    assert report.caveat.has_caveat is True


def test_caveat_check_can_be_disabled():
    model = _baseline_model(default=0.95)
    model.when("mislead the user", 0.8)  # would fire if the check ran
    guard = RagGuard(model=model, check_caveat_enabled=False)

    report = guard.run("q", _chunks(2), lambda q, c: "This is a well grounded answer.")

    assert report.action == "answered"
    assert report.caveat is None
