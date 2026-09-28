from rag_guard.decision_model import HeuristicDecisionModel


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
