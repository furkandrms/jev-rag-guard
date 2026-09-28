"""Minimal end-to-end example, no API keys required.

Simulates a tiny vector store (just a list of pre-written chunks -- swap
this for real retrieval from your own vector DB) and a tiny generator
(a string template -- swap for your real LLM call). Run with:

    python examples/quickstart.py
"""

from __future__ import annotations

from rag_guard import Chunk, HeuristicDecisionModel, RagGuard

# --- stand-ins for your real infrastructure -------------------------------

CORPUS = [
    Chunk(id="doc1", text="Paris is the capital and most populous city of France."),
    Chunk(id="doc2", text="France is a country in Western Europe with a population of about 68 million."),
    Chunk(id="doc3", text="The Eiffel Tower is a wrought-iron lattice tower in Paris, completed in 1889."),
    Chunk(id="doc4", text="Bananas are a good source of potassium and are grown in tropical climates."),
]


def fake_vector_search(query: str, k: int = 4) -> list[Chunk]:
    """Stand-in for a real vector store call -- just returns everything.

    In a real pipeline this is your existing retriever (pgvector, Milvus,
    Pinecone, whatever); rag-guard doesn't care how chunks were retrieved,
    only what it does with them afterward.
    """
    return CORPUS[:k]


def fake_generate(query: str, chunks: list[Chunk]) -> str:
    """Stand-in for your real LLM generation call."""
    if any("Paris" in c.text and "capital" in c.text for c in chunks):
        return "Paris is the capital of France. It is also home to the Eiffel Tower."
    return "I don't know."


# --- rag-guard wiring ------------------------------------------------------


def main() -> None:
    model = HeuristicDecisionModel()
    guard = RagGuard(
        model=model,
        relevance_threshold=0.3,
        sufficiency_threshold=0.3,
        grounding_threshold=0.3,
        grounding_min_coverage=0.5,
    )

    for query in [
        "What is the capital of France?",
        "How many moons does Jupiter have?",
    ]:
        chunks = fake_vector_search(query)
        report = guard.run(query, chunks, fake_generate)

        print(f"\nQuery: {query}")
        print(f"  Retrieved:  {len(chunks)} chunks")
        print(f"  Kept after relevance filter: {len(report.kept_chunks)}")
        print(f"  Sufficiency: {report.sufficiency.sufficient} (p={report.sufficiency.probability:.2f})")
        print(f"  Action: {report.action}")
        print(f"  Answer: {report.answer}")
        if report.grounding is not None:
            print(f"  Grounding coverage: {report.grounding.coverage:.2f}")


if __name__ == "__main__":
    main()
