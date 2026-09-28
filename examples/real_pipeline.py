"""End-to-end example against a real vector store and a real LLM.

Unlike `quickstart.py` (a hardcoded chunk list + a string-template
"generator", no dependencies, no API key), this wires rag-guard to:

  - a real embedding + local vector store (chromadb, see "Why chromadb"
    below) over the ~30-document corpus in `corpus.py`
  - a real LLM for `generate_fn`, via whichever of OPENAI_API_KEY /
    ANTHROPIC_API_KEY is set
  - a real decision model for rag-guard's own relevance/sufficiency/
    grounding checks: `JevDecisionModel` if TYPESAFE_API_KEY is set
    (native typed decisions -- no generation capability, so it's never
    used for `generate_fn`), otherwise whichever LLM is generating

and prints the same `GuardReport` trace `quickstart.py` does, so you can
compare the heuristic demo against a real pipeline side by side, or
compare decision-model backends against each other on the same queries.

## Why chromadb (not faiss-cpu)

Both are reasonable choices; chromadb has fewer setup steps for this
example specifically. faiss-cpu is just an index -- you still need to bring
your own embedding step (another API call, or a local sentence-transformers
/ torch install) before you have anything to add to it. chromadb ships a
default local embedding function (onnxruntime + a small MiniLM model, no
torch, no separate API key) behind the same `add()` / `query()` calls, so
`collection.add(documents=[...])` is the entire indexing step. If you
already have your own embedding pipeline, faiss-cpu (or your existing
vector DB) is just as good a fit for rag-guard -- it only ever sees the
`Chunk` objects you hand it.

## Setup

    pip install -e ".[openai]" chromadb   # or ".[anthropic]", ".[typesafe]"
    export OPENAI_API_KEY=...             # or ANTHROPIC_API_KEY
    export TYPESAFE_API_KEY=...           # optional: routes decision-model
                                           # checks through Jev instead
    python examples/real_pipeline.py

## Expected cost

This runs 4 queries end to end. Each query makes up to 1 chat completion
for generation, plus rag-guard's own decision-model calls (one Noul call
per candidate chunk for relevance, one for sufficiency, and one per
generated sentence for grounding -- with the default chunk count and short
answers here, that's on the order of 10-20 small completions per query).
Using `gpt-4o-mini` or `claude-haiku-4-5` (the defaults), the whole run
costs well under $0.01. Using a larger model for generation will cost more.
Jev's Noul/Choice calls are typed decisions, not chat completions -- they
are not priced or billed the same way as the LLM calls above; check your
TypeSafe plan for its own cost basis.
"""

from __future__ import annotations

import os

from corpus import DOCUMENTS

from rag_guard import Chunk, RagGuard
from rag_guard.decision_model import (
    AnthropicDecisionModel,
    DecisionModel,
    JevDecisionModel,
    OpenAIDecisionModel,
)
from rag_guard.pipeline import GenerateFn

COLLECTION_NAME = "rag_guard_real_pipeline_example"


def _build_vector_store():
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    if collection.count() == 0:
        ids, texts = zip(*DOCUMENTS)
        collection.add(ids=list(ids), documents=list(texts))
    return collection


def _make_retriever(collection):
    def retrieve(query: str, k: int = 4) -> list[Chunk]:
        results = collection.query(query_texts=[query], n_results=k)
        ids = results["ids"][0]
        texts = results["documents"][0]
        distances = results["distances"][0]
        return [
            Chunk(id=chunk_id, text=text, metadata={"distance": distance})
            for chunk_id, text, distance in zip(ids, texts, distances)
        ]

    return retrieve


def _make_generate_fn() -> tuple[str, GenerateFn, DecisionModel]:
    """Pick the LLM that generates answers, and the decision model it also backs by default.

    Returns (backend_name, generate_fn, fallback_decision_model) -- the
    fallback is used for rag-guard's own checks only if TYPESAFE_API_KEY
    isn't set (see `_make_decision_model`), since Jev has no generation
    capability of its own and can never be the answer to "what generates?".
    """
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI

        client = OpenAI()

        def generate(query: str, chunks: list[Chunk]) -> str:
            context = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Answer the question using only the context below. "
                            "If the context doesn't contain the answer, say so.\n\n"
                            f"Context:\n{context}\n\nQuestion: {query}"
                        ),
                    }
                ],
            )
            return response.choices[0].message.content

        return "openai", generate, OpenAIDecisionModel(client=client)

    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        client = anthropic.Anthropic()

        def generate(query: str, chunks: list[Chunk]) -> str:
            context = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=300,
                # No `temperature`: the current Anthropic API rejects it on
                # messages.create (see decision_model.py's AnthropicDecisionModel).
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Answer the question using only the context below. "
                            "If the context doesn't contain the answer, say so.\n\n"
                            f"Context:\n{context}\n\nQuestion: {query}"
                        ),
                    }
                ],
            )
            return "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )

        return "anthropic", generate, AnthropicDecisionModel(client=client)

    raise SystemExit(
        "real_pipeline.py needs a real LLM to generate answers: set OPENAI_API_KEY "
        "or ANTHROPIC_API_KEY. See the module docstring for setup and expected cost."
    )


def _make_decision_model(fallback: DecisionModel) -> tuple[str, DecisionModel]:
    """Pick the backend for rag-guard's own relevance/sufficiency/grounding checks.

    Prefers Jev (native typed decisions -- see README's "decision model is
    pluggable" section) when TYPESAFE_API_KEY is set; otherwise reuses
    whichever LLM is already generating answers, so no second API key is
    required just to run this example.
    """
    if os.environ.get("TYPESAFE_API_KEY"):
        return "jev", JevDecisionModel()
    return "same as generate_fn", fallback


def main() -> None:
    generate_backend, generate_fn, fallback_model = _make_generate_fn()
    decision_backend, model = _make_decision_model(fallback_model)
    collection = _build_vector_store()
    retrieve = _make_retriever(collection)

    print(f"generate_fn backend: {generate_backend}")
    print(f"decision model backend: {decision_backend}")

    guard = RagGuard(model=model)

    queries = [
        "What is the capital of France?",
        "How many moons does Jupiter have?",
        "What is the capital of the moon?",  # should fail sufficiency: no real answer exists
        "What ingredients does sourdough bread use to rise?",
    ]

    for query in queries:
        chunks = retrieve(query, k=4)
        report = guard.run(query, chunks, generate_fn)

        print(f"\nQuery: {query}")
        print(f"  Retrieved:  {[c.id for c in chunks]}")
        print(f"  Kept after relevance filter: {[c.id for c in report.kept_chunks]}")
        print(f"  Sufficiency: {report.sufficiency.sufficient} (p={report.sufficiency.probability:.2f})")
        print(f"  Action: {report.action}")
        print(f"  Answer: {report.answer}")
        if report.grounding is not None:
            print(f"  Grounding coverage: {report.grounding.coverage:.2f}")
            for claim in report.grounding.claims:
                flag = "OK" if claim.supported else "UNSUPPORTED"
                print(f"    [{flag}] ({claim.probability:.2f}) {claim.claim}")


if __name__ == "__main__":
    main()
