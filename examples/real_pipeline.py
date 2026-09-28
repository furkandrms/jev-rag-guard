"""End-to-end example against a real vector store and a real LLM.

Unlike `quickstart.py` (a hardcoded chunk list + a string-template
"generator", no dependencies, no API key), this wires rag-guard to:

  - a real embedding + local vector store (chromadb, see "Why chromadb"
    below) over the ~30-document corpus in `corpus.py`
  - a real LLM for `generate_fn`, via whichever of OPENAI_API_KEY /
    ANTHROPIC_API_KEY is set

and prints the same `GuardReport` trace `quickstart.py` does, so you can
compare the heuristic demo against a real pipeline side by side.

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

    pip install -e ".[openai]" chromadb   # or ".[anthropic]"
    export OPENAI_API_KEY=...             # or ANTHROPIC_API_KEY
    python examples/real_pipeline.py

## Expected cost

This runs 4 queries end to end. Each query makes up to 1 chat completion
for generation, plus rag-guard's own decision-model calls (one Noul call
per candidate chunk for relevance, one for sufficiency, and one per
generated sentence for grounding -- with the default chunk count and short
answers here, that's on the order of 10-20 small completions per query).
Using `gpt-4o-mini` or `claude-haiku-4-5` (the defaults), the whole run
costs well under $0.01. Using a larger model for `DECISION_MODEL_NAME` /
`GENERATE_MODEL_NAME` will cost more.
"""

from __future__ import annotations

import os

from corpus import DOCUMENTS

from rag_guard import Chunk, RagGuard
from rag_guard.decision_model import AnthropicDecisionModel, DecisionModel, OpenAIDecisionModel
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


def _pick_backend() -> tuple[DecisionModel, GenerateFn]:
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI

        client = OpenAI()
        model = OpenAIDecisionModel(client=client)

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

        return model, generate

    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        client = anthropic.Anthropic()
        model = AnthropicDecisionModel(client=client)

        def generate(query: str, chunks: list[Chunk]) -> str:
            context = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=300,
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
            return "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )

        return model, generate

    raise SystemExit(
        "real_pipeline.py needs a real LLM: set OPENAI_API_KEY or ANTHROPIC_API_KEY.\n"
        "See the module docstring for setup and expected cost."
    )


def main() -> None:
    model, generate_fn = _pick_backend()
    collection = _build_vector_store()
    retrieve = _make_retriever(collection)

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
