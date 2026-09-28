# rag-guard

A typed-decision accuracy layer for RAG pipelines. It doesn't retrieve and
it doesn't generate -- it wraps the seam between the two with three small,
cheap, typed checks, and gives you a full trace of what it decided and why.

## Why

Vector search finds passages that are embedding-close to a query. That is
not the same thing as "this passage answers the query," "we have enough of
these to answer confidently," or "the model's answer is actually supported
by what we retrieved." Most RAG hallucination doesn't come from the
generator lying -- it comes from being handed a context that doesn't
actually contain the answer, and doing its best anyway.

rag-guard adds three checkpoints around your existing pipeline:

| Stage | When | Question | Effect |
|---|---|---|---|
| **Relevance** | after retrieval | "Does this passage help answer the query?" | drops noise before it reaches the next stage |
| **Sufficiency** | before generation | "Is the kept context enough to answer confidently and completely?" | skips generation entirely on a "no" -- returns an honest "I don't know" instead |
| **Grounding** | after generation | "Is each claim in the answer actually supported by the context?" | flags (doesn't hide) answers that drift beyond what was retrieved |

Each check is a single bounded yes/no (or closed-set) judgment with a
calibrated probability attached -- not a free-form model call. That's the
same shape used by typed decision models like TypeSafe's Jev (Choice /
Score / Noul over program state), and it's what keeps these checks fast,
cheap, and auditable: you get a probability you can threshold and log, not
prose you have to parse.

## Install

```bash
pip install -e .
# optional, if you want an LLM-backed decision model instead of the built-in heuristic:
pip install -e ".[openai]"
pip install -e ".[anthropic]"
```

## Quickstart

```python
from rag_guard import Chunk, HeuristicDecisionModel, RagGuard

guard = RagGuard(model=HeuristicDecisionModel())

def my_generate(query: str, chunks: list[Chunk]) -> str:
    # your existing LLM call goes here
    ...

chunks = my_vector_store.search(query, k=5)   # your existing retrieval
report = guard.run(query, chunks, my_generate)

print(report.action)   # "insufficient_context" | "ungrounded_answer_flagged" | "answered"
print(report.answer)
```

Run `python examples/quickstart.py` for a full working example with no API
keys required (uses the built-in heuristic decision model and a stub
retriever/generator).

For a version wired to a real vector store (chromadb) and a real LLM
(OpenAI or Anthropic), see `examples/real_pipeline.py`:

```bash
pip install -e ".[examples,openai]"   # or ".[examples,anthropic]"
export OPENAI_API_KEY=...             # or ANTHROPIC_API_KEY
python examples/real_pipeline.py
```

It runs 4 queries end to end against a ~30-document corpus and prints the
same `GuardReport` trace as `quickstart.py`, so you can compare the
heuristic demo against a real pipeline side by side. The whole run costs
well under $0.01 with the default models (`gpt-4o-mini` /
`claude-haiku-4-5`) -- see the module docstring for the cost breakdown.

`report` is a `GuardReport` with the full trace: `report.relevance` (per-chunk
probabilities and keep/drop decisions), `report.sufficiency`,
`report.grounding` (per-claim probabilities), and `report.kept_chunks`.
Nothing is hidden from you -- rag-guard flags problems, it doesn't silently
swallow them. What to do with an `ungrounded_answer_flagged` result (show it
with a warning, retry generation, escalate to a stronger model, log it for
review) is a decision left to your application.

## The decision model is pluggable

`DecisionModel` is the one interface everything in this library is built
on: `noul(state, question) -> float` and `choice(state, question, options)
-> (key, distribution)`. Three implementations ship with the library:

- **`HeuristicDecisionModel`** (default, zero dependencies): deterministic
  lexical-overlap scoring. Good for tests, demos, and wiring up your
  pipeline before you've decided on a real backend. It has no real language
  understanding -- don't trust its judgments in production.
- **`OpenAIDecisionModel`** / **`AnthropicDecisionModel`**: ask a real LLM
  for a structured yes/no + probability. These work today with widely
  available APIs, but they're a compatibility adapter, not a native typed
  decision model -- you're paying LLM-call latency and cost for what should
  be a fast, cheap judgment.
- **A real typed-decision model** (e.g. TypeSafe's Jev, if you have API
  access): implement `DecisionModel` around `typesafe-sdk-python` and every
  check in this library gets both faster and cheaper for free, since that's
  exactly the shape typed decision models are built for. See
  `rag_guard/decision_model.py` for the interface to implement.

Swapping backends never touches `relevance.py`, `sufficiency.py`,
`grounding.py`, or `pipeline.py` -- they only ever call `model.noul(...)`.

## Tuning

Every threshold is a constructor argument on `RagGuard`:

```python
guard = RagGuard(
    model=my_model,
    relevance_threshold=0.5,       # per-chunk keep/drop cutoff
    sufficiency_threshold=0.6,     # minimum P(sufficient) to proceed to generation
    grounding_threshold=0.5,       # minimum P(supported) for one claim
    grounding_min_coverage=0.8,    # fraction of claims that must be grounded
    check_grounding_enabled=True,  # set False to skip stage 3 entirely
)
```

Start conservative (higher thresholds) in domains where a wrong answer is
costly, and loosen them once you've measured false-reject rates on your own
data -- the same "validate on your own data, calibrate thresholds" advice
that applies to any typed-decision system applies here too.

## What this deliberately doesn't do

- **No retrieval.** Bring your own vector store; rag-guard only sees the
  `Chunk` objects you hand it.
- **No generation.** You supply `generate_fn`; rag-guard calls it once per
  `run()`, after the sufficiency gate passes.
- **No reranking.** Relevance filtering drops chunks below a threshold; it
  doesn't reorder survivors. Pair rag-guard with a reranker upstream if you
  want both (filter after rerank, so you're not scoring passages your
  reranker already discarded).
- **No claim extraction beyond sentence splitting.** `grounding.py`'s
  `split_claims` is a naive sentence splitter. If your answers are long or
  multi-clause, swap in a proper claim-decomposition step (an LLM call that
  extracts atomic claims) before grounding-checking each one.

## Project layout

```
rag_guard/
  decision_model.py   # the DecisionModel interface + Heuristic/OpenAI/Anthropic implementations
  relevance.py         # stage 1: per-chunk relevance filter
  sufficiency.py        # stage 2: pre-generation sufficiency gate
  grounding.py           # stage 3: post-generation claim-by-claim grounding check
  pipeline.py            # RagGuard: orchestrates all three stages
  types.py               # Chunk, RelevanceResult, SufficiencyResult, GroundingResult, GuardReport
tests/                    # full suite runs against a scripted FakeDecisionModel, no API needed
examples/quickstart.py    # runnable end-to-end demo, no API keys required
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

All tests run against a fully-scripted `FakeDecisionModel` (see
`tests/conftest.py`), so the suite is deterministic and needs no API keys or
network access.
