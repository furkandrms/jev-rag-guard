# rag-guard

A typed-decision accuracy layer for RAG pipelines. It doesn't retrieve and
it doesn't generate -- it wraps the seam between the two with three small,
cheap, typed checks, and gives you a full trace of what it decided and why.

[![CI](https://github.com/furkandrms/jev-rag-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/furkandrms/jev-rag-guard/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Table of contents

- [Why](#why)
- [Install](#install)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [The decision model is pluggable](#the-decision-model-is-pluggable)
- [Tuning](#tuning)
- [What this deliberately doesn't do](#what-this-deliberately-doesnt-do)
- [Project layout](#project-layout)
- [Development](#development)
- [About the SKILL.md file](#about-the-skillmd-file)
- [License](#license)

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

Requires Python 3.10+.

```bash
pip install -e .
# optional, if you want a real decision model instead of the built-in heuristic:
pip install -e ".[openai]"
pip install -e ".[anthropic]"
pip install -e ".[typesafe]"   # native typed-decision model, see "decision model is pluggable" below
```

The core `rag_guard` package has **zero required dependencies**. Every
backend beyond the built-in heuristic is an opt-in extra.

If you're using a real backend or `examples/real_pipeline.py`, copy
[`.env.example`](.env.example) to `.env` and fill in whichever key(s) you
need:

```bash
cp .env.example .env
```

`.env` is gitignored -- it's never committed. rag-guard itself doesn't load
`.env` files for you (it has no dependencies to do so); `examples/real_pipeline.py`
reads these as plain environment variables, so either `export` them or use
a tool like `python-dotenv` / `direnv` to load `.env` into your shell.

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
export TYPESAFE_API_KEY=...           # optional: route decision-model checks through Jev
python examples/real_pipeline.py
```

It runs 4 queries end to end against a ~30-document corpus and prints the
same `GuardReport` trace as `quickstart.py`, so you can compare the
heuristic demo against a real pipeline side by side, or compare
decision-model backends against each other on the same queries. The whole
run costs well under $0.01 with the default models (`gpt-4o-mini` /
`claude-haiku-4-5`) -- see the module docstring for the full cost
breakdown.

`report` is a `GuardReport` with the full trace: `report.relevance` (per-chunk
probabilities and keep/drop decisions), `report.sufficiency`,
`report.grounding` (per-claim probabilities), and `report.kept_chunks`.
Nothing is hidden from you -- rag-guard flags problems, it doesn't silently
swallow them. What to do with an `ungrounded_answer_flagged` result (show it
with a warning, retry generation, escalate to a stronger model, log it for
review) is a decision left to your application.

## How it works

`RagGuard.run(query, chunks, generate_fn)` walks through the three stages
in order, short-circuiting as soon as a stage says there's no point going
further:

```
retrieved chunks
      │
      ▼
┌─────────────┐   drop chunks below relevance_threshold
│  Relevance  │   (per-chunk Noul call: "does this help answer the query?")
└─────────────┘
      │ kept chunks
      ▼
┌─────────────┐   below sufficiency_threshold?
│ Sufficiency │──────────────────────────► return "I don't know" + trace
└─────────────┘   (generate_fn is never called -- no wasted generation cost)
      │ sufficient
      ▼
┌─────────────┐
│ generate_fn │   your existing LLM call -- called exactly once
└─────────────┘
      │ answer
      ▼
┌─────────────┐   split into claims, check each against kept chunks
│  Grounding  │   below grounding_min_coverage? action = "ungrounded_answer_flagged"
└─────────────┘   (answer is still returned -- rag-guard flags, never hides)
      │
      ▼
  GuardReport(action, answer, relevance, sufficiency, grounding)
```

Every stage calls `model.noul(state, question)` (or `.choice(...)`) on
whatever `DecisionModel` you passed to `RagGuard(model=...)` -- see the next
section. `relevance.py`, `sufficiency.py`, and `grounding.py` never call an
LLM API directly; they only ever go through that one interface, which is
what makes backends swappable without touching the pipeline logic.

## The decision model is pluggable

`DecisionModel` is the one interface everything in this library is built
on: `noul(state, question) -> float` and `choice(state, question, options)
-> (key, distribution)`. Four implementations ship with the library:

- **`HeuristicDecisionModel`** (default, zero dependencies): deterministic
  lexical-overlap scoring. Good for tests, demos, and wiring up your
  pipeline before you've decided on a real backend. It has no real language
  understanding -- don't trust its judgments in production.
- **`OpenAIDecisionModel`** / **`AnthropicDecisionModel`**: ask a real LLM
  for a structured yes/no + probability. These work today with widely
  available APIs, but they're a compatibility adapter, not a native typed
  decision model -- you're paying LLM-call latency and cost for what should
  be a fast, cheap judgment. Requires `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.
- **`JevDecisionModel`**: a native typed-decision model, backed by
  TypeSafe's Jev. Noul/Choice are first-class request/response primitives
  on the wire (no prompt-building or JSON-parsing layer, unlike the OpenAI
  / Anthropic adapters), so checks get both faster and cheaper. Requires
  `pip install rag-guard[typesafe]` and a `TYPESAFE_API_KEY` (see
  https://console.typesafe.ai/).

See `rag_guard/decision_model.py` for the `DecisionModel` interface if you
want to implement your own backend.

Swapping backends never touches `relevance.py`, `sufficiency.py`,
`grounding.py`, or `pipeline.py` -- they only ever call `model.noul(...)`.
You can also mix backends: use Jev (or an LLM) for rag-guard's own checks
while a different LLM handles generation, since `generate_fn` and the
`DecisionModel` are independent constructor arguments -- see how
`examples/real_pipeline.py` picks each one separately.

### Comparing backends

All four backends were run against the same corpus and queries in
`examples/real_pipeline.py` during development. On simple factual queries,
`OpenAIDecisionModel`, `AnthropicDecisionModel`, and `JevDecisionModel` all
produced the same relevance/sufficiency/grounding verdicts, with decisive
probabilities (rarely landing near the 0.5 boundary). `HeuristicDecisionModel`
is not in that comparison on purpose -- it's a lexical stand-in, not a
judgment you should compare for accuracy.

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
- **No framework adapters (yet).** There's no built-in LangChain `Runnable`
  or LlamaIndex query-engine wrapper. `RagGuard.run()` only needs a list of
  `Chunk` objects and a `generate_fn(query, chunks) -> str`, so wrapping
  either framework's retriever/generator is a small amount of glue code on
  the calling side today.

## Project layout

```
rag_guard/
  decision_model.py     # DecisionModel interface + Heuristic/OpenAI/Anthropic/Jev implementations
  relevance.py           # stage 1: per-chunk relevance filter
  sufficiency.py          # stage 2: pre-generation sufficiency gate
  grounding.py             # stage 3: post-generation claim-by-claim grounding check
  pipeline.py               # RagGuard: orchestrates all three stages
  types.py                   # Chunk, RelevanceResult, SufficiencyResult, GroundingResult, GuardReport
tests/                        # full suite runs against a scripted FakeDecisionModel, no API needed
  conftest.py                 # FakeDecisionModel fixture used by every test module
  test_decision_model.py      # heuristic tests + mocked/live tests for every real backend
  test_relevance.py / test_sufficiency.py / test_grounding.py / test_pipeline.py
examples/
  quickstart.py                # zero-dependency, no API keys required
  real_pipeline.py              # real vector store (chromadb) + real LLM + optional Jev backend
  corpus.py                      # ~30-document corpus used by real_pipeline.py
.github/workflows/ci.yml          # pytest (3.10/3.11/3.12) + ruff + mypy, on push and PR
pyproject.toml                      # package metadata, optional-dependency extras, ruff/mypy config
```

## Development

```bash
pip install -e ".[dev]"
pytest -q          # 33 mocked tests always run; 3 live tests skip without API keys
ruff check .        # lint
mypy                 # strict type-check of rag_guard/
```

All non-live tests run against a fully-scripted `FakeDecisionModel` (see
`tests/conftest.py`), so the default suite is deterministic and needs no
API keys or network access. Three opt-in live tests
(`test_openai_live_*`, `test_anthropic_live_*`, `test_jev_live_*`) make a
real API call each and are skipped automatically unless the matching
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `TYPESAFE_API_KEY` is set --
they're what actually proves an adapter works against its live API, not
just that it parses a response correctly. CI never sets these keys, so it
never burns real API calls; wiring them up as CI secrets is a decision
left to whoever owns the repo.

## About the SKILL.md file

`SKILL.md` in the repo root is TypeSafe's own reference document for
building against their Jev API -- it points to the live documentation
(https://docs.typesafe.ai) for the current API contract, SDK usage, and
primitive definitions (Noul / Choice / Score), rather than duplicating
that reference here where it could drift out of date. It was used while
implementing `JevDecisionModel` in `rag_guard/decision_model.py`: the
correct package name (`typesafe-sdk`, not `typesafe-sdk-python`), the
`TypeSafeClient` / `Noul` / `Choice` shapes, and the `TYPESAFE_API_KEY`
convention all came from reading those live docs rather than being
guessed. It's kept in the repo as the canonical pointer for anyone
extending or debugging the Jev backend.

## License

MIT -- see [`LICENSE`](LICENSE).
