# Loci

**A local assistant whose memory lives on disk, not in your RAM.**

Loci is a low-footprint, fully local RAG system for modest machines (built and
tuned on a Mac mini M1 with 8GB). It pairs a small quantized LLM with a
knowledge base stored entirely in SQLite on your SSD — combining a structured
**facts layer** (who did what, where) with **vector + full-text search** — so
a 3B model can answer with the precision of something much larger, without
locking up your system.

The name comes from the *method of loci*, the ancient memory-palace technique:
knowledge filed in organized locations and retrieved by walking to the right
spot. That is literally how Loci works.

> **Status: Phase 3 complete.** Schema, config, instrumentation, ingestion
> pipeline, fact extraction, and entity resolution are implemented and tested.
> See [PLAN.md](PLAN.md) for the build phases.
>
> **Phase 2 demo:**
> ```bash
> # One-time setup
> uv run loci config init
> brew install cmake && CMAKE_ARGS="-DGGML_METAL=on" uv add llama-cpp-python
> uv run loci models pull          # download GGUF embedder + chat model
>
> # Ingest a document (embedder optional — facts and FTS work without it)
> loci ingest ~/loci/raw/sherlock.txt --meta book="The Sign of the Four"
>
> uv run pytest                    # 95 tests, all passing
> ```
>
> **Known Phase 2 limitation:** pronoun subjects ("He took…") produce no fact;
> coreference resolution is deferred to Phase 6. The vector/FTS layer covers
> those sentences. Passive sentences are extracted with the grammatical subject
> as the fact subject (semantic role inversion fixed in Phase 6).

---

## Why

Most local RAG stacks assume RAM is cheap: PyTorch runtimes, in-memory vector
indexes, embedding servers. On an 8GB machine that means swapping, beachballs,
and an unusable computer while the assistant "thinks."

Loci flips the assumption:

- **Knowledge lives on SSD.** One SQLite file holds the facts graph, the
  vector index (sqlite-vec), full-text search (FTS5), and full provenance.
  Idle RAM cost: effectively zero.
- **No PyTorch anywhere.** Chat and embedding models are GGUF files run
  through llama.cpp with Metal acceleration and mmap, so weights page from
  disk on demand.
- **Models load only when needed** and are freed when done. Peak RSS during
  chat stays under ~3.5GB; the rest of your machine keeps working.
- **Knowledge grows by connecting, not rebuilding.** Ingestion is append-only
  and idempotent. New facts link to existing entities; nothing is ever
  re-indexed from scratch.

## How it works

Every ingested sentence feeds two complementary layers over the same text:

```
                        ┌─────────────────────────────┐
  "Sherlock Holmes      │  FACTS LAYER (the dots)     │   precise, indexed,
   took his bottle  ──► │  subject: sherlock_holmes   │   ~milliseconds
   from the corner      │  predicate: take            │
   of the mantel-       │  object: bottle             │
   piece..."            │  qualifiers: {from: corner  │
        │               │   of the mantel-piece}      │
        │               └─────────────┬───────────────┘
        │                             │ every fact points back
        ▼                             ▼ to its source sentence
  ┌─────────────────────────────────────────────────┐
  │  VECTOR + FULL-TEXT LAYER (the safety net)      │   semantic recall for
  │  chunk embeddings (sqlite-vec) + FTS5/BM25      │   everything extraction
  └─────────────────────────────────────────────────┘   misses
```

Ask *"what did Sherlock Holmes take?"* and Loci parses out two identifiers —
a subject and a predicate — and resolves the answer with an indexed SQL
lookup before vectors are even needed. Ask a paraphrase the parser can't
handle and the embedding + keyword layers catch it. Either way, the LLM is
shown the **original source sentences** with citations, never just extracted
triples, so lossy extraction can't corrupt answers.

Entities accumulate aliases as Loci reads ("Holmes", "Sherlock",
"Mr. Sherlock Holmes" → one entity), so facts from every new document attach
to the knowledge you already have. That's the growth model: densify the
graph, never start over.

## Features

- **Hybrid retrieval** — entity/predicate SQL lookup + vector search + BM25,
  fused with Reciprocal Rank Fusion; `--explain` shows exactly why each
  result surfaced.
- **Grounded answers with citations** — every reply cites fact/chunk tags
  mapped to book/chapter/file; questions outside the knowledge base get an
  honest "not in my knowledge base."
- **Knowledge packs** — a pack is just a `.locipack.db` file with the same
  schema. Download an expert pack, `loci pack add` it, and queries fan out
  across all attached packs.
- **Everything configurable** — models, every storage location, chunking,
  retrieval parameters, logging: one commented `config.toml`, with CLI-flag
  and env-var overrides.
- **Built-in benchmarking** — `loci bench` measures ingest throughput, peak
  RSS, swap pressure, per-stage query latency, tokens/s, and grades answer
  quality 0–100 against your own QnA ground truth using Claude CLI as judge.

## Requirements

- macOS on Apple Silicon (primary target; Linux should work, untested)
- ~8GB RAM (a `--low-mem` profile targets even tighter setups)
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- ~3GB of SSD space for models, plus your knowledge bases
- Optional: [Claude CLI](https://docs.claude.com/en/docs/claude-code) for
  benchmark judging (the only feature that touches the network)

## Quickstart

```bash
# 1. Install
git clone https://github.com/<you>/loci && cd loci
uv sync

# 2. Create your config (edit paths/params as you like)
loci config init

# 3. Download the default models (chat + embedder, with checksum verification)
loci models pull

# 4. Feed it knowledge
loci ingest ~/loci/raw/pg2097_the_sign_of_four.txt \
  --meta book="The Sign of the Four"

# 5. Ask
loci ask "what did sherlock holmes take?"
#   Sherlock Holmes took his bottle from the corner of the mantel-piece
#   and his hypodermic syringe from its neat morocco case. [F3][F4]
#   Sources: The Sign of the Four, Ch. I — The Science of Deduction

# or have a conversation
loci chat
```

Useful extras:

```bash
loci ask "..." --explain      # show parse, SQL hits, fusion ranking
loci stats                    # entities, facts, chunks, db size
loci entities review          # resolve ambiguous entity links
loci pack add medicine.locipack.db
loci ask "..." --low-mem      # 1B model, 2048 ctx, for tighter machines
```

## Configuration

Everything lives in `config.toml` (`loci config init` writes a fully
commented copy; `loci config show` prints the effective merged values and
where each came from). Precedence: **CLI flag → env var (`LOCI_SECTION__KEY`)
→ config.toml → default**.

| Section      | What you control                                                        |
|--------------|-------------------------------------------------------------------------|
| `[paths]`    | models dir, raw knowledge dir, knowledge db, packs, chat context, bench logs, runtime logs |
| `[models]`   | chat / low-mem / embedder GGUFs, context size, GPU layers, sampling     |
| `[ingest]`   | chunk size, overlap, embedding batch size, spaCy model                  |
| `[retrieval]`| top-k per layer, context token budget, entity similarity threshold, RRF |
| `[logging]`  | level, structured JSONL, RSS logging                                    |
| `[bench]`    | QnA file, runs per question, RSS sampling rate, Claude judge settings   |

No module hardcodes a path or parameter — change a value, re-run the
benchmark, compare.

## Benchmarking

```bash
loci bench ingest ~/loci/raw/corpus.txt     # throughput, facts/s, RSS, swap, db growth
loci bench query --qna bench/qna.json        # the real-world eval
loci bench compare <runA> <runB>             # metric-by-metric regression diff
```

`bench query` runs your QnA set against the live system and reports, per
question and aggregate:

- **Latency breakdown** — parse, fact SQL, vector, FTS, fusion, model load,
  time-to-first-token, tokens/s.
- **Memory** — peak RSS per stage and swap delta (the true failure mode on
  8GB machines).
- **Mechanical quality** (offline, deterministic) — retrieval hit-rate, fact
  hit-rate, keyword recall, citation presence, and refusal correctness on
  trap questions marked `"answerable": false`.
- **Headline quality** — every answer graded **0–100 by Claude CLI in a
  single prompt** against your ground truth: correctness, groundedness,
  citations, with fabricated answers to unanswerable questions scored 0.

The QnA ground truth is a JSON file you build from the same raw documents you
ingest (`loci bench qna-skeleton` drafts entries from extracted facts for you
to review), so the benchmark measures *this system over this knowledge* — not
the model's pretraining. Every run snapshots its full config next to the
results, so every number is reproducible and every comparison honest.

## Knowledge packs

A pack is a self-contained SQLite knowledge base — facts, entities, vectors,
sources — that anyone can build with `loci pack export` and anyone can use
with `loci pack add`. The vision: small machines downloading expert knowledge
the way they download models today.

## Honest limitations

- Fact extraction (subject–verb–object via dependency parsing) misses
  passives, idioms, and pronoun references for now; the vector/FTS layer
  guarantees recall while facts add precision. LLM-assisted extraction and
  cheap coreference are planned (see PLAN.md, Phase 6).
- Extraction is English-only initially; the schema is language-agnostic.
- sqlite-vec search is brute-force — excellent up to a few hundred thousand
  chunks, with quantization/partitioning planned beyond that.
- A 3B model remains a 3B model: Loci makes it *grounded*, not omniscient.
  When the knowledge base doesn't contain the answer, Loci says so.

## License

TBD.
