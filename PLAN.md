# PLAN.md — "Loci": Low-RAM Local RAG with Structured Knowledge on SSD

Target machine: Mac mini M1, 8GB unified memory, fast SSD.
Goal: a local assistant whose knowledge lives on disk (SQLite), grows incrementally
by linking facts to existing entities, and answers questions grounded in stored
knowledge — while keeping peak RAM under ~3.5GB so the system stays responsive.

This document is the build plan for Claude Code. Implement it phase by phase.
Each phase has acceptance criteria; do not move on until they pass.

---

## 0. Core architecture (read first)

Two complementary knowledge layers over the SAME source text:

1. **Structured facts layer (the "dots")** — sentences are decomposed into
   facts: (subject_entity, predicate, object, qualifiers, location, source).
   Example from "Sherlock Holmes took his bottle from the corner of the
   mantel-piece":
   - subject: entity:sherlock_holmes
   - predicate: take (lemmatized)
   - object: bottle
   - qualifiers: {from: "corner of the mantel-piece"}
   - provenance: chunk_id → sentence → file/book/chapter
   A question like "what did Sherlock Holmes take?" parses to
   (subject=sherlock_holmes, predicate=take) → indexed SQL lookup. Cheap, fast,
   precise.

2. **Vector + full-text layer (the safety net)** — every chunk also gets an
   embedding (sqlite-vec) and an FTS5 entry. This catches everything the
   triple extractor misses (passives, idioms, multi-sentence meaning).

**Golden rule: facts are an INDEX into source text, never a replacement.**
Every fact stores its chunk_id. The LLM is always shown the original sentences
(plus the compact facts), so lossy extraction can't corrupt answers.

**Growth model:** ingestion is append-only and idempotent. New documents are
deduplicated by content hash. New facts attach to existing entities via an
alias-resolution step ("Holmes", "Sherlock", "Mr. Sherlock Holmes" → one
entity row). Nothing is ever rebuilt from scratch; the graph densifies.

**Knowledge packs:** a pack is just a standalone SQLite file with the same
schema. Users download `medicine.locipack.db`, the app `ATTACH`es it, and
queries fan out across attached packs. This is the "downloadable expert
knowledge" feature and it falls out of the schema for free.

---

## 1. RAM budget & model choices (hard constraints)

| Component                       | Choice                                      | Approx RAM |
|---------------------------------|---------------------------------------------|-----------|
| Chat LLM                        | Qwen2.5-3B-Instruct GGUF Q4_K_M via llama.cpp (Metal, mmap on) | ~2.1 GB |
| Fallback chat LLM (tighter)     | Llama-3.2-1B-Instruct Q4_K_M                | ~0.9 GB |
| Embedding model                 | bge-small-en-v1.5 GGUF (384-dim) via llama.cpp embedding mode | ~140 MB |
| Sentence parsing (ingest only)  | spaCy `en_core_web_sm`                      | ~150 MB (unloaded after ingest) |
| Vector store                    | sqlite-vec (disk-based, no resident index)  | ~0 idle |
| Full-text                       | SQLite FTS5                                 | ~0 idle |

Rules:
- **No PyTorch / sentence-transformers.** Everything inference goes through
  `llama-cpp-python` with `n_gpu_layers=-1` (Metal) and default mmap, so model
  weights page from SSD instead of being copied into RAM.
- Chat model and embedding model must support **load/unload on demand**:
  ingest uses embedder (+ spaCy), query uses embedder briefly then chat model.
  Never hold spaCy and the chat LLM simultaneously.
- Context window capped at 4096 tokens for the chat model (KV cache RAM).
- Provide a `--low-mem` flag that swaps to the 1B model and 2048 ctx.

---

## 2. Tech stack

- Python 3.11+, managed with `uv`. Single package `loci/`.
- `llama-cpp-python` (Metal build) — chat + embeddings.
- `sqlite-vec` extension — vector search inside SQLite.
- SQLite FTS5 — keyword search.
- `spacy` + `en_core_web_sm` — dependency parsing for fact extraction (ingest-time only).
- `typer` + `rich` — CLI.
- `pytest` — tests for every phase.
- Models stored in `~/.loci/models/`, downloaded with a `loci models pull` command (hugging face URLs hardcoded with sha256 checks).

---

## 3. Database schema (Phase 1 deliverable)

One file: `~/.loci/knowledge/main.db`. WAL mode, `PRAGMA mmap_size` set,
foreign keys on.

```sql
-- Provenance ------------------------------------------------------------
CREATE TABLE sources (
  id INTEGER PRIMARY KEY,
  path TEXT, title TEXT, author TEXT,
  meta JSON,                       -- book, chapter, story title, etc.
  sha256 TEXT UNIQUE NOT NULL,     -- dedup: re-ingesting same file is a no-op
  ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chunks (
  id INTEGER PRIMARY KEY,
  source_id INTEGER REFERENCES sources(id),
  ordinal INTEGER,                 -- position in source
  text TEXT NOT NULL,              -- ~3-6 sentences, ~512 token target
  sha256 TEXT UNIQUE NOT NULL      -- chunk-level dedup across sources
);

-- Entities & aliases (the "connect the dots" core) ----------------------
CREATE TABLE entities (
  id INTEGER PRIMARY KEY,
  canonical_name TEXT NOT NULL,    -- "Sherlock Holmes"
  kind TEXT DEFAULT 'unknown',     -- person|place|object|concept|org|unknown
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE aliases (
  entity_id INTEGER REFERENCES entities(id),
  alias TEXT NOT NULL,             -- lowercased: "holmes", "sherlock", "the detective"
  PRIMARY KEY (entity_id, alias)
);
CREATE INDEX idx_aliases_alias ON aliases(alias);

-- Facts ------------------------------------------------------------------
CREATE TABLE facts (
  id INTEGER PRIMARY KEY,
  chunk_id INTEGER REFERENCES chunks(id),   -- provenance, ALWAYS set
  sentence TEXT NOT NULL,                   -- the exact source sentence
  subject_id INTEGER REFERENCES entities(id),
  predicate TEXT NOT NULL,                  -- lemmatized verb: "take"
  object_id INTEGER REFERENCES entities(id),-- nullable
  object_text TEXT,                         -- literal when not an entity: "bottle"
  qualifiers JSON,                          -- {"from":"corner of the mantel-piece","manner":null,...}
  negated INTEGER DEFAULT 0,                -- "did not take" MUST set this
  confidence REAL DEFAULT 1.0,
  UNIQUE (chunk_id, subject_id, predicate, object_text, qualifiers)
);
CREATE INDEX idx_facts_subj_pred ON facts(subject_id, predicate);
CREATE INDEX idx_facts_pred_obj  ON facts(predicate, object_id);

-- Vector + FTS layers ----------------------------------------------------
CREATE VIRTUAL TABLE vec_chunks USING vec0(
  chunk_id INTEGER PRIMARY KEY,
  embedding FLOAT[384]
);
CREATE VIRTUAL TABLE vec_entities USING vec0(   -- for fuzzy entity resolution
  entity_id INTEGER PRIMARY KEY,
  embedding FLOAT[384]
);
CREATE VIRTUAL TABLE fts_chunks USING fts5(text, content='chunks', content_rowid='id');

-- Predicate synonyms (small curated + learned table) ----------------------
CREATE TABLE predicate_synonyms (
  predicate TEXT, synonym TEXT, PRIMARY KEY (predicate, synonym)
);  -- ("take","grab"), ("take","pick"), ("say","tell")...
```

**Acceptance (Phase 1):** module `loci/store.py` creates/migrates this
schema, loads sqlite-vec, exposes typed CRUD helpers, passes pytest suite
including: dedup on re-insert, FK enforcement, vec + FTS round-trip,
`ATTACH`ing a second db and querying across both. Additionally: `config.py`
loads and validates `config.toml` (§12, including `loci config init/show`),
and `bench.py` ships the `measure()` instrumentation context manager (§13)
that every subsequent phase wires into its CLI commands.

---

## 4. Phase 2 — Ingestion & fact extraction pipeline

CLI: `loci ingest path/to/file.txt --meta book="The Sign of the Four"`
Accept .txt and .md first; design the reader as a plugin interface (PDF/EPUB later).

Pipeline (streaming, constant memory — never load whole corpus into RAM):
1. **Hash & skip** if source sha256 already present (idempotent).
2. **Chunk**: split to ~512-token chunks on sentence boundaries, 1-sentence
   overlap. Store chunk + meta.
3. **Embed** chunks in batches of 16 with the GGUF embedder → `vec_chunks`.
4. **Extract facts** per sentence with spaCy dependency parse:
   - subject = nsubj/nsubjpass subtree head; resolve to entity (see §5)
   - predicate = lemma of root verb; conjuncts produce multiple facts
     ("took his bottle ... and his syringe ..." → 2 facts)
   - object = dobj/attr/pobj subtree text
   - qualifiers = prep phrases keyed by preposition ({"from": "...", "in": "..."})
   - negation: `neg` dependency → `negated=1`
   - skip sentences with no clear nsubj+verb (the vector layer covers them)
5. **Entity resolution** (§5) before inserting facts.
6. Per-file summary printed: chunks, facts, new entities, linked entities.

Defer (Phase 6, do NOT attempt now): pronoun coreference ("He took...").
For now, sentences whose subject is a pronoun produce no fact — vector layer
still captures them. Note this limitation in README.

**Acceptance:** ingest a Project Gutenberg Sherlock Holmes text in < 10 min on
M1 with peak RSS < 1.5GB (no chat model loaded); the example sentence from the
spec yields a fact equivalent to (sherlock_holmes, take, bottle,
{from: corner of the mantel-piece}); re-ingesting is a no-op.

---

## 5. Entity resolution — how knowledge "connects the dots"

`loci/resolve.py`, used at ingest and query time:

1. Normalize mention (lowercase, strip titles/punct).
2. **Exact alias hit** → existing entity. Done.
3. **Fuzzy**: token-subset match ("holmes" ⊂ "sherlock holmes") with a
   guard: only auto-merge if unambiguous (exactly one candidate).
4. **Embedding match**: embed mention, search `vec_entities`, cosine ≥ 0.92
   → link and ADD the mention as a new alias (the system literally learns
   connections as it reads).
5. Otherwise create a new entity + alias + entity embedding.
6. Ambiguous cases (2+ candidates) go to a `pending_links` table; CLI command
   `loci entities review` lets the user merge/split interactively. `loci
   entities merge A B` rewrites facts/aliases — this is how wrong dots get
   re-connected without re-ingesting.

**Acceptance:** ingesting two chapters mentioning "Holmes", "Sherlock Holmes",
"Mr. Sherlock Holmes" yields ONE entity with ≥3 aliases and facts from both
chapters attached to it.

---

## 6. Phase 3 — Hybrid retrieval

CLI: `loci ask "what did sherlock holmes take?"` (retrieval part).

1. **Question parse** (spaCy small, or pure-Python fallback): extract
   candidate entity mentions + main verb lemma + wh-type (what/where/who/when).
2. **Structured lookup**: resolve mentions to entity ids; expand predicate via
   `predicate_synonyms`; run indexed SELECT on `facts`
   (e.g. subject_id=?, predicate IN (...)). Score: exact predicate 1.0,
   synonym 0.8. Respect `negated`.
3. **Vector search**: embed question, top-12 from `vec_chunks`.
4. **FTS search**: top-12 BM25 from `fts_chunks`.
5. **Fuse** vector+FTS with Reciprocal Rank Fusion; facts ranked separately
   and always placed first in context if any matched.
6. Build context bundle ≤ ~1800 tokens:
   - matched facts rendered compactly: `[F3] Sherlock Holmes — take — bottle (from: corner of the mantel-piece) — "…exact sentence…" (The Sign of the Four, Ch. I)`
   - then top fused chunks with source tags `[C7]`.
7. `--explain` flag prints the parse, SQL hits, and fused ranking — essential
   for debugging quality.

**Acceptance:** for "what did sherlock holmes take?" the fact lookup alone
(no vectors) returns the bottle/syringe facts in <50 ms; for a paraphrase the
extractor can't parse ("what items did the detective grab?"), synonyms +
aliases + vector fallback still surface the right chunk.

---

## 7. Phase 4 — Grounded generation

1. Load chat model only now (embedder may be freed first under `--low-mem`).
2. System prompt: answer ONLY from provided facts/chunks; cite tags like
   [F3]/[C7]; if the knowledge base has nothing relevant, say so explicitly.
3. Stream tokens to terminal. After answer, print a "Sources" footer mapping
   tags → book/chapter.
4. `loci chat` = REPL with short rolling history (last 3 turns, summarized if
   over budget) so KV cache stays small.

**Acceptance:** answers cite tags; a question about content not in the DB gets
an explicit "not in my knowledge base" instead of a made-up answer; peak RSS
during chat < 3.5GB (verify with `psutil` and log it).

---

## 8. Phase 5 — Growth, packs, maintenance

- `loci ingest` already incremental (hashes). Add `loci stats` (entities,
  facts, chunks, db size, most-connected entities).
- **Packs**: `loci pack export --name sherlock --out sherlock.locipack.db`
  (copies schema subset), `loci pack add file.locipack.db` (registers; queries
  ATTACH and UNION across packs), `loci pack list/remove`.
- `loci entities review` (from §5) for pending merges.
- `VACUUM`/`ANALYZE` maintenance command.

## 9. Phase 6 — Quality upgrades (only after 1-5 work)

- LLM-assisted extraction pass: batch re-read chunks with the local 3B model
  overnight to extract facts spaCy missed (passives, copulas "X is Y",
  possession "X's Y"), writing with confidence=0.7. Idempotent via a
  `chunks.extracted_v` version column.
- Cheap coreference: within a chunk, resolve he/she/it to the most recent
  matching-kind entity; mark confidence=0.6.
- Predicate synonym learning from embedding clusters of predicates.
- Optional: PDF/EPUB readers.

## 10. Repo layout & conventions

```
loci/
  cli.py          # typer app: ingest, ask, chat, entities, pack, models, stats
  store.py        # schema, migrations, CRUD, attach logic
  ingest.py       # chunking, streaming pipeline
  extract.py      # spaCy SVO + qualifiers + negation
  resolve.py      # entity resolution
  retrieve.py     # parse, SQL, vec, FTS, RRF fusion, context builder
  generate.py     # llama.cpp chat wrapper, prompts, citation footer
  models.py       # model registry, download with sha256, load/unload
  config.py       # config.toml loading, validation, precedence (§12)
  bench.py        # measure() instrumentation + loci bench suite (§13)
bench/
  qna.json        # ground-truth QnA set built from the raw knowledge (§13)
config.example.toml  # documented default config, copied by `loci config init`
tests/            # per-module pytest; fixtures use a tiny 20-sentence corpus
README.md
```

Conventions for Claude Code:
- Every phase ends with passing tests + a short demo command in README.
- Every CLI command runs inside `bench.measure()`: wall time + peak RSS are
  printed at exit AND appended as JSONL to `runtime_logs_dir` (§12) — RAM is a
  feature here, so it is always recorded, not just during benchmarks.
- No global model singletons; explicit load()/free() context managers.
- Keep dependencies minimal; justify any addition in README.

## 11. Known limitations to state honestly in README

- SVO extraction misses passives, idioms, and pronouns until Phase 6 — the
  vector/FTS layer is the guarantee of recall, facts are the precision boost.
- English-only extraction initially (spaCy model choice); schema is language-agnostic.
- sqlite-vec is brute-force; fine up to a few hundred thousand chunks, plan
  for partitioned/quantized vectors if packs grow beyond that.

---

## 12. Configuration — everything editable in one file (Phase 1 deliverable)

Single human-editable `config.toml`. Lookup order for the file itself:
`--config PATH` flag → `$LOCI_CONFIG`-style env var `LOCI_CONFIG` →
`./config.toml` (project-local) → `~/.loci/config.toml`.

Value precedence (highest wins): **CLI flag → env var (`LOCI_SECTION__KEY`) →
config.toml → built-in default**. `config.py` validates types/paths on load
and fails fast with a clear message; `loci config init` writes a fully
commented `config.example.toml` copy; `loci config show` prints the effective
merged config and where each value came from.

```toml
[paths]
models_dir        = "~/.loci/models"          # GGUF files live here
raw_knowledge_dir = "~/loci/raw"              # source docs before ingest
knowledge_db      = "~/.loci/knowledge/main.db"  # ingested knowledge (SQLite)
packs_dir         = "~/.loci/packs"           # downloadable .locipack.db files
context_dir       = "~/.loci/context"         # chat sessions / rolling history
bench_logs_dir    = "~/.loci/logs/bench"      # benchmark run JSONL + reports
runtime_logs_dir  = "~/.loci/logs/runtime"    # debug/runtime logs

[models]
chat          = "qwen2.5-3b-instruct-q4_k_m.gguf"
chat_low_mem  = "llama-3.2-1b-instruct-q4_k_m.gguf"
embedder      = "bge-small-en-v1.5-q8_0.gguf"
n_ctx         = 4096        # 2048 in --low-mem profile
n_gpu_layers  = -1          # all layers on Metal
temperature   = 0.2
max_tokens    = 512
use_mmap      = true

[ingest]
chunk_tokens             = 512
chunk_overlap_sentences  = 1
embed_batch              = 16
spacy_model              = "en_core_web_sm"

[retrieval]
vec_top_k             = 12
fts_top_k             = 12
context_token_budget  = 1800
entity_sim_threshold  = 0.92
rrf_k                 = 60

[logging]
level         = "INFO"      # DEBUG dumps retrieval explanations to runtime log
log_peak_rss  = true
jsonl         = true        # structured logs, one event per line

[bench]
qna_file            = "bench/qna.json"
runs_per_question   = 3       # median reported, jitter visible
rss_sample_hz       = 10      # background RSS sampler frequency
judge               = "claude"      # "claude" (CLI judge) | "none" (offline, mechanical metrics only)
judge_cmd           = "claude -p"   # headless Claude CLI invocation, overridable
judge_single_prompt = true          # ALL Q/A pairs scored in ONE prompt
judge_max_chars     = 150000        # only above this is the batch split (fewest chunks possible)
```

Every module reads ONLY from the loaded config object — no hardcoded paths or
magic numbers anywhere else in the codebase. This makes the benchmark loop in
§13 meaningful: change a value, re-run `loci bench`, compare.

**Acceptance:** `loci config init/show` work; overriding `vec_top_k` via flag,
env var, and file each take effect with documented precedence; a bad path in
the file produces a clear error, not a stack trace.

---

## 13. Benchmark & evaluation suite — `loci bench`

Built incrementally: `measure()` instrumentation in Phase 1, `bench ingest`
right after Phase 2, full `bench query` after Phase 4. Benchmarks exist to
drive decisions (model swap? chunk size? top-k?), so every run is logged and
diffable.

### 13.1 Instrumentation core (`bench.measure()`)
Context manager wrapping any operation. Captures: wall time, CPU time,
**peak RSS** (background psutil sampler at `rss_sample_hz`), **swap delta**
(critical on 8GB — swapping is the failure mode we're designing against),
model load time, and arbitrary counters (chunks, facts, tokens). Emits one
JSON event to the runtime log; `loci bench` aggregates the same events.

### 13.2 `loci bench ingest <path>`
Ingests into a throwaway db (or `--db` override) and reports:
wall time, chunks/s, facts/s, sentences skipped, peak RSS, swap delta,
db size growth (bytes per 1k tokens of source), embed throughput.

### 13.3 `loci bench query --qna bench/qna.json`
The real-world eval. Each question runs `runs_per_question` times (cold +
warm); per question and aggregate it reports:

- **Latency breakdown**: question parse, fact SQL, vector search, FTS, fusion,
  context build, model load (cold), time-to-first-token, full generation,
  tokens/s.
- **Memory**: peak RSS per stage, swap delta.
- **Quality — mechanical layer** (deterministic, free, fully offline; scored
  against the QnA ground truth):
  - retrieval hit-rate: expected source/chunk present in fused top-k
  - fact hit-rate: expected (subject, predicate) matched by the SQL layer alone
  - answer keyword recall: fraction of `expected_keywords` in the answer
  - citation present: answer cites at least one [F]/[C] tag
  - **hallucination check**: questions marked `"answerable": false` must get a
    refusal; any substantive answer scores 0 and is flagged loudly.
- **Quality — headline score**: 0–100 per question, graded by Claude CLI in a
  single prompt (§13.5).

### 13.4 QnA ground-truth format (`bench/qna.json`)
Hand-built (or LLM-assisted then human-reviewed) from the SAME raw knowledge
that gets ingested, so the benchmark measures the system, not the model's
pretraining:

```json
[
  {
    "id": "q001",
    "type": "fact",
    "question": "What did Sherlock Holmes take from the mantel-piece?",
    "expected_keywords": ["bottle"],
    "expected_facts": [{"subject": "sherlock holmes", "predicate": "take"}],
    "expected_sources": ["pg2097_the_sign_of_four.txt"],
    "answerable": true
  },
  {
    "id": "q014",
    "type": "negative",
    "question": "What did Sherlock Holmes have for breakfast on the moon?",
    "answerable": false
  }
]
```

Include a mix: `fact` (structured layer should win), `paraphrase` (vector
layer should win), `multi_hop` (two facts needed), `negative` (must refuse).
`loci bench qna-skeleton <source>` can draft candidate QnA entries from
ingested facts for the user to review/edit — keeps benchmark building cheap.

### 13.5 Headline scoring 0–100 via Claude CLI — single prompt

Run AFTER all local answers are generated (judging never interferes with the
latency/RAM measurements). Flow in `bench.py`:

1. Assemble one judging payload: the rubric + a JSON array of every item
   `{id, type, question, expected_keywords, expected_facts, expected_sources,
   answerable, system_answer, citations_used}`.
2. Invoke Claude CLI headless exactly once:
   `claude -p "<rubric + payload>"` (command string from `bench.judge_cmd`).
   This is the ONLY networked step in the whole project; `--judge none` keeps
   the benchmark fully offline using the mechanical layer alone, and a missing
   `claude` binary degrades gracefully to the same with a warning.
3. The prompt instructs Claude to reply with STRICT JSON only — no prose, no
   code fences: `[{"id":"q001","score":87,"reason":"<one sentence>"}]`.
   Parser strips stray fences, validates the id set matches 1:1 with the
   submitted items, and retries once on malformed output before failing the
   judging step (mechanical metrics still reported).
4. Only if the payload exceeds `judge_max_chars` is it split — into the fewest
   possible chunks, each still a single prompt — and results merged.

Rubric embedded in the prompt (Claude Code: write it verbatim into
`bench.py`, tune wording during Phase 5):
- 100 = factually correct per ground truth, grounded in cited sources, no
  fabricated details. Deduct proportionally for missing expected facts,
  unsupported claims, or wrong/missing citations.
- `answerable: false` items: a clear refusal/"not in my knowledge base" = 100;
  ANY fabricated substantive answer = 0.
- `multi_hop` items: full credit only if both facts are connected; one fact
  alone caps at 50.
- Score the SYSTEM (retrieval + grounding), not writing style.

Outputs: per-question `judge_score` + `judge_reason` merged into the run
JSONL; aggregate mean/median overall and per question type; the raw judge
prompt + raw response saved next to the run log for audit. The judge identity
(Claude CLI version, model if reported) is recorded in the run's config
snapshot so cross-run comparisons stay honest — never compare judge scores
across different judge models without flagging it.

### 13.6 Run logs, reports, and comparison
- Each run writes `bench_logs_dir/<timestamp>_<label>.jsonl` plus a frozen
  snapshot of the effective config (so results are always reproducible).
- `loci bench report [run]` renders a markdown/rich-table summary.
- `loci bench compare <runA> <runB>` diffs two runs metric-by-metric with
  regression highlighting — the tool for "did switching to the 1B model or
  chunk_tokens=384 actually help?".

**Acceptance:** a full `bench query` over a 20-question QnA on the Sherlock
corpus completes unattended; all mechanical metrics are produced offline; with
`--judge claude`, exactly ONE Claude CLI call returns valid 0–100 scores for
all 20 questions (negative questions scored under the refusal rule) and the
report shows judge mean/median per question type; `bench compare` of two runs
with different `vec_top_k` shows the changed config key alongside the metric
deltas, including judge-score deltas.
