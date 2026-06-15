# planQ — SVO quarantine + fact provenance + deep ablation bench

> **◀ [ROADMAP](ROADMAP.md)** · prev: [planE_paraphrase.md](planE_paraphrase.md) · **next ▶ (gated) ingest-scale**
>
> **Phase:** Q · **Status:** `DONE (🛑 FAIL — minted-4=62.05, minted-2=62.55, both < M=64.35; halluc=2; gate bypass flaw identified)` · **Depends on:** P1 (consumes the 63 minted facts), B (vec_facts) · **Auto-advance:** no (🛑 gate at §5)

**Owner:** worker (Claude Sonnet 4.6) · **Repo:** `/Users/brunopinto/Repos/Loci` · **Branch:** `p0-fact-fts`
Run everything with `uv run`. DB override: `LOCI_PATHS__KNOWLEDGE_DB=/Volumes/SSD1TB001/loci/knowledge/scarlet_v2.db`.
Read `[[finding-p0-fts-facts]]`, `[[finding-fact-layer-dead]]`, `[[project-loci]]` first.

---

## 1. Why this exists (read first)

Every phase after **M** (baseline 64.35) shipped as injection-off because enabling `[F#]` injection hurts
paraphrase. The diagnosis (verified in code, 2026-06-15): the cause is **not** the injection mechanism. It is
**two compounding faults in the fact layer**:

1. **`confidence` is inverted.** The `facts` table has no provenance column. The only quality signal is
   `confidence`, and it ranks quality *backwards*:

   | Source | insert site | confidence | real quality |
   |---|---|---|---|
   | spaCy SVO | `ingest.py:236` | **1.0** | garbage (wrong-shaped triples) |
   | coref SVO | `ingest.py:306` | 0.6 | weak |
   | LLM-minted (P1) | `enhance.py:214` | **0.7** | good |

   The 63 good facts carry *lower* confidence than the ~1676 garbage ones.

2. **Selection is volume-driven with no quality gate.** Facts reach `[F#]` slots via `fact_fts_search_question`
   + `vec_fact_search_question` + `fact_lookup`, fused by RRF. The only screen is "skip lowercase subjects"
   (`retrieve.py:263`). At a **~26:1** SVO:minted ratio, the top-k candidates for any query are SVO by sheer
   count — the 63 good facts cannot win slots they are outnumbered in.

**The decisive experiment:** turn injection ON but make pure-SVO facts *ineligible* (inject minted-only). If
paraphrase stops regressing and fact rises, the mechanism is sound and the branch is viable — the only remaining
work is producing more clean facts at ingest. If minted-only *still* regresses paraphrase, the bottleneck is the
3B model / injection mechanism itself and no ingest work saves this branch (escalate to 7B+).

> **Anti-overfitting (hard rule).** Do not hand-seed or special-case the specific facts the 100-Q set asks for.
> Build a general source-gating mechanism. Improvement must show on **held-out** `qna_corpus100.json`, not just
> `qna_20.json`. See `[[finding-fact-layer-dead]]`: "fix the mechanism, don't hand-seed."

---

## 2. Codebase orientation (verified 2026-06-15)

| File | Relevant content |
|---|---|
| `loci/store.py` | `facts(id,chunk_id,sentence,subject_id,predicate,object_id,object_text,qualifiers,negated,confidence)` (line 82). Schema list + migrations driven by `db_meta` versions (`fact_fts_v`, `_FACT_FTS_VERSION` line 368). `insert_fact` (267, `confidence` param). `rebuild_fact_fts` (372), `fts_search_facts`, `vec_search_facts`. **Add `source` column + backfill here.** |
| `loci/ingest.py` | SVO insert `confidence=1.0` (236); coref insert `confidence=0.6` (306). **Tag these `source='svo'` / `'coref'`.** |
| `loci/enhance.py` | LLM-minted facts `confidence=0.7` (214). **Tag `source='llm'`.** |
| `loci/retrieve.py` | Candidate pull + hydration: `load_fact_hits_by_ids` (222, `FROM facts f`), `fact_lookup` (287, `FROM facts f`), `vec_fact_search_question` (373), `fact_fts_search_question` (447), `canonical_names_for_facts` (388). Lowercase-subject skip at 263. **Apply source filter here.** |
| `loci/config.py` | `RetrievalConfig`: `max_facts_in_context=0` (master injection switch; >0 = ON), `fact_fts_top_k=10`, `fact_vec_top_k=10`, `fact_vec_mode="off"`. **Add `fact_sources` knob.** Env override: `LOCI_RETRIEVAL__<FIELD>`. |
| `loci/cli.py` | `loci bench query/report/compare` (713–1063). `loci enhance`. |
| `loci/bench.py` | Buckets by `type`: `fact|paraphrase|multi_hop|negative`; `_compute_aggregate` emits `mean_judge_{qt}` (293). `compare_runs`. |

Counts to confirm at start: `sqlite3 $DB "SELECT confidence, COUNT(*) FROM facts GROUP BY confidence"`.

---

## 3. Implementation steps

### Step 1 — Provenance: add a `source` column, backfill, stop the inversion
- Add `source TEXT` to the `facts` schema (line 82) and a migration: `ALTER TABLE facts ADD COLUMN source TEXT`
  guarded by a new `db_meta` version key (`fact_source_v`), mirroring the `fact_fts_v` pattern.
- **Backfill** existing rows from current confidence: `1.0 → 'svo'`, `0.6 → 'coref'`, `0.7 → 'llm'`.
  (Verify these three buckets are the only confidences present before backfilling.)
- Add a `source: str` param to `insert_fact` (`store.py:267`); thread it through the three call sites:
  `ingest.py:236 → 'svo'`, `ingest.py:306 → 'coref'`, `enhance.py:214 → 'llm'`.
- Do **not** change `confidence` numerics yet (avoid churning the UNIQUE key / dedup). `source` is now the truth.

### Step 2 — Quarantine knob: gate injection candidates by source
- Add to `RetrievalConfig`: `fact_sources: str = "all"`  # `"all" | "minted" | "minted+coref"`.
  Map: `minted = {'llm'}`, `minted+coref = {'llm','coref'}`, `all = everything`. Default `"all"` preserves
  current behavior (no change unless explicitly flipped).
- Enforce at **hydration**, not deep in FTS5: `load_fact_hits_by_ids` and `fact_lookup` already `SELECT FROM facts f`
  — add `AND f.source IN (...)` when `fact_sources != "all"`. For the FTS/vec paths, **over-pull** (e.g. pull
  `fact_fts_top_k * 4`) before the source filter so quarantining doesn't starve the pool, then truncate to k.
- No new behavior when `max_facts_in_context == 0` (injection still fully off → still reproduces M exactly).

### Step 3 — Selective-injection guardrails (3B noise tolerance)
The 3B model degrades on *any* noisy `[F#]`. Even minted facts must be injected sparingly:
- Keep `max_facts_in_context` low (test `2` and `4`).
- Add a relevance gate in hydration: drop a candidate fact unless its subject or object entity also appears in
  the question's parsed entities (`find_mentioned_entity_ids`) **or** it cleared the vec/FTS similarity threshold.
  This prevents topically-loose minted facts from filling slots on paraphrase questions.

### Step 4 — (🛑 GATED follow-on) Scale minted facts at ingest
**Do not start until §5 gate passes.** If minted-only wins, 63 facts is too thin to carry the bucket
(71% of headroom). Expand `enhance.py`'s taxonomy-driven extraction (role/profession/occupation/identity/
alias_of/means/located_at/resides_at/relationship_to/affiliation/leader_of/owns/cause_of) to mint hundreds of
clean, coref-resolved, question-shaped `source='llm'` facts across the whole corpus, then `rebuild_fact_fts` +
`rebuild_fact_vec`. Re-run the deep bench. This is the durable lever and gets its own plan file.

---

## 4. Deep ablation bench (for continued planning)

Current bench is 20-Q / runs=1 — too coarse to separate a ±1 signal from judge noise (P1's −0.15 was "within
noise"). "Deep" = **more arms × more runs (variance) × both datasets × per-bucket × per-question delta**.

### Ablation matrix
Hold everything else at M defaults (`fact_vec_mode=off`, `rerank_mode=off`, `hyde_query=false`).

| Arm | `max_facts_in_context` | `fact_sources` | Purpose |
|---|---|---|---|
| **M** (control) | 0 | — | reproduce 64.35 |
| **all-4** | 4 | all | reproduce P1's −0.15 (sanity that the rig matches history) |
| **minted-4** | 4 | minted | ⭐ the hypothesis |
| **minted-2** | 2 | minted | tighter slot cap |
| **mintedcoref-4** | 4 | minted+coref | does coref help or re-add noise? |

Env overrides per arm (example, minted-4):
```
LOCI_PATHS__KNOWLEDGE_DB=/Volumes/SSD1TB001/loci/knowledge/scarlet_v2.db \
LOCI_RETRIEVAL__MAX_FACTS_IN_CONTEXT=4 \
LOCI_RETRIEVAL__FACT_SOURCES=minted \
uv run loci bench query --qna bench/qna_20.json --runs 3 --judge claude --label q_minted4_20
```

### Datasets & runs
- **`bench/qna_20.json` — runs=3** for every arm (variance bands; pick best 2 arms).
- **`bench/qna_corpus100.json` — runs=3** for **M + the best 1–2 arms only** (generalization; the deciding number).
  Held-out signal here is what counts, per the anti-overfitting rule.

### Outputs to capture (per arm)
- Overall mean judge + **per-bucket**: `fact / paraphrase / multi_hop / negative` (from `mean_judge_{qt}`).
- **Hallucination count** (guardrail: must not rise above M's 1).
- **Per-question delta vs M:** `uv run loci bench compare <M_log> <arm_log>` — list every question that moved ≥ ±10
  and the direction. This per-question table is the raw material for the next plan.
- Variance: with runs=3, report mean ± spread so "within noise" is quantified, not asserted.

---

## 5. Success criteria & 🛑 fork rule

Compare **minted-only** (best arm) vs **M (64.35)** on `qna_corpus100.json`:

- ✅ **PASS** (branch viable): overall ≥ M **and** paraphrase ≥ M−1 **and** halluc ≤ 1.
  → fact bucket should rise. Proceed to **Step 4** (scale minted facts at ingest); open its plan file.
- ⚠️ **PARTIAL**: fact rises but paraphrase still dips 1–3 within variance.
  → tighten Step 3 (drop to `max_facts=2`, stricter relevance gate); re-bench before deciding.
- 🛑 **FAIL** (branch not viable on 3B): minted-only still regresses paraphrase beyond variance.
  → the bottleneck is the model/mechanism, not the data. Stop ingest work. Record the finding and escalate to a
  7B+ base model (re-run this same matrix on 7B) — that becomes the next phase, not more fact engineering.

Default config stays **off** (`max_facts_in_context=0`, `fact_sources=all`) regardless — only flip it as a shipped
default if PASS holds on the 100-Q set.

---

## 6. Tests
`uv run pytest tests/test_store.py tests/test_retrieve.py tests/test_bench_suite.py -q` must stay green.
Add: (a) migration adds `source` + backfills the three buckets correctly; (b) `fact_sources=minted` excludes
`svo`/`coref` rows from hydration; (c) `fact_sources=all` + `max_facts=0` reproduces baseline candidate sets
byte-for-byte. (Pre-existing ~6 failures in `tests/test_generate.py` are stale — ignore, but don't let them mask
new breakage.)

## 7. Record & commit
Update the ROADMAP phase table (add row **Q**). Append results (all arms, per-bucket, per-Q movers) to memory
`finding-p0-fts-facts.md`. Commit on `p0-fact-fts`:
`feat(store+retrieve+config): Phase Q fact provenance + SVO quarantine + ablation bench`
ending `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
