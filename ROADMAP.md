# Loci retrieval-quality ROADMAP (driver)

**Purpose:** the single entry point a worker follows to raise bench quality on the 100-Q
*A Study in Scarlet* set. Phases run **in order**, one at a time. Each phase has its own
plan file (linked below) with full implementation detail, tests, bench, and success criteria.

**Repo:** `/Users/brunopinto/Repos/Loci` · **Branch:** `p0-fact-fts` (stay on it unless told otherwise)
**Run everything with `uv run`.** DB override: `LOCI_PATHS__KNOWLEDGE_DB=/Volumes/SSD1TB001/loci/knowledge/scarlet_v2.db`.

> **Reference baseline (old judge):** overall **70.95** — fact 54.3 · para 54.0 · multi 47.0 · neg 96.9 · halluc 1.
> Log: `/Volumes/SSD1TB001/loci/logs/bench/1781382958_scarlet_v2_100.jsonl`.
> **M baseline (keyed judge):** overall **64.35** — fact 44.4 · para 47.0 · multi 19.0 · neg 96.8 · halluc 1.
> Log: `/Volumes/SSD1TB001/loci/logs/bench/1781487152_baseline_v2_judgekey.jsonl`.
> Phase **M** replaces the old ruler; every later compare uses the M baseline (64.35).
>
> **Final result:** M baseline (64.35) remains the best. All subsequent phases shipped as F2=off. The binding constraint
> is that the 3B model degrades when context includes any noisy [F#] injection; 63 new P1 taxonomy facts are in
> scarlet_v2.db (1676→1739) and help specific questions, but enabling injection always hurts paraphrase more.
> Infrastructure in place: vec_facts, taxonomy enhance, rerank config, HyDE flag — all ready for a stronger base model.

---

## Phase order & status

| # | Phase | File | Status | Depends on | Auto-advance? |
|---|-------|------|--------|-----------|---------------|
| 0 | P0 fact-FTS (done; regressed → neutralized) | [planP0.md](planP0.md) | `DONE` | — | — |
| 1 | **M** — judge answer key (measurement) | [planM_judgekey.md](planM_judgekey.md) | `DONE (new baseline=64.35, fact=44.4, para=47.0, multi=19.0, neg=96.8)` | — | yes |
| 2 | **B** — vec-over-facts (P0.1) | [planB_vecfacts.md](planB_vecfacts.md) | `DONE (F1=off; surface=59.90, expand=61.95, both < M=64.35; facts MISSING not unretrievable)` | M | yes |
| 3 | **P1** — semantic fact minting | [planP1_factmint.md](planP1_factmint.md) | `DONE (63 facts minted; fact +1.2 on 100Q; qna_20=70.75; max_facts=0 prevents paraphrase noise)` | B (uses B's missing-fact backlog) | yes |
| 4 | **C** — rerank, don't pre-truncate (P2) | [planC_rerank.md](planC_rerank.md) | `DONE (F2=off; pool=62.80 -1.55 vs M; wider pool floods RRF with mediocre overlap → para -22)` | — (independent) | yes |
| 5 | **E** — paraphrase / multi-hop (P3) | [planE_paraphrase.md](planE_paraphrase.md) | `DONE (F2=off; HyDE=62.65 -1.70, halluc 1→3 violates guardrail; hyde_query=false)` | — | yes |

> **Keep this table current.** When a phase finishes, set its `Status` to `DONE (overall=NN.N, fact=NN.N)`
> and move to the next `TODO` row.

---

## Execution protocol (per phase)

1. **Open** the phase's plan file. Read it fully (and the memory files it links: `[[finding-p0-fts-facts]]`,
   `[[finding-fact-layer-dead]]`, `[[project-loci]]` under
   `/Users/brunopinto/.claude/projects/-Users-brunopinto-Repos-Loci/memory/`).
2. **Implement** exactly that phase's scope. Do **not** pull work forward from a later phase.
3. **Test:** `uv run pytest tests/test_store.py tests/test_retrieve.py tests/test_bench_suite.py -q`
   must stay green. (The ~6 failures in `tests/test_generate.py` are **pre-existing/stale** — ignore them,
   but don't let them hide *new* breakage; run the specific files above.)
4. **Bench** per the phase's §Benchmarking. Compare to the **current** baseline log
   (`uv run loci bench compare <baseline> <new>`).
5. **Evaluate** against the phase's Success criteria. Apply any fork rule below.
6. **Record:** update this table's Status row; append results to memory `finding-p0-fts-facts.md`.
7. **Commit** on `p0-fact-fts` with a clear message ending:
   `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
8. **Advance** to the next `TODO` phase automatically — unless a 🛑 gate (below) fires.

## 🛑 Stop-for-user gates (the ONLY reasons to pause)

Pause and ask the user **only** when one of these is hit; otherwise keep going:

- **G-AMBIGUOUS** — a phase's bench lands inside the noise band (within ±3 overall of baseline) **and** a
  `--runs 2` re-run doesn't resolve it. Report the numbers, ask whether to keep/revert.
- **G-PROMOTE** — before overwriting `main.db` (e.g. `cp scarlet_v2.db main.db`). Always confirm first.
- **G-PUSH** — before any `git push`, PR, or other outward/irreversible action. Always confirm.
- **G-DATA-MUTATION** — before a step that **rewrites facts in `scarlet_v2.db`** (P1 re-extraction).
  Make a timestamped backup copy first; then proceed (no pause needed *if* the backup succeeded).
- **G-BLOCKED** — an unresolvable failure (test you can't fix after a real attempt, missing model/judge,
  corrupt DB). Report the exact error and what you tried.

## Decision forks (worker decides automatically — no user pause)

- **F1 (after B):** if the better of B's `surface`/`expand` variants beats the M-baseline by **≥+5 overall
  with fact↑**, set that `fact_vec_mode` as the default and carry on. If neither moves the needle, set
  `fact_vec_mode="off"` and note "facts missing, not unretrievable" — P1 (next phase) is then doubly justified.
  Either way you proceed to **P1**; F1 only changes a config default and the P1 backlog emphasis.
- **F2 (after each phase):** if a change *regresses* overall, revert that change (keep the phase's tests/infra
  if harmless) before advancing. A phase that can't beat baseline ships as `mode/flag = off`, not as a regression.

---

## At-a-glance: why each phase exists

- **M** — the judge scores answerable Qs from its own Holmes knowledge (no answer key) → ±95/question noise →
  we can't see a real 5–8pt win. M adds a per-question reference answer. **Foundation for trusting every later number.**
- **B** — P0 proved FTS-over-facts can't bridge `understand`↔`profession`. Semantic (vec) retrieval can. Also the
  **diagnostic gate**: if even vec can't move facts, the facts are *missing*, not unretrievable → P1.
- **P1** — the real lever (fact bucket = 71% of headroom). Mint facts whose predicate/object vocabulary matches
  questions, with canonical-NAME objects, via coref-resolved LLM extraction. Validate on held-out Qs (no overfitting).
- **C** — rescue the "right chunk not in top-5" abstentions by widening the pool and reranking before the
  1800-tok cut. Retrieval-only; stacks on anything.
- **E** — paraphrase (0.10) + multi_hop (0.05) buckets = 15% of score. Query expansion / HyDE-lite +
  sub-question decomposition. Lowest weighted headroom; last.
