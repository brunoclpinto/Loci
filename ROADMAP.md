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
>
> **Phase Q (2026-06-15):** Added `source` provenance column (svo/coref/llm), `fact_sources` quarantine knob, relevance
> gate, and 4× FTS over-pull. Deep ablation bench (5 arms × qna_20 runs=3, 2 arms × qna_scarlet runs=3):
> minted-only injection showed qna_20 improvement (73.75, +10.0 vs M-20) but regressed on 100-Q (minted-4=62.05,
> minted-2=62.55, both < M=64.35, halluc=2). Gate bypass flaw fixed (`elif source_filter and not main_entity_ids:
> all_fact_hits = []`); re-bench: minted-4+fix=63.20 (−1.15, best arm), minted-2+fix=61.15. Halluc=2 persists —
> bound by 3B noise on negative questions + entity-matched wrong-fact injection. Definitive 🛑 FAIL on 3B.
> Infrastructure shipped off (defaults unchanged).
>
> **Model exploration (2026-06-15):** 4B models (Qwen3.5-4B, Qwen3-4B, Gemma3-4B) tested but make the system
> unusable under concurrent use. Qwen3.5-4B=64.0 (≈M, fact+6.8 but para−17); Qwen3-4B=65.55 but halluc=6
> (thinking mode active, needs /no_think); Gemma3-4B=58.15, halluc=9 (draws on training knowledge for negative Qs).
> Reverting to 3B footprint: benchmarking Llama-3.2-3B, Qwen3.5-2B, Qwen3-1.7B, Gemma3-1B candidates (Phase N).
>
> **Phase P2 (ran ahead of N, 2026-06-15):** Two new extraction passes added to enhance.py: entity-centric
> (cross-chunk coref, 110 entities → 174 new facts) and implied/archaic (138 chunks → 111 new facts). Total llm
> facts: 380→665; total facts: 1739→2024. Key fixes in data: RACHE|mean|revenge confirmed, Hope|work_as|jarvey
> and jarvey|mean|cab driver minted. Bench: minted-4 runs=1=65.25 (+0.90, noise); runs=3=61.70 (−2.65), halluc=2
> — same 3B bound as Phase Q. Data improvements are real but 3B model can't handle injection. Injection stays OFF.
>
> **Phase N (2026-06-16):** All 3B-footprint and Phi-4-mini candidates tested (runs=3 each, qna_scarlet 100-Q):
> Llama-3.2-3B=63.55 (halluc=3, para=35.5 −11.5, RSS=2594 MB);
> Qwen3.5-2B=52.55 (halluc=5, neg=86.5 — too small to follow citation rules);
> Qwen3-1.7B Q8_0=46.30 (thinking mode fires despite /no_think, over-refuses fact+para, fact=11.6);
> Gemma3-1B=41.0 (halluc=18, neg=64.9 — 1B can't suppress training knowledge);
> Phi-4-mini-instruct=53.15 with simple prompt (multi=0, RSS=3607 MB; verbose prompt causes 93% refusals;
> simple prompt collapses multi-hop and bloats RSS; also tested Qwen2.5-3B+simple=52.70 → verbose prompt
> is load-bearing for Qwen2.5-3B). No candidate beats Qwen2.5-3B. **Incumbent stays (verbose prompt kept).**
> Qwen2.5-7B (4.4 GB, on SSD) remains the only untested upgrade path.
>
> **Phase F (2026-06-16) — Fact Graph Closure + LLM-only FTS:** Built graph closure pass (run_closure_pass),
> prune pass (run_prune_pass), fts_facts_llm table, and source-filtered FTS routing. Bench sequence:
> (1) closure+minted-4=62.45 (−1.90 vs M, halluc=2): closure amplified P2 noise (Holmes/Lestrade as cab drivers).
> (2) prune(22 bad llm + 10 closure) + re-closure + minted-4=60.15 (−4.20, halluc=2): worse — removing noisy
> facts changed what gets injected for para questions, para 47→37.5 (−9.5). Both attempts 🛑 FAIL. 3B bound is
> immutable regardless of data quality: any imperfect fact injection causes regression. DB restored to pre-F state
> (2024 facts, 665 llm, 0 closure, closure_v cleared). Infrastructure shipped max_facts=0 (unchanged):
> run_closure_pass, run_prune_pass, fts_facts_llm ready for a stronger model. 218 tests passing.

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
| 6 | **Q** — SVO quarantine + fact provenance | [planQ_quarantine.md](planQ_quarantine.md) | `DONE (🛑 FAIL; best arm=minted-4+fix=63.20, −1.15 vs M; halluc=2 persists; 3B bound; escalate to 4-5B)` | P1 | no (🛑 gate) |
| 7 | **N** — base model selection | [planN_modelbench.md](planN_modelbench.md) | `DONE (🛑 no winner; Qwen2.5-3B stays incumbent; all candidates failed: Llama-3.2-3B=63.55 halluc=3, Qwen3.5-2B=52.55 halluc=5, Qwen3-1.7B=46.30 over-refuses, Gemma3-1B=41.0 halluc=18, Phi-4-mini=53.15 multi=0 RSS=3.6GB)` | Q | yes |
| 8 | **P2** — multi-pass ingest | [planP2_multipass.md](planP2_multipass.md) | `DONE (🛑 FAIL on injection; ran ahead of N; entity+implied passes minted 285 new llm facts (380→665, 1739→2024 total); RACHE+Hope cab driver now correct; but minted-4 runs=3=61.70 −2.65 vs M, halluc=2 — same 3B bound as Q; injection off; data in DB ready for N winner)` | N (needs winning model first) | yes |
| 9 | **F** — fact graph closure + LLM-only FTS + prune | [jaunty-honking-hare.md](../.claude/plans/jaunty-honking-hare.md) | `DONE (🛑 FAIL; closure=62.45 then prune+closure=60.15, both < M=64.35; 3B bound immutable; DB restored; infra ready for stronger model)` | P2 | no (🛑 gate) |

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
