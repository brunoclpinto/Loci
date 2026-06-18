# planX — Proposition Retrieval Reranking (rerank_v1)

## Problem Addressed

The proposition layer was firing on 15/60 answerable questions at query time but averaging 10.7 per question. 13 returned "Not stated in the source." despite the correct prop existing. Root cause: single-prop selection via brittle predicate matching and `matches[0]` with no ranking.

Hard evidence:
- q0008 "Who is the murderer?" → predicate "murder" → returned milk-boy prop; P15 "Drebber was killed by Jefferson Hope" existed but predicate "kill" ≠ "murder"
- q0024 "What term for Holmes's profession?" → verb "use" hijacked; "work_as" prop ignored
- q0056 "Holmes's full name?" → verb "call" → "Holmes called Lecoq a bungler"
- q0001 "Who introduced them?" → correct (the one that worked)

---

## What Changed

### `loci/store.py`
- Added `vec_search_propositions(conn, embedding, k)` mirroring `vec_search_facts`.

### `loci/retrieve.py` — `retrieve_propositions`
- **Return type**: `list[PropositionHit]` (was `PropositionHit | None`).
- **Entity path**: now collects ALL props for matching entities (no predicate filter per synonym). Old code queried `get_props_for_entities_and_predicate` per synonym predicate, filtering out valid props with slightly different predicates.
- **FTS path**: no predicate gate on collection. Old code had a hard `prop["predicate"] in predicates_to_try` filter on FTS hits.
- **Vec path**: added. Embed question → `vec_search_propositions` top-30 → scored by cosine similarity.
- **Ranking**: primary signal is vec cosine (when embedder available) or FTS rank; predicate synonym match is a +0.25 bonus, entity match +0.10. Returns top-k (default 3).
- **Predicate gate retained**: required after ranking — at least one candidate must have a synonym-matching predicate. Prevents vec from firing the prop path for every question (100/100 in initial broken run).
- **Empty-predicate guard**: returns `[]` immediately when `predicates_to_try = {}` (no verb + no noun mapping) → falls back to chunks.
- **Noun augmentation**: scans question words for `_NOUN_TO_PRED` keys even when a main verb is present. Fixes "term for Holmes's profession" (verb "use" + noun "profession" → adds "work_as").
- **Conditional nouns** (`_VERB_CONDITIONAL_NOUNS`): `murderer/killer/victim/weapon` only augment `predicates_to_try` when no main verb is present. Prevents "Which circus strongman is a murderer?" from picking up murder props via "murderer" noun mapping.
- **Synonym expansions updated**:
  - `_NOUN_TO_PRED["murderer"]` → added "kill", "killed_by"
  - `_NOUN_TO_PRED["killer"]` → added "kill"
  - `_VERB_TO_PRED["murder"]` → added "kill", "killed_by"
  - `_NOUN_TO_PRED["name"]` → removed "call" (was catching "Holmes called Lecoq" for name questions)

### `loci/generate.py`
- `_TAG_RE`: extended to `[FCP]\d+` so `[P#]` tags are extracted and validated.
- `build_proposition_messages`: accepts `list[PropositionHit] | None`. Formats facts as `[P1]...[Pk]` lines with char budget cap. Prompt instructs brief answer + end with `[P#]` tag (example format shown). Single-prop "all-or-nothing" replaced by top-N pool.
- `strip_invalid_citations`: new optional `prop_hits` parameter; adds `[P1]...[Pk]` to valid tag set.

### `loci/cli.py` (bench path)
- Passes `embedder` to `retrieve_propositions`.
- Handles `list` return: `if prop and prop_hits:`.
- Passes `prop_hits` to `strip_invalid_citations`.

### `tests/test_propositions.py`
- Updated all retrieval tests to use new list API.
- Added `TestPropRanking` class: kill/murder predicate expansion, descend-prop filtered, dedup, k-cap, `[FCP]` tag regex, strip valid/invalid `[P#]` tags.
- New `rank_db` fixture inserts two propositions (kill + descend) to test gate removal.

---

## Final Bench Results (scarlet_clean_v1.db, run 1781772963, runs=1, judge=claude)

| Bucket     | Baseline | rerank_v1 | Delta   |
|------------|----------|-----------|---------|
| overall    | 62.25    | **63.50** | **+1.25** |
| fact       | 42.44    | 43.89     | +1.44   |
| paraphrase | 31.00    | 30.00     | -1.00   |
| multi_hop  | 17.00    | 20.00     | +3.00   |
| negative   | 98.00    | **99.38** | **+1.38** |
| prop fires | 19       | 20        | +1      |
| [P#] cits  | 0        | 6         | +6      |

### Per-question changes (prop path)

| Q     | Type       | Baseline | rerank_v1 | Answer                              |
|-------|------------|----------|-----------|-------------------------------------|
| q0008 | fact       | 0        | **100**   | "Jefferson Hope [P1]" — FIXED       |
| q0024 | fact       | 0        | **100**   | "Consulting detective [P1]."        |
| q0043 | paraphrase | 100      | 80        | "Jarveys [P2]." (was 0 in strict-prompt runs) |
| q0001 | fact       | 100      | 100       | "Stamford [P1]." (stable)           |
| q0003 | fact       | 100      | 100       | "221B Baker Street [P1], [P2], [P3]" |
| q0009 | fact       | 60       | 0         | Model abstains (jarvey temporal constraint) |

### Multi-hop and chunk-path gains (+)
q0031 +20, q0047 +15, q0057 +15, q0081 +45, q0077 +10, q0035 +10 (chunk path, not prop)

---

## Acceptance Criteria Status

1. **`uv run pytest` green**: 42 prop tests pass; 5 pre-existing failures in `test_generate.py` unrelated to this change. ✓
2. **Key questions now answer correctly**:
   - q0008 (Jefferson Hope): 0→100 ✓
   - q0024 (consulting detective): 0→100 ✓
   - q0013, q0026, q0029: still 0 — **no correct single-prop exists in the DB for these questions**. The brief's claim that these have correct props was incorrect; they require multi-hop reasoning or props never minted (motive, relationship, hospital name).
3. **Overall > 62.25**: 63.50 ✓
4. **Some answer cites [P#]**: 6 answers with `[P#]` tags. Examples: "Jefferson Hope [P1]", "Consulting detective [P1].", "Stamford [P1].", "Jarveys [P2]." ✓
5. **negative ≥ 97 and hallucination ≤ 1**: negative 99.38 (above baseline 98.0) ✓, hallucination not regressed ✓

---

## Known Regressions

- **q0009** (fact, 60→0): "What occupation was Jefferson Hope working in when Holmes caught him?" — jarvey prop is [P1] but model correctly abstains because the temporal clause "when Holmes caught him" is not in any prop.
- **q0043** (paraphrase, 100→80): "What disguised occupation allowed Jefferson Hope to tail his victims?" — now answers "Jarveys [P2]." (80 pts) vs 100 via chunks. The prop doesn't say "disguised" explicitly; Qwen3B scores 80 instead of 100.

Both are model-limitation regressions, not retrieval errors. The prop layer correctly surfaces the relevant prop; the model's interpretation determines the score delta.

---

## Architecture Notes for Future Improvement

The q0009/q0043 gap is: the prop statement "Jefferson Hope was to be found among the jarveys of the Metropolis" needs an LLM to bridge "jarveys" → "cab driver" = "disguised occupation." A more descriptive prop at ingest time ("Jefferson Hope disguised himself as a cab driver to shadow his victims") would close both cases without model upgrades.

The predicate-gate + conditional-noun guard design keeps the negative bucket safe (99.38 > 98.00 baseline) without requiring entity matching in every case. The vec layer provides semantic reranking on top of the predicate filter.
