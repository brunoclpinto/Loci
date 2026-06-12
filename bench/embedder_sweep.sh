#!/usr/bin/env bash
# Ingest A Study in Scarlet into three separate DBs (one per embedder),
# then bench each and print a comparison.
#
# Usage:
#   bash bench/embedder_sweep.sh [--skip-ingest]
#
# --skip-ingest: skip ingest+enhance and go straight to bench (if DBs exist)

set -euo pipefail

SCARLET="/Volumes/SSD1TB001/loci/raw/pg244_a_study_in_scarlet.txt"
DB_DIR="/Volumes/SSD1TB001/loci/knowledge"
LOG_DIR="/Volumes/SSD1TB001/loci/logs/bench"
QNA="bench/qna.json"

declare -A EMBEDDERS=(
  [small]="bge-small-en-v1.5-q8_0.gguf"
  [base]="bge-base-en-v1.5-q8_0.gguf"
  [large]="bge-large-en-v1.5-q8_0.gguf"
)
declare -A DIMS=([small]=384 [base]=768 [large]=1024)

SKIP_INGEST=false
for arg in "$@"; do [[ "$arg" == "--skip-ingest" ]] && SKIP_INGEST=true; done

LOGS=()

for VARIANT in small base large; do
  EMBEDDER="${EMBEDDERS[$VARIANT]}"
  DIM="${DIMS[$VARIANT]}"
  DB="${DB_DIR}/scarlet_${VARIANT}.db"

  echo ""
  echo "════════════════════════════════════════════════════"
  echo "  Variant: bge-${VARIANT}  (${DIM}-dim)  →  ${DB}"
  echo "════════════════════════════════════════════════════"

  if [[ "$SKIP_INGEST" == "false" ]]; then
    echo "→ Ingest..."
    LOCI_MODELS__EMBEDDER="$EMBEDDER" \
    LOCI_MODELS__VEC_DIM="$DIM" \
    LOCI_PATHS__KNOWLEDGE_DB="$DB" \
      uv run loci ingest "$SCARLET"

    echo "→ Enhance..."
    LOCI_MODELS__EMBEDDER="$EMBEDDER" \
    LOCI_MODELS__VEC_DIM="$DIM" \
    LOCI_PATHS__KNOWLEDGE_DB="$DB" \
      uv run loci enhance
  fi

  echo "→ Bench..."
  LOCI_MODELS__EMBEDDER="$EMBEDDER" \
  LOCI_MODELS__VEC_DIM="$DIM" \
  LOCI_PATHS__KNOWLEDGE_DB="$DB" \
    uv run loci bench query --qna "$QNA" --runs 1 --label "emb_${VARIANT}"

  # Capture the log path for comparison
  LATEST=$(ls -t "${LOG_DIR}"/*.jsonl 2>/dev/null | grep -v judge | head -1)
  LOGS+=("$LATEST")
done

echo ""
echo "════════════════════════════════════════════════════"
echo "  Pairwise comparisons"
echo "════════════════════════════════════════════════════"

if [[ ${#LOGS[@]} -ge 2 ]]; then
  echo ""
  echo "── small vs base ──"
  uv run loci bench compare "${LOGS[0]}" "${LOGS[1]}"
fi
if [[ ${#LOGS[@]} -ge 3 ]]; then
  echo ""
  echo "── small vs large ──"
  uv run loci bench compare "${LOGS[0]}" "${LOGS[2]}"
  echo ""
  echo "── base vs large ──"
  uv run loci bench compare "${LOGS[1]}" "${LOGS[2]}"
fi
