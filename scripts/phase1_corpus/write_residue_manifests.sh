#!/usr/bin/env bash
# write_residue_manifests.sh
# ─────────────────────────────────────────────────────────────────────
# Write the four authentic-monolingual residue manifests for swahili,
# kikuyu, kamba, luo into /workspace/data/corpus/manifests/authentic_monolingual/
# Run from anywhere; idempotent (overwrites existing manifest files).

set -e

MANIFEST_DIR=/workspace/data/corpus/manifests/authentic_monolingual
mkdir -p "$MANIFEST_DIR"

# Per-language data (residue counts taken from the actual extraction output)
write_manifest () {
    local LANG=$1
    local BASE_TOTAL=$2
    local RESIDUE=$3
    local DROPPED=$4

    local FILE="$MANIFEST_DIR/${LANG}_raw_residue.yaml"

    cat > "$FILE" <<YAML
source_id: ${LANG}_raw_residue_v1
display_name: "LughaGen raw ${LANG^} residue (non-overlapping with synthetic, Jaccard < 0.5)"
origin: authentic
register: monolingual
language: ${LANG}

relation_to_synthetic:
  type: "residue_after_overlap_removal"
  base_file: /workspace/data/cleaned_final/${LANG}_clean.jsonl
  overlap_report: /workspace/data/corpus/audit/overlap_${LANG}.json
  threshold_used: 0.5
  records_in_base: ${BASE_TOTAL}
  records_unique: ${RESIDUE}
  records_dropped_as_overlap: ${DROPPED}

location:
  type: jsonl
  path: /workspace/data/secondary/monolingual/${LANG}_raw_residue.jsonl
  text_field: text
  quality_threshold:
    field: quality
    min: 0.8

collection:
  method: "Residue of /workspace/data/cleaned_final/${LANG}_clean.jsonl after MinHash-LSH overlap removal vs synthetic tier 3 output"
  date: 2026-05-16

license:
  type: "internal_research_only"

defaults:
  domain: mixed
  preprocessing:
    - unicode_nfc
    - whitespace_normalise
    - paraphrastic_overlap_removal_vs_synthetic
  augmentation: []

quality_contract:
  min_words: 5
  max_words: 2000
  required_fields: [text]
YAML
    echo "wrote $FILE"
}

# Args: language  records_in_base  records_unique  records_dropped
write_manifest swahili 8340511 882801 7457710
write_manifest kikuyu   140000  34284  105716
write_manifest kamba     54542  15462   39080
write_manifest luo      543763 221180  322583

echo
echo "All four manifests written. Verify with:"
echo "  ls -la $MANIFEST_DIR"
echo "  head -20 $MANIFEST_DIR/swahili_raw_residue.yaml"
