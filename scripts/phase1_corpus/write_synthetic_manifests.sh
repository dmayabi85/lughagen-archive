#!/usr/bin/env bash
# write_synthetic_manifests.sh
# ─────────────────────────────────────────────────────────────────────
# Write 8 synthetic manifests (4 monolingual + 4 code-switched) for the
# LughaGen tier 3 output, pointing at files produced by
# separate_synthetic_cs_v2.py. Record counts taken from the v2 run.
# Run anywhere; idempotent.

set -e

MONO_DIR=/workspace/data/corpus/manifests/synthetic_monolingual
CS_DIR=/workspace/data/corpus/manifests/synthetic_code_switched
mkdir -p "$MONO_DIR" "$CS_DIR"

# ── Synthetic monolingual ────────────────────────────────────────────
write_mono () {
    local LANG=$1
    local N=$2
    local FILE="$MONO_DIR/${LANG}_lughagen_synth_mono.yaml"

    cat > "$FILE" <<YAML
source_id: ${LANG}_lughagen_synth_mono_v2
display_name: "LughaGen synthetic ${LANG^} (monolingual fraction, v2)"
origin: synthetic
register: monolingual
language: ${LANG}

location:
  type: jsonl
  path: /workspace/data/corpus/staged/synth_mono/${LANG}_lughagen_synth.jsonl
  text_field: text
  quality_threshold:
    field: combined_quality
    min: 0.7

collection:
  method: "Tier 1/2/3 LughaGen synthetic pipeline; CS-vs-mono split by tier 2 pipeline labels (type, original_style)"
  date: 2026-05-13
  generator: meta-llama/Meta-Llama-3.1-8B-Instruct
  acceptance_pipeline: "tier2_glotlid_gated -> tier3_balance -> tier3_fluency"
  separator_script: separate_synthetic_cs_v2.py

license:
  type: "internal_research_only"
  reuse_terms: "JKUAT LughaGen research programme; not for redistribution"

defaults:
  domain: generated
  code_switch_pair: null
  preprocessing:
    - unicode_nfc
    - whitespace_normalise
    - tier2_glotlid_gate
    - tier3_fluency_filter
    - cs_separation_v2
  augmentation: []

provenance_notes: |
  Records in this bucket were prompted for monolingual output by the tier 2
  generator. Post-hoc English-marker validation confirms the predominantly
  monolingual character of this content (mismatch rate < 4%).

quality_contract:
  min_words: 5
  max_words: 1000
  max_repetition_ratio: 0.40
  required_fields: [text, language]

records_at_ingest:
  count_v2_run: ${N}
YAML
    echo "wrote $FILE"
}

# ── Synthetic code-switched ──────────────────────────────────────────
write_cs () {
    local LANG=$1
    local N=$2
    local FILE="$CS_DIR/${LANG}_lughagen_synth_cs.yaml"

    cat > "$FILE" <<YAML
source_id: ${LANG}_lughagen_synth_cs_v2
display_name: "LughaGen synthetic ${LANG^} (code-switched fraction, v2)"
origin: synthetic
register: code_switched
language: ${LANG}

location:
  type: jsonl
  path: /workspace/data/corpus/staged/synth_cs/${LANG}_lughagen_synth_cs.jsonl
  text_field: text
  quality_threshold:
    field: combined_quality
    min: 0.7

collection:
  method: "Tier 1/2/3 LughaGen synthetic pipeline; CS-vs-mono split by tier 2 pipeline labels (type, original_style)"
  date: 2026-05-13
  generator: meta-llama/Meta-Llama-3.1-8B-Instruct
  acceptance_pipeline: "tier2_glotlid_gated -> tier3_balance -> tier3_fluency"
  separator_script: separate_synthetic_cs_v2.py

license:
  type: "internal_research_only"
  reuse_terms: "JKUAT LughaGen research programme; not for redistribution"

defaults:
  domain: generated
  code_switch_pair: ${LANG}-english
  preprocessing:
    - unicode_nfc
    - whitespace_normalise
    - tier2_glotlid_gate
    - tier3_fluency_filter
    - cs_separation_v2
  augmentation: []

provenance_notes: |
  Records in this bucket were prompted for code-switched output by the tier 2
  generator. Post-hoc English-marker validation indicates many records contain
  predominantly monolingual content despite the CS prompt; this is a known
  pipeline-output limitation documented in /workspace/data/corpus/audit/
  synth_separation_qa.json. Authentic code-switched secondary sources are
  used to supplement this bucket (see manifests/authentic_code_switched/).

quality_contract:
  min_words: 5
  max_words: 1000
  max_repetition_ratio: 0.40
  required_fields: [text, language]

records_at_ingest:
  count_v2_run: ${N}
YAML
    echo "wrote $FILE"
}

# Counts from separate_synthetic_cs_v2.py run:
# swahili : 119,845 mono / 25,297 cs
# kikuyu  : 106,287 mono / 31,919 cs
# kamba   : 100,037 mono / 29,393 cs
# luo     : 111,491 mono / 31,361 cs

write_mono swahili 119845
write_mono kikuyu  106287
write_mono kamba   100037
write_mono luo     111491

write_cs   swahili  25297
write_cs   kikuyu   31919
write_cs   kamba    29393
write_cs   luo      31361

echo
echo "All 8 synthetic manifests written:"
echo "  $MONO_DIR/"
echo "  $CS_DIR/"
echo
echo "Verify:"
echo "  ls -la $MONO_DIR $CS_DIR"
echo "  head -25 $MONO_DIR/swahili_lughagen_synth_mono.yaml"
