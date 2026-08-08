#!/bin/bash
# Aug 15: Review thesis drafts for overclaims
# Searches for patterns that might indicate causal overreach

if [ ! -d /workspace/thesis_drafts ]; then
    echo "Create /workspace/thesis_drafts first, put chapter drafts there"
    exit 1
fi

echo "=== Causal Claim Review ==="
echo ""

echo "Patterns that overclaim causation:"
grep -r -n --color -iE "causes|proves|demonstrates that|shows that.*because|X.*leads to Y|X made Y|due to X" \
    /workspace/thesis_drafts/ 2>/dev/null | head -30 || echo "  (none found)"

echo ""
echo "Patterns that suggest superlative claims:"
grep -r -n --color -iE "best model|superior|winner|dominates|clearly better" \
    /workspace/thesis_drafts/ 2>/dev/null | head -30 || echo "  (none found)"

echo ""
echo "Patterns to consider rewording:"
grep -r -n --color -iE "significant|meaningful|remarkable|impressive|dramatic" \
    /workspace/thesis_drafts/ 2>/dev/null | head -30 || echo "  (none found)"

echo ""
echo "Suggested replacements:"
echo "  'X causes Y' → 'under our conditions, X is associated with Y'"
echo "  'X is best' → 'X achieved the lowest BPC in tested configurations'"
echo "  'X > Y' → 'X outperformed Y by ΔBPC (95% CI [...])'"
echo "  'proves' → 'provides evidence for'"
echo "  'significant' → 'reliable' or specify effect size"
