"""
Compute main effects and 2-way interactions from the (2^4) factorial experiment.
Reads existing factorial results.
"""

import json
from pathlib import Path
import numpy as np

# Load factorial results
factorial_file = Path("/workspace/results/factorial_design.csv")
# Or wherever the results are — check with:
# find /workspace/results -name "*factorial*"

# Placeholder — you'll fill in actual data structure once located
# Main effects computation:
# For each factor with 2 levels (say LR high vs low):
#   effect = mean(high) - mean(low)
#   
# Report:
#   Factor       Main Effect  ±SE
#   LR           +0.05        ±0.01
#   Batch        -0.02        ±0.01
#   Seq         +0.01        ±0.01
#   Optimizer   -0.03        ±0.01
#
# Plus 2-way interactions:
#   LR × Batch    +0.02
#   LR × Seq     +0.01
#   etc.

print("=== (2^4) Factorial Main Effects Analysis ===")
print("(Placeholder — populate with actual factorial data)")
print()

# Once data is loaded, actual computation would be:
# for factor in factors:
#     high = df[df[factor] == '+']['bpc'].mean()
#     low = df[df[factor] == '-']['bpc'].mean()
#     effect = high - low
#     print(f"  {factor:<20} {effect:+.4f}")
