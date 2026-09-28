# DenPAR leakage remediation denpar-leakage-remediation-v2

All three manual review decisions are recorded and applied. Validation 853 is canonical for the confirmed Validation–Testing pair; Testing 852 is excluded. Pairs 2 and 3 were confirmed different, released from quarantine, and retained in their original derived partitions.

## Derived counts

- Clean Training: 2
- Clean Validation: 0
- Clean historical exposed Testing: 1
- Total exclusions: 3
- Remaining quarantined images: 0
- Exclusions by partition: {'Training': 1, 'Validation': 1, 'Testing': 1}

## Gate

The derived manifests are ready for Stage 2B development, provided loaders enforce policy version `denpar-leakage-remediation-v2` and the recorded content hashes. Historical Testing remains exposed and is limited to transparent comparison. A future independent expert-labeled external holdout remains necessary for final unbiased evaluation.
