# DenPAR leakage remediation denpar-leakage-remediation-v1

## Reconciled duplicate counts

| Type | Unique images | Pairwise relationships | Connected groups | Same-partition pairs/groups | Cross-partition pairs/groups |
|---|---:|---:|---:|---:|---:|
| Exact | 18 | 9 | 9 | 5 / 5 | 4 / 4 |
| Near candidates | 19 | 11 | 9 | 8 / 6 | 3 / 3 |

The earlier count of eight same-partition candidates refers to near-duplicate pairwise relationships. The nine exact pairs consist of five same-partition and four cross-partition pairs. Every exact group currently contains two images, so exact pair and exact group totals are both nine.

## Derived manifest sizes

- Clean Training: 643
- Clean Validation: 146
- Clean historical exposed Testing: 196
- Exact exclusions: 9
- Quarantined pending review: 6 images in 3 cross-partition pairs

Exact exclusions by partition: {'Training': 5, 'Validation': 3, 'Testing': 1}. Quarantined by partition: {'Training': 2, 'Validation': 1, 'Testing': 3}.

## Safety status

Stage 2B is blocked until all three cross-partition near-duplicate pairs receive a saved manual decision. Official files and partitions were not changed. The existing Testing split is labeled `historical_exposed_test` and cannot be represented as a pristine final test set.
