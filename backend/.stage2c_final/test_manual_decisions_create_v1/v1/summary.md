# DenPAR leakage remediation denpar-leakage-remediation-v1

## Reconciled duplicate counts

| Type | Unique images | Pairwise relationships | Connected groups | Same-partition pairs/groups | Cross-partition pairs/groups |
|---|---:|---:|---:|---:|---:|
| Exact | 3 | 3 | 1 | 1 / 0 | 2 / 1 |
| Near candidates | 2 | 1 | 1 | 0 / 0 | 1 / 1 |

The earlier count of eight same-partition candidates refers to near-duplicate pairwise relationships. The nine exact pairs consist of five same-partition and four cross-partition pairs. Every exact group currently contains two images, so exact pair and exact group totals are both nine.

## Derived manifest sizes

- Clean Training: 1
- Clean Validation: 0
- Clean historical exposed Testing: 1
- Exact exclusions: 2
- Quarantined pending review: 2 images in 1 cross-partition pairs

Exact exclusions by partition: {'Training': 1, 'Validation': 1, 'Testing': 0}. Quarantined by partition: {'Training': 1, 'Validation': 0, 'Testing': 1}.

## Safety status

Stage 2B is blocked until all three cross-partition near-duplicate pairs receive a saved manual decision. Official files and partitions were not changed. The existing Testing split is labeled `historical_exposed_test` and cannot be represented as a pristine final test set.
