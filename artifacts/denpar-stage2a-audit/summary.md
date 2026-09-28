# DenPAR Stage 2A audit

Generated: 2026-09-21T11:49:15.729264+00:00

## Scope and method

This was a read-only identity, duplicate, annotation-integrity, and checkpoint-provenance audit. No radiograph was copied into this report. Exact identity uses SHA-256 of each file and SHA-256 of decoded canonical 8-bit grayscale pixels plus dimensions. Near duplicates are manual-review candidates produced by a 64-bit DCT pHash threshold of Hamming distance <= 6; they are not confirmed duplicates.

## Partition counts

| Partition | Expected | Observed | Match |
|---|---:|---:|---|
| Training | 650 | 650 | True |
| Validation | 150 | 150 | True |
| Testing | 200 | 200 | True |

Training images are read from `dataset/raw` because the supplied DenPAR `Training` directory contains annotations but no image folder. The immutable 650-entry training manifest is the supporting identity record.

## Findings

- Exact duplicate pairs: 9 unique pairs (4 across partitions); evidence is available by file hash and normalized-pixel hash
- Near-duplicate candidates: 11 (3 across partitions)
- Duplicate annotation mappings: 0
- Missing or orphaned annotation findings: 0
- Coordinate-integrity findings: 0
- Source files hashed and preserved: 8408; changed: 0
- Checkpoint training provenance verified from available artifacts: True
- Patient grouping available: False

## Leakage conclusion

confirmed_leakage_detected. Perceptual matches remain candidates for manual review. Patient-level separation is unverified because the workbook contains image-level arch, site, and visible-tooth fields but no patient identifier.

## Stage 2B gate

Review `manual_review_queue.csv`, especially every cross-partition candidate and annotation-integrity finding, before using this dataset for Stage 2B. This report does not alter the official partitions.
