# DenPAR Stage 2A audit

Generated: 2026-09-22T02:06:13.295017+00:00

## Scope and method

This was a read-only identity, duplicate, annotation-integrity, and checkpoint-provenance audit. No radiograph was copied into this report. Exact identity uses SHA-256 of each file and SHA-256 of decoded canonical 8-bit grayscale pixels plus dimensions. Near duplicates are manual-review candidates produced by a 64-bit DCT pHash threshold of Hamming distance <= 6; they are not confirmed duplicates.

## Partition counts

| Partition | Expected | Observed | Match |
|---|---:|---:|---|
| Training | 1 | 1 | True |
| Validation | 1 | 1 | True |
| Testing | 2 | 2 | True |

Training images are read from `dataset/raw` because the supplied DenPAR `Training` directory contains annotations but no image folder. The immutable 650-entry training manifest is the supporting identity record.

## Findings

- Exact duplicate pairs: 3 unique pairs (3 across partitions); evidence is available by file hash and normalized-pixel hash
- Near-duplicate candidates: 3 (2 across partitions)
- Duplicate annotation mappings: 2
- Missing or orphaned annotation findings: 3
- Coordinate-integrity findings: 1
- Source files hashed and preserved: 21; changed: 0
- Checkpoint training provenance verified from available artifacts: False
- Patient grouping available: False

## Leakage conclusion

confirmed_leakage_detected. Perceptual matches remain candidates for manual review. Patient-level separation is unverified because the workbook contains image-level arch, site, and visible-tooth fields but no patient identifier.

## Stage 2B gate

Review `manual_review_queue.csv`, especially every cross-partition candidate and annotation-integrity finding, before using this dataset for Stage 2B. This report does not alter the official partitions.
