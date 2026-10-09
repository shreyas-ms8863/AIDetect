# Community Forensics Maintainer Export Request

**Status:** Metadata-only request mapping; not an accepted/frozen experiment dataset.

Request exactly **800 image payloads**, identified by the CSV rows:

- REAL: 400 from `real_source=RAISE`.
- AI: 100 each from `GALIP`, `kandinsky_2_2`, `stable_cascade`, and `DeciDiffusionV2`, all paired with `RAISE`.

**Dataset:** `OwensLab/CommunityForensics-Eval`

- Dataset commit: `7d4a74a88d2cac93b513c0853bf92c260eaceea0`
- Parquet conversion commit: `49365d527202f179b93e478f5b32107eb92aaa3f`
- Config/split: `default` / `CompEval`
- Selection seed: `20261009`

## Retrieval identifiers

The strongest locator available in the source metadata is the pinned Parquet conversion revision plus `parquet_shard` and `row_offset_in_shard`. The CSV also includes `image_name`, source, generator/model, label, architecture, subset, and split. `candidate_id` is the manifest-derived `shard:offset` locator, not a source-provided stable ID.

`image_name` is not globally unique: this manifest has 58 repeated-name groups across generator partitions. Do not retrieve by filename alone. The source metadata provides no per-row SHA-256, perceptual hash, explicit original/pair ID, or individual image URL; the corresponding fields are therefore not claimed.

## Safety and pending work

- No image payloads were downloaded or saved for this mapping.
- No inference was performed.
- SHA-256 and perceptual-duplicate auditing remain pending.
- The candidates remain provisional until the returned payloads are verified against these identifiers and AIDetect history.
- No maintainer was contacted automatically.
