# Batch repeatability

Record independent preparation dates, precursor lots, operator and equipment changes, device identities and the selection rule for characterization. Freeze the acceptance rule before comparing batches and retain failed runs.

Use `process.register_plan` to distinguish the parent batch, specimens and devices. Attach an acceptance checkpoint and a quarantine branch. Report per-batch results with sample counts, within-batch spread and between-batch spread. Bind results to the actual specimen and process conditions.

Repeated readings of one device estimate measurement repeatability. Independent preparation batches are required to estimate batch-to-batch variation. A missing batch manifest or a change in selection criteria leaves the repeatability inference UNKNOWN.

Reference protocol: [Batch and device reporting in photodetector characterization](https://www.nature.com/articles/s41566-025-01759-1).
