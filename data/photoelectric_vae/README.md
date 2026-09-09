# Photoelectric VAE reproducibility bundle

This directory holds the source data needed to rebuild the packaged
photoelectric composition VAE and inverse index.

## Training inputs

- `training/jarvis_all_ir_candidates.csv`: 11,240 filtered 2D/3D JARVIS
  candidates and the 14 property fields used by the model.
- `training/raw/jarvis_dft3d_2025.zip`: externally downloaded JARVIS 3D
  snapshot, Figshare DOI
  `10.6084/m9.figshare.6815699.v11`.
- `training/raw/jarvis_dft2d_2022.zip`: externally downloaded JARVIS 2D
  snapshot, Figshare DOI
  `10.6084/m9.figshare.6815705.v8`.
- `training/DATA_DICTIONARY.md`: field definitions and units.
- `training/LICENSES_AND_PROVENANCE.md`: licensing, origin and citation rules.

The two raw JARVIS archives are intentionally excluded from Git and must be
downloaded into `training/raw/` for a complete rebuild. Their expected sizes
and SHA-256 hashes remain in `asset_manifest.json`. Both archives are CC BY
4.0. The generated candidate table and model artifacts retain the source
identifiers and DOI fields needed for attribution.

## Rebuild sequence

Run from the repository root after installing the generation extra:

```bash
uv sync --extra generation
uv run python scripts/vae/build_jarvis_candidates.py
uv run python scripts/vae/train_inverse_index.py
uv run python scripts/vae/train_conditional_vae.py --epochs 10
uv run python scripts/vae/verify_assets.py
uv run python scripts/vae/verify_assets.py --require-source-archives
```

The first verification command checks every packaged/committed asset and also
checks either raw archive when it is present. The strict form additionally
requires both external source archives, and is the appropriate final gate for
a full reproduction run.

The first command regenerates the committed candidate CSV from the raw
archives. The second rebuilds the inverse index, aligned metadata and
structures. The third trains the deployed CVAE using the original hyperparameter
settings (`seed=42`, `batch_size=256`, `learning_rate=1e-3`, `beta=0.02`,
`condition_dropout=0.45`, 10 epochs).

Generated formulas remain model proposals, not validated materials. Database,
stability, electronic, optical, defect, transport and device evidence must be
evaluated separately.
