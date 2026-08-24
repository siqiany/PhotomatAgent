# Packaged photoelectric VAE assets

These files are shipped inside `photomatagent` so composition generation works
after a normal clone or wheel installation without another local repository.

- `jarvis_cvae_v1/checkpoint.pt`: 89-element, 14-condition CVAE checkpoint.
- `jarvis_inverse_v1/inverse_index.npz`: normalized property/composition index.
- `jarvis_inverse_v1/candidate_metadata.json`: 11,240 JARVIS candidate records
  used for novelty checks and exact formula/system retrieval.
- `jarvis_inverse_v1/structures.jsonl`: aligned JARVIS structures used by the
  inverse retriever.
- Each model directory contains its original training metrics.

The source training inputs are committed under `data/photoelectric_vae/`.
Rebuild the artifacts from the repository root with:

```bash
uv run python scripts/vae/build_jarvis_candidates.py
uv run python scripts/vae/train_inverse_index.py
uv run python scripts/vae/train_conditional_vae.py --epochs 10
uv run python scripts/vae/verify_assets.py
```

The two raw archives are NIST JARVIS-DFT snapshots distributed under CC BY
4.0. Cite dataset DOI `10.6084/m9.figshare.6815699.v11` for the 3D data and
`10.6084/m9.figshare.6815705.v8` for the 2D data. See
`data/photoelectric_vae/training/LICENSES_AND_PROVENANCE.md` for attribution
and scientific-use limitations.

Training uses seeded NumPy/PyTorch sampling. Metrics should reproduce within
normal numerical tolerance; byte-identical checkpoints can still depend on the
PyTorch version and CPU backend.
