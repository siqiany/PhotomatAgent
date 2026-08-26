from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from photomatagent.scientific.capabilities.generation.inverse_retrieval import (
    InverseMaterialRetriever,
)
from photomatagent.scientific.capabilities.generation.tools import (
    PACKAGED_VAE_ASSET_ROOT,
    VAEFormulaTool,
    VAERetrieveTool,
    _resolve_vae_assets,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_packaged_assets_are_default_and_self_contained(monkeypatch):
    for name in (
        "PHOTOMATAGENT_VAE_ASSET_ROOT",
        "PHOTOELECTRIC_VAE_ASSET_ROOT",
        "VAE_ASSET_ROOT",
        "VAE_CHECKPOINT_PATH",
        "VAE_METADATA_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    checkpoint, metadata = _resolve_vae_assets()
    assert checkpoint == (
        PACKAGED_VAE_ASSET_ROOT / "jarvis_cvae_v1" / "checkpoint.pt"
    ).resolve()
    assert metadata == (
        PACKAGED_VAE_ASSET_ROOT
        / "jarvis_inverse_v1"
        / "candidate_metadata.json"
    ).resolve()
    assert REPOSITORY_ROOT in checkpoint.parents
    assert "GlassCrewAgent" not in str(checkpoint)


def test_asset_manifest_hashes_verify():
    completed = subprocess.run(
        [sys.executable, "scripts/vae/verify_assets.py"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "verified 9 VAE asset files" in completed.stdout


def test_checkpoint_and_inverse_index_share_training_schema():
    torch = pytest.importorskip("torch")
    checkpoint = torch.load(
        PACKAGED_VAE_ASSET_ROOT / "jarvis_cvae_v1" / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    arrays = np.load(
        PACKAGED_VAE_ASSET_ROOT
        / "jarvis_inverse_v1"
        / "inverse_index.npz"
    )
    assert checkpoint["property_fields"] == arrays["property_fields"].tolist()
    assert checkpoint["vocabulary"] == arrays["vocabulary"].tolist()
    assert checkpoint["training"]["record_count"] == len(
        arrays["compositions"]
    )
    assert not Path(checkpoint["training"]["data_source"]).is_absolute()


def test_packaged_inverse_index_predicts_without_external_files():
    retriever = InverseMaterialRetriever(
        PACKAGED_VAE_ASSET_ROOT / "jarvis_inverse_v1"
    )
    results = retriever.predict(
        {"gap_selected_eV": 0.5},
        top_k=2,
        max_energy_above_hull_eV_per_atom=None,
    )
    assert len(results) == 2
    assert all(item["metadata"]["jarvis_id"] for item in results)
    assert all("gap_selected_eV" in item["properties"] for item in results)


def test_packaged_vae_generates_formulas_from_multiple_properties():
    pytest.importorskip("torch")
    result = asyncio.run(
        VAEFormulaTool().execute(
            {
                "target_properties": {
                    "gap_selected_eV": 0.35,
                    "formation_energy_eV_per_atom": -0.8,
                    "density_g_cm3": 6.0,
                    "dielectric_mean": 15.0,
                    "avg_electron_mass_m0": 0.2,
                },
                "limit": 3,
                "sample_count": 256,
                "require_novel": False,
                "require_charge_neutral": False,
                "random_seed": 23,
            }
        )
    )
    assert not result.is_error
    assert len(result.data["proposals"]) == 3
    metadata = result.data["metadata"]
    assert metadata["conditioned_property_count"] == 6
    assert metadata["target_properties"]["density_g_cm3"] == 6.0
    assert metadata["target_properties"][
        "cutoff_wavelength_um_from_gap"
    ] == pytest.approx(1.239841984 / 0.35)
    assert metadata["clipped_condition_fields"] == []


def test_vae_retrieve_uses_packaged_candidate_metadata(monkeypatch):
    monkeypatch.delenv("VAE_INDEX_PATH", raising=False)
    metadata_path = (
        PACKAGED_VAE_ASSET_ROOT
        / "jarvis_inverse_v1"
        / "candidate_metadata.json"
    )
    first = json.loads(metadata_path.read_text(encoding="utf-8"))[0]
    result = asyncio.run(
        VAERetrieveTool().execute(
            {"formula": first["formula"], "max_results": 3}
        )
    )
    assert not result.is_error
    assert result.data["count"] >= 1
    assert result.data["matches"][0]["formula"] == first["formula"]
