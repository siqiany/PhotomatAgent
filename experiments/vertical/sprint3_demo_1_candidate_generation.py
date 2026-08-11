"""Sprint 3 Demo 1: candidate generation with formula consistency + lineage.

Target: 2-5 um infrared constraint -> VAE formula proposal -> MatterGen
structure -> formula consistency report -> candidate lineage.

Note on the VAE decoder: this environment has no torch checkpoint, so the
demo injects a deterministic decoder (documented as demo-only). The full
tool path returns a typed missing_prerequisites failure instead of guessing
when the checkpoint is absent -- that behavior is shown in the second part.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from photomatagent.scientific.capabilities.generation.formulas import (  # noqa: E402
    VAEFormulaGenerator,
)
from photomatagent.scientific.capabilities.generation.mattergen import (  # noqa: E402
    MatterGenGenerator,
)


def main() -> int:
    print("=" * 72)
    print("Demo 1: candidate generation (2-5 um infrared constraint)")
    print("=" * 72)

    target_wavelength_um = 2.5  # 0.496 eV photon
    print(f"\n[1] Target: {target_wavelength_um} um "
          f"(~{1.239841984 / target_wavelength_um:.3f} eV)")

    # Demo-only deterministic decoder: HgTe-heavy composition prior.
    vocabulary = ["Hg", "Te", "Pb", "Sb", "In", "As"]

    def demo_decoder(condition, count):
        fractions = np.array([0.48, 0.48, 0.01, 0.01, 0.01, 0.01])
        return np.tile(fractions, (count, 1))

    generator = VAEFormulaGenerator(
        vocabulary=vocabulary,
        known_formulas={"HgTe"},
        decoder=demo_decoder,
        require_charge_neutral=True,
        require_novel=False,
        random_seed=42,
    )
    proposals, metadata = generator.generate(
        target_wavelength_um=target_wavelength_um, limit=3
    )
    print("\n[2] VAE formula proposals (demo decoder; checkpoint path "
          "shows typed failure below):")
    for proposal in proposals:
        print(f"    {proposal.formula:8s} charge_neutral={proposal.charge_neutral} "
              f"novel={proposal.novel_against_training_data} "
              f"composition_error={proposal.composition_error:.4f}")
    print(f"    rejection counts: {metadata['rejection_counts']}")
    print(f"    defaults: {metadata['defaults_note']}")

    proposed_formula = proposals[0].formula

    print("\n[3] MatterGen structure generation (fixture manifest):")
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        cif = tmp / "0001.cif"
        cif.write_text(
            """data_HgTe
_cell_length_a   6.46
_cell_length_b   6.46
_cell_length_c   6.46
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Hg  Hg  0.0 0.0 0.0
Te  Te  0.25 0.25 0.25
""",
            encoding="utf-8",
        )
        manifest = tmp / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "pretrained_name": "dft_band_gap",
                    "properties_to_condition_on": {"dft_band_gap": 0.496},
                    "band_gap_target_source": "photon_energy_proxy",
                    "candidates": [
                        {
                            "structure_path": str(cif),
                            "candidate_id": "mg-0001",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        candidates, mg_metadata = MatterGenGenerator().generate(
            target_wavelength_um=target_wavelength_um,
            chemical_system="Hg-Te",
            proposed_formula=proposed_formula,
            manifest_path=manifest,
        )
        for candidate in candidates:
            print(f"    candidate {candidate['candidate_id']}:")
            print(f"      vae_proposed_formula:       {candidate['vae_proposed_formula']}")
            print(f"      mattergen_generated_formula:{candidate['mattergen_generated_formula']}")
            print(f"      formula_preserved:          {candidate['formula_preserved']}")
            print(f"      composition_distance:       {candidate['composition_distance']}")
            print(f"      validation_status:          "
                  f"{candidate['lineage']['validation_status']}")
            print(f"      warnings: {candidate['warnings'][0][:80]}...")

    print("\n[4] Missing checkpoint -> typed failure (no hallucination):")
    tool_generator = VAEFormulaGenerator(
        vocabulary=vocabulary, checkpoint_path="/nonexistent/model.pt"
    )
    try:
        tool_generator.generate(target_wavelength_um=target_wavelength_um)
    except Exception as exc:
        print(f"    {type(exc).__name__}: {exc}")
        print("    -> correct: the tool reports missing prerequisites")

    print("\nDemo 1 done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
