"""Sprint 3 Demo 4: vasprun -> dielectric -> n/k -> Meep thin-film R/T/A.

Uses a synthetic VASP optics vasprun.xml fixture. The dielectric -> n/k
conversion is deterministic code; the Meep simulation requires the ``meep``
package (isolated env), otherwise the tool returns a typed
MISSING_DEPENDENCY failure -- no numbers are invented.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from photomatagent.scientific.capabilities.optics.meep_thinfilm import (  # noqa: E402
    optical_point_from_vasprun,
)

VASPRUN_FIXTURE = """<modeling>
  <parameters>
    <i name="EMAX">6.0</i>
    <i name="NEDOS">61</i>
  </parameters>
  <calculation>
    <dielectricfunction>
      <varray name="real">
        {real_rows}
      </varray>
      <varray name="imag">
        {imag_rows}
      </varray>
    </dielectricfunction>
  </calculation>
</modeling>
"""


def main() -> int:
    print("=" * 72)
    print("Demo 4: VASP optics -> n/k -> Meep thin-film R/T/A")
    print("=" * 72)
    target_wavelength_um = 5.0  # LWIR

    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        # Simple Lorentzian-ish dielectric spectrum (fixture only).
        real_rows = []
        imag_rows = []
        for index in range(61):
            energy = index * 0.1
            epsilon_real = 12.0 - 4.0 / (1.0 + (energy - 2.5) ** 2)
            epsilon_imag = 2.0 / (1.0 + (energy - 2.5) ** 2)
            row_real = f"{epsilon_real:.5f} 0 0 0 {epsilon_real:.5f} 0 0 0 {epsilon_real:.5f}"
            row_imag = f"{epsilon_imag:.5f} 0 0 0 {epsilon_imag:.5f} 0 0 0 {epsilon_imag:.5f}"
            real_rows.append(f"<v> {row_real} </v>")
            imag_rows.append(f"<v> {row_imag} </v>")
        vasprun = tmp / "vasprun.xml"
        vasprun.write_text(
            VASPRUN_FIXTURE.format(
                real_rows="\n".join(real_rows),
                imag_rows="\n".join(imag_rows),
            ),
            encoding="utf-8",
        )

        print(f"\n[1] Target wavelength: {target_wavelength_um} um "
              f"(~{1.239841984 / target_wavelength_um:.3f} eV)")
        point = optical_point_from_vasprun(vasprun, target_wavelength_um)
        print(f"[2] vasprun -> n/k (deterministic conversion):")
        print(f"    n = {point['refractive_index']:.4f}, "
              f"k = {point['extinction_coefficient']:.4f} "
              f"(source: {point['source']})")

        print("\n[3] Meep thin-film simulation:")
        import asyncio

        from photomatagent.scientific.capabilities.optics.meep_thinfilm import (
            MeepThinFilmTool,
        )

        result = asyncio.run(
            MeepThinFilmTool().execute(
                {
                    "wavelength_um": target_wavelength_um,
                    "thickness_um": 1.0,
                    "vasprun_xml": str(vasprun),
                    "resolution": 20,
                }
            )
        )
        if result.is_error:
            print(f"    typed failure: {result.output[:160]}")
            print("    -> correct: meep is not installed here; no R/T/A is "
                  "invented (run in the isolated meep env for numbers)")
        else:
            payload = result.data
            print(f"    R = {payload['reflectance']:.4f}, "
                  f"T = {payload['transmittance']:.4f}, "
                  f"A = {payload['absorptance']:.4f}, "
                  f"residual = {payload['energy_conservation_residual']:.2e}")
            print(f"    evidence: {len(result.evidence)} item(s), "
                  f"fidelity={result.evidence[0].fidelity}")

    print("\nDemo 4 done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
