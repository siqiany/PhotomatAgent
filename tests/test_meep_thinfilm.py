"""Offline Meep thin-film tests: n/k conversion, flux normalization, contracts."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from photomatagent.scientific.capabilities.optics.meep_thinfilm import (
    dielectric_to_nk,
    interpolate_dielectric,
    normalize_fluxes,
    optical_point_from_vasprun,
)


def test_dielectric_to_nk_known_value():
    # Lossless: epsilon = n^2 -> k = 0.
    n, k = dielectric_to_nk(4.0, 0.0)
    assert n == pytest.approx(2.0)
    assert k == pytest.approx(0.0)
    # Complex: n=3.5, k=0.2 -> epsilon = (n+ik)^2.
    epsilon_real = 3.5**2 - 0.2**2
    epsilon_imag = 2 * 3.5 * 0.2
    n_back, k_back = dielectric_to_nk(epsilon_real, epsilon_imag)
    assert n_back == pytest.approx(3.5)
    assert k_back == pytest.approx(0.2)


def test_flux_normalization_energy_conservation():
    r, t, a, residual = normalize_fluxes(10.0, -1.5, 7.0)
    assert r == pytest.approx(0.15)
    assert t == pytest.approx(0.70)
    assert a == pytest.approx(0.15)
    assert residual == pytest.approx(0.0, abs=1e-12)
    assert r + t + a == pytest.approx(1.0)


def test_flux_normalization_clamps_to_unit_interval():
    r, t, a, residual = normalize_fluxes(10.0, -12.0, 8.0)
    assert r == 1.0 and t == 0.8 and a == 0.0
    assert residual != 0.0  # untruncated residual is preserved


def test_flux_normalization_rejects_zero_incident():
    with pytest.raises(ValueError, match="incident flux"):
        normalize_fluxes(0.0, 0.0, 0.0)


def test_interpolate_dielectric_isotropic_average():
    energies = [1.0, 2.0, 3.0, 4.0]
    real = [[10.0, 12.0, 14.0, 11.0, 10.0, 13.0]] * 4
    imag = [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]] * 4
    point = interpolate_dielectric(energies, real, imag, wavelength_um=0.62)
    assert point["energy_ev"] == pytest.approx(1.239841984 / 0.62)
    assert point["refractive_index"] > 0
    assert point["source"] == "dielectric spectrum"


def test_interpolate_dielectric_out_of_range():
    with pytest.raises(ValueError, match="outside the spectrum"):
        interpolate_dielectric(
            [1.0, 2.0], [[4.0, 0, 0]] * 2, [[0.0, 0, 0]] * 2, wavelength_um=0.3
        )


def test_optical_point_from_vasprun(tmp_path):
    vasprun = tmp_path / "vasprun.xml"
    vasprun.write_text(
        """<modeling>
  <parameters>
    <i name="EMAX">4.0</i>
    <i name="NEDOS">3</i>
  </parameters>
  <calculation>
    <dielectricfunction>
      <varray name="real">
        <v> 12.0 0.0 0.0 0.0 12.0 0.0 0.0 0.0 12.0 </v>
        <v> 10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0 </v>
        <v> 8.0 0.0 0.0 0.0 8.0 0.0 0.0 0.0 8.0 </v>
      </varray>
      <varray name="imag">
        <v> 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 </v>
        <v> 0.6 0.0 0.0 0.0 0.6 0.0 0.0 0.0 0.6 </v>
        <v> 0.2 0.0 0.0 0.0 0.2 0.0 0.0 0.0 0.2 </v>
      </varray>
    </dielectricfunction>
  </calculation>
</modeling>
""",
        encoding="utf-8",
    )
    point = optical_point_from_vasprun(vasprun, wavelength_um=0.62)
    assert point["source"].startswith("VASP:")
    assert point["energy_ev"] == pytest.approx(1.239841984 / 0.62)
    assert point["refractive_index"] > 0
    assert point["extinction_coefficient"] > 0


def test_optical_point_from_vasprun_missing_dielectric(tmp_path):
    vasprun = tmp_path / "vasprun.xml"
    vasprun.write_text("<modeling></modeling>", encoding="utf-8")
    with pytest.raises(ValueError, match="dielectricfunction"):
        optical_point_from_vasprun(vasprun, 0.62)


def test_meep_tool_missing_dependency_without_meep(tmp_path):
    from photomatagent.scientific.capabilities.optics.meep_thinfilm import (
        MeepThinFilmTool,
    )

    result = asyncio.run(
        MeepThinFilmTool().execute(
            {
                "wavelength_um": 2.0,
                "thickness_um": 1.0,
                "refractive_index": 3.5,
                "extinction_coefficient": 0.1,
            }
        )
    )
    assert result.is_error
    assert "meep" in result.output


def test_meep_tool_missing_constants_is_typed():
    from photomatagent.scientific.capabilities.optics.meep_thinfilm import (
        MeepThinFilmTool,
    )

    result = asyncio.run(
        MeepThinFilmTool().execute({"wavelength_um": 2.0, "thickness_um": 1.0})
    )
    assert result.is_error
    assert result.data["error_type"] == "missing_prerequisites"
