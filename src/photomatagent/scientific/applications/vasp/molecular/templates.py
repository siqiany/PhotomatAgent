"""Canonical stage templates for isolated-molecule workflows.

Parameter values follow the verified smoke run
(``gel_electrolyte_dft/codex_run/.../tfpma_smoke_corrected_static_clean/``):
PREC=Accurate, EDIFF=1E-6, ISMEAR=0/SIGMA=0.05, ISYM=0, LREAL=.FALSE.,
LASPH/ADDGRID=.TRUE., GGA=PE, IVDW=12, NCORE=2, IDIPOL=4/LDIPOL=.TRUE. with
``DIPOL = 0.5 0.5 0.5``, IBRION=2/ISIF=2 for relax and IBRION=-1/NSW=0 for
static stages. ``total_charge``-derived NELECT is an explicit parameter:
nothing is inferred from names.
"""

from __future__ import annotations

from typing import Any

from photomatagent.scientific.applications.vasp.molecular.models import (
    ResourceClass,
    StageName,
    StageSpec,
)


def _base_incar(
    *,
    spin: int,
    encut: float,
    nelect: float | None,
    lmono: bool,
    dipole: bool,
    dipol: str,
    ncore: int,
    functional: str = "PBE-D3(BJ)",
    nupdown: int | None = None,
    magmom: list[float] | None = None,
) -> dict[str, Any]:
    gga = "PE" if functional.startswith("PBE") else functional.upper()
    settings: dict[str, Any] = {
        "PREC": "Accurate",
        "ENCUT": encut,
        "EDIFF": 1e-6,
        "NELM": 120,
        "ALGO": "Normal",
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "ISPIN": spin,
        "ISYM": 0,
        "LREAL": False,
        "LASPH": True,
        "ADDGRID": True,
        "GGA": gga,
        "IVDW": 12,
        "NCORE": ncore,
    }
    if dipole:
        settings.update({"IDIPOL": 4, "LDIPOL": True, "DIPOL": dipol})
    if lmono:
        settings["LMONO"] = True
    if nelect is not None:
        settings["NELECT"] = nelect
    if nupdown is not None:
        settings["NUPDOWN"] = nupdown
    if magmom is not None:
        settings["MAGMOM"] = magmom
    return settings


def relax_incar(
    *,
    spin: int = 1,
    encut: float = 520.0,
    nelect: float | None = None,
    lmono: bool = False,
    dipole: bool = True,
    dipol: str = "0.5 0.5 0.5",
    ncore: int = 2,
    nsw: int = 200,
    nupdown: int | None = None,
    magmom: list[float] | None = None,
) -> dict[str, Any]:
    settings = _base_incar(
        spin=spin,
        encut=encut,
        nelect=nelect,
        lmono=lmono,
        dipole=dipole,
        dipol=dipol,
        ncore=ncore,
        nupdown=nupdown,
        magmom=magmom,
    )
    settings.update(
        {
            "IBRION": 2,
            "NSW": nsw,
            "EDIFFG": -0.02,
            "ISIF": 2,  # fixed vacuum box: the cell must not relax
            "LWAVE": False,
            "LCHARG": True,
        }
    )
    return settings


def static_incar(
    *,
    spin: int = 1,
    encut: float = 520.0,
    nelect: float | None = None,
    lmono: bool = False,
    dipole: bool = True,
    dipol: str = "0.5 0.5 0.5",
    ncore: int = 2,
) -> dict[str, Any]:
    settings = _base_incar(
        spin=spin,
        encut=encut,
        nelect=nelect,
        lmono=lmono,
        dipole=dipole,
        dipol=dipol,
        ncore=ncore,
    )
    settings.update(
        {
            "IBRION": -1,
            "NSW": 0,
            "ICHARG": 1,  # reuse CHGCAR from the relax stage
            "LWAVE": True,  # seed WAVECAR for HSE06/orbital stages
            "LCHARG": True,
        }
    )
    return settings


def static_preconverge_incar(
    *,
    spin: int = 1,
    encut: float = 520.0,
    nelect: float | None = None,
    lmono: bool = False,
    dipole: bool = False,  # deliberate default: cheap WAVECAR/CHGCAR source
    dipol: str = "0.5 0.5 0.5",
    ncore: int = 2,
    ediff: float = 1e-5,  # looser than the corrected static stage
) -> dict[str, Any]:
    """Preconvergence single point: produces clean WAVECAR/CHGCAR seeds.

    Energies from this stage are never production values; they only restart
    the corrected static stage. The dipole correction is OFF by default.
    """
    settings = _base_incar(
        spin=spin,
        encut=encut,
        nelect=nelect,
        lmono=lmono,
        dipole=dipole,
        dipol=dipol,
        ncore=ncore,
    )
    settings.update(
        {
            "EDIFF": ediff,
            "IBRION": -1,
            "NSW": 0,
            "ISTART": 0,
            "ICHARG": 1,  # reuses the relax-stage CHGCAR
            "LWAVE": True,
            "LCHARG": True,
        }
    )
    return settings


def corrected_static_incar(
    *,
    spin: int = 1,
    encut: float = 520.0,
    nelect: float | None = None,
    lmono: bool = False,
    dipole: bool = True,
    dipol: str = "0.5 0.5 0.5",
    ncore: int = 2,
    ediff: float = 1e-6,  # stricter than the preconvergence stage
) -> dict[str, Any]:
    """Corrected single point restarting from the preconvergence stage."""
    settings = _base_incar(
        spin=spin,
        encut=encut,
        nelect=nelect,
        lmono=lmono,
        dipole=dipole,
        dipol=dipol,
        ncore=ncore,
    )
    settings.update(
        {
            "EDIFF": ediff,
            "IBRION": -1,
            "NSW": 0,
            "ISTART": 1,  # WAVECAR from static_preconverge
            "ICHARG": 1,  # CHGCAR from static_preconverge
            "LWAVE": True,
            "LCHARG": True,
        }
    )
    return settings


def orbital_single_incar(
    *,
    spin: int = 1,
    encut: float = 520.0,
    nelect: float | None = None,
    lmono: bool = False,
    dipole: bool = True,
    dipol: str = "0.5 0.5 0.5",
    ncore: int = 2,
    iband: int,
) -> dict[str, Any]:
    """HOMO or LUMO single point: IBAND + LVHAR LOCPOT for vacuum alignment."""
    settings = _base_incar(
        spin=spin,
        encut=encut,
        nelect=nelect,
        lmono=lmono,
        dipole=dipole,
        dipol=dipol,
        ncore=ncore,
    )
    settings.update(
        {
            "IBRION": -1,
            "NSW": 0,
            "ISTART": 1,
            "ICHARG": 1,
            "LPARD": True,
            "LSEPB": True,
            "LSEPK": True,
            "KPUSE": 1,
            "IBAND": iband,
            "LVHAR": True,  # LOCPOT = ionic + Hartree only (vacuum level)
            "LWAVE": False,
            "LCHARG": False,
        }
    )
    return settings


def static_hse_incar(
    *,
    spin: int = 1,
    encut: float = 520.0,
    nelect: float | None = None,
    lmono: bool = False,
    dipole: bool = True,
    dipol: str = "0.5 0.5 0.5",
    ncore: int = 2,
) -> dict[str, Any]:
    settings = _base_incar(
        spin=spin,
        encut=encut,
        nelect=nelect,
        lmono=lmono,
        dipole=dipole,
        dipol=dipol,
        ncore=ncore,
    )
    settings.update(
        {
            "IBRION": -1,
            "NSW": 0,
            "ISTART": 1,
            "ICHARG": 1,
            "LHFCALC": True,
            "AEXX": 0.25,
            "HFSCREEN": 0.2,  # HSE06 screening
            "ALGO": "Damped",
            "PRECFOCK": "Fast",
            "LWAVE": False,
            "LCHARG": False,
        }
    )
    return settings


def orbital_incar(
    *,
    spin: int = 1,
    encut: float = 520.0,
    nelect: float | None = None,
    lmono: bool = False,
    dipole: bool = True,
    dipol: str = "0.5 0.5 0.5",
    ncore: int = 2,
    iband: int | list[int] | None = None,
) -> dict[str, Any]:
    settings = _base_incar(
        spin=spin,
        encut=encut,
        nelect=nelect,
        lmono=lmono,
        dipole=dipole,
        dipol=dipol,
        ncore=ncore,
    )
    settings.update(
        {
            "IBRION": -1,
            "NSW": 0,
            "ICHARG": 11,
            "LPARD": True,
            "LSEPB": True,
            "LSEPK": True,
            "KPUSE": 1,
            "LWAVE": False,
            "LCHARG": False,
        }
    )
    if iband is None:
        raise ValueError(
            "orbital stage requires IBAND (HOMO/LUMO band indices read from "
            "the converged static run)"
        )
    settings["IBAND"] = iband
    return settings


def esp_incar(
    *,
    spin: int = 1,
    encut: float = 520.0,
    nelect: float | None = None,
    lmono: bool = False,
    dipole: bool = True,
    dipol: str = "0.5 0.5 0.5",
    ncore: int = 2,
) -> dict[str, Any]:
    settings = _base_incar(
        spin=spin,
        encut=encut,
        nelect=nelect,
        lmono=lmono,
        dipole=dipole,
        dipol=dipol,
        ncore=ncore,
    )
    settings.update(
        {
            "IBRION": -1,
            "NSW": 0,
            "ICHARG": 1,
            "LVHAR": True,  # LOCPOT = ionic + Hartree potential only (ESP)
            "LWAVE": True,
            "LCHARG": True,
        }
    )
    return settings


def restart_incar(
    *,
    spin: int = 1,
    encut: float = 520.0,
    nelect: float | None = None,
    lmono: bool = False,
    dipole: bool = True,
    dipol: str = "0.5 0.5 0.5",
    ncore: int = 2,
) -> dict[str, Any]:
    settings = _base_incar(
        spin=spin,
        encut=encut,
        nelect=nelect,
        lmono=lmono,
        dipole=dipole,
        dipol=dipol,
        ncore=ncore,
    )
    settings.update(
        {
            "IBRION": -1,
            "NSW": 0,
            "ISTART": 1,
            "ICHARG": 1,
            "LWAVE": True,
            "LCHARG": True,
        }
    )
    return settings


def make_stage(
    name: StageName,
    *,
    incar: dict[str, Any],
    depends_on: StageName | None = None,
    required_upstream_outputs: list[str] | None = None,
    produced_outputs: list[str] | None = None,
    description: str = "",
    resource_class: ResourceClass | str = ResourceClass.STANDARD,
) -> StageSpec:
    """Build one typed stage spec."""
    cls = ResourceClass(resource_class)
    return StageSpec(
        name=name,
        depends_on=depends_on,
        description=description,
        required_upstream_outputs=required_upstream_outputs or [],
        produced_outputs=produced_outputs or [],
        incar=incar,
        resource_class=cls,
    )
