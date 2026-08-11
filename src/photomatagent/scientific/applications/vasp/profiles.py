"""VASP calculation profiles (Sprint 3 section 24-28).

Every INCAR setting below carries a documented source. Settings were
checked against the official VASP wiki (vasp.at/wiki) entries for
LSORBIT / LNONCOLLINEAR / LOPTICS / IBRION=0 (MD) and against the
Hefei-NAMD training documentation (Qijing Zheng, staff.ustc.edu.cn) and
the official Hefei-NAMD repository (QijingZheng/Hefei-NAMD). Do not edit
these values from memory without re-checking the cited sources.

Profile semantics:
* ``standard_semiconductor`` -- relax -> static -> band -> dos for ordinary
  semiconductors (donor migration, section 25).
* ``narrow_gap_soc`` -- strong-SOC narrow-gap systems (HgTe, HgCdTe, PbTe,
  PbSe, Bi-containing candidates) with explicit noncollinear SOC settings
  (section 26). Note the wiki requirement: LSORBIT=.TRUE. implies
  LNONCOLLINEAR=.TRUE. and requires the ``vasp_ncl`` binary; GGA_COMPAT
  must be .FALSE. with LASPH=.TRUE.; LREAL=.FALSE. is required; ISYM=-1 is
  recommended for final SOC runs; ISPIN=2 is not allowed together with
  noncollinear magnetism.
* ``optics`` -- LOPTICS workflow producing the dielectric spectrum in
  vasprun.xml (section 27); Meep consumes n/k derived from it.
* ``namd_preparation`` -- VASP MD trajectory + per-snapshot WAVECAR
  generation for Hefei-NAMD (section 28). Requirements documented from the
  official training notes: IBRION=0, NBLOCK=1 (every frame written to
  XDATCAR), LWAVE=.TRUE., consistent ENCUT/NBANDS/k-mesh/spin across the
  whole trajectory so every WAVECAR has the same size, and a static
  continuation at each selected frame for a consistent wavefunction. This
  profile is gated: it reports NEEDS_CONFIGURATION until the SCNet
  Hefei-NAMD module/environment has been confirmed by a probe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from photomatagent.scientific.remote.models import ResourceRequest


@dataclass(frozen=True)
class VaspProfile:
    """One named VASP calculation profile with sourced settings."""

    name: str
    description: str
    executable: str
    soc: bool
    stages: list[str]
    base_incar: dict[str, Any]
    default_resource: ResourceRequest
    sources: list[str]
    limitations: list[str] = field(default_factory=list)
    needs_configuration: bool = False


_WIKI_SOC = (
    "VASP wiki: LSORBIT page (LSORBIT=.TRUE. implies LNONCOLLINEAR=.TRUE.; "
    "use vasp_ncl; GGA_COMPAT=.FALSE., LASPH=.TRUE., LREAL=.FALSE.; ISYM=-1 "
    "recommended for final SOC runs; ISPIN=2 not allowed with noncollinear "
    "magnetism)"
)
_WIKI_OPTICS = (
    "VASP wiki: LOPTICS page (LOPTICS=.TRUE. computes the frequency "
    "dependent dielectric tensor in vasprun.xml; NEDOS/CSHIFT control the "
    "spectral grid)"
)
_WIKI_MD = (
    "VASP wiki: IBRION=0 molecular dynamics page (POTIM in fs, NSW steps, "
    "TEBEG/TEEND thermostat, SMASS Nose mass, NBLOCK frames per XDATCAR "
    "block, LWAVE=.TRUE. writes WAVECAR)"
)
_HEFEI_NAMD = (
    "Hefei-NAMD official training/repository (Qijing Zheng): requires "
    "reference POSCAR + XDATCAR trajectory + OUTCAR + one consistent "
    "WAVECAR per MD snapshot; inp + INICON are the NAMD runtime inputs; "
    "consistent cell/ENCUT/NBANDS/spin/k-mesh across all snapshots"
)


STANDARD_SEMICONDUCTOR = VaspProfile(
    name="standard_semiconductor",
    description=(
        "Relax -> static -> band -> dos workflow for ordinary "
        "semiconductors (donor migration; no SOC)."
    ),
    executable="vasp_std",
    soc=False,
    stages=["relax", "static", "band", "dos"],
    base_incar={
        "PREC": "Accurate",
        "ENCUT": 520,
        "EDIFF": 1e-5,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "ISPIN": 1,
        "LORBIT": 11,
    },
    default_resource=ResourceRequest(
        partition="kshcnormal", nodes=1, tasks_per_node=32, walltime_minutes=240
    ),
    sources=[
        "donor repository VaspInputGenerator (photoelectric-detection)",
        "VASP wiki: PREC/ENCUT/EDIFF/ISMEAR/SIGMA pages",
    ],
    limitations=[
        "ISMEAR=0 with SIGMA=0.05 targets semiconductors; metallic systems "
        "need ISMEAR=1/2 with a larger SIGMA",
        "no spin-orbit coupling; use narrow_gap_soc for strong-SOC systems",
    ],
)


NARROW_GAP_SOC = VaspProfile(
    name="narrow_gap_soc",
    description=(
        "Strong-SOC narrow-gap workflow (HgTe, HgCdTe, PbTe, PbSe, "
        "Bi-containing candidates) with explicit noncollinear SOC settings. "
        "Not applied automatically by element: the caller must select it "
        "for systems where SOC matters."
    ),
    executable="vasp_ncl",
    soc=True,
    stages=["relax", "static", "band", "dos"],
    base_incar={
        "PREC": "Accurate",
        "ENCUT": 520,
        "EDIFF": 1e-5,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "ISPIN": 1,  # noncollinear SOC is incompatible with ISPIN=2
        "LNONCOLLINEAR": True,  # implied by LSORBIT; kept explicit
        "LSORBIT": True,
        "GGA_COMPAT": False,
        "LASPH": True,
        "LREAL": False,
        "ISYM": -1,
        "LMAXMIX": 4,  # 4 for d electrons; set 6 for f-electron systems
        "LORBIT": 11,
        # SAXIS defaults to (0,0,1); set explicitly only when a specific
        # magnetization axis is required.
    },
    default_resource=ResourceRequest(
        partition="kshcnormal", nodes=1, tasks_per_node=32, walltime_minutes=480
    ),
    sources=[_WIKI_SOC],
    limitations=[
        "requires the vasp_ncl binary on SCNet (probed at submit time)",
        "LMAXMIX=4 targets d-electron systems; f-electron systems (rare "
        "earth) need LMAXMIX=6",
        "LSORBIT with relaxation is expensive; relax first without SOC when "
        "structure optimization dominates",
        "ISMEAR=0/SIGMA=0.05 is a starting point for narrow gaps; check "
        "occupation smearing in OUTCAR for each system",
    ],
)


OPTICS = VaspProfile(
    name="optics",
    description=(
        "Static SCF with LOPTICS producing the frequency-dependent "
        "dielectric tensor in vasprun.xml for n/k -> Meep."
    ),
    executable="vasp_std",
    soc=False,
    stages=["static", "optics"],
    base_incar={
        "PREC": "Accurate",
        "ENCUT": 520,
        "EDIFF": 1e-5,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "ISPIN": 1,
        "LREAL": False,
        "LOPTICS": True,
        "NEDOS": 2000,
        "CSHIFT": 0.1,
        "IBRION": -1,
        "NSW": 0,
        "LWAVE": False,
        "LCHARG": False,
    },
    default_resource=ResourceRequest(
        partition="kshcnormal", nodes=1, tasks_per_node=32, walltime_minutes=240
    ),
    sources=[_WIKI_OPTICS],
    limitations=[
        "optical spectrum needs a converged ground state (run "
        "standard_semiconductor static first or provide CHGCAR)",
        "no local-field effects in the independent-particle response",
        "SOC is off; combine with narrow_gap_soc settings manually when "
        "spin-orbit coupling affects the optical gap",
    ],
)


NAMD_PREPARATION = VaspProfile(
    name="namd_preparation",
    description=(
        "VASP AIMD trajectory + per-snapshot WAVECAR generation for "
        "Hefei-NAMD (carrier dynamics input). Requires confirmation of the "
        "SCNet Hefei-NAMD environment before production use."
    ),
    executable="vasp_std",
    soc=False,
    stages=["md", "snapshot"],
    base_incar={
        "PREC": "Accurate",
        "ENCUT": 500,
        "EDIFF": 1e-6,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "ISPIN": 1,
        "IBRION": 0,  # molecular dynamics
        "NSW": 1000,
        "POTIM": 1.0,  # fs; must match the NAMD time step
        "TEBEG": 300,
        "TEEND": 300,
        "SMASS": 0,
        "ISIF": 2,
        "NBLOCK": 1,  # every MD frame written to XDATCAR
        "LWAVE": True,  # WAVECAR written during the MD
        "LCHARG": False,
    },
    default_resource=ResourceRequest(
        partition="kshcnormal", nodes=1, tasks_per_node=32, walltime_minutes=1440
    ),
    sources=[_WIKI_MD, _HEFEI_NAMD],
    limitations=[
        "WAVECAR from MD frames is a predicted wavefunction; Hefei-NAMD "
        "preprocessing may require a static continuation at each selected "
        "geometry for a consistent WAVECAR (see the official scripts)",
        "every snapshot WAVECAR must have identical size: same "
        "cell/ENCUT/NBANDS/k-mesh/spin across the whole trajectory",
        "avoid Gamma-only k-points unless the installed Hefei-NAMD "
        "explicitly supports the reduced WAVECAR format",
        "include enough empty bands (NBANDS) for the excited-state window",
        "profile is gated: NEEDS_CONFIGURATION until the SCNet Hefei-NAMD "
        "module is confirmed by probe_environment",
    ],
    needs_configuration=True,
)


def profiles() -> list[VaspProfile]:
    return [STANDARD_SEMICONDUCTOR, NARROW_GAP_SOC, OPTICS, NAMD_PREPARATION]


def get_profile(name: str) -> VaspProfile:
    for profile in profiles():
        if profile.name == name:
            return profile
    raise ValueError(
        f"unknown VASP profile {name!r}; available: "
        + ", ".join(profile.name for profile in profiles())
    )
