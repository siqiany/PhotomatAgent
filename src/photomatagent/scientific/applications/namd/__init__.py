"""Hefei-NAMD application adapter (namespace ``namd``).

Sprint 3 Phase E: a reliable scientific backend, not an autonomous NAMD
research expert. Everything is gated on the SCNet environment probe:
without a confirmed Hefei-NAMD module the adapter reports
UNCONFIGURED / NEEDS_CONFIGURATION and never fabricates input formats or
carrier-dynamics numbers.

Requirements recorded from the official Hefei-NAMD training material
(Qijing Zheng, USTC) and repository:
* reference POSCAR + XDATCAR trajectory + OUTCAR from a VASP AIMD run
* one WAVECAR per selected MD snapshot, all identical in size (same
  cell/ENCUT/NBANDS/k-mesh/spin)
* runtime inputs ``inp`` + ``INICON`` are generated only after the module
  has been confirmed (their exact format is version-dependent)
"""

from __future__ import annotations
