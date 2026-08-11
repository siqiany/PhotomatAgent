"""Application adapters layered on top of the SCNet backend.

Applications (VASP, Hefei-NAMD, MAGUS) know the physics and the file
formats of one scientific program; they never implement generic SSH/Slurm.
The backend (``photomatagent.scientific.remote``) knows SSH/Slurm and never
assumes anything about VASP/NAMD input formats.
"""

from __future__ import annotations
