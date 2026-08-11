"""Candidate generation capability (namespace ``generation``).

Sprint 3 Phase G: migrate the donor VAE formula generator and MatterGen
wrapper. Generation is NOT property validation (section 83): VAE/MatterGen
candidates are proposals with ``UNVALIDATED_GENERATED_STRUCTURE`` status and
require CHGNet/DFT validation before any stability claim.
"""

from __future__ import annotations
