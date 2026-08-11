"""MAGUS structure-search application adapter (namespace ``magus``).

MAGUS is an optional external application (evolutionary structure
search, Xia et al., Comput. Phys. Commun.). It is never assumed to be
installed: the probe reports UNCONFIGURED with an installation /
registration requirement, and PhotoMatAgent starts normally regardless.
MAGUS candidates are proposals only -- they still require CHGNet / DFT
validation before any stability claim (section 42).
"""

from __future__ import annotations
