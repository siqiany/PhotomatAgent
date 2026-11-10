"""A lossless normal-incidence optical reference with no external dependencies."""

from __future__ import annotations

import cmath
import json
import math


def reflectance(n_left: float, n_film: float, n_right: float, phase: float) -> float:
    first = (n_left - n_film) / (n_left + n_film)
    second = (n_film - n_right) / (n_film + n_right)
    phase_factor = cmath.exp(2j * phase)
    return abs((first + second * phase_factor) / (1 + first * second * phase_factor)) ** 2


if __name__ == "__main__":
    result = reflectance(1.0, 2.0, 1.5, phase=0.0)
    direct_interface_reference = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
    if not math.isclose(result, direct_interface_reference, rel_tol=0, abs_tol=1e-12):
        raise SystemExit("Zero-thickness film did not recover the direct interface")
    print(json.dumps({"R": result, "unit": "1"}))
