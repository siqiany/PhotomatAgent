"""Host-side reference for a lossless, normal-incidence single film."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, FiniteFloat

from photomatagent.capability_acquisition.models import StrictModel
from photomatagent.capability_acquisition.sandbox import _sha256_file

if TYPE_CHECKING:
    from photomatagent.application.skills import CheckerSpec


class FresnelFilmInput(StrictModel):
    contract: Literal["lossless_normal_incidence_film"]
    n_incident: FiniteFloat = Field(ge=0.1, le=10)
    n_film: FiniteFloat = Field(ge=0.1, le=10)
    n_exit: FiniteFloat = Field(ge=0.1, le=10)
    thickness_nm: FiniteFloat = Field(ge=0, le=100_000)
    wavelength_nm: FiniteFloat = Field(ge=100, le=100_000)


def check_fresnel_film(spec: CheckerSpec, inputs: dict[str, Path], output: Path) -> dict[str, Any]:
    """Recompute reflectance using the characteristic matrix, independently of candidate code."""
    try:
        data = FresnelFilmInput.model_validate_json(inputs[spec.input_key].read_bytes())
        result = json.loads(output.read_bytes())
        observed = result[spec.field]
        if (type(observed) not in (int, float) or not math.isfinite(observed)
                or result["unit"] != "1" or result["scientific_status"] != "UNKNOWN"):
            raise ValueError("OPTICAL_OUTPUT_INVALID")
        phase = math.tau * data.n_film * data.thickness_nm / data.wavelength_nm
        cosine, sine = math.cos(phase), math.sin(phase)
        electric = cosine + 1j * data.n_exit * sine / data.n_film
        magnetic = data.n_exit * cosine + 1j * data.n_film * sine
        reflected = (data.n_incident * electric - magnetic) / (data.n_incident * electric + magnetic)
        expected = abs(reflected) ** 2
        return {"status": "PASS" if abs(observed - expected) <= 1e-9 else "FAIL",
                "checker_id": spec.checker_id, "method": "independent characteristic matrix",
                "expected": expected, "observed": observed, "unit": "1",
                "input_sha256": _sha256_file(inputs[spec.input_key]), "output_sha256": _sha256_file(output),
                "scientific_status": "UNKNOWN"}
    except (OSError, KeyError, ValueError, TypeError, OverflowError):
        return {"status": "UNKNOWN", "checker_id": spec.checker_id, "gap": "OPTICAL_CONTRACT_OR_EVIDENCE_INVALID"}
