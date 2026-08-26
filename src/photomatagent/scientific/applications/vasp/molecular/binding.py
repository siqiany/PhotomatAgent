"""Electronic binding energies with strict parameter-consistency checks.

Only electronic ΔE is computed (no vibrational or thermal corrections).
Two reference decompositions may be given; the difference between them is
reported as ΔΔE, which reduces the bare-ion reference error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class BindingReference(BaseModel):
    """One reference system (fragment/ion/neutral piece) of a binding energy."""

    name: str
    results_dir: str
    charge: int
    role: str = "fragment"  # fragment | ion | neutral | ...


class BindingEnergyInput(BaseModel):
    complex_name: str
    complex_dir: str
    references: list[BindingReference] = Field(min_length=1)
    alternative_references: list[BindingReference] = Field(default_factory=list)
    charge: int = 0  # complex charge; refs must sum to it


def _identity_of(directory: Path) -> dict[str, Any]:
    """Read the persisted results.json identity + method/energy."""
    results = json_load(directory / "results.json")
    if results is None:
        raise ValueError(f"results.json missing in {directory}")
    if not results.get("validated"):
        raise ValueError(
            f"results in {directory} are not validated; binding energies "
            "require validated E0 values"
        )
    energy = results.get("energy") or {}
    e0 = energy.get("e_0_ev")
    if e0 is None:
        raise ValueError(f"results in {directory} carry no E0")
    identity = results.get("identity") or {}
    method = results.get("method") or {}
    corrections = results.get("corrections") or {}
    return {
        "formula": identity.get("formula", "?"),
        "e0_ev": float(e0),
        "box_ang": method.get("box_ang"),
        "functional": method.get("functional"),
        "encut_ev": method.get("encut_ev"),
        "corrections": corrections,
        "explicit_reference_assumption": bool(
            results.get("explicit_reference_assumption")
        ),
        "reference_kind": results.get("reference_kind", ""),
        "not_a_vasp_result": bool(results.get("not_a_vasp_result")),
        "results_dir": str(directory),
    }


def json_load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _charge_signature(directory: Path) -> int | None:
    """Charge implied by the results' declared NELECT/meta (parity check)."""
    results = json_load(directory / "results.json")
    if results is None:
        return None
    identity = results.get("identity") or {}
    meta = Path(directory) / "POTCAR.meta"
    if meta.is_file():
        import json

        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
            neutral = payload.get("neutral_valence_electrons")
            nelect = payload.get("nelect")
            if neutral is not None and nelect is not None:
                return int(round(neutral - float(nelect)))
        except Exception:
            pass
    return identity.get("charge")


def compute_binding_energy(
    inputs: BindingEnergyInput,
) -> dict[str, Any]:
    """Compute electronic ΔE (+ΔΔE) with consistency checks and components."""
    complex_dir = Path(inputs.complex_dir).expanduser().resolve()
    complex_id = _identity_of(complex_dir)
    errors: list[str] = []
    warnings: list[str] = []

    def check_reference(binding: BindingReference) -> dict[str, Any]:
        directory = Path(binding.results_dir).expanduser().resolve()
        try:
            identity = _identity_of(directory)
        except ValueError as exc:
            errors.append(f"reference {binding.name}: {exc}")
            return {}
        for key, label, numeric in (
            ("box_ang", "box", True),
            ("functional", "functional", False),
            ("encut_ev", "ENCUT", True),
        ):
            complex_value = complex_id.get(key)
            reference_value = identity.get(key)
            if complex_value is None or reference_value is None:
                continue
            if numeric:
                mismatch = abs(float(complex_value) - float(reference_value)) > 1e-9
            else:
                mismatch = str(complex_value) != str(reference_value)
            if mismatch:
                errors.append(
                    f"{label} mismatch: complex {complex_value} vs "
                    f"reference {binding.name} {reference_value}"
                )
        charge = _charge_signature(directory)
        if charge is not None and binding.charge is not None and charge != binding.charge:
            errors.append(
                f"reference {binding.name}: declared charge {binding.charge} "
                f"conflicts with its own NELECT metadata ({charge})"
            )
        return identity

    resolved_primary: list[tuple[BindingReference, dict[str, Any]]] = []
    resolved_alt: list[tuple[BindingReference, dict[str, Any]]] = []
    reference_assumptions: list[str] = []
    primary_sum = 0.0
    for binding in inputs.references:
        identity = check_reference(binding)
        if not identity:
            continue
        primary_sum += identity["e0_ev"]
        if identity.get("explicit_reference_assumption"):
            reference_assumptions.append(f"{binding.name}")
        resolved_primary.append((binding, identity))
    declared_ref_charge = sum(ref.charge for ref in inputs.references)
    if inputs.alternative_references:
        declared_alt_charge = sum(ref.charge for ref in inputs.alternative_references)
        if declared_alt_charge != declared_ref_charge:
            errors.append(
                "alternative reference set must conserve the same total "
                f"charge ({declared_ref_charge} vs {declared_alt_charge})"
            )
    alt_sum = 0.0
    for binding in inputs.alternative_references:
        identity = check_reference(binding)
        if not identity:
            continue
        alt_sum += identity["e0_ev"]
        if identity.get("explicit_reference_assumption"):
            reference_assumptions.append(binding.name)
        resolved_alt.append((binding, identity))
    if declared_ref_charge != inputs.charge:
        errors.append(
            "complex charge is not conserved: "
            f"Σ(reference charges) = {declared_ref_charge} != complex {inputs.charge}"
        )

    delta_e = complex_id["e0_ev"] - primary_sum if resolved_primary else None
    delta_delta_e = None
    if resolved_alt and delta_e is not None:
        alternative_delta_e = complex_id["e0_ev"] - alt_sum
        delta_delta_e = delta_e - alternative_delta_e
    if delta_e is None:
        errors.append("no complete primary reference set; ΔE undefined")

    primary_components = []
    for binding, identity in resolved_primary:
        primary_components.append(
            {
                "name": binding.name,
                "role": binding.role,
                "charge": binding.charge,
                "formula": identity.get("formula", "?"),
                "e0_ev": identity.get("e0_ev"),
                "explicit_reference_assumption": bool(
                    identity.get("explicit_reference_assumption")
                ),
                "not_a_vasp_result": bool(identity.get("not_a_vasp_result")),
            }
        )
    alternative_components = []
    for binding, identity in resolved_alt:
        alternative_components.append(
            {
                "name": binding.name,
                "role": binding.role,
                "charge": binding.charge,
                "formula": identity.get("formula", "?"),
                "e0_ev": identity.get("e0_ev"),
                "explicit_reference_assumption": bool(
                    identity.get("explicit_reference_assumption")
                ),
            }
        )

    mentions_reference_assumption = bool(reference_assumptions)
    if mentions_reference_assumption:
        warnings.append(
            "declared reference model(s) present (E=0 convention): "
            + ", ".join(sorted(set(reference_assumptions)))
            + "; ΔΔE / ligand-exchange schemes are preferred over absolute "
            "binding energies, which carry high bare-ion reference risk"
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "method": {
            "kind": "electronic-only binding energy (no vibrational/thermal corrections)",
            "vasp": "Gamma-only fixed-box PBE-D3(BJ)",
            "corrections": complex_id["corrections"],
        },
        "complex": {
            "name": inputs.complex_name,
            "formula": complex_id["formula"],
            "charge": inputs.charge,
            "e0_ev": complex_id["e0_ev"],
        },
        "components": {
            "primary": primary_components,
            "alternative": alternative_components,
        },
        "results": {
            "delta_e_ev": delta_e,
            "delta_delta_e_ev": delta_delta_e,
            "definition": "ΔE = E0(complex) - Σ E0(references)",
            "definition_dde": "ΔΔE = ΔE(primary scheme) - ΔE(alternative scheme)",
            "electronic_only": True,
            "uses_declared_reference_assumption": mentions_reference_assumption,
            "high_risk_absolute_binding_energy": mentions_reference_assumption,
        },
        "limitations": [
            "electronic binding only; zero-point, thermal and entropy terms "
            "are not included",
            "no basis-set superposition correction",
            "fragment geometries are the ones actually computed; deformation "
            "energies are reported only if the fragments were relaxed",
        ],
        "reference_assumptions": sorted(set(reference_assumptions)),
    }
