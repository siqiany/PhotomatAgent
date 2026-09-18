import pytest
from pydantic import ValidationError

from photomatagent.scientific.capabilities.structure.construction_models import (
    ConstructionLimits, OrderingRequest, SiteReplacement, SubstitutionRequest,
    SupercellRequest,
)
from photomatagent.scientific.discovery.structures import StructureDerivation


def test_contracts_are_strict_and_bounded():
    assert SupercellRequest(path="a.cif", scaling=(1, 2, 1), task_slug="task").scaling == (1, 2, 1)
    with pytest.raises(ValidationError):
        SupercellRequest(path="a.cif", scaling=(0, 1, 1), task_slug="task")
    with pytest.raises(ValidationError):
        SupercellRequest(path="a.cif", scaling=(1, 1, 1), task_slug="../bad")
    with pytest.raises(ValidationError):
        SiteReplacement(index=0, from_element="Na", to_element="Na")
    with pytest.raises(ValidationError):
        SubstitutionRequest(path="a.cif", replacements=[{"index": 0, "from_element": "Na", "to_element": "Ag"}], expected_formula="NaAg", hypothesis_id="h", task_slug="x", extra="bad")
    with pytest.raises(ValidationError):
        OrderingRequest(path="a.cif", eligible_indices=[1, 1], from_element="Na", to_element="Ag", replacement_count=1, expected_formula="NaAg", hypothesis_id="h", task_slug="x")
    with pytest.raises(ValidationError):
        ConstructionLimits(max_atoms=513)

@pytest.mark.parametrize("value", [1.0, "1", True])
def test_construction_rejects_pseudo_integer_scaling(value):
    with pytest.raises(ValidationError):
        SupercellRequest(path="a.cif", scaling=(value, 1, 1), task_slug="x")

def test_derivation_is_strict_and_deeply_immutable():
    params = {"nested": {"x": 1}}
    origin = {"tool": "trusted"}
    lineage = {
        "candidate_id": "cand_x",
        "generation_parameters": {"seed": 1},
        "source_artifacts": ["input.cif"],
    }
    record = StructureDerivation(id="der_x", candidate_id="cand_x", input_sha256="a"*64, structure_hash="b"*64, output_path="user_output/t/structures/op_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/structure_0000.cif", operation="make_supercell", normalized_composition=(("Si", 1),), lineage=lineage, parameters=params, origin=origin)
    params["nested"]["x"] = 2
    origin["tool"] = "changed"
    assert record.parameters["nested"]["x"] == 1
    with pytest.raises(TypeError):
        record.parameters["nested"]["x"] = 3
    with pytest.raises(TypeError):
        record.parameters |= {"new": 1}
    with pytest.raises(TypeError):
        record.origin["tool"] = "changed"
    with pytest.raises(TypeError):
        record.lineage.generation_parameters["seed"] = 2
    with pytest.raises(TypeError):
        record.lineage.source_artifacts += ["other.cif"]
    assert record.model_dump(mode="json")["structure_hash"] == "b"*64

def test_derivation_rejects_non_json_like_nested_values():
    with pytest.raises(TypeError):
        StructureDerivation(id="d", candidate_id="c", input_sha256="a"*64, structure_hash="b"*64, output_path="user_output/x", operation="make_supercell", normalized_composition=(("Si", 1),), lineage={"candidate_id": "c"}, parameters={"bad": {1, 2}})

@pytest.mark.parametrize("composition", [
    (("Xx", 1),),
    tuple((symbol, 1) for symbol in (
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    )),
])
def test_derivation_rejects_noncanonical_composition(composition):
    with pytest.raises((ValidationError, ValueError)):
        StructureDerivation(id="d", candidate_id="c", input_sha256="a"*64, structure_hash="b"*64, output_path="user_output/x/structures/op_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/structure_0000.cif", operation="make_supercell", normalized_composition=composition, lineage={"candidate_id": "c"})

def test_derivation_rejects_inconsistent_lineage():
    with pytest.raises(ValidationError):
        StructureDerivation(id="d", candidate_id="c", input_sha256="a"*64, structure_hash="b"*64, output_path="user_output/x/structures/op_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/structure_0000.cif", operation="make_supercell", normalized_composition=(("Si", 1),), lineage={"candidate_id": "other"})


def test_derivation_model_copy_revalidates_identity_and_freezes_updates():
    record = StructureDerivation(
        id="d",
        candidate_id="c",
        input_sha256="a" * 64,
        structure_hash="b" * 64,
        output_path="user_output/x/structures/op_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/structure_0000.cif",
        operation="make_supercell",
        normalized_composition=(("Si", 1),),
        lineage={"candidate_id": "c"},
    )
    with pytest.raises(ValidationError):
        record.model_copy(update={"candidate_id": "attacker"})
    caller_origin = {"source": "caller"}
    copied = record.model_copy(update={"origin": caller_origin})
    caller_origin["late"] = "mutation"
    assert copied.origin == {"source": "caller"}
    with pytest.raises(TypeError):
        copied.origin["x"] = "blocked"

@pytest.mark.parametrize("path", [
    "", "../x", "/tmp/x", "C:\\tmp\\x", "user_output//x", "user_output/./x",
    "tmp/foo.cif",
    "user_output/task/structures/op_bad/structure_0000.cif",
    "user_output/task/structures/op_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/bad.cif",
])
def test_derivation_rejects_unsafe_output_path(path):
    with pytest.raises(ValidationError):
        StructureDerivation(
            id="d",
            candidate_id="c",
            input_sha256="a" * 64,
            structure_hash="b" * 64,
            output_path=path,
            operation="make_supercell",
            normalized_composition=(("Si", 1),),
            lineage={"candidate_id": "c"},
        )
