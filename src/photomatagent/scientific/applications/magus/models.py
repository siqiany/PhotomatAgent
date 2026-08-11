"""Typed MAGUS request models (Sprint 4, sections 18-23).

These are PhotoMatAgent domain models, NOT the raw MAGUS YAML schema: the
deterministic renderer (``render.py``) maps them onto the keys actually
accepted by the installed MAGUS version. Composition parsing is delegated
to pymatgen so the LLM never parses chemical formulas by hand.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

StructureType = Literal["bulk", "cluster", "surface"]
FormulaType = Literal["fix", "var"]
CalculatorName = Literal["vasp", "emt", "lj", "gulp", "mtp", "abacus", "espresso"]
ExecutionMode = Literal["serial", "parallel"]

# Structure types MAGUS 2.1.0 demonstrably supports (from the installed
# official examples: bulk examples 01--*/03--*, cluster example 05--1-LJ26,
# surface examples 06--*). ``probe_environment`` re-verifies against the
# installed CLI/examples; a type is only exposed when confirmed.
SUPPORTED_STRUCTURE_TYPES: tuple[str, ...] = ("bulk", "cluster", "surface")


class MagusComposition(BaseModel):
    """Deterministic element/formula decomposition of one composition."""

    symbols: list[str] = Field(min_length=1)
    formula: list[int] = Field(min_length=1)
    formula_type: FormulaType = "fix"

    @classmethod
    def from_formula(
        cls, composition: str, formula_type: FormulaType = "fix"
    ) -> "MagusComposition":
        """Parse ``composition`` deterministically with pymatgen.

        ``InAs`` -> symbols ["In", "As"], formula [1, 1]. The element order
        is pymatgen's canonical order (ascending electronegativity), so the
        renderer and the pseudopotential checker always agree.
        """
        from pymatgen.core import Composition

        if not composition or not composition.strip():
            raise ValueError("composition must not be empty")
        parsed = Composition(composition)
        amounts = parsed.get_el_amt_dict()
        symbols = [str(element.symbol) for element in parsed.elements]
        formula = [int(round(amounts[symbol])) for symbol in symbols]
        if any(value < 1 for value in formula):
            raise ValueError(f"composition {composition!r} has a non-positive amount")
        return cls(symbols=symbols, formula=formula, formula_type=formula_type)

    @field_validator("symbols")
    @classmethod
    def _symbols_are_elements(cls, symbols: list[str]) -> list[str]:
        from pymatgen.core import Element

        for symbol in symbols:
            try:
                Element(symbol)
            except Exception as exc:
                raise ValueError(f"not an element symbol: {symbol!r}") from exc
        return symbols

    @field_validator("formula")
    @classmethod
    def _formula_positive(cls, formula: list[int]) -> list[int]:
        if any(value < 1 for value in formula):
            raise ValueError("formula amounts must be >= 1")
        return formula


class MagusSlabConfig(BaseModel):
    """Surface-reconstruction slab metadata (MAGUS 2.1.0 06--* examples)."""

    bulk_file: str = Field(min_length=1)  # remote POSCAR of the bulk structure
    cutslices: int = Field(default=2, ge=1)
    bulk_layernum: int = Field(default=2, ge=1)
    buffer_layernum: int = Field(default=1, ge=0)
    rcs_layernum: int = Field(default=1, ge=0)
    direction: list[int] = Field(default_factory=lambda: [1, 0, 0])
    vacuum_thickness: float = Field(default=10.0, gt=0)
    spg_type: str = "plane"
    rcs_x: list[int] = Field(default_factory=lambda: [2])
    rcs_y: list[int] = Field(default_factory=lambda: [1])


class MagusPseudopotentialRequirement(BaseModel):
    """One required POTCAR setup: element + explicit setup label."""

    element: str
    setup: str = ""
    resolved_path: str = ""


class _MagusBaseRequest(BaseModel):
    structure_type: StructureType = "bulk"
    composition: MagusComposition
    min_atoms: int | None = Field(default=None, ge=1)
    max_atoms: int | None = Field(default=None, ge=1)
    spacegroup: list[str] = Field(default_factory=lambda: ["1-230"])

    @field_validator("max_atoms")
    @classmethod
    def _max_atoms_ge_min(cls, value: int | None, info: Any) -> int | None:
        if value is not None and info.data.get("min_atoms") is not None:
            if value < info.data["min_atoms"]:
                raise ValueError("max_atoms must be >= min_atoms")
        return value


class MagusGenerateRequest(_MagusBaseRequest):
    """Structure-generation request (``magus generate``; no calculator)."""

    number: int = Field(default=5, ge=1, le=100)
    d_ratio: float = Field(default=0.5, gt=0)
    volume_ratio: float = Field(default=8.0, gt=0)

    @classmethod
    def from_composition(
        cls,
        composition: str,
        *,
        structure_type: StructureType = "bulk",
        number: int = 5,
        min_atoms: int | None = None,
        max_atoms: int | None = None,
        formula_type: FormulaType = "fix",
    ) -> "MagusGenerateRequest":
        return cls(
            structure_type=structure_type,
            composition=MagusComposition.from_formula(composition, formula_type),
            number=number,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
        )


class MagusSearchRequest(_MagusBaseRequest):
    """Structure-search request (``magus search``; requires a calculator).

    Domain fields (init_size / population_size / generations) are mapped
    onto the installed MAGUS keys (initSize / popSize / numGen) by the
    renderer; they are never passed through raw.
    """

    formula_type: FormulaType = "fix"
    init_size: int = Field(default=4, ge=1, le=100)
    population_size: int = Field(default=4, ge=1, le=100)
    generations: int = Field(default=1, ge=1, le=50)
    save_good: int = Field(default=2, ge=1, le=100)
    pressure_gpa: float = Field(default=0.0, ge=0)
    calculator: CalculatorName = "vasp"
    execution_mode: ExecutionMode = "serial"
    d_ratio: float = Field(default=0.6, gt=0)
    volume_ratio: float = Field(default=3.0, gt=0)
    rand_ratio: float = Field(default=0.4, ge=0, le=1)
    add_sym: bool = True
    pseudopotentials: list[MagusPseudopotentialRequirement] = Field(
        default_factory=list
    )
    slab: MagusSlabConfig | None = None

    @field_validator("save_good")
    @classmethod
    def _save_good_le_pop(cls, value: int, info: Any) -> int:
        if value > info.data.get("population_size", value):
            raise ValueError("save_good must be <= population_size")
        return value

    @classmethod
    def from_composition(
        cls,
        composition: str,
        *,
        structure_type: StructureType = "bulk",
        calculator: CalculatorName = "vasp",
        execution_mode: ExecutionMode = "serial",
        init_size: int = 4,
        population_size: int = 4,
        generations: int = 1,
        save_good: int = 2,
        pressure_gpa: float = 0.0,
        formula_type: FormulaType = "fix",
        min_atoms: int | None = None,
        max_atoms: int | None = None,
    ) -> "MagusSearchRequest":
        return cls(
            structure_type=structure_type,
            composition=MagusComposition.from_formula(composition, formula_type),
            calculator=calculator,
            execution_mode=execution_mode,
            init_size=init_size,
            population_size=population_size,
            generations=generations,
            save_good=save_good,
            pressure_gpa=pressure_gpa,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
        )


class MagusExecutionConfig(BaseModel):
    """Immutable configuration snapshot for one MAGUS job."""

    backend: str = "scnet"
    magus_root: str = ""
    executable: str = ""
    version: str = ""
    operation: str = ""  # generate | search
    search_type: str = "bulk"
    calculator: str = ""
    execution_mode: str = "serial"
    input_artifact: str = "input.yaml"
    output_artifact: str = "gen.traj"
    generation_parameters: dict[str, Any] = Field(default_factory=dict)
    remote_root: str = "~/photomatagent"
    job_system: str = "SLURM"
    limitations: list[str] = Field(default_factory=list)
