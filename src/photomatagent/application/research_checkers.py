"""Host-owned numerical references. Never import or execute candidate code here."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Protocol


class Spec(Protocol):
    @property
    def checker_id(self) -> str: ...
    @property
    def input_key(self) -> str: ...
    @property
    def tolerance(self) -> float: ...


def _load(path: Path) -> Any:
    if path.stat().st_size > 2_000_000:
        raise ValueError("CHECKER_INPUT_TOO_LARGE")
    return json.loads(path.read_text())


def _number(value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("NUMBER_REQUIRED")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError("FINITE_NUMBER_REQUIRED")
    return number


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result(spec: Spec, inputs: dict[str, Path], output: Path, mismatches: list[str], method: str) -> dict[str, Any]:
    return {"status": "FAIL" if mismatches else "PASS", "checker_id": spec.checker_id,
            "checker_version": "1", "scientific_status": "UNKNOWN", "method": method,
            "mismatches": mismatches, "input_sha256": {k: _digest(v) for k, v in inputs.items()},
            "output_sha256": _digest(output),
            "limitations": ["Checks a bounded mathematical contract only; input parameters and device performance require independent scientific evidence."]}


def _solve(matrix: list[list[complex]], vector: list[complex]) -> list[complex]:
    """Small complex node system, partial pivoting; no closed-form circuit formula."""
    size = len(vector)
    rows = [list(row) + [value] for row, value in zip(matrix, vector, strict=True)]
    scale = max(abs(x) for row in matrix for x in row)
    for col in range(size):
        pivot = max(range(col, size), key=lambda i: abs(rows[i][col]))
        if abs(rows[pivot][col]) <= scale * 1e-14:
            raise ValueError("SINGULAR_OR_ILL_CONDITIONED_CIRCUIT")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        divisor = rows[col][col]
        rows[col] = [value / divisor for value in rows[col]]
        for i in range(size):
            if i != col:
                factor = rows[i][col]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[col], strict=True)]
    return [row[-1] for row in rows]


def check_rc_network(spec: Spec, inputs: dict[str, Path], output: Path) -> dict[str, Any]:
    """Input SI passive RC graph + 1 A injection; output complex transimpedance samples."""
    try:
        data = _load(inputs[spec.input_key])
        nodes = data["nodes"]
        if (not isinstance(nodes, list) or not 1 <= len(nodes) <= 8
                or any(not isinstance(n, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", n) for n in nodes)
                or len(set(nodes)) != len(nodes) or "gnd" in nodes):
            raise ValueError("NODE_CONTRACT_INVALID")
        index = {node: i for i, node in enumerate(nodes)}
        injection = index[data["injection_node"]]
        observed = index[data["output_node"]]
        frequencies = data["frequencies_hz"]
        components = data["components"]
        if not isinstance(frequencies, list) or not 2 <= len(frequencies) <= 64:
            raise ValueError("FREQUENCY_CONTRACT_INVALID")
        frequencies = [_number(f) for f in frequencies]
        if min(frequencies) < 0 or len(set(frequencies)) != len(frequencies):
            raise ValueError("FREQUENCY_CONTRACT_INVALID")
        if not isinstance(components, list) or not 1 <= len(components) <= 64:
            raise ValueError("COMPONENT_CONTRACT_INVALID")
        edges: list[tuple[int | None, int | None, str, float]] = []
        for item in components:
            kind = item["kind"]
            if kind not in {"R", "C"} or item["a"] == item["b"]:
                raise ValueError("UNSUPPORTED_COMPONENT")
            a = None if item["a"] == "gnd" else index[item["a"]]
            b = None if item["b"] == "gnd" else index[item["b"]]
            edges.append((a, b, kind, _number(item["value_si"], positive=True)))
        actual = _load(output)
        samples = actual["samples"]
        if not isinstance(samples, list) or len(samples) != len(frequencies):
            return _result(spec, inputs, output, ["samples: row count"], "complex nodal KCL")
        mismatches = []
        for row, frequency in enumerate(frequencies):
            matrix = [[0j] * len(nodes) for _ in nodes]
            vector = [0j] * len(nodes)
            vector[injection] = 1
            for a, b, kind, value in edges:
                admittance = complex(1 / value) if kind == "R" else 2j * math.pi * frequency * value
                for endpoint in (a, b):
                    if endpoint is not None:
                        matrix[endpoint][endpoint] += admittance
                if a is not None and b is not None:
                    matrix[a][b] -= admittance
                    matrix[b][a] -= admittance
            reference = _solve(matrix, vector)[observed]
            expected = {"frequency_hz": frequency, "real_ohm": reference.real, "imag_ohm": reference.imag}
            for key, value in expected.items():
                found = _number(samples[row][key])
                # Tolerance is capped independently of candidate settings.
                if not math.isclose(found, value, rel_tol=min(spec.tolerance, 1e-9), abs_tol=1e-12):
                    mismatches.append(f"samples[{row}].{key}")
        if actual.get("scientific_status") != "UNKNOWN":
            mismatches.append("scientific_status")
        return _result(spec, inputs, output, mismatches, "complex nodal KCL; unit current excitation; SI components")
    except (OSError, ValueError, TypeError, KeyError, OverflowError, IndexError, ZeroDivisionError):
        return {"status": "UNKNOWN", "checker_id": spec.checker_id, "gap": "RC_CONTRACT_OR_EVIDENCE_INVALID"}


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _tables(text: str) -> dict[str, list[list[str]]]:
    """Strict subset: heading names identify plain GFM tables, ignoring code fences."""
    tables: dict[str, list[list[str]]] = {}
    heading = ""
    lines = text.splitlines()
    in_code = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
            continue
        if in_code:
            continue
        match = re.fullmatch(r"#{1,6}\s+(.+?)\s*", line)
        if match:
            heading = match[1]
        if i + 1 >= len(lines) or not line.strip().startswith("|"):
            continue
        separator = _cells(lines[i + 1])
        if not separator or not all(re.fullmatch(r":?-{3,}:?", c) for c in separator):
            continue
        if not heading or heading in tables:
            raise ValueError("AMBIGUOUS_TABLE_HEADING")
        rows = [_cells(line)]
        for rest in lines[i + 2:]:
            if not rest.strip().startswith("|"):
                break
            rows.append(_cells(rest))
        tables[heading] = rows
    return tables


def check_report_tables(spec: Spec, inputs: dict[str, Path], output: Path) -> dict[str, Any]:
    """Re-read report input, compare to frozen expected tables, never trust checker PASS."""
    try:
        data = _load(inputs[spec.input_key])
        report = inputs["report"]
        if report.stat().st_size > 2_000_000:
            raise ValueError("REPORT_TOO_LARGE")
        tables = _tables(report.read_text())
        expected = data["tables"]
        if not isinstance(expected, list) or not 1 <= len(expected) <= 64:
            raise ValueError("TABLE_CONTRACT_INVALID")
        headings: set[str] = set()
        mismatches: list[str] = []
        for table in expected:
            heading = table["heading"]
            columns = table["columns"]
            rows = table["rows"]
            if (not isinstance(heading, str) or heading in headings or not heading
                    or not isinstance(columns, list) or not columns or len(columns) > 32
                    or any(not isinstance(c, str) for c in columns) or len(set(columns)) != len(columns)
                    or not isinstance(rows, list) or len(rows) > 1000
                    or any(not isinstance(r, list) or len(r) != len(columns) for r in rows)):
                raise ValueError("TABLE_CONTRACT_INVALID")
            headings.add(heading)
            for row in rows:
                for cell in row:
                    if not isinstance(cell, str):
                        _number(cell)
            found = tables.get(heading)
            if found is None:
                mismatches.append(f"{heading}: missing table")
                continue
            if found[0] != columns:
                mismatches.append(f"{heading}: columns")
            if len(found) - 1 != len(rows):
                mismatches.append(f"{heading}: row count")
            for i, row in enumerate(rows):
                if i + 1 >= len(found):
                    break
                if len(found[i + 1]) != len(columns):
                    mismatches.append(f"{heading}[{i}]: column count")
                    continue
                for j, reference in enumerate(row):
                    cell = found[i + 1][j]
                    if isinstance(reference, str):
                        equal = cell == reference
                    else:
                        try:
                            equal = math.isclose(_number(float(cell)), _number(reference), rel_tol=min(spec.tolerance, 1e-9), abs_tol=1e-12)
                        except ValueError:
                            equal = False
                    if not equal:
                        mismatches.append(f"{heading}[{i}].{columns[j]}")
        actual = _load(output)
        # Candidate audit must accurately describe even a failing report.
        if not isinstance(actual, dict):
            raise ValueError("AUDIT_OBJECT_REQUIRED")
        if actual.get("mismatches") != mismatches or actual.get("scientific_status") != "UNKNOWN":
            return _result(spec, inputs, output, ["candidate audit differs from host parsing"], "report body cell comparison")
        return _result(spec, inputs, output, mismatches, "report body cell comparison against frozen structured input")
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return {"status": "UNKNOWN", "checker_id": spec.checker_id, "gap": "TABLE_CONTRACT_OR_EVIDENCE_INVALID"}
