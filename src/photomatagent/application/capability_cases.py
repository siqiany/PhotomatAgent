"""Host-owned development and holdout cases; candidates cannot replace these."""
from __future__ import annotations
import copy
from typing import Any

SUITE_VERSION = 'research-capability-v1'


def rc(rj: float, rs: float, rl: float, c: float) -> dict[str, Any]:
    return {'nodes': ['a', 'b'], 'injection_node': 'a', 'output_node': 'b',
            'frequencies_hz': [0, 731.0, 1e6, 1e8], 'components': [
                {'kind': 'R', 'a': 'a', 'b': 'gnd', 'value_si': rj},
                {'kind': 'R', 'a': 'a', 'b': 'b', 'value_si': rs},
                {'kind': 'R', 'a': 'b', 'b': 'gnd', 'value_si': rl},
                {'kind': 'C', 'a': 'a', 'b': 'gnd', 'value_si': c}]}


def cases(kind: str, *, holdout: bool = False) -> list[dict[str, Any]]:
    if kind == 'fresnel_film':
        base = {'contract': 'lossless_normal_incidence_film', 'n_incident': 1.0, 'n_film': 2.0,
                'n_exit': 1.7 if holdout else 1.5, 'thickness_nm': 0, 'wavelength_nm': 1000}
        transfer = base | {'thickness_nm': 137 if holdout else 125, 'wavelength_nm': 914 if holdout else 1000}
        return [{'id': 'bare-interface', 'role': 'regression', 'inputs': {'data': base}, 'expected': 'PASS'},
                {'id': 'interference', 'role': 'transfer', 'inputs': {'data': transfer}, 'expected': 'PASS'},
                {'id': 'matched-film', 'role': 'transfer', 'inputs': {'data': transfer | {'n_film': base['n_exit']}}, 'expected': 'PASS'},
                {'id': 'invalid-wavelength', 'role': 'negative', 'inputs': {'data': transfer | {'wavelength_nm': 0}}, 'expected': 'REJECT'}]
    if kind == 'rc_network':
        data = rc(43.0, 570.0, 87.0, 2.7e-10) if holdout else rc(100.0, 100.0, 50.0, 1e-9)
        new = rc(1013.0, 73.0, 191.0, 4.1e-9) if holdout else rc(17.0, 290.0, 130.0, 3e-10)
        graph = {'nodes': ['a', 'b', 'c'], 'injection_node': 'a', 'output_node': 'c',
                 'frequencies_hz': [0, 37.0, 5e4], 'components': [
                     {'kind': 'R', 'a': 'a', 'b': 'b', 'value_si': 11.0 if holdout else 100.0},
                     {'kind': 'R', 'a': 'b', 'b': 'c', 'value_si': 73.0},
                     {'kind': 'R', 'a': 'c', 'b': 'gnd', 'value_si': 191.0},
                     {'kind': 'C', 'a': 'b', 'b': 'gnd', 'value_si': 5e-9}]}
        invalid = copy.deepcopy(data); invalid['components'][0]['value_si'] = -1
        return [{'id': 'regression', 'role': 'regression', 'inputs': {'data': data}, 'expected': 'PASS'},
                {'id': 'parameters', 'role': 'transfer', 'inputs': {'data': new}, 'expected': 'PASS'},
                {'id': 'topology', 'role': 'transfer', 'inputs': {'data': graph}, 'expected': 'PASS'},
                {'id': 'invalid-resistor', 'role': 'negative', 'inputs': {'data': invalid}, 'expected': 'REJECT'}]
    if kind == 'report_tables':
        heading = 'Holdout measurements' if holdout else 'Readout'
        value = 12.73 if holdout else 60.0
        data = {'tables': [{'heading': heading, 'columns': ['parameter', 'value'], 'rows': [['reference', value], ['tau_s', 6e-8]]}]}
        report = f'# {heading}\n\n| parameter | value |\n| --- | --- |\n| reference | {value} |\n| tau_s | 6e-8 |\n'
        new = {'tables': [{'heading': heading + ' new', 'columns': ['sample', 'mass', 'unit'], 'rows': [['B', 0.0123, 'kg'], ['C', 9.81, 'kg']]}]}
        new_report = f'## {heading} new\n| sample | mass | unit |\n| --- | --- | --- |\n| B | 0.0123 | kg |\n| C | 9.81 | kg |\n'
        return [{'id': 'regression', 'role': 'regression', 'inputs': {'data': data, 'report': report}, 'expected': 'PASS'},
                {'id': 'new-table', 'role': 'transfer', 'inputs': {'data': new, 'report': new_report}, 'expected': 'PASS'},
                *[{'id': ident, 'role': 'negative', 'inputs': {'data': data, 'report': damaged}, 'expected': 'AUDIT_FAIL'} for ident, damaged in [
                    ('number', report.replace(str(value), str(value + 0.00002))),
                    ('column', report.replace(' | value', '')),
                    ('table', f'# {heading}\nNo table here.\n')]]]
    raise ValueError('CAPABILITY_SUITE_UNSUPPORTED')
