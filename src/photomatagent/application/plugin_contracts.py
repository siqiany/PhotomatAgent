"""Frozen plugin interfaces; validation never imports generated plugin code."""
from __future__ import annotations
import json
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PluginInterface(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    input_schema: dict[str, Any] = Field(default_factory=lambda: {'type': 'object'})
    output_schema: dict[str, Any] = Field(default_factory=lambda: {'type': 'object'})
    argv_choices: list[list[str]] = Field(default=[[]], min_length=1, max_length=32)
    units: dict[str, str] = Field(default_factory=dict, max_length=32)
    assumptions: list[str] = Field(default_factory=list, max_length=32)
    limitations: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode='after')
    def safe_schemas(self) -> PluginInterface:
        import jsonschema
        for schema in (self.input_schema, self.output_schema):
            raw = json.dumps(schema, allow_nan=False)
            if len(raw) > 32000 or any(key in raw for key in ('"$ref"', '"pattern"', '"patternProperties"')):
                raise ValueError('PLUGIN_SCHEMA_UNSUPPORTED')
            jsonschema.Draft202012Validator.check_schema(schema)
        if any(len(argv) > 32 or any(len(arg) > 1000 or '\0' in arg for arg in argv) for argv in self.argv_choices):
            raise ValueError('PLUGIN_ARGV_INVALID')
        return self


def validate_interface(interface: dict[str, Any], inputs: dict[str, Any], output: Any = None) -> None:
    import jsonschema
    spec = PluginInterface.model_validate(interface)
    jsonschema.Draft202012Validator(spec.input_schema).validate(inputs)
    if output is not None:
        jsonschema.Draft202012Validator(spec.output_schema).validate(output)


def interface_for(checker_id: str, argv: list[str]) -> PluginInterface:
    properties: dict[str, Any] = {'data': {'type': 'object'}}
    required = ['data']
    output: dict[str, Any] = {'type': 'object', 'required': ['scientific_status'],
                             'properties': {'scientific_status': {'const': 'UNKNOWN'}}}
    if checker_id == 'rc_network':
        properties['data'] = {'type': 'object', 'required': ['nodes', 'components', 'frequencies_hz', 'injection_node', 'output_node'],
            'properties': {'nodes': {'type':'array','minItems':1,'maxItems':8,'uniqueItems':True,'items':{'type':'string'}},
                           'injection_node':{'type':'string'},'output_node':{'type':'string'},
                           'frequencies_hz':{'type':'array','minItems':2,'maxItems':64,'items':{'type':'number'}},
                           'components':{'type':'array','minItems':1,'maxItems':64,'items':{'type':'object','required':['kind','a','b','value_si'],
                               'properties':{'kind':{'enum':['R','C']},'a':{'type':'string'},'b':{'type':'string'},'value_si':{'type':'number'}}}}}}
        output['required'].append('samples')
        output['properties']['samples'] = {'type': 'array', 'minItems': 2, 'maxItems': 64}
    elif checker_id == 'report_tables':
        properties['data'] = {'type': 'object', 'required': ['tables']}
        properties['report'] = {'type': 'string', 'maxLength': 2_000_000}
        required.append('report')
        output['required'].append('mismatches')
        output['properties']['mismatches'] = {'type': 'array'}
    elif checker_id == 'fresnel_film':
        from photomatagent.application.optical_reference import FresnelFilmInput
        properties['data'] = FresnelFilmInput.model_json_schema()
        output['required'].extend(['R', 'unit'])
        output['properties'].update({'R': {'type': 'number'}, 'unit': {'const': '1'}})
    else:
        return PluginInterface(argv_choices=[argv])
    return PluginInterface(input_schema={'type': 'object', 'required': required, 'properties': properties},
                           output_schema=output, argv_choices=[argv],
                           units=({'resistance':'ohm','capacitance':'F','frequency':'Hz','transimpedance':'ohm'} if checker_id=='rc_network'
                                  else {'thickness_nm': 'nm', 'wavelength_nm': 'nm', 'R': '1'} if checker_id == 'fresnel_film' else {}),
                           assumptions=['Only the host checker contract is tested.'],
                           limitations=['No device performance or empirical correctness is inferred.'])
