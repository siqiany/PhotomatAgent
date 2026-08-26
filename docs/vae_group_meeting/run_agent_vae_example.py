"""Runnable example of the agent-facing multi-property VAE tool call."""

from __future__ import annotations

import asyncio
import json

from photomatagent.scientific.capabilities.generation.tools import VAEFormulaTool


TOOL_ARGUMENTS = {
    "target_properties": {
        "gap_selected_eV": 0.35,
        "formation_energy_eV_per_atom": -1.20,
        "energy_above_hull_eV_per_atom": 0.05,
        "density_g_cm3": 6.0,
        "dielectric_mean": 15.0,
        "avg_electron_mass_m0": 0.20,
    },
    "limit": 8,
    "sample_count": 4096,
    "require_novel": True,
    "require_charge_neutral": True,
    "random_seed": 23,
}


async def main() -> None:
    result = await VAEFormulaTool().execute(TOOL_ARGUMENTS)
    if result.is_error:
        raise RuntimeError(result.output)
    print(json.dumps(result.data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
