"""Single Slurm renderer for isolated-molecule submissions.

Deliberately shares ONE template with the periodic path: the molecular
``run.slurm`` is rendered by the same :func:`render_slurm_script` plus the
same product-environment and remote-POTCAR-assembly preambles that
:class:`VaspApplication` uses. There is no second, divergent copy of the
Slurm template.

The rendered script only fingerprints what is safe: the job name, resource
request, module/environment name and the element sequence. POTCAR content
never appears in the script, registry, logs or model output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from photomatagent.scientific.applications.vasp.application import (
    env_source_preamble,
    potcar_assembly_preamble,
)
from photomatagent.scientific.applications.vasp.molecular.psp_metadata import (
    PspError,
)
from photomatagent.scientific.applications.vasp.psp import (
    is_safe_potcar_symbol,
    resolve_local_psp_library,
)
from photomatagent.scientific.remote.models import ResourceRequest
from photomatagent.scientific.remote.scheduler import render_slurm_script


def potcar_symbols_from_stage(stage_dir: str | Path) -> list[str]:
    """Deterministic element sequence for remote POTCAR assembly.

    Reads POTCAR.meta ``sequence`` (a JSON list) or, as a fallback, the
    ``sequence: C O H Li`` line in POTCAR.policy. Returns [] when neither
    exists so remote assembly is simply not attempted.
    """
    directory = Path(stage_dir)
    meta = directory / "POTCAR.meta"
    if meta.is_file():
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            payload = {}
        sequence = payload.get("sequence")
        if isinstance(sequence, list) and sequence:
            symbols = [
                str(item)
                for item in sequence
                if isinstance(item, str) and item.isalpha()
            ]
            if symbols:
                return symbols
    policy = directory / "POTCAR.policy"
    if policy.is_file():
        match = re.search(
            r"(?m)^sequence:\s*([A-Za-z\s]+)\s*$",
            policy.read_text(encoding="utf-8"),
        )
        if match:
            symbols = [
                token for token in match.group(1).split() if token.isalpha()
            ]
            if symbols:
                return symbols
    return []


def render_stage_slurm(
    *,
    job_name: str,
    resource: ResourceRequest,
    stage_dir: str | Path,
    module_name: str = "",
    env_script: str = "",
    remote_psp_dir: str = "",
    potcar_symbols: list[str] | None = None,
) -> str:
    """Render the deterministic ``run.slurm`` for one molecular stage.

    POTCAR handling resolves from the local files and the remote
    pseudopotential configuration:
      * a materialized local ``POTCAR`` is used as-is (mode ``local``);
      * otherwise, if a remote PSP dir is configured and the element
        sequence is known, the script assembles POTCAR on the cluster
        (mode ``remote``);
      * otherwise the script omits POTCAR handling (mode ``none``) and the
        preflight/submission layer refuses the request.
    """
    executable = "vasp_std"
    symbols = (
        list(potcar_symbols)
        if potcar_symbols is not None
        else potcar_symbols_from_stage(stage_dir)
    )
    local_potcar = (Path(stage_dir) / "POTCAR").is_file()
    preamble_parts: list[str] = []
    if env_script:
        preamble_parts.append(env_source_preamble(env_script, executable))
    if remote_psp_dir and symbols and not local_potcar:
        preamble_parts.append(
            potcar_assembly_preamble(remote_psp_dir, symbols)
        )
    # SBATCH options are fully rendered through the shared template; the
    # launcher is the canonical ``srun --mpi=pmi2 vasp_std`` for VASP 5.4.4
    # on SCNet. ``launcher`` is validated by ``render_slurm_script``.
    return render_slurm_script(
        job_name=job_name,
        resource=resource,
        module_load="" if env_script else module_name,
        executable=executable,
        preamble="\n".join(part for part in preamble_parts if part),
    )


def potcar_mode_of_stage(
    stage_dir: str | Path,
    *,
    remote_psp_dir: str,
    psp_dir: str | Path | None = None,
) -> str:
    """Declare the POTCAR strategy for one staged input directory.

    ``local`` when a curated POTCAR exists in the stage directory already;
    ``remote`` when the script can assemble it on the cluster; otherwise,
    when a local PAW-PBE library is resolvable (``psp_dir`` with a known
    layout), ``local`` reports that the POTCAR can be materialized from the
    local library before upload; ``none`` only when no strategy exists
    (submission is refused: no fabricated POTCAR is ever used).
    """
    directory = Path(stage_dir)
    if (directory / "POTCAR").is_file():
        return "local"
    if remote_psp_dir and potcar_symbols_from_stage(directory):
        return "remote"
    if psp_dir is not None and resolve_local_psp_library(psp_dir) is not None:
        return "local"
    return "none"


def local_potcar_materializable(psp_dir: str | Path | None) -> bool:
    """True when the configured local library resolves to a known layout."""
    if psp_dir is None:
        return False
    return resolve_local_psp_library(psp_dir) is not None


def materialize_stage_potcar(
    stage_dir: str | Path,
    psp_dir: str | Path | None,
    symbols: list[str],
) -> bool:
    """Assemble POTCAR from the local library in POSCAR element order.

    Returns True when the file was written by this call (the caller then
    owns cleanup after upload); False when a POTCAR already existed or the
    library cannot be resolved. POTCAR content is a bytes-level copy only:
    it never enters logs, the registry, JSON payloads or model output.
    """
    directory = Path(stage_dir)
    if (directory / "POTCAR").is_file():
        return False
    if psp_dir is None or not symbols:
        return False
    resolved = resolve_local_psp_library(psp_dir)
    if resolved is None:
        return False
    library, _ = resolved
    target = directory / "POTCAR"
    try:
        with target.open("wb") as destination:
            for symbol in symbols:
                if not is_safe_potcar_symbol(symbol):
                    raise PspError(
                        f"unsafe POTCAR symbol {symbol!r}",
                        code="PSP_SYMBOL_UNSAFE",
                    )
                source = library / symbol / "POTCAR"
                if not source.is_file():
                    raise PspError(
                        f"missing PAW-PBE dataset for {symbol}: {source}",
                        code="PSP_DATASET_MISSING",
                    )
                destination.write(source.read_bytes())
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return True


def cleanup_materialized_potcar(
    stage_dir: str | Path, *, materialized: bool
) -> bool:
    """Remove a POTCAR that this session materialized (never a curated one).

    The file is only deleted when the caller tells us it created it; a
    user-provided POTCAR is left untouched. Deletion is safe because the
    assembled file is deterministic and cheap to rebuild for a retry, and it
    keeps POTCAR bytes out of the workspace between attempts.
    """
    if not materialized:
        return False
    path = Path(stage_dir) / "POTCAR"
    if path.is_file():
        path.unlink()
        return True
    return False
