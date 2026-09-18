"""Scientific capability configuration (limits, secrets, MCP servers)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from photomatagent.mcp.config import MCPServerConfig, load_mcp_servers


def _boolish(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "")
    try:
        return int(value) if value.strip() else default
    except ValueError:
        return default


def _bounded_int_env(
    name: str, default: int, *, minimum: int, maximum: int
) -> int:
    """Read a new strict integer setting and validate its safe bounds."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _bounded_float_env(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    """Read a strict finite float setting and validate its safe bounds."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _first_env(name: str, *aliases: str) -> str | None:
    """Return the first configured value, retaining compatibility aliases."""
    for candidate in (name, *aliases):
        value = os.environ.get(candidate)
        if value is not None:
            return value
    return None


def _text_env(name: str, default: str, *aliases: str) -> str:
    value = _first_env(name, *aliases)
    if value is None:
        return default
    return value.strip() or default


def _optional_text_env(name: str, *aliases: str) -> str | None:
    """Read an optional text setting, treating blank values as unset."""
    value = _first_env(name, *aliases)
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _bounded_int_env_with_aliases(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
    aliases: tuple[str, ...] = (),
) -> int:
    """Apply strict parsing while accepting a documented legacy alias."""
    selected = name
    if os.environ.get(name) is None:
        for alias in aliases:
            if os.environ.get(alias) is not None:
                selected = alias
                break
    return _bounded_int_env(
        selected, default, minimum=minimum, maximum=maximum
    )


@dataclass(frozen=True)
class ScientificConfig:
    """Hard limits and integration settings for scientific capabilities."""

    materials_api_key_env: str = "MATERIALS_API_KEY"
    materials_max_results: int = 10
    literature_max_papers: int = 5
    literature_max_chars: int = 4000
    # Literature RAG / Qdrant configuration.
    literature_root: str = "dataset/paper"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key_env: str = "QDRANT_API_KEY"
    qdrant_collection_prefix: str = "photomat_literature"
    qdrant_timeout_seconds: int = 20
    rag_allow_external: bool = False
    embedding_provider: str = "local"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_vector_dim: int = 384
    embedding_base_url: str = ""
    embedding_api_key_env: str = "RAG_EMBEDDING_API_KEY"
    reranker_provider: str = "local"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_base_url: str = ""
    reranker_api_key_env: str = "RAG_RERANK_API_KEY"
    rag_batch_size: int = 128
    rag_tool_max_documents: int = 20
    literature_search_top_k: int = 5
    literature_passage_chars: int = 600
    structure_output_dir: str = "output/scientific"
    chgnet_model_name: str = "0.3.0"
    chgnet_device: str = "cpu"
    chgnet_max_structures: int = 32
    chgnet_relax_fmax: float = 0.1
    chgnet_relax_steps: int = 200
    structure_max_atoms: int = 128
    structure_max_raw_configurations: int = 4096
    structure_max_outputs: int = 32
    # MatterGen is deliberately an isolated executable integration.  Keep its
    # settings separate from the main Python environment and pass only the
    # configured executable/cache through the narrow runner boundary.
    mattergen_executable: str = "mattergen-generate"
    mattergen_hf_home: str | None = None
    mattergen_pretrained_name: str = "dft_band_gap"
    mattergen_candidate_limit: int = 8
    mattergen_timeout_seconds: float = 3600.0
    mattergen_guidance_factor: float = 2.0
    mattergen_seed: int = 42
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)

    @classmethod
    def from_environment(
        cls, *, workspace: Path | str | None = None
    ) -> "ScientificConfig":
        """Build a config from process environment plus the workspace ``.env``.

        The workspace ``.env`` is loaded into the process environment first
        (non-overriding: existing env vars win), so keys such as
        ``MATERIALS_API_KEY`` are picked up without a shell export.
        """
        root = Path(workspace or Path.cwd())
        _load_dotenv_if_present(root)
        servers = load_mcp_servers(root)
        return cls(
            materials_api_key_env=os.environ.get(
                "PHOTOMATAGENT_MATERIALS_KEY_ENV", "MATERIALS_API_KEY"
            ),
            # These values cap model-visible output.  Keep parsing strict so
            # a typo cannot silently widen/disable a safety bound.
            materials_max_results=_bounded_int_env(
                "PHOTOMATAGENT_MATERIALS_MAX_RESULTS",
                10,
                minimum=1,
                maximum=10,
            ),
            literature_max_papers=_bounded_int_env(
                "PHOTOMATAGENT_LITERATURE_MAX_PAPERS",
                5,
                minimum=1,
                maximum=10,
            ),
            literature_max_chars=_bounded_int_env(
                "PHOTOMATAGENT_LITERATURE_MAX_CHARS",
                4000,
                minimum=200,
                maximum=20_000,
            ),
            literature_root=os.environ.get(
                "PHOTOMATAGENT_LITERATURE_DIR", "dataset/paper"
            ).strip()
            or "dataset/paper",
            qdrant_url=_text_env(
                "PHOTOMATAGENT_QDRANT_URL", "http://127.0.0.1:6333"
            ),
            qdrant_api_key_env=_text_env(
                "PHOTOMATAGENT_QDRANT_API_KEY_ENV", "QDRANT_API_KEY"
            ),
            qdrant_collection_prefix=_text_env(
                "PHOTOMATAGENT_QDRANT_COLLECTION_PREFIX",
                "photomat_literature",
            ),
            qdrant_timeout_seconds=_bounded_int_env(
                "PHOTOMATAGENT_QDRANT_TIMEOUT_SECONDS",
                20,
                minimum=1,
                maximum=300,
            ),
            rag_allow_external=_boolish(
                os.environ.get("PHOTOMATAGENT_RAG_ALLOW_EXTERNAL"), False
            ),
            embedding_provider=_text_env(
                "PHOTOMATAGENT_RAG_EMBEDDING_PROVIDER", "local"
            ),
            embedding_model=_text_env(
                "PHOTOMATAGENT_RAG_EMBEDDING_MODEL",
                "intfloat/multilingual-e5-small",
                "PHOTOMATAGENT_EMBEDDING_MODEL",
            ),
            embedding_vector_dim=_bounded_int_env_with_aliases(
                "PHOTOMATAGENT_RAG_EMBEDDING_VECTOR_DIM",
                384,
                minimum=1,
                maximum=8192,
                aliases=("PHOTOMATAGENT_EMBEDDING_VECTOR_DIM",),
            ),
            embedding_base_url=_text_env(
                "PHOTOMATAGENT_RAG_EMBEDDING_BASE_URL", ""
            ),
            embedding_api_key_env=_text_env(
                "PHOTOMATAGENT_RAG_EMBEDDING_API_KEY_ENV",
                "RAG_EMBEDDING_API_KEY",
            ),
            reranker_provider=_text_env(
                "PHOTOMATAGENT_RAG_RERANK_PROVIDER",
                "local",
                "PHOTOMATAGENT_RAG_RERANKER_PROVIDER",
            ),
            reranker_model=_text_env(
                "PHOTOMATAGENT_RAG_RERANK_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "PHOTOMATAGENT_RAG_RERANKER_MODEL",
                "PHOTOMATAGENT_RERANKER_MODEL",
            ),
            reranker_base_url=_text_env(
                "PHOTOMATAGENT_RAG_RERANK_BASE_URL",
                "",
                "PHOTOMATAGENT_RAG_RERANKER_BASE_URL",
            ),
            reranker_api_key_env=_text_env(
                "PHOTOMATAGENT_RAG_RERANK_API_KEY_ENV",
                "RAG_RERANK_API_KEY",
                "PHOTOMATAGENT_RAG_RERANKER_API_KEY_ENV",
            ),
            rag_batch_size=_bounded_int_env(
                "PHOTOMATAGENT_RAG_BATCH_SIZE",
                128,
                minimum=16,
                maximum=512,
            ),
            rag_tool_max_documents=_bounded_int_env(
                "PHOTOMATAGENT_RAG_TOOL_MAX_DOCUMENTS",
                20,
                minimum=1,
                maximum=100,
            ),
            literature_search_top_k=_bounded_int_env(
                "PHOTOMATAGENT_LITERATURE_TOP_K",
                5,
                minimum=1,
                maximum=10,
            ),
            literature_passage_chars=_bounded_int_env(
                "PHOTOMATAGENT_LITERATURE_PASSAGE_CHARS",
                600,
                minimum=50,
                maximum=600,
            ),
            chgnet_model_name=_text_env(
                "PHOTOMATAGENT_CHGNET_MODEL_NAME", "0.3.0"
            ),
            chgnet_device=_text_env(
                "PHOTOMATAGENT_CHGNET_DEVICE", "cpu"
            ),
            chgnet_max_structures=_bounded_int_env(
                "PHOTOMATAGENT_CHGNET_MAX_STRUCTURES",
                32,
                minimum=1,
                maximum=32,
            ),
            chgnet_relax_fmax=_bounded_float_env(
                "PHOTOMATAGENT_CHGNET_RELAX_FMAX",
                0.1,
                minimum=0.0001,
                maximum=1.0,
            ),
            chgnet_relax_steps=_bounded_int_env(
                "PHOTOMATAGENT_CHGNET_RELAX_STEPS",
                200,
                minimum=1,
                maximum=200,
            ),
            structure_max_atoms=_bounded_int_env(
                "PHOTOMATAGENT_STRUCTURE_MAX_ATOMS", 128, minimum=1, maximum=512
            ),
            structure_max_raw_configurations=_bounded_int_env(
                "PHOTOMATAGENT_STRUCTURE_MAX_RAW_CONFIGURATIONS", 4096,
                minimum=1, maximum=4096,
            ),
            structure_max_outputs=_bounded_int_env(
                "PHOTOMATAGENT_STRUCTURE_MAX_OUTPUTS", 32, minimum=1, maximum=32
            ),
            mattergen_executable=_text_env(
                "PHOTOMATAGENT_MATTERGEN_EXECUTABLE",
                "mattergen-generate",
                "MATTERGEN_EXECUTABLE",
            ),
            mattergen_hf_home=_optional_text_env(
                "PHOTOMATAGENT_MATTERGEN_HF_HOME",
                "MATTERGEN_HF_HOME",
                "HF_HOME",
            ),
            mattergen_pretrained_name=_mattergen_pretrained_name(),
            mattergen_candidate_limit=_bounded_int_env(
                "PHOTOMATAGENT_MATTERGEN_CANDIDATE_LIMIT",
                8,
                minimum=1,
                maximum=32,
            ),
            mattergen_timeout_seconds=_bounded_float_env(
                "PHOTOMATAGENT_MATTERGEN_TIMEOUT_SECONDS",
                3600.0,
                minimum=1.0,
                maximum=7200.0,
            ),
            mattergen_guidance_factor=_bounded_float_env(
                "PHOTOMATAGENT_MATTERGEN_GUIDANCE_FACTOR",
                2.0,
                minimum=0.0,
                maximum=20.0,
            ),
            mattergen_seed=_bounded_int_env(
                "PHOTOMATAGENT_MATTERGEN_SEED",
                42,
                minimum=0,
                maximum=2**31 - 1,
            ),
            mcp_servers=servers,
        )

    def materials_api_key(self) -> str:
        return os.environ.get(self.materials_api_key_env, "").strip()


def _load_dotenv_if_present(root: Path) -> None:
    """Load ``root/.env`` into the process environment without overriding.

    Missing files are ignored; existing environment variables always win so
    explicit exports (or CI secrets) take precedence over the file.
    """
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


def _mattergen_pretrained_name() -> str:
    value = _text_env(
        "PHOTOMATAGENT_MATTERGEN_PRETRAINED_NAME",
        "dft_band_gap",
        "MATTERGEN_PRETRAINED_NAME",
    )
    if value not in {"dft_band_gap", "chemical_system"}:
        raise ValueError(
            "PHOTOMATAGENT_MATTERGEN_PRETRAINED_NAME must be "
            "dft_band_gap or chemical_system"
        )
    return value
