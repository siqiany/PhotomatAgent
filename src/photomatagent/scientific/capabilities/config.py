"""Scientific capability configuration (limits, secrets, MCP servers)."""

from __future__ import annotations

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


@dataclass(frozen=True)
class ScientificConfig:
    """Hard limits and integration settings for scientific capabilities."""

    materials_api_key_env: str = "MATERIALS_API_KEY"
    materials_max_results: int = 10
    literature_max_papers: int = 5
    literature_max_chars: int = 4000
    # Literature RAG V1: configurable dataset/index/embedding locations.
    literature_root: str = "dataset/paper"
    literature_index_dir: str = "output/literature_index"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_vector_dim: int = 384
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    literature_search_top_k: int = 5
    literature_passage_chars: int = 600
    structure_output_dir: str = "output/scientific"
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
            materials_max_results=_int_env(
                "PHOTOMATAGENT_MATERIALS_MAX_RESULTS", 10
            ),
            literature_max_papers=_int_env(
                "PHOTOMATAGENT_LITERATURE_MAX_PAPERS", 5
            ),
            literature_max_chars=_int_env(
                "PHOTOMATAGENT_LITERATURE_MAX_CHARS", 4000
            ),
            literature_root=os.environ.get(
                "PHOTOMATAGENT_LITERATURE_DIR", "dataset/paper"
            ).strip()
            or "dataset/paper",
            literature_index_dir=os.environ.get(
                "PHOTOMATAGENT_LITERATURE_INDEX_DIR", "output/literature_index"
            ).strip()
            or "output/literature_index",
            embedding_model=os.environ.get(
                "PHOTOMATAGENT_EMBEDDING_MODEL",
                "intfloat/multilingual-e5-small",
            ).strip()
            or "intfloat/multilingual-e5-small",
            embedding_vector_dim=_int_env(
                "PHOTOMATAGENT_EMBEDDING_VECTOR_DIM", 384
            ),
            reranker_model=os.environ.get(
                "PHOTOMATAGENT_RERANKER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ).strip()
            or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            literature_search_top_k=_int_env(
                "PHOTOMATAGENT_LITERATURE_TOP_K", 5
            ),
            literature_passage_chars=_int_env(
                "PHOTOMATAGENT_LITERATURE_PASSAGE_CHARS", 600
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
