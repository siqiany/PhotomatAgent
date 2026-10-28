#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
cd -- "$project_root"

if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' '找不到 uv。请先安装 uv，再重新运行 scripts/start-workbench.sh。' >&2
  exit 2
fi

if ! uv run --no-sync --extra web python -c \
  'import sys; import fastapi, multipart, uvicorn; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' \
  >/dev/null 2>&1; then
  printf '%s\n' 'Python 3.12 或 Web 依赖未就绪。请在项目目录运行 `uv sync --extra web --inexact`，并确认 Python 3.12 可用。' >&2
  exit 2
fi

exec uv run --no-sync --extra web python -m photomatagent.web.launcher "$@"
