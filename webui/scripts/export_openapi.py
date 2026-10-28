from __future__ import annotations

import json
from pathlib import Path
import tempfile

from photomatagent.application.runner import RunManager
from photomatagent.application.service import WorkbenchService
from photomatagent.application.store import WorkbenchStore
from photomatagent.web.app import create_app
from photomatagent.web.artifacts import ArtifactService


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="photomatagent-openapi-") as temporary:
        root = Path(temporary)
        store = WorkbenchStore(root / "workbench.sqlite3")
        manager = RunManager(store, runtime_builder=lambda **_: None)
        service = WorkbenchService(
            store=store,
            manager=manager,
            workspace_root=root,
            sessions_dir=root / "sessions",
            artifacts=ArtifactService(store, root),
        )
        try:
            schema = create_app(service=service, manager=manager).openapi()
            (repository_root / "webui" / "openapi.json").write_text(
                json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        finally:
            store.close()


if __name__ == "__main__":
    main()
