"""Chat management adapter and offline storage maintenance commands."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import typer
from rich.console import Console
from photomatagent.cli.research import WorkbenchClient

app = typer.Typer(help="工作区数据诊断、导入、备份和离线恢复；聊天中使用 /data。")


class InformationCommands:
    def __init__(self, console: Console) -> None:
        self.console = console
        self.project_id: str | None = None
        self.client = WorkbenchClient("http://127.0.0.1:8765")

    async def request(self, path: str, body: Any = None, method: str | None = None) -> Any:
        return await asyncio.to_thread(self.client.request,path,body,method=method)

    async def execute(self, command: str, args: list[str]) -> None:
        sub = args[0] if args else "list"
        if command == "/search":
            workspace = "--workspace" in args
            words = [arg for arg in args if arg!="--workspace"]
            params = {"q":" ".join(words)}
            if self.project_id and not workspace:
                params["project_id"] = self.project_id
            result = await self.request("/search?"+urlencode(params))
        elif command == "/project":
            if sub == "list":
                result = await self.request("/projects?archived=true")
            elif sub == "create" and len(args)>1:
                result = await self.request("/projects",{"name":" ".join(args[1:])})
            elif sub == "use" and len(args)==2:
                result = await self.request("/projects/"+quote(args[1],safe=""))
                self.project_id = result["project_id"]
            elif sub in {"show","rename","archive","unarchive"} and len(args)>=2:
                path = "/projects/"+quote(args[1],safe="")
                result = await self.request(path)
                if sub=="rename" and len(args)>2:
                    result = await self.request(path,{"expected_revision":result["revision"],"name":" ".join(args[2:])},"PATCH")
                elif sub in {"archive","unarchive"}:
                    result = await self.request(path+"/"+sub,{"expected_revision":result["revision"]})
            else:
                raise ValueError("用法：/project list|create 名称|use ID|show ID|rename ID 名称|archive ID|unarchive ID")
        elif command == "/task":
            if sub=="list":
                params = {"view":"page"}
                if self.project_id:
                    params["project_id"]=self.project_id
                result = await self.request("/tasks?"+urlencode(params))
            elif len(args)>=2:
                path="/tasks/"+quote(args[1],safe="")
                detail=await self.request(path)
                task=detail["task"]
                if sub=="show":
                    result=detail
                elif sub in {"archive","unarchive"}:
                    result=await self.request(path+"/"+sub,{"expected_revision":task["revision"]})
                elif sub=="input" and len(args)==3:
                    result=await self.request(path+"/inputs",{"source_artifact_id":args[2]})
                elif sub=="replace-input" and len(args)==4:
                    result=await self.request(path+"/inputs",{"replaces_artifact_id":args[2],"source_artifact_id":args[3]})
                elif sub=="move" and len(args)==3:
                    target=await self.request("/projects/"+quote(args[2],safe=""))
                    result=await self.request(path+"/move",{"expected_revision":task["revision"],"project_id":target["project_id"],"target_revision":target["revision"]})
                elif sub in {"rename","state","tag"} and len(args)>2:
                    field={"rename":"title","state":"management_status","tag":"tags"}[sub]
                    value: Any = args[2:] if sub=="tag" else " ".join(args[2:])
                    result=await self.request(path,{"expected_revision":task["revision"],field:value},"PATCH")
                else:
                    raise ValueError("用法：/task show|rename|move|state|tag|archive|unarchive ID [参数]，或 /task replace-input 任务ID 失效输入ID 新产物ID")
            else:
                raise ValueError("用法：/task list|show ID|rename ID 标题|move ID 课题ID|state ID OPEN/PAUSED/DONE|tag ID 标签...")
        else:
            if sub=="restore":
                self.console.print("停止工作台与独立科研写入后运行：photomatagent data restore BACKUP_RELATIVE_PATH DESTINATION_RELATIVE_PATH；仅恢复到新目录，不替换现有数据。",markup=False)
                return
            if sub in {"status","doctor"}:
                result=await self.request("/storage/"+("diagnostics" if sub=="doctor" else "status"))
            elif sub in {"import","refresh","reindex","export","backup"}:
                if "--full" in args:
                    self.console.print("完整备份需要停止工作台与独立科研写入，再运行 photomatagent data backup --full。在线 /data backup 创建主库快照。",markup=False)
                    return
                result=await self.request("/storage/"+sub,{})
            else:
                raise ValueError("用法：/data status|doctor|import|refresh|reindex|export|backup|restore --help")
        self.console.print(json.dumps(result,ensure_ascii=False,indent=2),markup=False)


@app.command("backup")
def backup(full: bool = False, workspace: Path = Path(".")) -> None:
    from photomatagent.application.store import WorkbenchStore
    from photomatagent.persistence.maintenance import StorageService
    root=workspace.resolve()
    store=WorkbenchStore(root/".photomatagent/workbench.sqlite3")
    try:
        typer.echo(json.dumps(StorageService(store,root).backup(full),ensure_ascii=False))
    finally:
        store.close()


@app.command("restore")
def restore(backup: str, destination: str, workspace: Path = Path(".")) -> None:
    from photomatagent.persistence.maintenance import restore_preview
    typer.echo(json.dumps(restore_preview(workspace.resolve(),Path(backup),Path(destination)),ensure_ascii=False))
