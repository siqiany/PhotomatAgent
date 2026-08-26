"""Detached, persistent job monitoring built on the registry + backend.

Monitoring must not require the LLM to poll inside a single model loop.
``JobMonitor.start`` spawns a background asyncio task that watches one
``request_id`` through the registry, persists every transition and returns a
``MonitoringHandle``; the caller can later ask for the latest snapshot
(``latest()``), wait for the next transition (``wait_next()``) or stop the
task (``stop()``). Status-query failures are recorded as structured errors
and never overwrite the last known good state.

The handle survives as long as the owning process; for truly process-detached
supervision the registry (SQLite) is the durable state and a later process can
simply construct a fresh monitor with the same ``request_id``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from photomatagent.scientific.remote.lifecycle import (
    JobBackend,
    StatusRefresh,
    SubmitOnceSession,
)
from photomatagent.scientific.remote.registry import (
    JobLifecycleState,
    JobRegistry,
)


@dataclass
class MonitoringHandle:
    """Handle to a background monitor task."""

    request_id: str
    task: asyncio.Task[Any] | None = None
    _queue: asyncio.Queue[StatusRefresh] = field(default_factory=asyncio.Queue)
    _latest: StatusRefresh | None = None

    def latest(self) -> StatusRefresh | None:
        return self._latest

    async def wait_next(
        self, timeout_seconds: float | None = None
    ) -> StatusRefresh:
        """Wait for the next persisted transition (or the poll error)."""
        return await asyncio.wait_for(self._queue.get(), timeout=timeout_seconds)

    def stop(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()


class JobMonitor:
    """Poll one request through the registry; persist every state change."""

    def __init__(
        self,
        session: SubmitOnceSession,
        *,
        poll_interval_seconds: float = 30.0,
        on_transition: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self.session = session
        self.poll_interval_seconds = poll_interval_seconds
        self.on_transition = on_transition
        self._handles: dict[str, MonitoringHandle] = {}

    def start(self, request_id: str) -> MonitoringHandle:
        """Start a detached background monitor for one request."""
        if request_id in self._handles:
            return self._handles[request_id]
        queue: asyncio.Queue[StatusRefresh] = asyncio.Queue()
        handle = MonitoringHandle(request_id=request_id, _queue=queue)
        handle.task = asyncio.create_task(
            self._run(request_id, handle, queue)
        )
        self._handles[request_id] = handle
        return handle

    def stop(self, request_id: str) -> None:
        handle = self._handles.pop(request_id, None)
        if handle is not None:
            handle.stop()

    def stop_all(self) -> None:
        for handle in list(self._handles.values()):
            handle.stop()
        self._handles.clear()

    async def poll_once(self, request_id: str) -> StatusRefresh:
        """Single pull + persist; safe to call from anywhere."""
        return await self.session.refresh_status(request_id)

    async def run_until_terminal(
        self, request_id: str, *, poll_until: float | None = None
    ) -> StatusRefresh:
        """Poll until a terminal lifecycle state or an optional time budget."""
        import time

        deadline = (
            None if poll_until is None else time.monotonic() + poll_until
        )
        while True:
            refresh = await self.poll_once(request_id)
            record = self.session.registry.get(request_id)
            terminal = (
                record is not None and record.state.terminal
            )
            if terminal or (deadline is not None and time.monotonic() >= deadline):
                return refresh
            await asyncio.sleep(self.poll_interval_seconds)

    async def _run(
        self,
        request_id: str,
        handle: MonitoringHandle,
        queue: asyncio.Queue[StatusRefresh],
    ) -> None:
        last_state: str | None = None
        try:
            while True:
                refresh = await self.poll_once(request_id)
                handle._latest = refresh
                queue.put_nowait(refresh)
                record = self.session.registry.get(request_id)
                state = record.state.value if record is not None else None
                if state != last_state and state is not None:
                    last_state = state
                    if self.on_transition is not None:
                        try:
                            await self.on_transition(request_id, state)
                        except Exception:
                            pass
                    if handle._queue.qsize() > 64:
                        handle._queue.get_nowait()
                if record is not None and (
                    record.state.terminal
                    or record.state
                    is JobLifecycleState.UNKNOWN_RECONCILIATION_REQUIRED
                ):
                    return
                await asyncio.sleep(self.poll_interval_seconds)
        except asyncio.CancelledError:
            return
        except Exception:
            return
