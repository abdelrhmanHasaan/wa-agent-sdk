"""Manages the Node.js/Baileys bridge subprocess over a local WebSocket.

The bridge (see ``node_bridge/bridge.mjs``) owns the WhatsApp Web multi-device
connection. This module spawns it, keeps a JSON request/response protocol and
re-emits WhatsApp events to a caller-supplied async handler.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

import websockets

from .exceptions import (
    BridgeError,
    BridgeInstallError,
    BridgeNotRunningError,
    NodeNotFoundError,
)

log = logging.getLogger("wa_agent.bridge")

EventHandler = Callable[[str, dict], Awaitable[None]]

_BRIDGE_FILES = ("bridge.mjs", "package.json")


def bundled_bridge_dir() -> Path:
    override = os.environ.get("WA_BRIDGE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / "node_bridge"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class BaileysBridge:
    """Lifecycle manager for one WhatsApp session backed by Baileys."""

    def __init__(
        self,
        *,
        auth_dir: Path,
        event_handler: EventHandler | None = None,
        log_level: str = "WARN",
        max_restarts: int = 5,
    ) -> None:
        self.auth_dir = Path(auth_dir)
        self.event_handler = event_handler
        self.log_level = log_level.upper()
        self.max_restarts = max_restarts

        self._bridge_dir = bundled_bridge_dir()
        for required in _BRIDGE_FILES:
            if not (self._bridge_dir / required).exists():
                raise BridgeError(f"Bridge files missing in {self._bridge_dir} ({required})")

        self._proc: asyncio.subprocess.Process | None = None
        self._ws: Any = None
        self._log_fh: Any = None
        self._tasks: set[asyncio.Task] = set()
        self._pending: dict[int, asyncio.Future] = {}
        self._req_counter = itertools.count(1)
        self._stopping = False
        self._running = False
        self._restarts = 0
        self._installed_checked = False
        self._log_path: Path | None = None
        self._token = uuid.uuid4().hex

    @property
    def running(self) -> bool:
        return self._running and self._ws is not None

    async def start(self) -> None:
        if self.running:
            return
        self._stopping = False
        self.auth_dir.mkdir(parents=True, exist_ok=True)
        await self._ensure_dependencies()
        await self._launch()
        self._spawn_supervisor()

    async def _ensure_dependencies(self) -> None:
        if self._installed_checked:
            return
        if not (self._bridge_dir / "node_modules").exists():
            npm = shutil.which("npm")
            if npm is None:
                raise NodeNotFoundError(
                    "npm was not found on PATH. Install Node.js LTS from https://nodejs.org "
                    "(the SDK needs Node >= 18) and try again."
                )
            log.info("Installing bridge dependencies (first run only)...")
            proc = await asyncio.create_subprocess_exec(
                npm,
                "install",
                "--no-audit",
                "--no-fund",
                "--loglevel=error",
                cwd=str(self._bridge_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                tail = stderr.decode("utf-8", errors="replace")[-1500:]
                raise BridgeInstallError(
                    f"npm install failed (exit {proc.returncode}).\n{tail}"
                )
        node = shutil.which("node")
        if node is None:
            raise NodeNotFoundError(
                "Node.js was not found on PATH. Install Node.js >= 18 from https://nodejs.org"
            )
        self._installed_checked = True

    async def _launch(self) -> None:
        port = _free_port()
        logs_dir = self.auth_dir.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = logs_dir / f"bridge-{port}.log"
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:  # noqa: BLE001
                pass
        self._log_fh = open(self._log_path, "ab")

        env = os.environ.copy()
        env.update(
            WA_BRIDGE_PORT=str(port),
            WA_BRIDGE_TOKEN=self._token,
            WA_AUTH_DIR=str(self.auth_dir),
            WA_LOG_LEVEL="error",
            FORCE_COLOR="0",
        )

        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        self._proc = await asyncio.create_subprocess_exec(
            shutil.which("node") or "node",
            str(self._bridge_dir / "bridge.mjs"),
            cwd=str(self._bridge_dir),
            env=env,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        log.info("Bridge process started on port %d (pid %s)", port, self._proc.pid)

        uri = f"ws://127.0.0.1:{port}/?token={self._token}"
        deadline = asyncio.get_running_loop().time() + 30
        last_exc: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            if self._proc.returncode is not None:
                raise BridgeError(self._crash_message())
            try:
                self._ws = await websockets.connect(uri, max_size=64 * 1024 * 1024)
                break
            except OSError as exc:
                last_exc = exc
                await asyncio.sleep(0.4)
        else:
            raise BridgeError(f"Bridge did not open its WebSocket in time: {last_exc}")

        self._running = True
        self._spawn_reader()
        self._spawn_writer_watchdog()

    def _crash_message(self) -> str:
        tail = ""
        if self._log_path and self._log_path.exists():
            lines = self._log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
            tail = "\n".join(lines)
        return f"Bridge process exited unexpectedly.\nLast log lines:\n{tail}"

    def _spawn_task(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _spawn_reader(self) -> None:
        assert self._ws is not None

        async def reader() -> None:
            try:
                async for raw in self._ws:
                    await self._dispatch(raw)
            except (websockets.ConnectionClosed, OSError):
                pass
            finally:
                self._running = False
                for fut in list(self._pending.values()):
                    if not fut.done():
                        fut.set_exception(BridgeNotRunningError("Bridge connection closed"))
                self._pending.clear()

        self._spawn_task(reader())

    def _spawn_writer_watchdog(self) -> None:
        async def watchdog() -> None:
            while True:
                await asyncio.sleep(1.0)
                proc = self._proc
                if proc is not None and proc.returncode is not None:
                    self._running = False
                    return

        self._spawn_task(watchdog())

    def _spawn_supervisor(self) -> None:
        async def supervisor() -> None:
            while not self._stopping:
                await asyncio.sleep(2.0)
                healthy = self._running or (self._ws is not None)
                if healthy and self._proc is not None and self._proc.returncode is None:
                    continue
                if self._stopping:
                    return
                self._restarts += 1
                if self._restarts > self.max_restarts:
                    log.error("Bridge restart limit (%d) reached", self.max_restarts)
                    await self._emit("fatal", {"error": "bridge restart limit reached"})
                    return
                backoff = min(30.0, 2.0**self._restarts)
                log.warning("Bridge down; restarting in %.1fs (attempt %d)", backoff, self._restarts)
                await asyncio.sleep(backoff)
                if self._stopping:
                    return
                try:
                    await self._launch()
                    self._restarts = 0
                except BridgeError as exc:
                    log.error("Bridge relaunch failed: %s", exc)
                    await self._emit("fatal", {"error": str(exc)})
                    return

        self._spawn_task(supervisor())

    async def _emit(self, event: str, payload: dict) -> None:
        if self.event_handler is None:
            return
        try:
            await self.event_handler(event, payload)
        except Exception:  # noqa: BLE001 - user hook must not kill the loop
            log.exception("Event handler raised")

    async def _dispatch(self, raw: str | bytes) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        mtype = data.get("type")
        ref = data.get("ref")
        if ref is not None:
            fut = self._pending.pop(ref, None)
            if fut is not None and not fut.done():
                fut.set_result(data)
            return
        if mtype == "result":
            return
        await self._emit(mtype or "unknown", {k: v for k, v in data.items() if k != "type"})

    async def _rpc(self, payload: dict, timeout: float = 60.0) -> dict:
        if not self.running or self._ws is None:
            raise BridgeNotRunningError("Bridge is not connected; call start() first")
        ref = next(self._req_counter)
        payload = {**payload, "ref": ref}
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[ref] = fut
        try:
            await self._ws.send(json.dumps(payload))
            response = await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError as exc:
            raise BridgeError(f"Bridge RPC timed out after {timeout}s ({payload.get('type')})") from exc
        finally:
            self._pending.pop(ref, None)
        if not response.get("ok", False):
            error = response.get("error", "unknown bridge error")
            raise BridgeError(f"Bridge '{payload.get('type')}' failed: {error}")
        return response

    async def wait_ready(self, timeout: float = 60.0) -> bool:
        """Wait until the bridge answers at least one ping."""
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            try:
                await self._rpc({"type": "ping"}, timeout=5.0)
                return True
            except (BridgeNotRunningError, BridgeError):
                if self._stopping:
                    return False
                await asyncio.sleep(0.3)
        return False

    async def send_text(self, to: str, text: str) -> str | None:
        resp = await self._rpc({"type": "send_text", "to": to, "text": text})
        return resp.get("id")

    async def send_media(
        self,
        to: str,
        media_type: str,
        data_b64: str,
        *,
        mimetype: str | None = None,
        filename: str | None = None,
        caption: str | None = None,
        ptt: bool = False,
    ) -> str | None:
        resp = await self._rpc(
            {
                "type": "send_media",
                "to": to,
                "media_type": media_type,
                "data_b64": data_b64,
                "mimetype": mimetype,
                "filename": filename,
                "caption": caption,
                "ptt": ptt,
            }
        )
        return resp.get("id")

    async def download_media(self, message_id: str, timeout: float = 120.0) -> tuple[str, str | None]:
        resp = await self._rpc({"type": "download_media", "id": message_id}, timeout=timeout)
        return resp.get("data_b64", ""), resp.get("mimetype")

    async def set_presence(self, presence: str, chat_jid: str | None = None) -> None:
        await self._rpc({"type": "set_presence", "presence": presence, "jid": chat_jid})

    async def mark_read(self, message_id: str, chat_jid: str, sender_jid: str | None = None) -> None:
        await self._rpc(
            {
                "type": "mark_read",
                "id": message_id,
                "chat_jid": chat_jid,
                "sender_jid": sender_jid,
            }
        )

    async def logout(self) -> None:
        """Unlink the device and wipe stored credentials."""
        try:
            await self._rpc({"type": "logout"}, timeout=20.0)
        except (BridgeError, BridgeNotRunningError):
            pass
        self._running = False

    async def stop(self) -> None:
        self._stopping = True
        self._running = False
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close(code=1001)
            except Exception:  # noqa: BLE001
                pass
        proc, self._proc = self._proc, None
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._log_fh = None
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
