#!/usr/bin/env python3
"""Consume live metrics from voxel-editor-mcp and keep a bounded JSONL trace."""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

PROTOCOL = "2025-11-25"
ROTATE_BYTES = 8 * 1024 * 1024
ROTATE_FILES = 4
MAX_RECORD_BYTES = 1024 * 1024
MAX_DURATION_SECONDS = 24 * 60 * 60


def require_e_path(value: str, label: str, *, must_exist: bool) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path on E:")
    resolved = path.resolve(strict=must_exist)
    if resolved.drive.upper() != "E:":
        raise ValueError(f"{label} must resolve to E: (C: and other drives are rejected)")
    return resolved


def validate_duration(value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value <= 0 or value > MAX_DURATION_SECONDS):
        raise ValueError(f"duration-seconds must be in (0, {MAX_DURATION_SECONDS}]")


class StreamState:
    """Tracks a runtime session and advances only to sequences actually returned."""

    def __init__(self) -> None:
        self.session_id: str | None = None
        self.cursor = 0

    def accept(self, response: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
        session = response.get("session_id")
        if not isinstance(session, str) or not session:
            raise ValueError("metrics response has no session_id")
        reset = self.session_id != session
        if reset:
            self.session_id = session
            self.cursor = 0
            return [], True

        samples = response.get("samples")
        if not isinstance(samples, list):
            raise ValueError("metrics response has no samples array")
        valid: list[dict[str, Any]] = []
        advanced = self.cursor
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            sequence = sample.get("sequence")
            if isinstance(sequence, int) and not isinstance(sequence, bool) and sequence > advanced:
                valid.append(sample)
                advanced = max(advanced, sequence)
        self.cursor = advanced
        return valid, False


class RotatingJsonl:
    def __init__(self, directory: Path, stem: str = "runtime-metrics.jsonl") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / stem

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        if len(line) > MAX_RECORD_BYTES:
            raise ValueError(f"JSONL record exceeds {MAX_RECORD_BYTES} bytes")
        current_size = self.path.stat().st_size if self.path.exists() else 0
        if current_size + len(line) > ROTATE_BYTES:
            oldest = self.path.with_name(f"{self.path.name}.{ROTATE_FILES - 1}")
            oldest.unlink(missing_ok=True)
            for index in range(ROTATE_FILES - 2, 0, -1):
                source = self.path.with_name(f"{self.path.name}.{index}")
                if source.exists():
                    os.replace(source, self.path.with_name(f"{self.path.name}.{index + 1}"))
            if self.path.exists():
                os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))
        with self.path.open("ab") as stream:
            stream.write(line)


class McpStdio:
    def __init__(self, executable: Path, project_root: Path) -> None:
        self.process = subprocess.Popen(
            [str(executable), "--project-root", str(project_root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self.responses: queue.Queue[dict[str, Any] | BaseException] = queue.Queue()
        self.next_id = 0
        self.reader = threading.Thread(target=self._read_responses, daemon=True)
        self.reader.start()
        try:
            self._request(
                "initialize",
                {"protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": {"name": "voxel-runtime-watch", "version": "1.0"}},
                timeout=5,
            )
            self._send_notification("notifications/initialized")
        except BaseException:
            self.close()
            raise

    def _read_responses(self) -> None:
        assert self.process.stdout is not None
        try:
            for raw_line in iter(self.process.stdout.readline, b""):
                try:
                    message = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    self.responses.put(RuntimeError(f"invalid MCP JSON response: {exc}"))
                    continue
                if isinstance(message, dict) and "id" in message:
                    self.responses.put(message)
        except BaseException as exc:  # relay unexpected stream failures to the caller
            self.responses.put(exc)
        finally:
            self.responses.put(RuntimeError("MCP process closed stdout"))

    def _send_notification(self, method: str) -> None:
        self._write_message({"jsonrpc": "2.0", "method": method, "params": {}})

    def _write_message(self, message: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        data = memoryview(json.dumps(message, separators=(",", ":")).encode() + b"\n")
        while data:
            written = self.process.stdin.write(data)
            if not written:
                raise RuntimeError("MCP process stopped accepting requests")
            data = data[written:]
        self.process.stdin.flush()

    def _request(self, method: str, params: dict[str, Any], timeout: float = 12) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError(f"MCP process exited with code {self.process.returncode}")
        self.next_id += 1
        request_id = self.next_id
        request = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        try:
            self._write_message(request)
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("MCP process stopped accepting requests") from exc
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"MCP request {method} timed out")
            try:
                message = self.responses.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(f"MCP request {method} timed out") from exc
            if isinstance(message, BaseException):
                raise RuntimeError(str(message)) from message
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(f"MCP {method} error: {message['error'].get('message', message['error'])}")
            result = message.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"MCP {method} returned no result object")
            return result

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            message = result.get("content", [{}])[0].get("text", "tool call failed")
            raise RuntimeError(str(message))
        content = result.get("structuredContent")
        if isinstance(content, dict):
            if content.get("ok") is False:
                raise RuntimeError(str(content.get("error", "runtime query failed")))
            return content
        raise RuntimeError("MCP tool response has no structuredContent")

    def close(self) -> None:
        if self.process.poll() is None:
            if self.process.stdin is not None:
                try:
                    self.process.stdin.close()
                except OSError:
                    pass
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()


def run_watch(client: Any, writer: Any, duration_seconds: float | None) -> None:
    state = StreamState()
    started = time.monotonic()
    next_summary = started
    retry_delay = 1.0
    summary("watch started; waiting for native runtime")
    while duration_seconds is None or time.monotonic() - started < duration_seconds:
        try:
            status = client.call_tool("get_runtime_status", {})
            session = status.get("session_id")
            if not isinstance(session, str) or not session:
                raise RuntimeError("runtime status has no session_id")
            if state.session_id != session:
                state.session_id = session
                state.cursor = 0
                writer.write({"type": "session", "session_id": session, "root": status.get("root"), "observed_at_ms": int(time.time() * 1000)})
                summary(f"runtime connected; session={session}")
            response = client.call_tool(
                "read_runtime_metrics",
                {"after_sequence": state.cursor, "max_samples": 16, "wait_ms": 1000},
            )
            cursor_before = state.cursor
            samples, reset = state.accept(response)
            if reset:
                writer.write({"type": "session", "session_id": state.session_id, "root": response.get("root"), "observed_at_ms": int(time.time() * 1000)})
                summary(f"runtime session changed; cursor reset; session={state.session_id}")
                continue
            if response.get("gap"):
                writer.write({"type": "gap", "session_id": state.session_id, "lost_samples": response.get("lost_samples", 0), "after_sequence": cursor_before, "earliest_sequence": response.get("earliest_sequence"), "observed_at_ms": int(time.time() * 1000)})
            for sample in samples:
                writer.write({"type": "sample", "session_id": state.session_id, **sample})
            if samples and time.monotonic() >= next_summary:
                summary(f"received {len(samples)} samples; cursor={state.cursor}; latest={response.get('latest_sequence')}")
                next_summary = time.monotonic() + 60
            retry_delay = 1.0
        except (RuntimeError, TimeoutError, ValueError) as exc:
            if time.monotonic() >= next_summary:
                summary(f"runtime unavailable; retrying: {exc}")
                next_summary = time.monotonic() + 30
            remaining = None if duration_seconds is None else max(0.0, duration_seconds - (time.monotonic() - started))
            time.sleep(retry_delay if remaining is None else min(retry_delay, remaining))
            retry_delay = min(retry_delay * 1.5, 5.0)


def summary(message: str) -> None:
    if sys.stdout is not None:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcp-exe", required=True, help="Absolute E: path to voxel-editor-mcp.exe")
    parser.add_argument("--project-root", required=True, help="Absolute E: project root")
    parser.add_argument("--output-dir", required=True, help="Explicit E: directory for rotating JSONL logs")
    parser.add_argument("--duration-seconds", type=float, help=f"Optional smoke-run limit (maximum {MAX_DURATION_SECONDS} seconds)")
    args = parser.parse_args(argv)
    try:
        args.mcp_exe = require_e_path(args.mcp_exe, "--mcp-exe", must_exist=True)
        args.project_root = require_e_path(args.project_root, "--project-root", must_exist=True)
        if not args.mcp_exe.is_file():
            raise ValueError("--mcp-exe must point to a file")
        if not args.project_root.is_dir():
            raise ValueError("--project-root must point to a directory")
        args.output_dir = require_e_path(args.output_dir, "--output-dir", must_exist=False)
        validate_duration(args.duration_seconds)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    client: McpStdio | None = None
    try:
        client = McpStdio(args.mcp_exe, args.project_root)
        writer = RotatingJsonl(args.output_dir)
        run_watch(client, writer, args.duration_seconds)
        return 0
    except KeyboardInterrupt:
        summary("stopping")
        return 130
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"watch_runtime: {exc}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
