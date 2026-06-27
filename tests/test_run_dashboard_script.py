from __future__ import annotations

import shutil
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_dashboard.ps1"
PWSH = shutil.which("pwsh")
WINDOWS_POWERSHELL = shutil.which("powershell") or shutil.which("powershell.exe")


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/_stcore/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


class _OccupiedHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def _run_dashboard_dry_run(
    port: int,
    search_limit: int = 3,
    executable: str | None = PWSH,
) -> subprocess.CompletedProcess[str]:
    if executable is None:
        pytest.skip("PowerShell executable is not available")
    return subprocess.run(
        [
            executable,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Port",
            str(port),
            "-PortSearchLimit",
            str(search_limit),
            "-DryRun",
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )


def _http_server_with_free_next(handler: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, int]:
    for _ in range(50):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = int(server.server_address[1])
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", port + 1))
        except OSError:
            server.server_close()
        else:
            probe.close()
            return server, port
    raise RuntimeError("Could not reserve an HTTP server port with a free next port")


def test_run_dashboard_dry_run_reuses_healthy_dashboard_port():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = _run_dashboard_dry_run(port)
    finally:
        server.shutdown()
        server.server_close()

    assert result.returncode == 0, result.stderr
    assert f"Reusing healthy dashboard: http://localhost:{port}" in result.stdout


def test_run_dashboard_dry_run_selects_next_free_port_when_target_is_occupied():
    server, port = _http_server_with_free_next(_OccupiedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = _run_dashboard_dry_run(port, search_limit=2)
    finally:
        server.shutdown()
        server.server_close()

    assert result.returncode == 0, result.stderr
    assert f"Port {port} is occupied; trying next port." in result.stdout
    assert f"Starting dashboard: http://localhost:{port + 1}" in result.stdout


def test_run_dashboard_script_is_ascii_for_windows_powershell_compatibility():
    assert SCRIPT.read_bytes().isascii()


def test_run_dashboard_dry_run_works_with_windows_powershell_when_available():
    if WINDOWS_POWERSHELL is None:
        pytest.skip("Windows PowerShell is not available")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _HealthHandler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = _run_dashboard_dry_run(port, executable=WINDOWS_POWERSHELL)
    finally:
        server.shutdown()
        server.server_close()

    assert result.returncode == 0, result.stderr
    assert f"Reusing healthy dashboard: http://localhost:{port}" in result.stdout
