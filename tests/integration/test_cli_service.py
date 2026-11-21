"""Exercise installed CLI subprocesses against a separately running Emergence service."""

import json
import os
import socket
import subprocess
import sys
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from typing import Final
from uuid import UUID

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

PROJECT: Final = Path(__file__).resolve().parents[2]
CLI: Final = Path(sys.executable).with_name("emergence")
OPERATOR: Final = "synthetic-cli-operator-aaaaaaaaaaaaaaaa"
READER: Final = "synthetic-cli-reader-bbbbbbbbbbbbbbbb"
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class _Provider(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        # Deliberately omit usage: the service must hold the reservation for resolution.
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps(
            {
                "id": "offline-response",
                "model": "offline-model",
                "choices": [{"message": {"content": OPERATOR}, "finish_reason": "stop"}],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def upstream() -> Iterator[str]:
    with ThreadingHTTPServer(("127.0.0.1", 0), _Provider) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join(timeout=5)


@dataclass(frozen=True, slots=True)
class _Service:
    config: Path
    url: str
    env: dict[str, str]

    def run(
        self, *args: str, token_env: str | None = None, stdin: str = ""
    ) -> subprocess.CompletedProcess[str]:
        auth = ("--token-env", token_env) if token_env else ()
        return subprocess.run(
            [CLI, "--config", str(self.config), *auth, *args],
            cwd=PROJECT,
            env=self.env,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("EMERGENCE_")}


def _port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@contextmanager
def _start(service: _Service, *, legacy: bool = False) -> Generator[None]:
    launch = [] if legacy else ["serve"]
    with (
        subprocess.Popen(
            [CLI, *launch, "--config", str(service.config)],
            cwd=PROJECT,
            env=service.env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process,
        httpx.Client(base_url=service.url, trust_env=False, timeout=0.25) as client,
    ):
        try:
            deadline = monotonic() + 10
            wake = Event()
            ready = False
            while monotonic() < deadline and process.poll() is None:
                try:
                    ready = client.get("/health/ready").is_success
                except httpx.TransportError:
                    ready = False
                if ready:
                    break
                wake.wait(0.05)
            if not ready:
                if process.poll() is None:
                    process.terminate()
                output, _ = process.communicate(timeout=5)
                pytest.fail(f"Emergence did not become ready:\n{output}")
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=5)


@pytest.fixture(scope="module")
def launch_config(tmp_path_factory: pytest.TempPathFactory, upstream: str) -> _Service:
    directory = tmp_path_factory.mktemp("e006-cli")
    private = directory / "private"
    private.mkdir(mode=0o700)
    port = _port()
    path = directory / "cli.yaml"
    # JSON is a YAML subset; the normal service YAML loader reads this temporary configuration.
    path.write_text(
        json.dumps(
            {
                "port": port,
                "database_path": str(directory / "core.sqlite3"),
                "spool_path": str(private / "spool"),
                "disk_headroom_bytes": 1,
                "operator_token_env": "CLI_FIXTURE_OPERATOR",
                "reader_token_env": "CLI_FIXTURE_READER",
                "providers": {
                    "offline": {
                        "kind": "zai",
                        "base_url": upstream,
                        "credential_env": "CLI_FIXTURE_UPSTREAM",
                        "quota_policy": "local_limits_only",
                        "safety_limit_confirmed": True,
                        "models": {
                            "alias": {
                                "id": "offline-model",
                                "max_input_tokens": 32,
                                "max_output_tokens": 8,
                                "input_overhead_tokens": 0,
                                "input_bound_verified": True,
                                "input_bound_evidence": "offline fixture",
                                "output_bound_verified": True,
                                "bound_evidence": "offline fixture",
                            }
                        },
                    },
                },
                "prices": [
                    {
                        "version": "offline-price",
                        "provider": "offline",
                        "model": "alias",
                        "effective_at": "2020-01-01T00:00:00Z",
                        "input_per_million": "1",
                        "output_per_million": "1",
                        "source": "synthetic fixture",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    env = _clean_env() | {
        "CLI_FIXTURE_OPERATOR": OPERATOR,
        "CLI_FIXTURE_READER": READER,
        "CLI_FIXTURE_UPSTREAM": "synthetic-upstream-credential",
        "CLI_FIXTURE_INVALID": "invalid-token-xxxxxxxxxxxxxxxxxxxxxxxx",
    }
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=PROJECT,
        env=env | {"EMERGENCE_CONFIG": str(path)},
        check=True,
        timeout=15,
    )
    return _Service(path, f"http://127.0.0.1:{port}", env)


@pytest.fixture(scope="module")
def service(launch_config: _Service) -> Iterator[_Service]:
    with _start(launch_config):
        yield launch_config


def _success(result: subprocess.CompletedProcess[str]) -> dict[str, JsonValue]:
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    return JSON_OBJECT.validate_json(result.stdout)


def _error(result: subprocess.CompletedProcess[str], status: int) -> dict[str, JsonValue]:
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert result.stdout == ""
    body = JSON_OBJECT.validate_json(result.stderr)
    assert body["status"] == status
    assert OPERATOR not in result.stderr
    assert READER not in result.stderr
    return body


def test_public_health_and_legacy_startup(launch_config: _Service) -> None:
    # Keep this listener independent from the module-scoped explicitly launched service.
    port = _port()
    path = launch_config.config.with_name("legacy.yaml")
    config = JSON_OBJECT.validate_json(launch_config.config.read_bytes())
    config.update(port=port, providers={}, prices=[])
    config["database_path"] = str(path.with_suffix(".sqlite3"))
    config["spool_path"] = str(path.parent / "private" / "legacy-spool")
    path.write_text(json.dumps(config), encoding="utf-8")
    env = _clean_env()
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=PROJECT,
        env=env | {"EMERGENCE_CONFIG": str(path)},
        check=True,
        timeout=15,
    )
    legacy = _Service(path, f"http://127.0.0.1:{port}", env)
    with _start(legacy, legacy=True):
        assert _success(legacy.run("health"))["status"] == "ready"
        live = subprocess.run(
            [CLI, "health", "--live"],
            cwd=PROJECT,
            env=env | {"EMERGENCE_URL": legacy.url},
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        assert _success(live)["status"] == "alive"


def test_missing_token_cannot_read_providers(service: _Service) -> None:
    anonymous = _Service(service.config, service.url, _clean_env())
    _error(anonymous.run("providers"), 401)


@pytest.mark.parametrize(
    ("command", "field"),
    [
        (("providers",), "providers"),
        (("models",), "models"),
        (("resources",), "observations"),
        (("budgets", "show", "--provider", "offline"), "budgets"),
    ],
)
def test_reader_can_inspect_metadata(
    service: _Service, command: tuple[str, ...], field: str
) -> None:
    assert isinstance(_success(service.run(*command, token_env="CLI_FIXTURE_READER"))[field], list)


def test_invalid_and_reader_tokens_cannot_infer_or_adjust(service: _Service) -> None:
    args = ("inference", "test", "--provider", "offline", "--model", "alias")
    _error(service.run(*args, token_env="CLI_FIXTURE_INVALID", stdin="offline"), 401)
    _error(service.run(*args, token_env="CLI_FIXTURE_READER", stdin="offline"), 403)
    _error(
        service.run(
            "budgets",
            "adjust",
            "--provider",
            "offline",
            "--amount",
            "1",
            "--kind",
            "grant",
            "--reason",
            "reader cannot change the balance",
            token_env="CLI_FIXTURE_READER",
        ),
        403,
    )
    _error(service.run("usage", token_env="CLI_FIXTURE_READER"), 403)


def test_operator_adjusts_exact_decimal_budget(service: _Service) -> None:
    adjusted = _success(
        service.run(
            "budgets",
            "adjust",
            "--provider",
            "offline",
            "--amount",
            "0.123456789012",
            "--kind",
            "grant",
            "--reason",
            "offline CLI grant",
        )
    )
    assert adjusted["provider"] == "offline"
    assert isinstance(amount := adjusted["adjustments"], str)
    assert Decimal(amount) == Decimal("0.123456789012")


def test_operator_replays_inference_and_audits_no_charge_resolution(
    service: _Service, tmp_path: Path
) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("offline", encoding="utf-8")
    args = (
        "inference",
        "test",
        "--provider",
        "offline",
        "--model",
        "alias",
        "--max-output-tokens",
        "8",
        "--request-id",
        "offline-cli-key",
    )
    first = service.run(*args, "--prompt-file", str(prompt))
    assert first.returncode == 1 and first.stderr == "", (first.stdout, first.stderr)
    assert OPERATOR not in first.stdout
    body = JSON_OBJECT.validate_json(first.stdout)
    assert body["state"] == "ambiguous"
    assert body["replayed"] is False
    assert isinstance(request_id := body["request_id"], str)

    replay = service.run(*args, stdin="offline")
    assert replay.returncode == 1 and replay.stderr == ""
    receipt = JSON_OBJECT.validate_json(replay.stdout)
    assert receipt["replayed"] is True and receipt["request_id"] == request_id
    assert _error(service.run(*args, stdin="not the same input"), 409)["error"] == (
        "idempotency_conflict"
    )
    assert _success(service.run("inference", "status", request_id))["state"] == "ambiguous"
    resolution = _success(
        service.run(
            "inference",
            "resolve",
            request_id,
            "--no-charge",
            "--reason",
            "confirmed offline",
        )
    )
    assert resolution["state"] == "resolved"


def test_usage_passes_pagination_and_filters_to_service(service: _Service) -> None:
    page = _success(
        service.run(
            "usage",
            "--provider",
            "offline",
            "--model",
            "alias",
            "--start",
            "2020-01-01T00:00:00Z",
            "--end",
            "2020-01-02T00:00:00Z",
            "--limit",
            "2",
            "--offset",
            "3",
        )
    )
    assert (page["items"], page["limit"], page["offset"], page["has_more"]) == ([], 2, 3, False)


def test_missing_reply_exposes_retry_key_without_echoing_credentials(
    launch_config: _Service,
) -> None:
    with socket.socket() as closed:
        # Hold an unused local port without listening, so no request can leave this machine.
        closed.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{closed.getsockname()[1]}"
        args = (
            "--url",
            url,
            "inference",
            "test",
            "--provider",
            "offline",
            "--model",
            "alias",
        )
        missing = launch_config.run(*args, stdin="offline")
        redacted = launch_config.run(*args, "--request-id", OPERATOR, stdin="offline")

    for result in (missing, redacted):
        assert result.returncode == 3, (result.stdout, result.stderr)
        assert result.stdout == ""
        assert OPERATOR not in result.stderr
        assert JSON_OBJECT.validate_json(result.stderr)["outcome_uncertain"] is True
    key = JSON_OBJECT.validate_json(missing.stderr)["idempotency_key"]
    assert isinstance(key, str) and UUID(key).version == 4
    assert JSON_OBJECT.validate_json(redacted.stderr)["idempotency_key"] == "[REDACTED]"
