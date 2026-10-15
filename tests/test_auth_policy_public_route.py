"""The logged-out login page can read ``GET /api/auth/policy``.

``static/login.html`` fetches the policy before anyone is signed in, to apply
the server's password minimum and reserved usernames during setup and signup.
The route describes its data as public, but ``AuthMiddleware`` had no
exemption for it, so the request returned 401 and the page silently fell back
to hard-coded constants (QA audit 2026-10-12, item 7).

The exemption is scoped to that one method and path. These tests run the real
``AuthMiddleware`` from ``app.py`` around the real auth router with a
configured account and no session, then check that the policy answers while
its neighbours, other methods on the same path, and look-alike paths stay
protected. ``app`` reads ``AUTH_ENABLED`` at import, so the probe runs in a
subprocess with an isolated environment, as in ``test_auth_root_path.py``.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]

POLICY_KEYS = {"password_min_length", "reserved_usernames", "signup_enabled", "session_days"}

_PROBE = textwrap.dedent(
    """
    import json
    import os
    import sys

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import app as app_module
    from core.auth import AuthManager
    from routes.auth_routes import setup_auth_routes

    manager = AuthManager(os.path.join(sys.argv[1], "auth-probe.json"))
    manager.create_user("probe-admin", "probe-password-123", is_admin=True)
    app_module.auth_manager = manager

    downstream = FastAPI()
    downstream.state.auth_manager = manager
    downstream.include_router(setup_auth_routes(manager))
    downstream.add_middleware(app_module.AuthMiddleware)
    client = TestClient(downstream)

    cases = {
        "get_policy": ("GET", "/api/auth/policy"),
        "post_policy": ("POST", "/api/auth/policy"),
        "put_policy": ("PUT", "/api/auth/policy"),
        "policy_child": ("GET", "/api/auth/policy/extra"),
        "policy_lookalike": ("GET", "/api/auth/policyx"),
        "users": ("GET", "/api/auth/users"),
        "totp_status": ("GET", "/api/auth/2fa/status"),
        "integrations": ("GET", "/api/auth/integrations"),
        "change_password": ("POST", "/api/auth/change-password"),
    }
    out = {}
    for name, (method, path) in cases.items():
        response = client.request(method, path)
        try:
            body = response.json()
        except ValueError:
            body = None
        out[name] = {"status": response.status_code, "body": body}
    print("RESULT=" + json.dumps(out, sort_keys=True))
    """
)


def _run_probe(tmp_path):
    env = os.environ.copy()
    env.update({
        "AUTH_ENABLED": "true",
        "CHROMADB_CONNECT_TIMEOUT": "0.01",
        "CHROMADB_HOST": "127.0.0.1",
        "CHROMADB_PORT": "9",
        "DATABASE_URL": f"sqlite:///{tmp_path / 'app.db'}",
        "LOCALHOST_BYPASS": "false",
        "ODYSSEUS_DATA_DIR": str(tmp_path),
        "ODYSSEUS_DISABLE_MCP": "1",
        "OPENAI_API_KEY": "",
        "PYTHONPATH": str(ROOT),
        "PYTHON_DOTENV_DISABLED": "1",
    })
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(tmp_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    line = next(
        (row for row in result.stdout.splitlines() if row.startswith("RESULT=")),
        None,
    )
    assert line is not None, result.stdout
    return json.loads(line.removeprefix("RESULT="))


def test_policy_is_public_and_only_the_policy_get_is_exempt(tmp_path):
    results = _run_probe(tmp_path)

    policy = results["get_policy"]
    assert policy["status"] == 200, policy
    # Nothing beyond static policy constants and the already-public signup
    # flag: no usernames, sessions, or settings.
    assert set(policy["body"]) == POLICY_KEYS
    assert isinstance(policy["body"]["password_min_length"], int)
    assert isinstance(policy["body"]["session_days"], int)
    assert isinstance(policy["body"]["signup_enabled"], bool)
    assert "probe-admin" not in json.dumps(policy["body"])

    for name in (
        "post_policy",
        "put_policy",
        "policy_child",
        "policy_lookalike",
        "users",
        "totp_status",
        "integrations",
        "change_password",
    ):
        assert results[name]["status"] == 401, (name, results[name])
