"""Shared fixtures.

Everything here is opt-in (no autouse) so the existing pure-logic tests keep
running exactly as before. The fixtures exist so the API, graph and tool tests
never touch Postgres, SMTP, Langfuse or an LLM endpoint.
"""

import time
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt

from src.auth.rbac import access_scopes_for
from src.auth.security import Principal
from src.schemas.enums import Role

# ---------------------------------------------------------------------------
# Auth0 stand-in: a throwaway RSA key pair signs the test tokens and its public
# half is planted in the JWKS cache, so no test ever talks to auth0.com.
# ---------------------------------------------------------------------------

TEST_AUTH0_DOMAIN = "test-tenant.auth0.com"
TEST_API_AUDIENCE = "https://api.test-policy.local"
TEST_ROLE_NAMESPACE = "https://stateful-agent.com"
TEST_KID = "test-key-1"

_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_other_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _pem(key) -> str:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def _b64(number: int, length: int) -> str:
    import base64

    return base64.urlsafe_b64encode(number.to_bytes(length, "big")).rstrip(b"=").decode()


def fake_jwks() -> dict:
    numbers = _private_key.public_key().public_numbers()

    return {
        "keys": [
            {
                "kty": "RSA",
                "kid": TEST_KID,
                "use": "sig",
                "alg": "RS256",
                "n": _b64(numbers.n, 256),
                "e": _b64(numbers.e, 3),
            }
        ]
    }


def mint_auth0_token(
    role: Role | str | list | None = Role.COMPLIANCE_OFFICER,
    user_id: str = "mohit",
    departments: list[str] | None = None,
    *,
    audience: str = TEST_API_AUDIENCE,
    issuer: str | None = None,
    expires_in: timedelta = timedelta(hours=1),
    kid: str = TEST_KID,
    wrong_key: bool = False,
    extra_claims: dict | None = None,
    omit_roles_claim: bool = False,
) -> str:
    """Sign a token the way Auth0 would (RS256, roles under the namespace claim)."""
    if role is None:
        roles: list[str] = []
    elif isinstance(role, list):
        roles = [r.value if isinstance(r, Role) else str(r) for r in role]
    else:
        roles = [role.value if isinstance(role, Role) else str(role)]

    now = datetime.now(UTC)
    claims = {
        "iss": issuer or f"https://{TEST_AUTH0_DOMAIN}/",
        "sub": user_id,
        "aud": audience,
        "iat": now - timedelta(seconds=5),
        "exp": now + expires_in,
        f"{TEST_ROLE_NAMESPACE}/roles": roles,
        f"{TEST_ROLE_NAMESPACE}/departments": departments or [],
    }
    if omit_roles_claim:
        # like an Auth0 Action that only calls api.idToken.setCustomClaim
        claims.pop(f"{TEST_ROLE_NAMESPACE}/roles")

    claims.update(extra_claims or {})

    key = _other_private_key if wrong_key else _private_key

    return jwt.encode(claims, _pem(key), algorithm="RS256", headers={"kid": kid})


@pytest.fixture
def auth0(monkeypatch):
    """Point the backend at the fake tenant and pre-fill the JWKS cache."""
    from configs.settings import settings
    from src.auth import security

    monkeypatch.setattr(settings, "auth0_domain", TEST_AUTH0_DOMAIN)
    monkeypatch.setattr(settings, "api_audience", TEST_API_AUDIENCE)
    monkeypatch.setattr(settings, "algorithms", "RS256")
    monkeypatch.setattr(settings, "auth0_role_namespace", TEST_ROLE_NAMESPACE)
    monkeypatch.setattr(security, "_jwks_cache", {"keys": fake_jwks()["keys"], "fetched_at": time.time()})

    def _no_network(force: bool = False):
        return security._jwks_cache

    monkeypatch.setattr(security, "_fetch_jwks", _no_network)

    return mint_auth0_token


class _NoDatabase(RuntimeError):
    pass


@pytest.fixture
def no_db(monkeypatch):
    """Make every database touch fail fast instead of waiting on a connection pool.

    ``configs.database.get_pool`` is what every ``read_only_connection`` /
    ``writable_connection`` goes through, so raising there covers audit writes,
    the escalation queue, conversation history and SLO history in one place.
    The audit writer is silenced separately so a stubbed graph run does not
    trip the audit circuit breaker three times per test.
    """
    from configs import database
    from src.core import audit

    def _refuse():
        raise _NoDatabase("tests never open a database connection")

    monkeypatch.setattr(database, "get_pool", _refuse)
    monkeypatch.setattr(database, "open_pool", _refuse)
    monkeypatch.setattr(audit, "write_audit", lambda record: None)

    return _refuse


@pytest.fixture
def no_tracing(monkeypatch):
    """Keep Langfuse out of the picture: no handler, no flush, no network."""
    from src.observability import langfuse_callback, tracing

    monkeypatch.setattr(tracing, "get_callback_handlers", lambda: [])
    monkeypatch.setattr(
        langfuse_callback,
        "setup_langfuse_callback",
        lambda config, **kwargs: (config, None),
    )
    monkeypatch.setattr(langfuse_callback, "flush_langfuse_traces", lambda handler: None)


@pytest.fixture
def memory_graph(monkeypatch, no_db, no_tracing):
    """Reset the compiled-graph singleton and force the in-memory checkpointer.

    Returns the ``src.graph.builder`` module so a test can swap node functions
    on it *before* the first ``get_compiled_graph()`` call.
    """
    from langgraph.checkpoint.memory import MemorySaver

    from src.graph import builder

    monkeypatch.setattr(builder, "_compiled", None)
    monkeypatch.setattr(builder, "_checkpointer", None)
    monkeypatch.setattr(builder, "build_checkpointer", MemorySaver)

    yield builder

    builder._compiled = None
    builder._checkpointer = None


def make_principal(role: Role, user_id: str = "mohit", departments: list[str] | None = None) -> Principal:
    return Principal(
        user_id=user_id,
        role=role,
        departments=departments or [],
        access_scopes=access_scopes_for(role),
    )


def base_state(query: str = "What is the retention period for customer invoices?", **overrides) -> dict:
    """A fully-populated AgentState the way ``/ask`` would build it."""
    from src.graph.state import initial_state

    started = time.monotonic()

    state = initial_state(
        request_id=overrides.pop("request_id", "req-test"),
        thread_id=overrides.pop("thread_id", "thread-test"),
        user_id=overrides.pop("user_id", "mohit"),
        role=overrides.pop("role", Role.COMPLIANCE_OFFICER.value),
        access_scopes=overrides.pop("access_scopes", access_scopes_for(Role.COMPLIANCE_OFFICER)),
        raw_query=query,
        started_ts=started,
        deadline_ts=overrides.pop("deadline_ts", started + 30.0),
        token_budget=overrides.pop("token_budget", 48000),
    )

    state.update(overrides)

    return state
