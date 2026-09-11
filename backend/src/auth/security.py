# security.py
#
# Auth0 JWT validation + RBAC for the FastAPI backend.
#
# This follows the same approach as the practice-1 "securing the trip travel
# planner agent" backend (security/security.py there):
#
#   1. the React frontend logs the user in with the Auth0 SDK (useAuth hook)
#   2. it calls getAccessTokenSilently() and sends "Authorization: Bearer <token>"
#   3. here we download the JWKS (public keys) of the Auth0 tenant, check the
#      signature / audience / issuer / expiry with python-jose
#   4. the Auth0 Action "add-roles-to-tokens" puts the user's roles into the
#      token under "<namespace>/roles" - we read that claim and turn it into
#      the backend Role (store_associate ... admin) + its access scopes
#
# Nothing here mints tokens any more. Auth0 does that.

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from configs.settings import settings
from src.auth.rbac import access_scopes_for, can_ingest, can_review, resolve_role
from src.schemas.enums import Role

logger = logging.getLogger(__name__)

CHALLENGE = {"WWW-Authenticate": "Bearer"}

MISSING_TOKEN_DETAIL = (
    "this endpoint needs an 'Authorization: Bearer <token>' header and the request carried none. "
    "Log in through Auth0 in the frontend - the useAuth hook calls getAccessTokenSilently() and "
    "sends that access token on every call. In the Swagger page the token is not attached "
    "automatically: click Authorize and paste the Auth0 access token on its own, without the word Bearer."
)

# Simple JWKS cache (same idea as practice-1)
_jwks_cache: dict[str, Any] = {"keys": [], "fetched_at": 0.0}
JWKS_CACHE_SECONDS = 60 * 60  # 1 hour


def auth0_is_configured() -> bool:
    return bool(settings.auth0_domain and settings.api_audience)


def jwks_url() -> str:
    return f"https://{settings.auth0_domain}/.well-known/jwks.json"


def issuer() -> str:
    # Auth0 always puts a trailing slash in the "iss" claim
    return f"https://{settings.auth0_domain}/"


def algorithms() -> list[str]:
    return [a.strip() for a in settings.algorithms.split(",") if a.strip()] or ["RS256"]


def roles_claim_name() -> str:
    # e.g. https://stateful-agent.com/roles
    return f"{settings.auth0_role_namespace.rstrip('/')}/roles"


def departments_claim_name() -> str:
    # optional extra claim the Action can add from user.app_metadata.departments
    return f"{settings.auth0_role_namespace.rstrip('/')}/departments"


def _fetch_jwks(force: bool = False) -> dict[str, Any]:
    """Fetch JWKS from Auth0 with simple caching."""
    now = time.time()

    if force or not _jwks_cache["keys"] or (now - _jwks_cache["fetched_at"] > JWKS_CACHE_SECONDS):
        resp = requests.get(jwks_url(), timeout=5)
        resp.raise_for_status()
        _jwks_cache["keys"] = resp.json().get("keys", [])
        _jwks_cache["fetched_at"] = now
        logger.info("jwks fetched url=%s keys=%d", jwks_url(), len(_jwks_cache["keys"]))

    return _jwks_cache


def _get_rsa_key(unverified_header: dict[str, Any]) -> dict[str, Any] | None:
    jwks = _fetch_jwks()

    for key in jwks.get("keys", []):
        if key.get("kid") == unverified_header.get("kid"):
            return {
                "kty": key.get("kty"),
                "kid": key.get("kid"),
                "use": key.get("use"),
                "n": key.get("n"),
                "e": key.get("e"),
            }

    return None


class BearerToken(HTTPBearer):
    """HTTPBearer with a friendlier 401 when the header is missing."""

    async def __call__(self, request: Request) -> HTTPAuthorizationCredentials:
        try:
            return await super().__call__(request)
        except HTTPException:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=MISSING_TOKEN_DETAIL,
                headers=CHALLENGE,
            ) from None


bearer_scheme = BearerToken(auto_error=True)


def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict[str, Any]:
    """
    FastAPI dependency to validate a JWT from Auth0.
    Returns the decoded payload (claims) if validation succeeded.
    Raises HTTPException(401) on failure.
    """
    if not auth0_is_configured():
        logger.error("auth rejected: AUTH0_DOMAIN / API_AUDIENCE are not set in backend/.env")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth0 is not configured on the server. Set AUTH0_DOMAIN and API_AUDIENCE in backend/.env "
            "(same values as the frontend .env) and restart.",
        )

    token = credentials.credentials

    try:
        unverified_header = jwt.get_unverified_header(token)
    except JWTError:
        logger.warning("auth rejected: invalid token header")

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token header.",
            headers=CHALLENGE,
        ) from None

    try:
        rsa_key = _get_rsa_key(unverified_header)

        if not rsa_key:
            # refresh JWKS once and retry (in case the key rotated)
            _fetch_jwks(force=True)
            rsa_key = _get_rsa_key(unverified_header)
    except requests.RequestException as exc:
        logger.error("auth failed: could not download JWKS from %s: %s", jwks_url(), exc)

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"could not download the Auth0 signing keys from {jwks_url()}: {exc}",
        ) from exc

    if not rsa_key:
        logger.warning("auth rejected: no JWKS key matches kid=%s", unverified_header.get("kid"))

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unable to find appropriate key",
            headers=CHALLENGE,
        )

    try:
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=algorithms(),
            audience=settings.api_audience,
            issuer=issuer(),
        )
    except jwt.ExpiredSignatureError:
        logger.warning("auth rejected: token expired")

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers=CHALLENGE,
        ) from None
    except jwt.JWTClaimsError:
        logger.warning("auth rejected: wrong audience or issuer (expected aud=%s iss=%s)", settings.api_audience, issuer())

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect claims. Check audience and issuer.",
            headers=CHALLENGE,
        ) from None
    except Exception:
        logger.warning("auth rejected: token could not be parsed / signature invalid")

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unable to parse authentication token.",
            headers=CHALLENGE,
        ) from None

    return payload


@dataclass(slots=True)
class Principal:
    """Who is calling, as far as the rest of the backend cares."""

    user_id: str
    role: Role
    departments: list[str]
    access_scopes: list[str]
    auth0_roles: list[str] = field(default_factory=list)
    email: str = ""
    name: str = ""


def _as_list(value) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    return [str(v) for v in value]


def _no_role_detail(claims: dict[str, Any], raw_roles: Any, auth0_roles: list[str]) -> str:
    """Say WHY there is no role - the causes are fixed in different places in the Auth0 dashboard."""
    claim = roles_claim_name()
    valid = [r.value for r in Role]

    # 1. the claim is not in the access token at all
    if raw_roles is None:
        other_claims = [key for key in claims if key != claim and (key == "roles" or key.endswith("/roles"))]

        if other_claims:
            return (
                f"the access token has no '{claim}' claim, but it does have {other_claims}. The Auth0 Action uses "
                "a different namespace: make AUTH0_ROLE_NAMESPACE (backend/.env) and VITE_AUTH0_ROLE_NAMESPACE "
                "(frontend/.env) match it, restart both, and log in again."
            )

        return (
            f"the access token has no '{claim}' claim at all, so the Auth0 Action 'add-roles-to-tokens' is not "
            "adding roles to the ACCESS token (the backend never sees the ID token). In Auth0: Actions -> Library -> "
            "add-roles-to-tokens must call api.accessToken.setCustomClaim(...) and be Deployed, and it must be in "
            "Actions -> Triggers -> post-login. Then log out and log in again."
        )

    # 2. the Action runs, but the user has no role assigned
    if not auth0_roles:
        return (
            f"this Auth0 user has no role assigned: the token carries an empty '{claim}' claim. Assign one of "
            f"{valid} in Auth0 (User Management -> Users -> pick the user -> Roles tab -> Assign Roles), then log "
            "out and log in again so Auth0 issues a new token."
        )

    # 3. roles are there but the names do not match ours
    return (
        f"this Auth0 user has no recognised role: the token carries roles {auth0_roles} under the '{claim}' "
        f"claim, and none of them is one of {valid}. Rename the role in Auth0 or assign one of those, then log "
        "out and log in again."
    )


def current_principal(claims: dict[str, Any] = Depends(validate_token)) -> Principal:
    """Turn the Auth0 claims into a Principal (role + scopes). 403 when the user has no known role."""
    raw_roles = claims.get(roles_claim_name())
    auth0_roles = _as_list(raw_roles)
    role = resolve_role(auth0_roles)

    if role is None:
        logger.warning(
            "authorization denied: sub=%s has no recognised role (claim %s=%r, claim present=%s)",
            claims.get("sub"),
            roles_claim_name(),
            auth0_roles,
            raw_roles is not None,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=_no_role_detail(claims, raw_roles, auth0_roles),
        )

    namespace = settings.auth0_role_namespace.rstrip("/")

    return Principal(
        user_id=str(claims.get("sub", "")),
        role=role,
        departments=_as_list(claims.get(departments_claim_name())),
        access_scopes=access_scopes_for(role),
        auth0_roles=auth0_roles,
        email=str(claims.get(f"{namespace}/email") or claims.get("email") or ""),
        name=str(claims.get(f"{namespace}/name") or claims.get("name") or ""),
    )


def require_reviewer(principal: Principal = Depends(current_principal)) -> Principal:
    if not can_review(principal.role):
        logger.warning(
            "authorization denied: user=%s role=%s needs a reviewer role", principal.user_id, principal.role.value
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"this token carries the role '{principal.role.value}', and reviewing needs one of "
            f"compliance_officer, legal_reviewer or admin",
        )

    return principal


def require_corpus_admin(principal: Principal = Depends(current_principal)) -> Principal:
    if not can_ingest(principal.role):
        logger.warning(
            "authorization denied: user=%s role=%s needs admin to change the corpus",
            principal.user_id,
            principal.role.value,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"this token carries the role '{principal.role.value}', and changing the policy "
            f"corpus needs 'admin'. Uploading a document changes what every other role can retrieve.",
        )

    return principal
