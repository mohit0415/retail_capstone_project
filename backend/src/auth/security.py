from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from configs.settings import settings
from src.auth.rbac import access_scopes_for
from src.schemas.enums import Role

bearer_scheme = HTTPBearer(auto_error=True)


@dataclass(slots=True)
class Principal:
    user_id: str
    role: Role
    departments: list[str]
    access_scopes: list[str]


def issue_token(user_id: str, role: Role, departments: list[str]) -> tuple[str, int]:
    expires_in = settings.jwt_expiry_minutes * 60
    now = datetime.now(timezone.utc)

    claims = {
        "sub": user_id,
        "role": role.value,
        "departments": departments,
        "scopes": access_scopes_for(role),
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
    }

    token = jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)

    return token, expires_in


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")


def current_principal(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> Principal:
    claims = decode_token(credentials.credentials)

    try:
        role = Role(claims["role"])
    except (KeyError, ValueError):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="role claim missing or unknown")

    return Principal(
        user_id=claims["sub"],
        role=role,
        departments=claims.get("departments", []),
        access_scopes=claims.get("scopes", access_scopes_for(role)),
    )


def require_reviewer(principal: Principal = Depends(current_principal)) -> Principal:
    from src.auth.rbac import can_review

    if not can_review(principal.role):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="reviewer role required")

    return principal


def require_corpus_admin(principal: Principal = Depends(current_principal)) -> Principal:
    from src.auth.rbac import can_ingest

    if not can_ingest(principal.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="admin role required to change the policy corpus",
        )

    return principal
