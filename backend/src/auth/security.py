from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from configs.settings import settings
from src.auth.rbac import access_scopes_for
from src.schemas.enums import Role

CHALLENGE = {"WWW-Authenticate": "Bearer"}

MISSING_TOKEN_DETAIL = (
    "this endpoint needs an 'Authorization: Bearer <token>' header and the request carried none "
    "that could be read. Mint one with POST /auth/token, then send it on every call. In the Swagger "
    "page the token is not attached automatically: click Authorize and paste the access_token value "
    "on its own, without the word Bearer."
)

DECODE_HINTS = {
    "Invalid header padding": "something is stuck to the front of the token. Swagger adds the word "
    "Bearer itself, so paste the value on its own.",
    "Invalid crypto padding": "something is stuck to the end of the token - a quotation mark, a "
    "space or a newline picked up while copying. Copy from the first character of 'eyJ' to the last "
    "character before the closing quote, or use the copy button on the response.",
    "Invalid payload padding": "the token was copied across a line break, so part of the middle is "
    "missing.",
    "Not enough segments": "only part of the token was copied. A JWT is three parts separated by "
    "dots and all three are needed.",
    "Signature verification failed": "the token is well formed but does not match this server's "
    "JWT_SECRET, so either characters were altered while copying or the secret changed after the "
    "token was minted.",
}

GENERIC_DECODE_HINT = "check the value against what /auth/token returned, character for character."


class BearerToken(HTTPBearer):
    async def __call__(self, request: Request) -> HTTPAuthorizationCredentials:
        try:
            return await super().__call__(request)
        except HTTPException:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=MISSING_TOKEN_DETAIL,
                headers=CHALLENGE,
            ) from None


@dataclass(slots=True)
class Principal:
    user_id: str
    role: Role
    departments: list[str]
    access_scopes: list[str]


bearer_scheme = BearerToken(auto_error=True)


def issue_token(user_id: str, role: Role, departments: list[str]) -> tuple[str, int]:
    expires_in = settings.jwt_expiry_minutes * 60
    now = datetime.now(UTC)

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
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"this token expired. Tokens last {settings.jwt_expiry_minutes} minutes; mint a new one with POST /auth/token.",
            headers=CHALLENGE,
        ) from None
    except jwt.InvalidTokenError as exc:
        hint = DECODE_HINTS.get(str(exc), GENERIC_DECODE_HINT)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"this token could not be verified ({exc}): {hint}",
            headers=CHALLENGE,
        ) from exc


def current_principal(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> Principal:
    claims = decode_token(credentials.credentials)

    try:
        role = Role(claims["role"])
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="role claim missing or unknown"
        ) from exc

    return Principal(
        user_id=claims["sub"],
        role=role,
        departments=claims.get("departments", []),
        access_scopes=claims.get("scopes", access_scopes_for(role)),
    )


def require_reviewer(principal: Principal = Depends(current_principal)) -> Principal:
    from src.auth.rbac import can_review

    if not can_review(principal.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"this token carries the role '{principal.role.value}', and reviewing needs one of "
            f"compliance_officer, legal_reviewer or admin",
        )

    return principal


def require_corpus_admin(principal: Principal = Depends(current_principal)) -> Principal:
    from src.auth.rbac import can_ingest

    if not can_ingest(principal.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"this token carries the role '{principal.role.value}', and changing the policy "
            f"corpus needs 'admin'. Uploading a document changes what every other role can retrieve.",
        )

    return principal
