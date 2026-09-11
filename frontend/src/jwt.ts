// src/jwt.ts
// Tiny helpers to PEEK inside an Auth0 token. There is NO signature check here -
// the backend does the real validation. We only use this to show, on the
// "No role" screen, which roles the ID token and the access token carry.

export function decodeJwtPayload(token: string | undefined | null): Record<string, unknown> | null {
  if (!token) return null
  try {
    const part = token.split('.')[1]
    if (!part) return null
    // base64url -> base64
    const base64 = part.replace(/-/g, '+').replace(/_/g, '/')
    const padded = base64 + '==='.slice((base64.length + 3) % 4)
    // atob gives a binary string, this turns it back into proper utf-8 text
    const json = decodeURIComponent(
      atob(padded)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join(''),
    )
    return JSON.parse(json)
  } catch (e) {
    console.log('could not decode token', e)
    return null
  }
}

// null  = the claim is NOT in the token at all
// []    = the claim is there but empty
export function rolesFromClaims(claims: Record<string, unknown> | null | undefined, claimName: string): string[] | null {
  if (!claims) return null
  const value = claims[claimName]
  if (value === undefined || value === null) return null
  if (Array.isArray(value)) return value.map(String)
  return [String(value)]
}

// roles that sit under some OTHER key (wrong namespace in the Auth0 Action)
export function otherRoleClaims(claims: Record<string, unknown> | null | undefined, claimName: string): string[] {
  if (!claims) return []
  return Object.keys(claims).filter((key) => key !== claimName && (key === 'roles' || key.endsWith('/roles')))
}
