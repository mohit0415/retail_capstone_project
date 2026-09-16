// src/hooks/useAuth.tsx
//
// useAuth() = useAuth0() + the backend side of the login.
//
//   1. Auth0 gives us the user + the roles claim (added by the Auth0 Action
//      "add-roles-to-tokens" under "<namespace>/roles")
//   2. after login we POST the Azure OpenAI credentials from the login page
//      to the backend (/auth/azure) and then GET /auth/me which tells us the
//      effective role + which screens this role may open (RBAC)
//
// Everything the pages need (token, role, screens, logout...) comes from here.

/* eslint-disable react-refresh/only-export-components */
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react'
import { useAuth0 } from '@auth0/auth0-react'
import { ApiError, getMe, sendAzureCredentials } from '../apiService'
import { clearAzureCreds, loadAzureCreds } from '../azureCreds'
import { decodeJwtPayload, otherRoleClaims, rolesFromClaims } from '../jwt'
import type { MeResponse, Role, Screen } from '../types'

export const ROLE_NAMESPACE = (
  (import.meta.env.VITE_AUTH0_ROLE_NAMESPACE as string | undefined) || 'https://stateful-agent.com'
)
  .trim()
  .replace(/\/+$/, '')

export const ROLES_CLAIM = `${ROLE_NAMESPACE}/roles`

export type BackendStatus =
  | 'idle' // nothing done yet
  | 'connecting' // sending azure creds / loading /auth/me
  | 'ready' // all good
  | 'no_role' // token is fine but the user has no role assigned in Auth0
  | 'azure_rejected' // backend said the azure creds are wrong
  | 'error' // backend down etc

// what the two Auth0 tokens say about roles - shown on the "No role" screen
export interface TokenCheck {
  idTokenRoles: string[] | null // null = claim not in the ID token at all
  accessTokenRoles: string[] | null // null = claim not in the access token at all (this is what the backend reads)
  otherRoleClaims: string[] // roles found under a different namespace
}

export interface AuthState {
  // auth0 stuff
  isLoading: boolean
  isAuthenticated: boolean
  user: Record<string, unknown> | undefined
  displayName: string
  tokenRoles: string[] // roles straight from the ID token
  login: (returnTo?: string) => Promise<void>
  logout: () => Promise<void>
  getToken: () => Promise<string>
  // backend stuff
  me: MeResponse | null
  role: Role | null
  screens: Screen[]
  backendStatus: BackendStatus
  backendError: string
  tokenCheck: TokenCheck | null
  connectBackend: () => Promise<void>
  // non-fatal notice from POST /auth/azure (e.g. the embedding width no longer fits the corpus)
  backendWarning: string
  dismissBackendWarning: () => void
  refreshSession: () => Promise<void>
  hasScreen: (screen: Screen) => boolean
  isAdmin: boolean
}

// getAccessTokenSilently() failures come from Auth0 itself, not our backend.
// Its raw messages ("Consent required", "Login required") say nothing about
// what to do, so translate the two the user can actually act on.
function friendlyAuth0Error(err: unknown): string {
  const code = (err as { error?: string } | null)?.error
  if (code === 'consent_required')
    return 'Auth0 needs you to approve API access for this app once — click "Auth0 login" and accept the consent screen. (Auth0 always asks on localhost, then remembers your answer.)'
  if (code === 'login_required')
    return 'Your Auth0 session has expired — click "Auth0 login" to sign in again.'
  return err instanceof Error ? err.message : String(err)
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const {
    isLoading,
    isAuthenticated,
    user,
    loginWithRedirect,
    logout: auth0Logout,
    getAccessTokenSilently,
    getIdTokenClaims,
  } = useAuth0()

  const [me, setMe] = useState<MeResponse | null>(null)
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('idle')
  const [backendError, setBackendError] = useState('')
  const [backendWarning, setBackendWarning] = useState('')
  const [tokenCheck, setTokenCheck] = useState<TokenCheck | null>(null)
  const connecting = useRef(false)

  // Task 2 (practice-1) - read the roles the Auth0 Action put in the token
  const tokenRoles = useMemo(
    () => rolesFromClaims(user as Record<string, unknown> | undefined, ROLES_CLAIM) || [],
    [user],
  )

  const getToken = useCallback(async () => {
    // Task 3 - Get access token silently
    const token = await getAccessTokenSilently()
    console.log('access token (silent):', token)
    return token
  }, [getAccessTokenSilently])

  const login = useCallback(
    async (returnTo?: string) => {
      await loginWithRedirect({
        appState: { returnTo: returnTo || '/chat' },
        authorizationParams: { prompt: 'login' },
      })
    },
    [loginWithRedirect],
  )

  const logout = useCallback(async () => {
    clearAzureCreds()
    setMe(null)
    setBackendStatus('idle')
    setTokenCheck(null)
    // returnTo must be in "Allowed Logout URLs" of the Auth0 application.
    // practice-1 registered http://localhost:5173 (no path), so we use the origin only.
    await auth0Logout({ logoutParams: { returnTo: window.location.origin } })
  }, [auth0Logout])

  // look inside both tokens so the "No role" screen can say exactly what is wrong
  const checkTokens = useCallback(
    async (accessToken: string) => {
      let idClaims: Record<string, unknown> | null = null
      try {
        idClaims = ((await getIdTokenClaims()) as Record<string, unknown> | undefined) || null
      } catch (e) {
        console.log('could not read the ID token claims', e)
      }
      const accessClaims = decodeJwtPayload(accessToken)
      const check: TokenCheck = {
        idTokenRoles: rolesFromClaims(idClaims, ROLES_CLAIM),
        accessTokenRoles: rolesFromClaims(accessClaims, ROLES_CLAIM),
        otherRoleClaims: otherRoleClaims(accessClaims, ROLES_CLAIM),
      }
      console.log('token roles check', check)
      setTokenCheck(check)
    },
    [getIdTokenClaims],
  )

  // POST /auth/azure (creds from the login page) then GET /auth/me
  const connectBackend = useCallback(async () => {
    if (connecting.current) return
    connecting.current = true
    setBackendStatus('connecting')
    setBackendError('')
    setBackendWarning('')

    try {
      const token = await getToken()
      await checkTokens(token)
      const creds = loadAzureCreds()

      if (creds) {
        console.log('sending azure credentials to backend...')
        const applied = await sendAzureCredentials(token, creds, true)
        console.log(
          'azure applied:', applied.small_deployment, '/', applied.strong_deployment,
          '| embedding', applied.embedding_deployment, `(${applied.embedding_dimensions} dims)`,
          '| llamaparse', applied.llamaparse_source,
        )
        setBackendWarning(applied.warning || '')
      }

      const profile = await getMe(token)
      console.log('backend says my role is', profile.role, 'screens:', profile.screens)
      setMe(profile)
      setBackendStatus('ready')
    } catch (err) {
      console.log('connectBackend failed', err)
      if (err instanceof ApiError) {
        setBackendError(err.detail)
        if (err.status === 403) {
          setBackendStatus('no_role')
        } else if (err.status === 400 || err.status === 422) {
          // /auth/azure did not like the credentials -> back to the login page
          clearAzureCreds()
          setBackendStatus('azure_rejected')
        } else {
          setBackendStatus('error')
        }
      } else {
        const code = (err as { error?: string } | null)?.error
        if (code === 'consent_required' || code === 'login_required') {
          // Silent auth runs in a hidden iframe that cannot show Auth0's
          // consent or login dialog (and on localhost Auth0 never skips
          // consent). A full redirect CAN show it, and Auth0 remembers the
          // grant, so this only ever happens once per user.
          console.log(`silent auth said ${code} - redirecting to Auth0 to resolve it`)
          try {
            await loginWithRedirect({ appState: { returnTo: window.location.pathname } })
            return
          } catch (redirectErr) {
            // redirect could not start - fall through to the banner below
            console.log('redirect to Auth0 failed', redirectErr)
          }
        }
        setBackendError(friendlyAuth0Error(err))
        setBackendStatus('error')
      }
    } finally {
      connecting.current = false
    }
  }, [getToken, checkTokens, loginWithRedirect])

  // "I fixed the role in Auth0": the cached token still has the OLD roles,
  // so ask Auth0 for a brand new one. If that is not possible silently
  // (Allowed Web Origins / third party cookies) we just do a normal login again.
  const refreshSession = useCallback(async () => {
    setBackendStatus('connecting')
    setBackendError('')
    setBackendWarning('')
    try {
      console.log(
        'access token (forced refresh):',
        await getAccessTokenSilently({ cacheMode: 'off', timeoutInSeconds: 10 }),
      )
    } catch (err) {
      console.log('silent token refresh failed, doing a full login instead', err)
      await loginWithRedirect({
        appState: { returnTo: window.location.pathname },
        authorizationParams: { prompt: 'login' },
      })
      return
    }
    await connectBackend()
  }, [getAccessTokenSilently, loginWithRedirect, connectBackend])

  const screens: Screen[] = useMemo(() => (me ? me.screens : []), [me])

  const hasScreen = useCallback((screen: Screen) => screens.includes(screen), [screens])

  const displayName =
    (user && ((user.name as string) || (user.nickname as string) || (user.email as string))) || me?.user_id || ''

  const value: AuthState = {
    isLoading,
    isAuthenticated,
    user: user as Record<string, unknown> | undefined,
    displayName,
    tokenRoles,
    login,
    logout,
    getToken,
    me,
    role: me ? me.role : null,
    screens,
    backendStatus,
    backendError,
    tokenCheck,
    connectBackend,
    backendWarning,
    dismissBackendWarning: () => setBackendWarning(''),
    refreshSession,
    hasScreen,
    isAdmin: me?.role === 'admin' || tokenRoles.includes('admin'),
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext)
  if (!ctx) {
    throw new Error('useAuth must be used inside <AuthProvider>')
  }
  return ctx
}

// pretty labels for the role chips
export const ROLE_LABELS: Record<Role, string> = {
  store_associate: 'Store Associate',
  store_manager: 'Store Manager',
  compliance_officer: 'Compliance Officer',
  legal_reviewer: 'Legal Reviewer',
  admin: 'Admin',
}
