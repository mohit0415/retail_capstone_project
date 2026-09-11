// src/auth/devMockAuth.tsx
//
// DEV ONLY. When VITE_AUTH_MODE=mock this fakes the Auth0 context so you can
// look at the screens without an Auth0 tenant (e.g. for screenshots or when
// working on CSS). The backend still needs a real token, so use it with a
// mock backend or with AUTH0_DOMAIN left empty... it is NOT for production.
//
// VITE_MOCK_ROLE=admin | store_manager | ... | none   (none = user without a role)
//
// It provides the same context object that @auth0/auth0-react's provider
// gives, so useAuth0() in the app keeps working unchanged.

import { Auth0Context, initialContext, type Auth0ContextInterface, type IdToken, type User } from '@auth0/auth0-react'
import type { ReactNode } from 'react'
import { ROLES_CLAIM } from '../hooks/useAuth'

const MOCK_ROLE = (import.meta.env.VITE_MOCK_ROLE as string | undefined) || 'admin'
const MOCK_ROLES = MOCK_ROLE === 'none' ? [] : [MOCK_ROLE]

function base64url(text: string) {
  return btoa(text).replace(/=+$/, '').replace(/\+/g, '-').replace(/\//g, '_')
}

// unsigned fake token, only so the "what the tokens say" box has something to read
const MOCK_ACCESS_TOKEN =
  base64url(JSON.stringify({ alg: 'none', typ: 'JWT' })) +
  '.' +
  base64url(JSON.stringify({ sub: 'auth0|mock-user', [ROLES_CLAIM]: MOCK_ROLES })) +
  '.mock'

export function DevMockAuth0Provider({ children }: { children: ReactNode }) {
  const user: User = {
    sub: 'auth0|mock-user',
    name: 'Mohit (mock)',
    email: 'mock@example.com',
    [ROLES_CLAIM]: MOCK_ROLES,
  }

  const value: Auth0ContextInterface<User> = {
    ...initialContext,
    isLoading: false,
    isAuthenticated: true,
    user,
    getAccessTokenSilently: (async () => MOCK_ACCESS_TOKEN) as Auth0ContextInterface['getAccessTokenSilently'],
    getIdTokenClaims: async () => ({ __raw: 'mock', ...user }) as IdToken,
    loginWithRedirect: async () => {
      console.log('[mock auth] loginWithRedirect called')
    },
    logout: async () => {
      console.log('[mock auth] logout called')
      window.location.href = '/login'
    },
  }

  return <Auth0Context.Provider value={value}>{children}</Auth0Context.Provider>
}
