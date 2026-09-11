import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, useNavigate } from 'react-router-dom'
import { Auth0Provider, type AppState } from '@auth0/auth0-react'
import './styles.css'
import App from './App.tsx'
import { AuthProvider } from './hooks/useAuth.tsx'
import { DevMockAuth0Provider } from './auth/devMockAuth.tsx'

const domain = import.meta.env.VITE_AUTH0_DOMAIN as string | undefined
const clientId = import.meta.env.VITE_AUTH0_CLIENT_ID as string | undefined
const audience = import.meta.env.VITE_AUTH0_AUDIENCE as string | undefined
const authMode = (import.meta.env.VITE_AUTH_MODE as string | undefined) || 'auth0'

if (authMode !== 'mock' && (!domain || !clientId || !audience)) {
  throw new Error('Missing required Auth0 environment variables. Please check your .env file.')
}

// Task 2: Wrap the App component with Auth0Provider and configure it with domain,
// clientId, and audience. onRedirectCallback sends the user back to the page
// they wanted after the Auth0 Universal Login.
// eslint-disable-next-line react-refresh/only-export-components
function Auth0ProviderWithNavigate({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate()

  const onRedirectCallback = (appState?: AppState) => {
    navigate((appState && appState.returnTo) || '/chat', { replace: true })
  }

  if (authMode === 'mock') {
    console.log('%c[dev] VITE_AUTH_MODE=mock - Auth0 is faked, do not use this in production', 'color:orange')
    return <DevMockAuth0Provider>{children}</DevMockAuth0Provider>
  }

  return (
    <Auth0Provider
      domain={domain!}
      clientId={clientId!}
      authorizationParams={{
        redirect_uri: window.location.origin,
        audience: audience,
      }}
      cacheLocation="localstorage"
      onRedirectCallback={onRedirectCallback}
    >
      {children}
    </Auth0Provider>
  )
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Auth0ProviderWithNavigate>
        <AuthProvider>
          <App />
        </AuthProvider>
      </Auth0ProviderWithNavigate>
    </BrowserRouter>
  </StrictMode>,
)
