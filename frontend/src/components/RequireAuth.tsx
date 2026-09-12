// src/components/RequireAuth.tsx
// Gate for every protected page:
//   not logged in           -> /login
//   no azure creds in tab   -> /login (they must be typed on the login page)
//   backend not connected   -> "connecting" screen (POST /auth/azure + GET /auth/me)
//   no role in Auth0        -> friendly 403 screen

import { useEffect, type ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'
import { loadAzureCreds } from '../azureCreds'
import NoRoleScreen from './NoRoleScreen'

export default function RequireAuth({ children }: { children: ReactNode }) {
  const auth = useAuth()
  const location = useLocation()
  const creds = loadAzureCreds()

  const needsConnect = auth.isAuthenticated && !!creds && auth.backendStatus === 'idle'

  useEffect(() => {
    if (needsConnect) {
      auth.connectBackend()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [needsConnect])

  if (!auth.isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />
  }

  if (!creds) {
    // logged in with Auth0 but this tab has no Azure credentials -> login page again
    return <Navigate to="/login" replace state={{ from: location.pathname, reason: 'azure' }} />
  }

  if (auth.backendStatus === 'azure_rejected') {
    return <Navigate to="/login" replace state={{ from: location.pathname, error: auth.backendError }} />
  }

  if (auth.backendStatus === 'idle' || auth.backendStatus === 'connecting') {
    return (
      <div className="center-screen">
        <div className="card center-card">
          <div className="brand" style={{ justifyContent: 'center' }}>
            <div className="brand-mark">RP</div>
          </div>
          <h2>Connecting to the backend</h2>
          <div className="connect-steps">
            <div className="stage done">
              <span className="ring" /> Signed in with Auth0 as <b>{auth.displayName}</b>
            </div>
            <div className="stage running">
              <span className="ring" /> Verifying the Azure OpenAI credentials from the login page
            </div>
            <div className="stage">
              <span className="ring" /> Resolving your role and screens (RBAC)
            </div>
          </div>
          <p className="muted small">This makes one tiny embedding call and one chat call to Azure.</p>
        </div>
      </div>
    )
  }

  if (auth.backendStatus === 'no_role') {
    // looks inside both tokens and says exactly what to change in Auth0
    return <NoRoleScreen />
  }

  if (auth.backendStatus === 'error') {
    return (
      <div className="center-screen">
        <div className="card center-card">
          <h2>Backend not reachable</h2>
          <div className="alert error" style={{ textAlign: 'left' }}>
            {auth.backendError || 'unknown error'}
          </div>
          <p className="muted small">
            Is the FastAPI backend running on <span className="mono">{import.meta.env.VITE_API_URL || 'http://localhost:8000'}</span>?
          </p>
          <div className="row" style={{ justifyContent: 'center' }}>
            <button className="btn primary" onClick={() => auth.connectBackend()}>
              Retry
            </button>
            <button className="btn ghost" onClick={() => auth.logout()}>
              Log out
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <>
      {auth.backendWarning && (
        <div className="alert warn" style={{ margin: '8px 12px', display: 'flex', gap: 12, alignItems: 'flex-start' }}>
          <span style={{ flex: 1 }}>{auth.backendWarning}</span>
          <button className="btn sm ghost" onClick={auth.dismissBackendWarning}>
            dismiss
          </button>
        </div>
      )}
      {children}
    </>
  )
}
