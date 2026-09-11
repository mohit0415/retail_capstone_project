// src/components/RequireScreen.tsx
// Second RBAC gate: the backend told us (in /auth/me) which screens this role
// may open. If the role does not have the screen we show the "not allowed" page.
// (The backend also checks every endpoint itself, this is just for the UI.)

import type { ReactNode } from 'react'
import { Navigate } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'
import type { Screen } from '../types'

export default function RequireScreen({ screen, children }: { screen: Screen; children: ReactNode }) {
  const { hasScreen } = useAuth()

  if (!hasScreen(screen)) {
    console.log('RBAC: screen', screen, 'is not allowed for this role')
    return <Navigate to="/not-allowed" replace state={{ screen }} />
  }

  return <>{children}</>
}
