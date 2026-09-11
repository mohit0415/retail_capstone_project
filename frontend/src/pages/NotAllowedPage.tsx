import { Link, useLocation } from 'react-router-dom'
import { ROLE_LABELS, useAuth } from '../hooks/useAuth'

export default function NotAllowedPage() {
  const { role } = useAuth()
  const location = useLocation()
  const screen = (location.state as { screen?: string } | null)?.screen

  return (
    <div className="card center-card" style={{ margin: '60px auto' }}>
      <div style={{ fontSize: 40 }}>🔒</div>
      <h2>Not allowed for your role</h2>
      <p className="muted">
        The screen <b>{screen || 'you asked for'}</b> is not part of the{' '}
        <b>{role ? ROLE_LABELS[role] : 'current'}</b> role. The backend enforces this too (it would answer 403).
      </p>
      <Link className="btn primary" to="/chat">
        Back to chat
      </Link>
    </div>
  )
}
