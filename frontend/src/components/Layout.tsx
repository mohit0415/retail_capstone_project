// src/components/Layout.tsx
// Sidebar + page outlet. The nav items are filtered by the screens the
// backend gave us for the role (RBAC), locked ones are shown greyed out so
// the user can see what other roles get.

import { NavLink, Outlet } from 'react-router-dom'
import { ROLE_LABELS, useAuth } from '../hooks/useAuth'
import type { Screen } from '../types'

const NAV: { screen: Screen; to: string; label: string; icon: string; hint: string }[] = [
  { screen: 'chat', to: '/chat', label: 'Policy Chat', icon: '💬', hint: 'every role' },
  { screen: 'dashboard', to: '/dashboard', label: 'Compliance Dashboard', icon: '📊', hint: 'store manager +' },
  { screen: 'review', to: '/review', label: 'Reviewer Console', icon: '🛡️', hint: 'reviewers' },
  { screen: 'slo', to: '/slo', label: 'SLO & Cost', icon: '📈', hint: 'reviewers' },
  { screen: 'corpus', to: '/corpus', label: 'Corpus Admin', icon: '🗂️', hint: 'admin only' },
]

function initials(name: string) {
  const parts = name.trim().split(/[\s@._-]+/).filter(Boolean)
  if (parts.length === 0) return '?'
  return (parts[0][0] + (parts[1]?.[0] || '')).toUpperCase()
}

export default function Layout() {
  const auth = useAuth()

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">RP</div>
          <div>
            <div className="brand-title">Retail Policy Intelligence</div>
            <div className="brand-sub">decision support · v4.1</div>
          </div>
        </div>

        <div className="nav-label">Workspace</div>
        {NAV.map((item) => {
          const allowed = auth.hasScreen(item.screen)
          if (!allowed) {
            return (
              <div key={item.to} className="nav-item locked" title={`needs: ${item.hint}`}>
                <span className="ico">🔒</span>
                <span>{item.label}</span>
              </div>
            )
          }
          return (
            <NavLink key={item.to} to={item.to} className={({ isActive }) => 'nav-item' + (isActive ? ' active' : '')}>
              <span className="ico">{item.icon}</span>
              <span>{item.label}</span>
            </NavLink>
          )
        })}

        <div className="sidebar-footer">
          <div className="user-pill">
            <div className="avatar">{initials(auth.displayName || 'U')}</div>
            <div>
              <div className="user-name" title={auth.me?.user_id}>
                {auth.displayName}
              </div>
              {auth.role && (
                <span className="chip violet" style={{ marginTop: 2 }}>
                  {ROLE_LABELS[auth.role]}
                </span>
              )}
            </div>
          </div>
          <button className="btn sm ghost" onClick={() => auth.logout()}>
            Log out
          </button>
        </div>
      </aside>

      <main className="main">
        <Outlet />
      </main>
    </div>
  )
}
