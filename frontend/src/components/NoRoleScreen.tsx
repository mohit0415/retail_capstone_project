// src/components/NoRoleScreen.tsx
//
// Shown when the backend answers 403 because the token has no usable role.
// There are 4 different reasons and each one is fixed in a different place in
// the Auth0 dashboard, so we look inside BOTH tokens and tell the user which one.

import type { ReactNode } from 'react'
import { ROLES_CLAIM, ROLE_NAMESPACE, useAuth } from '../hooks/useAuth'

const VALID_ROLES = ['store_associate', 'store_manager', 'compliance_officer', 'legal_reviewer', 'admin']

const ACTION_CODE = `exports.onExecutePostLogin = async (event, api) => {
  const namespace = '${ROLE_NAMESPACE}';
  if (event.authorization) {
    api.idToken.setCustomClaim(\`\${namespace}/roles\`, event.authorization.roles);
    api.accessToken.setCustomClaim(\`\${namespace}/roles\`, event.authorization.roles);
  }
};`

function showRoles(roles: string[] | null | undefined) {
  if (roles === undefined) return 'not checked'
  if (roles === null) return 'claim missing'
  if (roles.length === 0) return '[ ] empty'
  return '[ ' + roles.join(', ') + ' ]'
}

export default function NoRoleScreen() {
  const auth = useAuth()
  const check = auth.tokenCheck
  const idRoles = check ? check.idTokenRoles : undefined
  const accessRoles = check ? check.accessTokenRoles : undefined

  let title = 'This account has no usable role'
  let fix: ReactNode = null

  if (check && accessRoles === null && check.otherRoleClaims.length > 0) {
    title = 'The roles are under a different namespace'
    fix = (
      <ol>
        <li>
          The access token has roles under <span className="mono">{check.otherRoleClaims.join(', ')}</span>, the app
          expects <span className="mono">{ROLES_CLAIM}</span>.
        </li>
        <li>
          Set <span className="mono">AUTH0_ROLE_NAMESPACE</span> (backend/.env) and{' '}
          <span className="mono">VITE_AUTH0_ROLE_NAMESPACE</span> (frontend/.env) to that namespace, restart both.
        </li>
        <li>Click “Sign in again”.</li>
      </ol>
    )
  } else if (check && accessRoles === null && idRoles && idRoles.length > 0) {
    title = 'Your Auth0 Action only puts the roles in the ID token'
    fix = (
      <ol>
        <li>
          The ID token has <span className="mono">{showRoles(idRoles)}</span> but the backend reads the{' '}
          <b>access token</b>, which has no roles claim.
        </li>
        <li>
          Auth0 → <b>Actions → Library → add-roles-to-tokens</b>: make it look like the code below (the{' '}
          <span className="mono">api.accessToken</span> line is the missing one) → <b>Deploy</b>.
        </li>
        <li>Click “Sign in again”.</li>
      </ol>
    )
  } else if (check && accessRoles === null) {
    title = 'The Auth0 Action is not adding roles to the tokens'
    fix = (
      <ol>
        <li>
          Auth0 → <b>Actions → Library</b>: create or open <b>add-roles-to-tokens</b> with the code below → <b>Deploy</b>.
        </li>
        <li>
          Auth0 → <b>Actions → Triggers → post-login</b>: drag the Action into the flow → <b>Apply</b>.
        </li>
        <li>Click “Sign in again”.</li>
      </ol>
    )
  } else if (check && accessRoles && accessRoles.length === 0) {
    title = 'This user has no role assigned in Auth0'
    fix = (
      <ol>
        <li>
          Auth0 → <b>User Management → Users</b> → click <b>{auth.displayName || 'the user'}</b>.
        </li>
        <li>
          Open the <b>Roles</b> tab → <b>Assign Roles</b> → pick one of <span className="mono">{VALID_ROLES.join(', ')}</span>.
          If the role is not in the list, create it first under <b>User Management → Roles</b>.
        </li>
        <li>
          Come back and click “I fixed it — refresh token” (a token is issued at login, so the old one still has no role).
        </li>
      </ol>
    )
  } else if (check && accessRoles && accessRoles.length > 0) {
    title = 'The role name is not one this app knows'
    fix = (
      <ol>
        <li>
          The token carries <span className="mono">{showRoles(accessRoles)}</span>.
        </li>
        <li>
          In Auth0 assign one of <span className="mono">{VALID_ROLES.join(', ')}</span> (exact names), then click “I fixed
          it — refresh token”.
        </li>
      </ol>
    )
  }

  const showActionCode = check && accessRoles === null && check.otherRoleClaims.length === 0

  return (
    <div className="center-screen">
      <div className="card center-card" style={{ maxWidth: 640 }}>
        <div style={{ fontSize: 34 }}>🔐</div>
        <h2>{title}</h2>
        <p className="muted">
          Signed in as <b>{auth.displayName}</b>. Login worked, but the backend (RBAC) needs a role in the access token.
        </p>

        <div style={{ textAlign: 'left' }}>
          <h3 className="card-title" style={{ marginTop: 14 }}>
            What the tokens say
          </h3>
          <dl className="kv">
            <dt>ID token roles</dt>
            <dd className="mono">{showRoles(idRoles)}</dd>
            <dt>Access token roles</dt>
            <dd className="mono">
              {showRoles(accessRoles)} <span className="muted">← the backend checks this one</span>
            </dd>
            <dt>Claim name</dt>
            <dd className="mono">{ROLES_CLAIM}</dd>
          </dl>

          {fix && (
            <div className="alert warn" style={{ marginTop: 14 }}>
              <b>How to fix</b>
              {fix}
            </div>
          )}

          {showActionCode && (
            <details className="box" style={{ marginTop: 10 }} open>
              <summary>Action code (Login / post-login trigger)</summary>
              <pre className="json">{ACTION_CODE}</pre>
            </details>
          )}

          {auth.backendError && (
            <details className="box" style={{ marginTop: 10 }}>
              <summary>Backend message (403)</summary>
              <p className="small" style={{ margin: 0 }}>
                {auth.backendError}
              </p>
            </details>
          )}
        </div>

        <div className="row" style={{ justifyContent: 'center', marginTop: 16 }}>
          <button className="btn primary" onClick={() => auth.refreshSession()}>
            I fixed it — refresh token
          </button>
          <button className="btn" onClick={() => auth.login(window.location.pathname)}>
            Sign in again / switch user
          </button>
          <button className="btn ghost" onClick={() => auth.logout()}>
            Log out
          </button>
        </div>
      </div>
    </div>
  )
}
