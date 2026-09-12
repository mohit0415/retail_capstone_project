// src/pages/LoginPage.tsx
//
// The only place where the Azure OpenAI credentials are typed.
//   step 1: Azure OpenAI endpoint / key / deployments  -> sessionStorage
//   step 2: "Continue with Auth0" (Universal Login)
// After the redirect back, RequireAuth sends the creds to POST /auth/azure.

import { useEffect, useState, type FormEvent } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'
import { DEFAULT_AZURE, EMBEDDING_CHOICES, dimensionsFor, loadAzureCreds, maskKey, saveAzureCreds } from '../azureCreds'
import { getHealth } from '../apiService'
import type { AzureCredentials, HealthResponse } from '../types'

export default function LoginPage() {
  const auth = useAuth()
  const location = useLocation()
  const state = (location.state as { from?: string; reason?: string; error?: string } | null) || {}

  const saved = loadAzureCreds()
  const [creds, setCreds] = useState<AzureCredentials>(saved || DEFAULT_AZURE)
  const [showKey, setShowKey] = useState(false)
  const [showParseKey, setShowParseKey] = useState(false)
  const [error, setError] = useState(state.error || '')
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthErr, setHealthErr] = useState('')
  const [submitting, setSubmitting] = useState(false)

  // small backend ping so the user knows if the API is up before logging in
  useEffect(() => {
    getHealth()
      .then((h) => setHealth(h))
      .catch((e) => setHealthErr(e instanceof Error ? e.message : String(e)))
  }, [])

  // already signed in + creds in this tab + backend happy -> go in
  if (auth.isAuthenticated && saved && auth.backendStatus === 'ready') {
    return <Navigate to={state.from && state.from !== '/login' ? state.from : '/chat'} replace />
  }

  function update(field: keyof AzureCredentials, value: string) {
    setCreds((c) => ({ ...c, [field]: value }))
  }

  function validate(): string {
    if (!creds.endpoint.trim().startsWith('https://')) return 'The endpoint must start with https:// (e.g. https://my-resource.openai.azure.com/)'
    if (creds.api_key.trim().length < 8) return 'Please paste the Azure OpenAI API key'
    if (!creds.small_deployment.trim() || !creds.strong_deployment.trim() || !creds.embedding_deployment.trim())
      return 'All three deployment names are needed'
    return ''
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault()
    const problem = validate()
    if (problem) {
      setError(problem)
      return
    }
    setError('')
    setSubmitting(true)

    const embedding = creds.embedding_deployment.trim()
    const clean: AzureCredentials = {
      endpoint: creds.endpoint.trim(),
      api_key: creds.api_key.trim(),
      api_version: creds.api_version.trim() || DEFAULT_AZURE.api_version,
      small_deployment: creds.small_deployment.trim(),
      strong_deployment: creds.strong_deployment.trim(),
      embedding_deployment: embedding,
      // sent explicitly so the width the user was shown is the width the backend uses
      embedding_dimensions: dimensionsFor(embedding) || undefined,
      llamaparse_api_key: creds.llamaparse_api_key.trim(),
    }
    saveAzureCreds(clean)
    console.log('azure creds saved to sessionStorage (key masked):', maskKey(clean.api_key))

    if (auth.isAuthenticated) {
      // already have an Auth0 session, just (re)connect the backend.
      // when it works backendStatus becomes "ready" and the <Navigate> above kicks in
      await auth.connectBackend()
      setSubmitting(false)
      return
    }

    // Task 2 (practice-1): Auth0 Universal Login
    await auth.login(state.from && state.from !== '/login' ? state.from : '/chat')
  }

  const chosenDimensions = dimensionsFor(creds.embedding_deployment)

  // errors from the backend connect (wrong azure key etc) also show up here
  const shownError =
    error || (auth.backendStatus === 'azure_rejected' || auth.backendStatus === 'error' ? auth.backendError : '')

  return (
    <div className="login-wrap">
      <section className="login-hero">
        <div className="brand" style={{ padding: 0 }}>
          <div className="brand-mark">RP</div>
          <div>
            <div className="brand-title">Retail Policy Intelligence</div>
            <div className="brand-sub">agentic compliance decision support</div>
          </div>
        </div>
        <h1>
          Ask the policy corpus,
          <br />
          <span>get a certified answer.</span>
        </h1>
        <p>
          Every question runs through guardrails, risk classification, RAG / SQL routing, validation and a
          confidence gate. High-risk answers wait for a human reviewer. What you can see depends on your role.
        </p>

        <div className="role-matrix">
          <div className="card">
            <b>Store Associate</b>
            <span>Chat over privacy, infosec and anti-bribery policies</span>
          </div>
          <div className="card">
            <b>Store Manager</b>
            <span>+ vendor &amp; retention policies, compliance dashboard</span>
          </div>
          <div className="card">
            <b>Compliance Officer / Legal Reviewer</b>
            <span>+ GDPR, ISO 27001, reviewer console, SLO dashboard</span>
          </div>
          <div className="card">
            <b>Admin</b>
            <span>everything + corpus ingestion and cache control</span>
          </div>
        </div>

        <div className="row" style={{ marginTop: 6 }}>
          {health ? (
            <>
              <span className="chip teal">
                <span className="dot ok" /> backend {health.status}
              </span>
              <span className="chip">
                db {health.database} · queue {health.escalation_queue_depth ?? '-'}
              </span>
              <span className={'chip ' + (health.azure_configured ? 'sky' : 'amber')}>
                azure on server: {health.azure_configured ? 'configured' : 'waiting for your credentials'}
              </span>
            </>
          ) : healthErr ? (
            <span className="chip rose">
              <span className="dot bad" /> backend offline ({healthErr})
            </span>
          ) : (
            <span className="chip">
              <span className="spinner" /> checking backend
            </span>
          )}
        </div>
      </section>

      <section className="login-panel">
        <form className="login-card" onSubmit={onSubmit}>
          <div className="steps">
            <div className="step active">
              <span className="n">1</span> Azure OpenAI
            </div>
            <div className={'step ' + (auth.isAuthenticated ? 'done' : '')}>
              <span className="n">2</span> Auth0 login
            </div>
            <div className="step">
              <span className="n">3</span> Role &amp; screens
            </div>
          </div>

          <h2 style={{ margin: '0 0 4px', fontSize: 18 }}>Azure OpenAI credentials</h2>
          <p className="muted small" style={{ marginTop: 0 }}>
            Typed here only. They stay in this browser tab and are sent to the backend after you sign in
            with Auth0 (one small test call verifies them).
          </p>

          {state.reason === 'azure' && !shownError && (
            <div className="alert info" style={{ marginBottom: 12 }}>
              You are signed in as <b>{auth.displayName}</b>. Enter the Azure OpenAI credentials for this
              session to continue.
            </div>
          )}

          {shownError && (
            <div className="alert error" style={{ marginBottom: 12 }}>
              {shownError}
            </div>
          )}

          <div className="field">
            <label className="label">Endpoint</label>
            <input
              className="input mono"
              placeholder="https://my-resource.openai.azure.com/"
              value={creds.endpoint}
              onChange={(e) => update('endpoint', e.target.value)}
              autoComplete="off"
            />
          </div>

          <div className="field">
            <label className="label">API key</label>
            <div className="row" style={{ flexWrap: 'nowrap' }}>
              <input
                className="input mono"
                type={showKey ? 'text' : 'password'}
                placeholder="paste the key"
                value={creds.api_key}
                onChange={(e) => update('api_key', e.target.value)}
                autoComplete="off"
              />
              <button type="button" className="btn sm" onClick={() => setShowKey((s) => !s)}>
                {showKey ? 'hide' : 'show'}
              </button>
            </div>
          </div>

          <div className="grid-2">
            <div className="field">
              <label className="label">API version</label>
              <input className="input mono" value={creds.api_version} onChange={(e) => update('api_version', e.target.value)} />
            </div>
            <div className="field">
              <label className="label">Embedding deployment</label>
              <select
                className="input mono"
                value={creds.embedding_deployment}
                onChange={(e) => update('embedding_deployment', e.target.value)}
              >
                {EMBEDDING_CHOICES.map((c) => (
                  <option key={c.deployment} value={c.deployment}>
                    {c.deployment}
                  </option>
                ))}
              </select>
              <span className="hint">
                {chosenDimensions
                  ? `${chosenDimensions} dimensions - the corpus must be ingested with this same model`
                  : 'vector width is derived on the server'}
              </span>
            </div>
            <div className="field">
              <label className="label">Small deployment (gpt-4o-mini tier)</label>
              <input
                className="input mono"
                value={creds.small_deployment}
                onChange={(e) => update('small_deployment', e.target.value)}
              />
            </div>
            <div className="field">
              <label className="label">Strong deployment (gpt-4o tier)</label>
              <input
                className="input mono"
                value={creds.strong_deployment}
                onChange={(e) => update('strong_deployment', e.target.value)}
              />
            </div>
          </div>

          <div className="field">
            <label className="label">LlamaParse API key (optional)</label>
            <div className="row" style={{ flexWrap: 'nowrap' }}>
              <input
                className="input mono"
                type={showParseKey ? 'text' : 'password'}
                placeholder="llx-... (your own LlamaCloud key)"
                value={creds.llamaparse_api_key}
                onChange={(e) => update('llamaparse_api_key', e.target.value)}
                autoComplete="off"
              />
              <button type="button" className="btn sm" onClick={() => setShowParseKey((s) => !s)}>
                {showParseKey ? 'hide' : 'show'}
              </button>
            </div>
            <span className="hint">
              Used only for uploads that carry tables or diagrams, and billed to this key. Leave it blank and
              those uploads are refused with a message instead of falling back to the server&apos;s key.
            </span>
          </div>

          <button className="btn primary" type="submit" disabled={submitting} style={{ width: '100%', justifyContent: 'center', marginTop: 4 }}>
            {submitting ? (
              <>
                <span className="spinner" /> working...
              </>
            ) : auth.isAuthenticated ? (
              <>Continue as {auth.displayName || 'signed-in user'} →</>
            ) : (
              <>Save &amp; continue with Auth0 →</>
            )}
          </button>

          {auth.isAuthenticated && (
            <button type="button" className="btn ghost sm" style={{ width: '100%', justifyContent: 'center', marginTop: 8 }} onClick={() => auth.logout()}>
              Not you? Sign out of Auth0
            </button>
          )}

          <p className="hint" style={{ textAlign: 'center', marginTop: 12 }}>
            Roles are managed in Auth0 (store_associate → admin). No Azure values live in the frontend .env.
          </p>
        </form>
      </section>
    </div>
  )
}
