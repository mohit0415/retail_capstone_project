// src/pages/DashboardPage.tsx
// Vendor & Compliance dashboard (store_manager, compliance_officer, admin).
//   GET /health              -> system state (db, queue depth, thresholds, cache/cost stats)
//   local chat threads       -> "my requests" with risk / path / outcome
//   GET /requests/{id}       -> status of the ones that went to a reviewer
// The vendor / compliance questions themselves run through POST /ask (RBAC-scoped NL2SQL)
// so the quick actions just open the chat with the question prefilled.

import { Fragment, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ApiError, getHealth, getRequestStatus } from '../apiService'
import { ROLE_LABELS, useAuth } from '../hooks/useAuth'
import { loadThreads } from '../threadStore'
import type { AnswerResponse, HealthResponse, PendingReviewResponse, RequestStatus } from '../types'

interface Row {
  when: number
  question: string
  status: string
  risk: string
  path: string
  confidence: number | null
  requestId: string
  threadId: string | null
  ms: number | null
}

function riskClass(level?: string) {
  const l = (level || '').toLowerCase()
  if (l === 'low') return 'teal'
  if (l === 'medium') return 'amber'
  if (!l || l === '-') return ''
  return 'rose'
}

const QUICK: Record<string, string[]> = {
  store_manager: [
    'Which vendors are pending compliance review?',
    'List Low and Medium risk vendors whose contract expires this quarter.',
    'What does the vendor policy require before onboarding a supplier?',
  ],
  compliance_officer: [
    'Which vendors are High or Critical risk and overdue for review?',
    'How many audit log entries show failed access attempts this month?',
    'Which retention records are past their deletion date?',
  ],
  admin: [
    'Show all vendors with status pending review, by risk category.',
    'Which compliance reviews closed as non-compliant this quarter?',
    'Which retention records are past their deletion date?',
  ],
}

export default function DashboardPage() {
  const auth = useAuth()
  const navigate = useNavigate()
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState('')
  const [polls, setPolls] = useState<Record<string, RequestStatus>>({})

  const userId = auth.me?.user_id || 'anon'

  // flatten the local chat threads into request rows
  const rows: Row[] = useMemo(() => {
    const out: Row[] = []
    for (const t of loadThreads(userId)) {
      const msgs = t.messages
      for (let i = 0; i < msgs.length; i++) {
        const m = msgs[i]
        if (m.role !== 'assistant' || !m.data) continue
        const q = i > 0 ? msgs[i - 1].content : '(question)'
        const d = m.data
        if (d.status === 'answered') {
          const a = d as AnswerResponse
          out.push({
            when: t.createdAt,
            question: q,
            status: 'answered',
            risk: a.risk_level,
            path: a.evidence_path,
            confidence: a.confidence,
            requestId: a.request_id,
            threadId: a.thread_id,
            ms: a.timings ? a.timings.total_ms : null,
          })
        } else if (d.status === 'pending_review') {
          const p = d as PendingReviewResponse
          out.push({
            when: t.createdAt,
            question: q,
            status: m.releasedAnswer ? 'released' : 'pending_review',
            risk: p.risk_level,
            path: p.trace?.evidence_path || '-',
            confidence: null,
            requestId: p.request_id,
            threadId: p.thread_id,
            ms: p.timings ? p.timings.total_ms : null,
          })
        } else if (d.status === 'refused') {
          out.push({ when: t.createdAt, question: q, status: 'refused', risk: '-', path: '-', confidence: null, requestId: d.request_id, threadId: null, ms: null })
        } else {
          out.push({ when: t.createdAt, question: q, status: 'clarification', risk: '-', path: '-', confidence: null, requestId: d.request_id, threadId: d.thread_id, ms: null })
        }
      }
    }
    return out.sort((a, b) => b.when - a.when)
  }, [userId])

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((e) => setHealthError(e instanceof ApiError ? e.detail : String(e)))
  }, [])

  // refresh the status of the escalated ones
  useEffect(() => {
    const pending = rows.filter((r) => r.status === 'pending_review')
    if (pending.length === 0) return
    let stop = false
    async function refresh() {
      try {
        const token = await auth.getToken()
        for (const r of pending) {
          if (stop) return
          const s = await getRequestStatus(token, r.requestId)
          setPolls((p) => ({ ...p, [r.requestId]: s }))
        }
      } catch (e) {
        console.log('status refresh failed', e)
      }
    }
    refresh()
    const timer = setInterval(refresh, 15000)
    return () => {
      stop = true
      clearInterval(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows])

  const counts = useMemo(() => {
    const c = { total: rows.length, answered: 0, pending: 0, refused: 0, high: 0 }
    for (const r of rows) {
      if (r.status === 'answered' || r.status === 'released') c.answered++
      if (r.status === 'pending_review') c.pending++
      if (r.status === 'refused') c.refused++
      if (r.risk.toLowerCase() === 'high' || r.risk.toLowerCase() === 'critical') c.high++
    }
    return c
  }, [rows])

  const optimization = (health?.optimization || {}) as Record<string, unknown>
  const quick = QUICK[auth.role || 'store_manager'] || QUICK.store_manager

  return (
    <>
      <div className="topbar">
        <div>
          <h1 className="page-title">Compliance Dashboard</h1>
          <p className="page-sub">
            System health, your recent requests and quick vendor / compliance questions (scoped to the {auth.role ? ROLE_LABELS[auth.role] : ''} role).
          </p>
        </div>
      </div>

      {healthError && <div className="alert error" style={{ marginBottom: 14 }}>backend health unavailable: {healthError}</div>}

      <div className="grid-4">
        <div className="stat">
          <div className="stat-label">Backend</div>
          <div className="stat-value">
            <span className={'dot ' + (health ? (health.status === 'ok' ? 'ok' : 'warn') : 'bad')} style={{ marginRight: 8 }} />
            {health ? health.status : 'offline'}
          </div>
          <div className="stat-sub">db {health?.database || '-'} · azure {health?.azure_configured ? 'configured' : 'not configured'}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Reviewer queue</div>
          <div className="stat-value">{health?.escalation_queue_depth ?? '–'}</div>
          <div className="stat-sub">escalations waiting for a human</div>
        </div>
        <div className="stat">
          <div className="stat-label">Confidence gate</div>
          <div className="stat-value">{health ? Math.round(health.confidence_threshold * 100) + '%' : '–'}</div>
          <div className="stat-sub">below this an answer escalates · as of {health?.as_of_date || '–'}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Cache hits / spend</div>
          <div className="stat-value">{String(optimization.cache_hits ?? '–')}</div>
          <div className="stat-sub">
            ${Number(optimization.usd_spent || 0).toFixed(4)} spent · small tier {Math.round(Number(optimization.small_tier_share || 0) * 100)}%
          </div>
        </div>
      </div>

      <div className="grid-2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3 className="card-title">Your requests</h3>
          <div className="row" style={{ marginBottom: 10 }}>
            <span className="chip">{counts.total} total</span>
            <span className="chip teal">{counts.answered} answered</span>
            <span className="chip amber">{counts.pending} pending review</span>
            <span className="chip rose">{counts.high} high risk</span>
            <span className="chip">{counts.refused} refused</span>
          </div>
          {rows.length === 0 ? (
            <p className="muted small">No requests yet on this device. Ask something in the chat and it shows up here.</p>
          ) : (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>question</th>
                    <th>status</th>
                    <th>risk</th>
                    <th>path</th>
                    <th>conf.</th>
                    <th>time</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.slice(0, 25).map((r) => {
                    const live = polls[r.requestId]
                    const status = live && live.status === 'answered' && r.status === 'pending_review' ? 'released' : r.status
                    return (
                      <tr key={r.requestId}>
                        <td style={{ maxWidth: 320 }}>
                          <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={r.question}>
                            {r.question}
                          </div>
                          <div className="badge-num">{r.requestId.slice(0, 13)}…</div>
                        </td>
                        <td>
                          <span className={'chip ' + (status === 'answered' || status === 'released' ? 'teal' : status === 'pending_review' ? 'amber' : status === 'refused' ? 'rose' : '')}>{status}</span>
                        </td>
                        <td>
                          <span className={'chip ' + riskClass(r.risk)}>{r.risk}</span>
                        </td>
                        <td className="mono">{r.path}</td>
                        <td>{r.confidence !== null ? Math.round(r.confidence * 100) + '%' : '–'}</td>
                        <td className="muted nowrap">{r.ms !== null ? (r.ms / 1000).toFixed(1) + ' s' : '–'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div>
          <div className="card">
            <h3 className="card-title">Vendor & compliance quick questions</h3>
            <p className="muted small" style={{ marginTop: 0 }}>
              These go through the NL2SQL path with your role's table grants
              {auth.me && auth.me.access_scopes.some((s) => s.startsWith('table:'))
                ? ` (${auth.me.access_scopes.filter((s) => s.startsWith('table:')).map((s) => s.slice(6)).join(', ')})`
                : ''}
              .
            </p>
            {quick.map((q) => (
              <button key={q} className="suggest" onClick={() => navigate('/chat', { state: { prefill: q } })}>
                {q} →
              </button>
            ))}
          </div>

          {health && (
            <div className="card">
              <h3 className="card-title">Deadlines & budget</h3>
              <dl className="kv">
                {Object.entries(health.deadlines_seconds).map(([k, v]) => (
                  <Fragment key={k}>
                    <dt>{k}</dt>
                    <dd>{v} s</dd>
                  </Fragment>
                ))}
                <dt>token budget</dt>
                <dd>{health.default_token_budget}</dd>
                <dt>routing</dt>
                <dd>{String(optimization.routing_strategy || '-')}</dd>
                <dt>audit log</dt>
                <dd className="mono small">{JSON.stringify(health.audit_log)}</dd>
              </dl>
            </div>
          )}
        </div>
      </div>
    </>
  )
}
