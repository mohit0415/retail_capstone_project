// src/pages/ReviewPage.tsx
// Reviewer console (compliance_officer / legal_reviewer / admin).
//   GET  /review/queue          -> pending escalations
//   GET  /review/{request_id}   -> the full review package
//   POST /review/{request_id}   -> accept / edit / reject  (the graph resumes on the backend)

import { useCallback, useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { ApiError, getReviewPackage, getReviewQueue, submitReview } from '../apiService'
import { useAuth } from '../hooks/useAuth'
import type { QueueItem, ReviewDecision, ReviewOutcome, ReviewPackage } from '../types'

function riskClass(level?: string) {
  const l = (level || '').toLowerCase()
  if (l === 'low') return 'teal'
  if (l === 'medium') return 'amber'
  return 'rose'
}

function ago(iso: string) {
  const ms = Date.now() - new Date(iso).getTime()
  const mins = Math.round(ms / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins} min ago`
  const hours = Math.round(mins / 60)
  if (hours < 24) return `${hours} h ago`
  return `${Math.round(hours / 24)} d ago`
}

export default function ReviewPage() {
  const auth = useAuth()
  const [queue, setQueue] = useState<QueueItem[]>([])
  const [loadingQueue, setLoadingQueue] = useState(true)
  const [queueError, setQueueError] = useState('')
  const [selected, setSelected] = useState<string | null>(null)
  const [pkg, setPkg] = useState<ReviewPackage | null>(null)
  const [pkgError, setPkgError] = useState('')
  const [decision, setDecision] = useState<ReviewDecision>('accept')
  const [edited, setEdited] = useState('')
  const [notes, setNotes] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [outcome, setOutcome] = useState<ReviewOutcome | null>(null)
  const [submitError, setSubmitError] = useState('')

  const loadQueue = useCallback(async () => {
    setLoadingQueue(true)
    setQueueError('')
    try {
      const token = await auth.getToken()
      const items = await getReviewQueue(token)
      setQueue(items)
    } catch (e) {
      setQueueError(e instanceof ApiError ? e.detail : String(e))
    } finally {
      setLoadingQueue(false)
    }
  }, [auth])

  useEffect(() => {
    loadQueue()
    const timer = setInterval(loadQueue, 20000) // refresh the queue every 20s
    return () => clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function open(requestId: string) {
    setSelected(requestId)
    setPkg(null)
    setPkgError('')
    setOutcome(null)
    setSubmitError('')
    setDecision('accept')
    setNotes('')
    try {
      const token = await auth.getToken()
      const p = await getReviewPackage(token, requestId)
      setPkg(p)
      setEdited(p.draft_answer || '')
    } catch (e) {
      setPkgError(e instanceof ApiError ? e.detail : String(e))
    }
  }

  async function submit() {
    if (!pkg) return
    setSubmitting(true)
    setSubmitError('')
    try {
      const token = await auth.getToken()
      const result = await submitReview(token, pkg.request_id, decision, edited, notes)
      setOutcome(result)
      console.log('review outcome', result)
      await loadQueue()
    } catch (e) {
      setSubmitError(e instanceof ApiError ? `${e.status}: ${e.detail}` : String(e))
    } finally {
      setSubmitting(false)
    }
  }

  const validation = pkg?.validation_output || {}
  const defects = (validation.defects as unknown[] | undefined) || []

  return (
    <>
      <div className="topbar">
        <div>
          <h1 className="page-title">Reviewer Console</h1>
          <p className="page-sub">Escalated requests wait here. Your decision resumes the graph and re-runs the output guardrail.</p>
        </div>
        <button className="btn" onClick={loadQueue} disabled={loadingQueue}>
          {loadingQueue ? <span className="spinner" /> : '↻'} Refresh queue
        </button>
      </div>

      <div className="review-layout">
        <aside>
          <div className="card">
            <div className="row spread">
              <h3 className="card-title" style={{ margin: 0 }}>
                Pending queue
              </h3>
              <span className="chip amber">{queue.length}</span>
            </div>
            <div style={{ marginTop: 10 }}>
              {queueError && <div className="alert error">{queueError}</div>}
              {!queueError && !loadingQueue && queue.length === 0 && (
                <p className="muted small">Nothing to review. High-risk or low-confidence answers will show up here.</p>
              )}
              {queue.map((item) => (
                <div key={item.request_id} className={'queue-item ' + (item.request_id === selected ? 'active' : '')} onClick={() => open(item.request_id)}>
                  <div className="row spread">
                    <span className={'chip ' + riskClass(item.risk_level)}>{item.risk_level}</span>
                    <span className="muted small">{ago(item.queued_at)}</span>
                  </div>
                  <div className="q">{item.reason}</div>
                  <div className="badge-num" style={{ marginTop: 4 }}>
                    {item.user_id} · {item.request_id.slice(0, 8)}…
                  </div>
                </div>
              ))}
            </div>
          </div>
        </aside>

        <section>
          {!selected && (
            <div className="card center-card" style={{ margin: '40px auto' }}>
              <div style={{ fontSize: 34 }}>🛡️</div>
              <h2>Pick a request</h2>
              <p className="muted">Select an escalation on the left to see the draft, the evidence and the validation output.</p>
            </div>
          )}

          {selected && pkgError && <div className="alert error">{pkgError}</div>}
          {selected && !pkg && !pkgError && (
            <div className="row">
              <span className="spinner" /> loading review package…
            </div>
          )}

          {pkg && (
            <>
              <div className="card">
                <div className="row spread">
                  <div className="row">
                    <span className={'chip ' + riskClass(pkg.risk_level)}>risk {pkg.risk_level}</span>
                    <span className="chip violet">confidence {Math.round(pkg.confidence * 100)}%</span>
                    <span className="chip sky">path {pkg.evidence_path || '-'}</span>
                    {pkg.degraded && <span className="chip amber">degraded</span>}
                    {pkg.reflection_count > 0 && <span className="chip">{pkg.reflection_count} reflections</span>}
                    {pkg.panel_repair_count > 0 && <span className="chip">{pkg.panel_repair_count} panel repairs</span>}
                  </div>
                  <span className="badge-num">{pkg.request_id}</span>
                </div>
                <h3 style={{ margin: '14px 0 4px' }}>Question</h3>
                <p style={{ margin: 0 }}>{pkg.original_query}</p>
                {pkg.standalone_query && pkg.standalone_query !== pkg.original_query && (
                  <p className="muted small">↳ rewritten as: {pkg.standalone_query}</p>
                )}

                <h3 style={{ margin: '14px 0 6px' }}>System draft {pkg.evidence_path === 'high_risk_panel' && <span className="chip violet">panel consensus</span>}</h3>
                <div className="draft-box">
                  {pkg.draft_answer ? <ReactMarkdown>{pkg.draft_answer}</ReactMarkdown> : <span className="muted">(no draft — use “edit” to write the answer)</span>}
                </div>
              </div>

              <div className="grid-2" style={{ marginTop: 14 }}>
                <div className="card">
                  <h3 className="card-title">Validation output</h3>
                  {defects.length > 0 ? (
                    <ul style={{ margin: 0, paddingLeft: 18 }}>
                      {defects.map((d, i) => (
                        <li key={i} className="small">
                          {typeof d === 'string' ? d : JSON.stringify(d)}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="muted small">no defects listed</p>
                  )}
                  <details className="box" style={{ marginTop: 8 }}>
                    <summary>raw validation</summary>
                    <pre className="json">{JSON.stringify(validation, null, 2)}</pre>
                  </details>
                </div>

                <div className="card">
                  <h3 className="card-title">Evidence</h3>
                  <p className="small muted" style={{ marginTop: 0 }}>
                    {pkg.retrieved_documents.length} retrieved chunk(s){pkg.sql_evidence ? ' · SQL evidence attached' : ''}
                  </p>
                  {pkg.retrieved_documents.slice(0, 6).map((doc, i) => (
                    <div className="citation" key={i}>
                      <b>{String(doc.document_title || doc.citation || doc.doc_type || 'chunk')}</b>{' '}
                      <span className="muted small">
                        {doc.clause_number ? `§${String(doc.clause_number)}` : ''} {doc.section ? String(doc.section) : ''}
                      </span>
                      <p>{String(doc.content || doc.excerpt || '').slice(0, 260)}</p>
                    </div>
                  ))}
                  {pkg.sql_evidence && (
                    <details className="box" style={{ marginTop: 8 }}>
                      <summary>SQL evidence</summary>
                      <pre className="json">{JSON.stringify(pkg.sql_evidence, null, 2)}</pre>
                    </details>
                  )}
                  {pkg.panel_verdict && (
                    <details className="box" style={{ marginTop: 8 }}>
                      <summary>Multi-agent panel verdict</summary>
                      <pre className="json">{JSON.stringify(pkg.panel_verdict, null, 2)}</pre>
                    </details>
                  )}
                  <details className="box" style={{ marginTop: 8 }}>
                    <summary>Reasoning trace ({pkg.reasoning_trace.length} steps)</summary>
                    <pre className="json">{JSON.stringify(pkg.reasoning_trace, null, 2)}</pre>
                  </details>
                </div>
              </div>

              <div className="card" style={{ marginTop: 14 }}>
                <h3 className="card-title">Your decision</h3>
                {outcome ? (
                  <div className={'alert ' + (outcome.released ? 'ok' : 'warn')}>
                    <b>{outcome.decision}</b> recorded — {outcome.note}
                    {outcome.answer && (
                      <div className="draft-box" style={{ marginTop: 10 }}>
                        <ReactMarkdown>{outcome.answer}</ReactMarkdown>
                      </div>
                    )}
                    <div className="row" style={{ marginTop: 10 }}>
                      <span className="chip">resumed: {String(outcome.resumed)}</span>
                      <span className="chip">released: {String(outcome.released)}</span>
                      {outcome.answer_source && <span className="chip violet">source {outcome.answer_source}</span>}
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="decision-btns">
                      <button className={'btn ' + (decision === 'accept' ? 'primary' : '')} onClick={() => setDecision('accept')} disabled={!pkg.draft_answer}>
                        ✓ Accept draft
                      </button>
                      <button className={'btn ' + (decision === 'edit' ? 'amber' : '')} onClick={() => setDecision('edit')}>
                        ✎ Edit & release
                      </button>
                      <button className={'btn ' + (decision === 'reject' ? 'danger' : '')} onClick={() => setDecision('reject')}>
                        ✕ Reject
                      </button>
                    </div>

                    {decision === 'edit' && (
                      <div className="field" style={{ marginTop: 12 }}>
                        <label className="label">Edited answer (this is what the asker will receive)</label>
                        <textarea className="textarea" rows={8} value={edited} onChange={(e) => setEdited(e.target.value)} />
                      </div>
                    )}

                    <div className="field" style={{ marginTop: 12 }}>
                      <label className="label">Reviewer notes (kept in the audit trail)</label>
                      <input className="input" value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="optional" />
                    </div>

                    {submitError && <div className="alert error" style={{ marginBottom: 10 }}>{submitError}</div>}

                    <button className="btn primary" onClick={submit} disabled={submitting || (decision === 'edit' && !edited.trim())}>
                      {submitting ? (
                        <>
                          <span className="spinner" /> submitting…
                        </>
                      ) : (
                        `Submit "${decision}"`
                      )}
                    </button>
                    <span className="muted small" style={{ marginLeft: 10 }}>
                      as {auth.displayName}
                    </span>
                  </>
                )}
              </div>
            </>
          )}
        </section>
      </div>
    </>
  )
}
