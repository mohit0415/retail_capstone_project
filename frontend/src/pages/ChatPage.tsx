// src/pages/ChatPage.tsx
//
// The main screen (every role gets it).
//   - POST /ask with the Auth0 token, thread_id (multi-turn), document_scope, use_cache
//   - while the graph runs we animate the pipeline stages
//   - the certified answer is then revealed chunk by chunk (streamText) so it
//     feels like a live stream, with citations / sql proof / decision trace
//   - 202 pending_review  -> we show a static "sent to human review" message (no spinner,
//     no polling). The user can press "Check status" once to see if a reviewer released it.
//   - every answer / escalation shows the multi-agent workflow (trace.agent_steps)
//   - "I don't know" answers (nothing in the documents) get a "not in your documents" chip
//   - refused / clarification_required are shown as their own cards

import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { useLocation } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import { v4 as uuidv4 } from 'uuid'
import { ApiError, askQuestion, getEvaluation, getRequestStatus, streamText } from '../apiService'
import AgentWorkflow from '../components/AgentWorkflow'
import { ROLE_LABELS, useAuth } from '../hooks/useAuth'
import type { AnswerResponse, ClarificationResponse, PendingReviewResponse } from '../types'

import { loadThreads, newThread, saveThreads, type Message, type Thread } from '../threadStore'

const DOC_LABELS: Record<string, string> = {
  privacy_policy: 'Privacy',
  infosec_policy: 'InfoSec',
  anti_bribery_policy: 'Anti-bribery',
  vendor_policy: 'Vendor',
  retention_policy: 'Retention',
  gdpr: 'GDPR',
  iso_27001: 'ISO 27001',
}

const STAGES = [
  'Input guardrail (injection / PII / scope)',
  'Query rewrite & intent classification',
  'Entity resolution & risk assessment',
  'Planner & path routing (RAG / SQL / hybrid)',
  'Retrieval, reranking & drafting',
  'Compliance validation & confidence score',
  'Release gate',
]

const SUGGESTIONS: Record<string, string[]> = {
  store_associate: [
    'What is the retention period for customer transaction records?',
    'Can I accept a gift from a supplier during a tender?',
    'Who can access customer personal data in the store?',
  ],
  store_manager: [
    'Which vendors are pending compliance review?',
    'What does the vendor policy say about onboarding a new supplier?',
    'List Low and Medium risk vendors with an expired contract.',
  ],
  compliance_officer: [
    'Which GDPR article covers breach notification and what is our internal deadline?',
    'Show the audit log entries for failed access attempts this quarter.',
    'Does the retention policy conflict with GDPR article 17 for customer invoices?',
  ],
  legal_reviewer: [
    'Summarise the ISO 27001 access control requirements we map to.',
    'What retention records are overdue for deletion?',
    'What is the escalation path for a suspected bribery incident?',
  ],
  admin: [
    'Which policy documents are indexed and what versions are active?',
    'List high risk vendors and the clause that governs their review cadence.',
    'What is the retention period for audit logs?',
  ],
}

function riskClass(level?: string) {
  if (!level) return ''
  const l = level.toLowerCase()
  if (l === 'low') return 'teal'
  if (l === 'medium') return 'amber'
  return 'rose'
}

// ----- RAGAS answer quality (scored in the background, fetched lazily) -----

const EVAL_RETRY_DELAYS_MS = [3000, 5000, 8000, 12000]

const QUALITY_METRICS = [
  {
    key: 'faithfulness',
    label: 'faithfulness',
    hint: 'How many of the claims in this answer are supported by the retrieved clauses.',
  },
  {
    key: 'answer_accuracy',
    label: 'accuracy',
    hint: 'A yes/no judge verdict: does the answer correctly answer the question, with every factual statement backed by the retrieved clauses.',
  },
  {
    key: 'context_precision',
    label: 'precision',
    hint: 'How many of the retrieved clauses were actually relevant to this answer.',
  },
  {
    key: 'context_recall',
    label: 'recall',
    hint: 'Whether retrieval brought back the clauses the answer needed. Measured against the generated answer — a live request has no golden reference.',
  },
] as const

function qualityClass(v: number) {
  if (v >= 0.8) return 'teal'
  if (v >= 0.5) return 'amber'
  return 'rose'
}

// ----- component -----

export default function ChatPage() {
  const auth = useAuth()
  const location = useLocation()
  const userId = auth.me?.user_id || 'anon'

  const [threads, setThreads] = useState<Thread[]>(() => {
    const existing = loadThreads(userId)
    return existing.length ? existing : [newThread()]
  })
  const [activeId, setActiveId] = useState<string>(() => threads[0].id)
  const [input, setInput] = useState<string>(() => (location.state as { prefill?: string } | null)?.prefill || '')
  const [scope, setScope] = useState<string[]>([])
  const [useCache, setUseCache] = useState(true)
  const [sending, setSending] = useState(false)
  const [stageIdx, setStageIdx] = useState(-1)
  const [showTrace, setShowTrace] = useState<Record<string, boolean>>({})
  // which escalated message is being checked right now (one request, no polling)
  const [checkingId, setCheckingId] = useState<string | null>(null)

  const scrollRef = useRef<HTMLDivElement>(null)
  const cancelStream = useRef<{ cancelled: boolean }>({ cancelled: false })

  const active = threads.find((t) => t.id === activeId) || threads[0]

  const docScopes = useMemo(
    () => (auth.me?.access_scopes || []).filter((s) => s.startsWith('doc:')).map((s) => s.slice(4)),
    [auth.me],
  )
  const tableScopes = useMemo(
    () => (auth.me?.access_scopes || []).filter((s) => s.startsWith('table:')).map((s) => s.slice(6)),
    [auth.me],
  )
  const riskScopes = useMemo(
    () => (auth.me?.access_scopes || []).filter((s) => s.startsWith('risk:')).map((s) => s.slice(5)),
    [auth.me],
  )

  // persist
  useEffect(() => {
    saveThreads(userId, threads)
  }, [threads, userId])

  // auto scroll
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [active?.messages, stageIdx])

  // answers restored from localStorage may still be waiting for their quality
  // scores (the page was closed while the judge ran) - resume the fetch once
  useEffect(() => {
    for (const t of threads) {
      for (const m of t.messages) {
        if (m.kind !== 'answer' || m.status !== 'done' || m.evaluationDone || !m.data) continue
        const d = m.data as AnswerResponse
        if (d.status !== 'answered' || d.not_found) {
          updateMessage(t.id, m.id, (x) => ({ ...x, evaluationDone: true }))
          continue
        }
        void fetchQuality(t.id, m.id, d.cache?.source_request_id || d.request_id)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // fake progress of the pipeline while we wait for /ask
  useEffect(() => {
    if (!sending) {
      setStageIdx(-1)
      return
    }
    setStageIdx(0)
    const timer = setInterval(() => {
      setStageIdx((i) => (i < STAGES.length - 2 ? i + 1 : i))
    }, 2200)
    return () => clearInterval(timer)
  }, [sending])

  function updateThread(threadId: string, fn: (t: Thread) => Thread) {
    setThreads((prev) => prev.map((t) => (t.id === threadId ? fn(t) : t)))
  }

  function updateMessage(threadId: string, messageId: string, fn: (m: Message) => Message) {
    updateThread(threadId, (t) => ({ ...t, messages: t.messages.map((m) => (m.id === messageId ? fn(m) : m)) }))
  }

  function startNewThread() {
    const t = newThread()
    setThreads((prev) => [t, ...prev])
    setActiveId(t.id)
    setInput('')
  }

  function deleteThread(id: string) {
    setThreads((prev) => {
      const rest = prev.filter((t) => t.id !== id)
      const next = rest.length ? rest : [newThread()]
      if (id === activeId) setActiveId(next[0].id)
      return next
    })
  }

  function toggleScope(doc: string) {
    setScope((s) => (s.includes(doc) ? s.filter((d) => d !== doc) : [...s, doc]))
  }

  // reveal the answer like a stream
  async function revealAnswer(threadId: string, messageId: string, text: string) {
    cancelStream.current = { cancelled: false }
    updateMessage(threadId, messageId, (m) => ({ ...m, content: '', status: 'streaming' }))
    await streamText(
      text,
      (piece) => {
        updateMessage(threadId, messageId, (m) => ({ ...m, content: m.content + piece }))
      },
      { signal: cancelStream.current },
    )
    updateMessage(threadId, messageId, (m) => ({ ...m, status: 'done' }))
  }

  // the RAGAS judge runs as a backend background task; poll a few times with
  // increasing delays and give up quietly (quality chips just don't appear)
  async function fetchQuality(threadId: string, messageId: string, requestId: string) {
    for (const delay of EVAL_RETRY_DELAYS_MS) {
      await new Promise((resolve) => setTimeout(resolve, delay))
      try {
        const token = await auth.getToken()
        const evaluation = await getEvaluation(token, requestId)
        if (evaluation.status !== 'pending') {
          updateMessage(threadId, messageId, (m) => ({ ...m, evaluation, evaluationDone: true }))
          return
        }
      } catch (e) {
        console.log('quality fetch failed', e)
      }
    }
    updateMessage(threadId, messageId, (m) => ({ ...m, evaluationDone: true }))
  }

  // ask the backend ONCE whether a reviewer released the escalated answer
  // (we used to poll every 6 seconds with a spinner - now it only happens when the user clicks)
  async function checkStatus(threadId: string, messageId: string, requestId: string) {
    setCheckingId(messageId)
    try {
      const token = await auth.getToken()
      const status = await getRequestStatus(token, requestId)
      updateMessage(threadId, messageId, (m) => ({ ...m, poll: status, polledAt: Date.now() }))
      if (status.status === 'answered') {
        updateMessage(threadId, messageId, (m) => ({ ...m, releasedAnswer: status.answer || '' }))
        await revealAnswer(threadId, messageId, status.answer || '(the reviewer released an empty answer)')
      }
    } catch (e) {
      console.log('status check failed', e)
      updateMessage(threadId, messageId, (m) => ({ ...m, polledAt: Date.now() }))
    }
    setCheckingId(null)
  }

  async function send(textOverride?: string) {
    const text = (textOverride ?? input).trim()
    if (!text || sending || !active) return

    const threadId = active.id
    const userMsg: Message = { id: uuidv4(), role: 'user', content: text, status: 'done' }
    const botMsg: Message = { id: uuidv4(), role: 'assistant', content: '', status: 'thinking' }

    updateThread(threadId, (t) => ({
      ...t,
      title: t.messages.length === 0 ? text.slice(0, 60) : t.title,
      messages: [...t.messages, userMsg, botMsg],
    }))
    setInput('')
    setSending(true)

    try {
      const token = await auth.getToken()
      const result = await askQuestion(
        token,
        {
          query: text,
          thread_id: active.thread_id,
          document_scope: scope,
          use_cache: useCache,
        },
        uuidv4(),
      )
      const body = result.body
      console.log('ask ->', result.httpStatus, body.status, body.request_id)

      if (body.status === 'answered') {
        updateThread(threadId, (t) => ({ ...t, thread_id: body.thread_id }))
        updateMessage(threadId, botMsg.id, (m) => ({ ...m, kind: 'answer', data: body, requestId: body.request_id }))
        setSending(false)
        if (!body.not_found) {
          // a cache hit re-uses the scores of the request that produced the answer
          void fetchQuality(threadId, botMsg.id, body.cache?.source_request_id || body.request_id)
        }
        await revealAnswer(threadId, botMsg.id, body.answer)
        return
      }

      if (body.status === 'pending_review') {
        updateThread(threadId, (t) => ({ ...t, thread_id: body.thread_id }))
        updateMessage(threadId, botMsg.id, (m) => ({
          ...m,
          kind: 'pending',
          data: body,
          requestId: body.request_id,
          status: 'done',
          content: body.reason,
        }))
        setSending(false)
        // no polling here on purpose - the message stays as it is
        return
      }

      if (body.status === 'clarification_required') {
        updateThread(threadId, (t) => ({ ...t, thread_id: body.thread_id }))
        updateMessage(threadId, botMsg.id, (m) => ({
          ...m,
          kind: 'clarification',
          data: body,
          requestId: body.request_id,
          status: 'done',
          content: body.question,
        }))
        setSending(false)
        return
      }

      // refused
      updateMessage(threadId, botMsg.id, (m) => ({
        ...m,
        kind: 'refused',
        data: body,
        requestId: body.request_id,
        status: 'done',
        content: body.reason,
      }))
      setSending(false)
    } catch (err) {
      console.log('ask failed', err)
      let text = 'Something went wrong while asking the backend.'
      if (err instanceof ApiError) {
        if (err.status === 401) text = 'Your session expired (401). Please log in again. ' + err.detail
        else if (err.status === 403) text = 'Not allowed for your role (403): ' + err.detail
        else if (err.status === 422) text = 'The backend rejected the request (422): ' + err.detail
        else if (err.status === 429) text = 'Rate limit hit (429): ' + err.detail
        else text = `Backend error ${err.status}: ${err.detail}` + (err.requestId ? ` (request ${err.requestId})` : '')
      } else if (err instanceof Error) {
        text = err.message.includes('fetch') ? 'Cannot reach the backend. Is it running?' : err.message
      }
      updateMessage(threadId, botMsg.id, (m) => ({ ...m, kind: 'error', status: 'error', content: text }))
      setSending(false)
    }
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const suggestions = SUGGESTIONS[auth.role || 'store_associate'] || SUGGESTIONS.store_associate

  // ----- render helpers -----

  function renderAnswerMeta(m: Message) {
    const d = m.data as AnswerResponse
    const trace = d.trace
    const ev = m.evaluation
    // "degraded" only because the role has no SQL tables (not because of the time/token budget)
    const partialEvidence =
      !!(trace?.path_decision as { partial_evidence?: boolean } | null)?.partial_evidence && !trace?.budget_stops?.length
    // real billed usage from the cost ledger (a cache hit bills no new tokens)
    const usage = trace?.llm_usage as { total_tokens?: number; prompt_tokens?: number; completion_tokens?: number; usd?: number } | null
    const usageTokens = Number(usage?.total_tokens || 0)
    return (
      <>
        <div className="meta-row">
          <span className={'chip ' + riskClass(d.risk_level)}>risk {d.risk_level}</span>
          {d.not_found ? (
            <span className="chip amber" title="the system answered “I don't know” instead of guessing">
              {d.not_found_reason === 'no_access'
                ? 'outside your access'
                : d.not_found_reason === 'no_records' || d.not_found_reason === 'no_evidence'
                  ? 'not in the records'
                  : 'not in your documents'}
            </span>
          ) : (
            <span className="chip violet">confidence {Math.round(d.confidence * 100)}%</span>
          )}
          <span className="chip sky">path {d.evidence_path}</span>
          {d.degraded &&
            (partialEvidence ? (
              <span className="chip amber" title="your role cannot read the records, so only the policy half of the answer was used">
                policy only
              </span>
            ) : (
              <span className="chip amber">degraded</span>
            ))}
          {d.cache && (
            <span className="chip teal" title={`matched: ${d.cache.matched_query}`}>
              cache {d.cache.kind} · saved ~{Math.round(d.cache.saved_ms_estimate)} ms
            </span>
          )}
          {d.timings && <span className="chip">{Math.round(d.timings.total_ms)} ms</span>}
          {usageTokens > 0 && (
            <span
              className="chip"
              title={`billed for this answer: ${usage?.prompt_tokens ?? 0} prompt + ${usage?.completion_tokens ?? 0} completion tokens`}
            >
              {usageTokens.toLocaleString()} tok · ${Number(usage?.usd || 0).toFixed(4)}
            </span>
          )}
          {d.history_turns_used > 0 && <span className="chip">{d.history_turns_used} turns of history</span>}
          {!d.not_found && !m.evaluationDone && <span className="chip shimmer">scoring quality…</span>}
          {ev?.status === 'done' &&
            QUALITY_METRICS.map(({ key, label, hint }) => {
              const value = ev[key]
              if (value === null || value === undefined) return null
              return (
                <span key={key} className={'chip ' + qualityClass(value)} title={hint}>
                  {label} {Math.round(value * 100)}%
                </span>
              )
            })}
          {ev?.status === 'skipped' && (
            <span className="chip" title="no retrieved policy clauses to ground against (record-only answer)">
              quality n/a
            </span>
          )}
        </div>

        {d.uncertainty_note && (
          <div className="alert warn" style={{ marginTop: 10 }}>
            {d.uncertainty_note}
          </div>
        )}

        {d.citations.length > 0 && (
          <div style={{ marginTop: 6 }}>
            {d.citations.map((c, i) => (
              <div className="citation" key={i}>
                <b>
                  {c.document_title} §{c.clause_number}
                </b>{' '}
                <span className="muted small">
                  {c.section} · v{c.version}
                </span>
                <p>{c.excerpt}</p>
              </div>
            ))}
          </div>
        )}

        {d.sql_evidence && (
          <details className="box" style={{ marginTop: 10 }}>
            <summary>
              SQL proof · template {d.sql_evidence.template_id} · {d.sql_evidence.row_count} rows · as of {d.sql_evidence.as_of}
            </summary>
            <pre className="json">{d.sql_evidence.statement}</pre>
            <pre className="json">{JSON.stringify(d.sql_evidence.parameters, null, 2)}</pre>
          </details>
        )}

        {ev?.status === 'done' && (
          <details className="box quality-box" style={{ marginTop: 10 }}>
            <summary>
              Answer quality (RAGAS) · {ev.contexts_scored} clause{ev.contexts_scored === 1 ? '' : 's'} judged
              {ev.judge_model ? ` by ${ev.judge_model}` : ''}
              {ev.duration_ms ? ` · ${(ev.duration_ms / 1000).toFixed(1)}s` : ''}
              {ev.request_id !== d.request_id ? ' · from cached source answer' : ''}
            </summary>
            {QUALITY_METRICS.map(({ key, label, hint }) => {
              const value = ev[key]
              return (
                <div key={key} className="quality-row">
                  <div className="row spread small">
                    <span>
                      {label === 'precision' ? 'context precision' : label === 'recall' ? 'context recall' : label}
                      {key === 'context_recall' && (
                        <span className="muted" title={hint}>
                          {' '}
                          (vs generated answer)
                        </span>
                      )}
                    </span>
                    <span className={value === null || value === undefined ? 'muted' : ''}>
                      {value === null || value === undefined ? 'not scored' : `${Math.round(value * 100)}%`}
                    </span>
                  </div>
                  <div className={'bar ' + (value !== null && value !== undefined ? qualityClass(value) : '')}>
                    <span style={{ width: `${Math.round((value || 0) * 100)}%` }} />
                  </div>
                  <p className="quality-hint">{hint}</p>
                </div>
              )
            })}
          </details>
        )}

        {trace && <AgentWorkflow steps={trace.agent_steps} />}

        {trace && (
          <details
            className="box"
            style={{ marginTop: 10 }}
            open={!!showTrace[m.id]}
            onToggle={(e) => setShowTrace((s) => ({ ...s, [m.id]: (e.target as HTMLDetailsElement).open }))}
          >
            <summary>
              Decision trace · intent {String((trace.intent as { intent?: string } | null)?.intent || '-')} · tokens{' '}
              {trace.tokens_spent}/{trace.token_budget} · reflections {trace.reflection_count}
              {trace.model_routing.length > 0 && ` · routing ${trace.model_routing.map((r) => `${r.node}=${r.tier}`).join(', ')}`}
            </summary>
            <pre className="json">{JSON.stringify(trace, null, 2)}</pre>
          </details>
        )}

        <div className="row spread" style={{ marginTop: 8 }}>
          <span className="badge-num">request {d.request_id}</span>
          {d.standalone_query && d.standalone_query !== '' && (
            <span className="muted small" title="how the graph rewrote your question">
              ↳ {d.standalone_query}
            </span>
          )}
        </div>
      </>
    )
  }

  function renderPending(m: Message) {
    const d = m.data as PendingReviewResponse
    const released = !!m.releasedAnswer
    const checking = checkingId === m.id
    return (
      <>
        {!released ? (
          <>
            <div className="row" style={{ marginBottom: 6 }}>
              <span className={'chip ' + riskClass(d.risk_level)}>risk {d.risk_level}</span>
              <span className="chip amber">sent to human review</span>
            </div>
            <p>
              <b>This answer could not be certified automatically.</b> {d.reason}
            </p>
            {m.poll && m.poll.status === 'rejected' ? (
              <p className="muted small">
                A reviewer rejected this answer, so nothing was released
                {m.poll.reviewed_at ? ` (${new Date(m.poll.reviewed_at).toLocaleString()})` : ''}. Try asking the question in a
                different way, or ask a compliance officer directly.
              </p>
            ) : (
              <p className="muted small">
                A reviewer will look at it{d.escalation_reference ? ` (ref ${d.escalation_reference})` : ''}. Nothing else is
                needed from you - you can keep asking other questions.
              </p>
            )}
            <div className="row">
              <button
                type="button"
                className="btn sm"
                onClick={() => m.requestId && checkStatus(activeId, m.id, m.requestId)}
                disabled={checking || !m.requestId}
              >
                {checking ? 'checking…' : 'Check status'}
              </button>
              {!checking && m.poll && m.poll.status === 'pending_review' && (
                <span className="muted small">
                  still with the reviewer
                  {m.polledAt ? ` (checked ${new Date(m.polledAt).toLocaleTimeString()})` : ''}
                </span>
              )}
              {!checking && m.poll && m.poll.status === 'rejected' && (
                <span className="chip rose">rejected by the reviewer</span>
              )}
              {!checking && !m.poll && m.polledAt && <span className="muted small">could not check right now</span>}
            </div>
          </>
        ) : (
          <>
            <div className="row" style={{ marginBottom: 6 }}>
              <span className="chip teal">released by reviewer · {m.poll?.reviewer_decision}</span>
              <span className={'chip ' + riskClass(d.risk_level)}>risk {d.risk_level}</span>
            </div>
            <ReactMarkdown>{m.content}</ReactMarkdown>
          </>
        )}
        {d.trace && <AgentWorkflow steps={d.trace.agent_steps} />}
        <div className="badge-num" style={{ marginTop: 8 }}>
          request {d.request_id}
        </div>
      </>
    )
  }

  function renderClarification(m: Message) {
    const d = m.data as ClarificationResponse
    return (
      <>
        <p>
          <b>I need one clarification:</b> {d.question}
        </p>
        <div className="row">
          {d.candidates.map((c, i) => {
            const label = String(c.name || c.label || c.id || JSON.stringify(c))
            return (
              <button key={i} className="chip clickable violet" onClick={() => send(`I mean ${label}`)} disabled={sending}>
                {label}
              </button>
            )
          })}
        </div>
      </>
    )
  }

  function renderAssistant(m: Message) {
    if (m.status === 'thinking') {
      return (
        <div className="pipeline">
          {STAGES.map((s, i) => (
            <div key={s} className={'stage ' + (i < stageIdx ? 'done' : i === stageIdx ? 'running' : '')}>
              <span className="ring" /> {s}
            </div>
          ))}
        </div>
      )
    }

    if (m.kind === 'error') {
      return <div className="alert error">{m.content}</div>
    }

    if (m.kind === 'refused') {
      return (
        <>
          <div className="row" style={{ marginBottom: 6 }}>
            <span className="chip rose">refused by guardrail</span>
          </div>
          <p>{m.content}</p>
          {m.requestId && <div className="badge-num">request {m.requestId}</div>}
        </>
      )
    }

    if (m.kind === 'pending') return renderPending(m)
    if (m.kind === 'clarification') return renderClarification(m)

    // answer
    return (
      <>
        <div className={m.status === 'streaming' ? 'stream-cursor' : ''}>
          <ReactMarkdown>{m.content}</ReactMarkdown>
        </div>
        {m.status === 'done' && m.data && renderAnswerMeta(m)}
      </>
    )
  }

  return (
    <>
      <div className="topbar">
        <div>
          <h1 className="page-title">Policy Chat</h1>
          <p className="page-sub">
            Ask about policies and records. Answers are certified before release — high risk goes to a reviewer.
          </p>
        </div>
        <div className="row">
          <button className="btn" onClick={startNewThread}>
            + New conversation
          </button>
        </div>
      </div>

      <div className="chat-layout">
        <section className="chat-column">
          <div className="chat-scroll" ref={scrollRef}>
            {active.messages.length === 0 && (
              <div className="empty-chat">
                <div style={{ fontSize: 34 }}>🛡️</div>
                <h3>Hi {auth.displayName.split(' ')[0] || 'there'}</h3>
                <p>
                  You are signed in as <b>{auth.role ? ROLE_LABELS[auth.role] : '…'}</b>. Ask a question or pick one on the
                  right. Follow-ups stay in the same thread so the agent remembers context.
                </p>
              </div>
            )}

            {active.messages.map((m) => (
              <div key={m.id} className={'msg ' + m.role}>
                <div className="who">{m.role === 'user' ? 'you' : 'AI'}</div>
                <div className="bubble">{m.role === 'user' ? m.content : renderAssistant(m)}</div>
              </div>
            ))}
          </div>

          <div className="composer">
            <div className="scope-row">
              <span className="muted small">Narrow to:</span>
              {docScopes.map((doc) => (
                <button
                  key={doc}
                  type="button"
                  className={'chip clickable ' + (scope.includes(doc) ? 'on' : '')}
                  onClick={() => toggleScope(doc)}
                  title={doc}
                >
                  {DOC_LABELS[doc] || doc}
                </button>
              ))}
              <label className="chip clickable" style={{ marginLeft: 'auto' }}>
                <input type="checkbox" checked={useCache} onChange={(e) => setUseCache(e.target.checked)} /> use answer cache
              </label>
            </div>
            <div className="composer-box">
              <textarea
                className="textarea"
                placeholder="Ask a policy or compliance question… (Enter to send, Shift+Enter for a new line)"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKey}
                disabled={sending}
                rows={2}
              />
              <button className="btn primary" onClick={() => send()} disabled={sending || input.trim().length < 3}>
                {sending ? (
                  <>
                    <span className="spinner" /> thinking
                  </>
                ) : (
                  'Ask →'
                )}
              </button>
            </div>
            {active.thread_id && (
              <div className="hint">
                thread <span className="mono">{active.thread_id}</span> · follow-ups use the conversation history
              </div>
            )}
          </div>
        </section>

        <aside className="side-column">
          <div className="card">
            <h3 className="card-title">Try asking</h3>
            {suggestions.map((s) => (
              <button key={s} className="suggest" onClick={() => setInput(s)} disabled={sending}>
                {s}
              </button>
            ))}
          </div>

          <div className="card">
            <h3 className="card-title">Your access</h3>
            <div className="row" style={{ marginBottom: 8 }}>
              <span className="chip violet">{auth.role ? ROLE_LABELS[auth.role] : '-'}</span>
              {auth.me?.departments.map((d) => (
                <span key={d} className="chip">
                  dept {d}
                </span>
              ))}
            </div>
            <div className="small muted" style={{ marginBottom: 4 }}>
              Documents
            </div>
            <div className="row" style={{ marginBottom: 8 }}>
              {docScopes.map((d) => (
                <span key={d} className="chip">
                  {DOC_LABELS[d] || d}
                </span>
              ))}
            </div>
            <div className="small muted" style={{ marginBottom: 4 }}>
              Tables
            </div>
            <div className="row" style={{ marginBottom: 8 }}>
              {tableScopes.length ? (
                tableScopes.map((t) => (
                  <span key={t} className="chip mono">
                    {t}
                  </span>
                ))
              ) : (
                <span className="chip rose">no SQL tables — policy documents only</span>
              )}
            </div>
            {riskScopes.length > 0 && (
              <>
                <div className="small muted" style={{ marginBottom: 4 }}>
                  Vendor risk categories
                </div>
                <div className="row">
                  {riskScopes.map((r) => (
                    <span key={r} className={'chip ' + riskClass(r)}>
                      {r}
                    </span>
                  ))}
                </div>
              </>
            )}
          </div>

          <div className="card threads">
            <div className="row spread">
              <h3 className="card-title" style={{ margin: 0 }}>
                Conversations
              </h3>
              <span className="muted small">{threads.length}</span>
            </div>
            <div style={{ marginTop: 8 }}>
              {threads.map((t) => (
                <div key={t.id} className={'thread-item ' + (t.id === active.id ? 'active' : '')} onClick={() => setActiveId(t.id)}>
                  <span className="row spread" style={{ flexWrap: 'nowrap' }}>
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{t.title}</span>
                    <button
                      className="btn sm ghost"
                      style={{ padding: '0 6px' }}
                      title="delete"
                      onClick={(e) => {
                        e.stopPropagation()
                        deleteThread(t.id)
                      }}
                    >
                      ×
                    </button>
                  </span>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </>
  )
}
