// src/pages/SloPage.tsx
// Observability: GET /metrics/slo (t1..t4 stage latencies vs targets) and
// GET /metrics/optimization (caches, model routing, cost).

import { useCallback, useEffect, useState } from 'react'
import { ApiError, getLangfuseLatency, getOptimizationReport, getRagasReport, getSloReport } from '../apiService'
import { useAuth } from '../hooks/useAuth'
import type { LangfuseLatencyReport, OptimizationReport, RagasReport, SloReport } from '../types'

function ms(v: number | null | undefined) {
  if (v === null || v === undefined) return '–'
  return v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`
}

function pct(v: unknown) {
  if (v === null || v === undefined) return '–'
  const n = Number(v)
  if (!isFinite(n)) return '–'
  return `${Math.round(n * 100)}%`
}

function num(v: unknown, digits = 0) {
  const n = Number(v)
  if (!isFinite(n)) return '–'
  return n.toFixed(digits)
}

const QUALITY_LABELS: Record<string, string> = {
  faithfulness: 'faithfulness — claims supported by the retrieved clauses and record rows',
  answer_accuracy: 'accuracy — judge verdict that the answer is correct (share of accurate answers)',
  context_precision: 'retrieval precision — relevant share of the raw pre-rerank candidates',
  context_recall: 'context recall — needed clauses retrieved (vs generated answer)',
}

function tokens(v: unknown) {
  const n = Number(v)
  if (!isFinite(n) || n <= 0) return '–'
  return n >= 10000 ? `${(n / 1000).toFixed(1)}k` : `${Math.round(n)}`
}

// one horizontal bar: a 0..1 RAGAS mean against its target
function QualityBar({ label, mean, p50, target, meets }: { label: string; mean: number | null; p50: number | null; target: number; meets: boolean | null }) {
  const cls = mean === null ? '' : mean >= 0.8 ? 'teal' : mean >= 0.5 ? 'amber' : 'rose'
  return (
    <div style={{ marginBottom: 12 }} title={`mean ${pct(mean)} · p50 ${pct(p50)} · target ${pct(target)}`}>
      <div className="row spread small" style={{ marginBottom: 4 }}>
        <span>{label}</span>
        <span className="muted">
          mean {pct(mean)} / target {pct(target)}{' '}
          {meets === null ? <span className="chip">no data</span> : meets ? <span className="chip teal">✓ meets</span> : <span className="chip rose">⚠ below</span>}
        </span>
      </div>
      <div className={'bar ' + cls} style={{ position: 'relative' }}>
        <span style={{ width: `${Math.min(100, (mean || 0) * 100)}%` }} />
        <i
          style={{
            position: 'absolute',
            left: `${target * 100}%`,
            top: -3,
            width: 2,
            height: 14,
            background: 'var(--text)',
            opacity: 0.7,
          }}
          title="target"
        />
      </div>
    </div>
  )
}

// one horizontal bar: p95 against its target (single hue = magnitude, the
// meets/breach state is written as text + icon, not colour alone)
function LatencyBar({ label, p50, p95, p99, target, meets }: { label: string; p50: number | null; p95: number | null; p99: number | null; target: number; meets: boolean | null }) {
  const max = Math.max(target * 1.3, p99 || 0, p95 || 0, 1)
  const w = (v: number | null) => `${Math.min(100, ((v || 0) / max) * 100)}%`
  return (
    <div style={{ marginBottom: 12 }} title={`p50 ${ms(p50)} · p95 ${ms(p95)} · p99 ${ms(p99)} · target ${ms(target)}`}>
      <div className="row spread small" style={{ marginBottom: 4 }}>
        <span>{label}</span>
        <span className="muted">
          p95 {ms(p95)} / target {ms(target)}{' '}
          {meets === null ? <span className="chip">no data</span> : meets ? <span className="chip teal">✓ meets</span> : <span className="chip rose">⚠ breached</span>}
        </span>
      </div>
      <div className="bar" style={{ position: 'relative' }}>
        <span style={{ width: w(p95) }} />
        <i
          style={{
            position: 'absolute',
            left: w(target),
            top: -3,
            width: 2,
            height: 14,
            background: 'var(--text)',
            opacity: 0.7,
          }}
          title="target"
        />
      </div>
    </div>
  )
}

export default function SloPage() {
  const auth = useAuth()
  const [hours, setHours] = useState(24)
  const [slo, setSlo] = useState<SloReport | null>(null)
  const [opt, setOpt] = useState<OptimizationReport | null>(null)
  const [ragas, setRagas] = useState<RagasReport | null>(null)
  const [langfuse, setLangfuse] = useState<LangfuseLatencyReport | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const token = await auth.getToken()
      // quality and portal metrics are additive - a failure there should not blank the page
      const [s, o, r, l] = await Promise.all([
        getSloReport(token, hours),
        getOptimizationReport(token),
        getRagasReport(token, hours).catch(() => null),
        getLangfuseLatency(token, hours).catch(() => null),
      ])
      setSlo(s)
      setOpt(o)
      setRagas(r)
      setLangfuse(l)
    } catch (e) {
      setError(e instanceof ApiError ? `${e.status}: ${e.detail}` : String(e))
    } finally {
      setLoading(false)
    }
  }, [auth, hours])

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hours])

  const caches = (opt?.caches || {}) as Record<string, Record<string, unknown>>
  const totals = (caches.totals || {}) as Record<string, unknown>
  const cost = (opt?.cost || {}) as Record<string, unknown>
  const routing = (opt?.routing || {}) as Record<string, unknown>
  const outcomesTotal = slo ? slo.outcomes.reduce((a, o) => a + o.count, 0) : 0

  // the five headline fields: cost (max tokens), latency p95, and the RAGAS trio
  const quality = (name: string) => ragas?.metrics.find((m) => m.metric === name) || null
  const accuracy = quality('answer_accuracy')
  const precision = quality('context_precision')
  const recall = quality('context_recall')
  // ledger view first (always on), Langfuse portal as the fallback source
  const maxTokens = Number(cost.tokens_per_request_max) > 0 ? Number(cost.tokens_per_request_max) : langfuse?.total_tokens_max

  const qualityStat = (label: string, m: typeof accuracy, hint: string) => (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{pct(m?.mean)}</div>
      <div className="stat-sub">
        {m?.mean === null || m?.mean === undefined
          ? hint
          : `target ${pct(m.target_mean)} ${m.meets_target === null ? '' : m.meets_target ? '· ✓ meets' : '· ⚠ below'}`}
      </div>
    </div>
  )

  return (
    <>
      <div className="topbar">
        <div>
          <h1 className="page-title">SLO & Cost</h1>
          <p className="page-sub">
            Cost (max tokens), p95 latency and RAGAS accuracy / precision / recall, plus stage latencies and cache &
            model-routing savings.
          </p>
        </div>
        <div className="row">
          {[1, 6, 24, 168].map((h) => (
            <button key={h} className={'chip clickable ' + (hours === h ? 'on' : '')} onClick={() => setHours(h)}>
              {h < 24 ? `${h} h` : `${h / 24} d`}
            </button>
          ))}
          <button className="btn sm" onClick={load} disabled={loading}>
            {loading ? <span className="spinner" /> : '↻'}
          </button>
        </div>
      </div>

      {error && <div className="alert error">{error}</div>}

      {slo && (
        <>
          {/* headline: cost (max tokens), latency p95 and the RAGAS quality trio */}
          <div className="grid-5" style={{ marginBottom: 14 }}>
            <div className="stat">
              <div className="stat-label">Cost · max tokens</div>
              <div className="stat-value">{tokens(maxTokens)}</div>
              <div className="stat-sub">
                per request · mean {tokens(cost.tokens_per_request_mean)} · peak ${num(cost.usd_per_request_peak, 4)}
              </div>
            </div>
            <div className="stat">
              <div className="stat-label">Latency · p95</div>
              <div className="stat-value">{ms(slo.total.p95_ms)}</div>
              <div className="stat-sub">
                target {ms(slo.total.target_p95_ms)}{' '}
                {slo.total.meets_slo === null ? '' : slo.total.meets_slo ? '· ✓ meets' : '· ⚠ breached'}
              </div>
            </div>
            {qualityStat('Accuracy · RAGAS', accuracy, 'no scored answers yet')}
            {qualityStat('Precision · RAGAS', precision, 'no scored answers yet')}
            {qualityStat('Recall · RAGAS', recall, 'no scored answers yet')}
          </div>

          <div className="grid-4">
            <div className="stat">
              <div className="stat-label">Requests in window</div>
              <div className="stat-value">{slo.sample_size}</div>
              <div className="stat-sub">last {slo.window_hours} h</div>
            </div>
            <div className="stat">
              <div className="stat-label">Total p95</div>
              <div className="stat-value">{ms(slo.total.p95_ms)}</div>
              <div className="stat-sub">
                target {ms(slo.total.target_p95_ms)} {slo.total.meets_slo === null ? '' : slo.total.meets_slo ? '· ✓ meets' : '· ⚠ breached'}
              </div>
            </div>
            <div className="stat">
              <div className="stat-label">Breached stages</div>
              <div className="stat-value">{slo.breached_stages.length}</div>
              <div className="stat-sub">{slo.breached_stages.join(', ') || 'none'}</div>
            </div>
            <div className="stat">
              <div className="stat-label">Breached paths</div>
              <div className="stat-value">{slo.breached_paths.length}</div>
              <div className="stat-sub">{slo.breached_paths.join(', ') || 'none'}</div>
            </div>
          </div>

          <div className="grid-2" style={{ marginTop: 14 }}>
            <div className="card">
              <h3 className="card-title">Latency · Langfuse portal</h3>
              {!langfuse || !langfuse.enabled ? (
                <p className="muted small">
                  Langfuse is not configured (set the LANGFUSE keys and TRACING_ENABLED). The database numbers on the left
                  remain the source of truth.
                </p>
              ) : (
                <>
                  <div className="row" style={{ gap: 24, marginBottom: 8 }}>
                    <div>
                      <div className="stat-label">p50</div>
                      <div className="stat-value">{ms(langfuse.p50_ms)}</div>
                    </div>
                    <div>
                      <div className="stat-label">p95</div>
                      <div className="stat-value">{ms(langfuse.p95_ms)}</div>
                    </div>
                    <div>
                      <div className="stat-label">traces</div>
                      <div className="stat-value">{langfuse.trace_count ?? '–'}</div>
                    </div>
                    <div>
                      <div className="stat-label">max tokens</div>
                      <div className="stat-value">{tokens(langfuse.total_tokens_max)}</div>
                    </div>
                    <div>
                      <div className="stat-label">cost</div>
                      <div className="stat-value">
                        {langfuse.total_cost_usd === null ? '–' : `$${num(langfuse.total_cost_usd, 4)}`}
                      </div>
                    </div>
                  </div>
                  {langfuse.error && <div className="alert warn">{langfuse.error}</div>}
                  <p className="hint">
                    trace <span className="mono">{langfuse.trace_name}</span> · last {langfuse.window_hours} h, as the
                    portal measures the same graph runs from the outside ·{' '}
                    <a href={langfuse.host} target="_blank" rel="noreferrer">
                      open Langfuse ↗
                    </a>
                  </p>
                </>
              )}
            </div>

            <div className="card">
              <h3 className="card-title">Answer quality · RAGAS</h3>
              {!ragas || ragas.scored === 0 ? (
                <p className="muted small">
                  no scored answers in this window — every certified answer is judged in the background a few seconds
                  after it is released
                </p>
              ) : (
                <>
                  {ragas.metrics.map((m) => (
                    <QualityBar
                      key={m.metric}
                      label={QUALITY_LABELS[m.metric] || m.metric}
                      mean={m.mean}
                      p50={m.p50}
                      target={m.target_mean}
                      meets={m.meets_target}
                    />
                  ))}
                  <div className="row" style={{ marginTop: 10 }}>
                    <span className="chip">{ragas.scored} scored</span>
                    {ragas.skipped > 0 && <span className="chip">{ragas.skipped} record-only (skipped)</span>}
                    {ragas.failed > 0 && <span className="chip amber">{ragas.failed} failed</span>}
                    {ragas.low_faithfulness_rate !== null && (
                      <span
                        className={'chip ' + (ragas.low_faithfulness_rate > 0.05 ? 'rose' : 'teal')}
                        title={`answers with faithfulness under ${Math.round(ragas.low_faithfulness_ceiling * 100)}%`}
                      >
                        low faithfulness {Math.round(ragas.low_faithfulness_rate * 100)}%
                      </span>
                    )}
                  </div>
                  {ragas.recent.length > 0 && (
                    <details className="box" style={{ marginTop: 10 }}>
                      <summary>recent evaluations</summary>
                      <div className="table-wrap">
                        <table className="table">
                          <thead>
                            <tr>
                              <th>request</th>
                              <th>path</th>
                              <th>faith</th>
                              <th>acc</th>
                              <th>prec</th>
                              <th>recall</th>
                            </tr>
                          </thead>
                          <tbody>
                            {ragas.recent.slice(0, 8).map((r) => (
                              <tr key={r.request_id}>
                                <td className="mono" title={r.request_id}>
                                  {r.request_id.slice(0, 8)}…
                                </td>
                                <td className="mono">{r.evidence_path || '–'}</td>
                                <td>{pct(r.faithfulness)}</td>
                                <td>{pct(r.answer_accuracy)}</td>
                                <td>{pct(r.context_precision)}</td>
                                <td>{pct(r.context_recall)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </details>
                  )}
                  <p className="hint">
                    judged per answer by the small-tier model; recall is measured against the generated answer — golden
                    recall lives in the offline eval set
                  </p>
                </>
              )}
            </div>
          </div>

          <div className="grid-2" style={{ marginTop: 14 }}>
            <div className="card">
              <h3 className="card-title">Stage latency · p95 vs target</h3>
              {slo.stages.map((s) => (
                <LatencyBar key={s.stage} label={`${s.stage} — ${s.meaning}`} p50={s.p50_ms} p95={s.p95_ms} p99={s.p99_ms} target={s.target_p95_ms} meets={s.meets_slo} />
              ))}
              <LatencyBar label="total" p50={slo.total.p50_ms} p95={slo.total.p95_ms} p99={slo.total.p99_ms} target={slo.total.target_p95_ms} meets={slo.total.meets_slo} />
              <p className="hint">{slo.total_target_basis}</p>
            </div>

            <div className="card">
              <h3 className="card-title">Outcomes</h3>
              {slo.outcomes.length === 0 && <p className="muted small">no requests in this window</p>}
              {slo.outcomes.map((o) => (
                <div key={o.outcome} style={{ marginBottom: 10 }}>
                  <div className="row spread small">
                    <span>{o.outcome}</span>
                    <span className="muted">
                      {o.count} · {outcomesTotal ? Math.round((o.count / outcomesTotal) * 100) : 0}%
                    </span>
                  </div>
                  <div className="bar">
                    <span style={{ width: `${outcomesTotal ? (o.count / outcomesTotal) * 100 : 0}%` }} />
                  </div>
                </div>
              ))}

              <h3 className="card-title" style={{ marginTop: 18 }}>
                Per evidence path
              </h3>
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>path</th>
                      <th>n</th>
                      <th>p95</th>
                      <th>target</th>
                      <th>slo</th>
                    </tr>
                  </thead>
                  <tbody>
                    {slo.evidence_paths.map((p) => (
                      <tr key={p.evidence_path}>
                        <td className="mono">{p.evidence_path}</td>
                        <td>{p.count}</td>
                        <td>{ms(p.total_p95_ms)}</td>
                        <td>{ms(p.target_p95_ms)}</td>
                        <td>{p.meets_slo === null ? '–' : p.meets_slo ? '✓ meets' : '⚠ breached'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="card" style={{ marginTop: 14 }}>
            <h3 className="card-title">Stage table</h3>
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>stage</th>
                    <th>meaning</th>
                    <th>p50</th>
                    <th>p95</th>
                    <th>p99</th>
                    <th>target p95</th>
                    <th>status</th>
                  </tr>
                </thead>
                <tbody>
                  {slo.stages.map((s) => (
                    <tr key={s.stage}>
                      <td className="mono">{s.stage}</td>
                      <td className="muted">{s.meaning}</td>
                      <td>{ms(s.p50_ms)}</td>
                      <td>{ms(s.p95_ms)}</td>
                      <td>{ms(s.p99_ms)}</td>
                      <td>{ms(s.target_p95_ms)}</td>
                      <td>{s.meets_slo === null ? '–' : s.meets_slo ? '✓ meets' : '⚠ breached'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="hint">measured from {slo.measured_from}</p>
          </div>
        </>
      )}

      {opt && (
        <>
          <h2 style={{ fontSize: 16, margin: '22px 0 10px' }}>Cost & latency optimisation</h2>
          <div className="grid-4">
            <div className="stat">
              <div className="stat-label">Cache hits</div>
              <div className="stat-value">{num(totals.hits)}</div>
              <div className="stat-sub">saved ~{ms(Number(totals.saved_ms || 0))} · {num(totals.saved_tokens)} tokens</div>
            </div>
            <div className="stat">
              <div className="stat-label">USD spent (process)</div>
              <div className="stat-value">${num((cost.process as Record<string, unknown> | undefined)?.usd, 4)}</div>
              <div className="stat-sub">mean ${num(cost.usd_per_request_mean, 4)} / request</div>
            </div>
            <div className="stat">
              <div className="stat-label">Small-tier share</div>
              <div className="stat-value">{pct(cost.small_tier_share)}</div>
              <div className="stat-sub">strategy {String(routing.strategy || routing.routing_strategy || '–')}</div>
            </div>
            <div className="stat">
              <div className="stat-label">Caches enabled</div>
              <div className="stat-value" style={{ fontSize: 14, lineHeight: 1.5 }}>
                {Object.entries((caches.enabled as Record<string, boolean>) || {})
                  .filter(([, v]) => v)
                  .map(([k]) => k)
                  .join(' · ') || 'none'}
              </div>
              <div className="stat-sub">generated {new Date(opt.generated_at).toLocaleTimeString()}</div>
            </div>
          </div>

          <div className="grid-3" style={{ marginTop: 14 }}>
            <div className="card">
              <h3 className="card-title">Caches</h3>
              <pre className="json">{JSON.stringify(opt.caches, null, 2)}</pre>
            </div>
            <div className="card">
              <h3 className="card-title">Model routing</h3>
              <pre className="json">{JSON.stringify(opt.routing, null, 2)}</pre>
            </div>
            <div className="card">
              <h3 className="card-title">Cost ledger</h3>
              <pre className="json">{JSON.stringify(opt.cost, null, 2)}</pre>
            </div>
          </div>
        </>
      )}
    </>
  )
}
