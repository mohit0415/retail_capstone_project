// src/components/AgentWorkflow.tsx
//
// Shows the multi-agent workflow for one answer.
// The backend sends it as trace.agent_steps (one entry per step, in the order they ran).
//   - the chip row is always visible: which agents took part
//   - "show steps" opens the full timeline with what every step decided and how long it took

import { useState } from 'react'
import type { AgentStep } from '../types'

function formatMs(ms: number) {
  if (!ms || ms < 1) return '0 ms'
  if (ms < 1000) return Math.round(ms) + ' ms'
  return (ms / 1000).toFixed(1) + ' s'
}

// the short names used in the chip row
function shortName(step: AgentStep) {
  return step.agent.replace(' Agent', '')
}

export default function AgentWorkflow({ steps }: { steps?: AgentStep[] }) {
  const [open, setOpen] = useState(false)

  if (!steps || steps.length === 0) return null

  // since_start_ms of the last step = total time of the graph run
  const totalMs = steps[steps.length - 1].since_start_ms
  const stopped = steps.filter((s) => s.status === 'budget_stopped').length

  // chip row: every real agent + the reflection step, once each (with a x2 when it ran twice)
  const counts: Record<string, number> = {}
  const order: AgentStep[] = []
  for (const step of steps) {
    if (step.kind !== 'agent' && step.kind !== 'reflect') continue
    if (!counts[step.agent]) order.push(step)
    counts[step.agent] = (counts[step.agent] || 0) + 1
  }

  return (
    <div className="workflow">
      <div className="workflow-head">
        <span className="muted small">Multi-agent workflow</span>
        <span className="workflow-chain">
          {order.map((step, i) => (
            <span key={step.agent} className="workflow-link">
              {i > 0 && <span className="workflow-arrow">→</span>}
              <span className={'chip ' + (step.kind === 'reflect' ? 'amber' : 'violet')}>
                {shortName(step)}
                {counts[step.agent] > 1 ? ` ×${counts[step.agent]}` : ''}
              </span>
            </span>
          ))}
        </span>
        <span className="muted small">
          {steps.length} steps · {formatMs(totalMs)}
          {stopped > 0 ? ` · ${stopped} stopped by budget` : ''}
        </span>
        <button type="button" className="btn sm ghost" onClick={() => setOpen(!open)}>
          {open ? 'hide steps' : 'show steps'}
        </button>
      </div>

      {open && (
        <ol className="workflow-steps">
          {steps.map((step, i) => (
            <li key={i} className={'workflow-step kind-' + step.kind + (step.status === 'budget_stopped' ? ' stopped' : '')}>
              <span className="workflow-dot" />
              <div className="workflow-body">
                <div className="workflow-title">
                  <b>{step.agent}</b>
                  {step.attempt > 1 && <span className="chip amber">attempt {step.attempt}</span>}
                  {step.tier && <span className="chip sky">model: {step.tier}</span>}
                  {step.status === 'budget_stopped' && <span className="chip rose">stopped</span>}
                  <span className="workflow-time">{formatMs(step.elapsed_ms)}</span>
                </div>
                {step.summary && <div className="workflow-summary">{step.summary}</div>}
                {step.children && step.children.length > 0 && (
                  <ul className="workflow-children">
                    {step.children.map((child, j) => (
                      <li key={j}>
                        <b>{child.agent}:</b> {child.summary}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
