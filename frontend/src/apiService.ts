// src/apiService.ts
// All the calls to the FastAPI backend live here.
// Every protected call sends the Auth0 access token as "Authorization: Bearer <token>"
// (same as Task 3 / Task 4 in practice-1).

import type {
  AskRequest,
  AskResponse,
  AzureCredentials,
  AzureCredentialsResponse,
  CacheInvalidationResponse,
  CorpusStatusResponse,
  HealthResponse,
  IngestResponse,
  MeResponse,
  OptimizationReport,
  QueueItem,
  RebuildResponse,
  RequestStatus,
  ReviewDecision,
  ReviewOutcome,
  ReviewPackage,
  SloReport,
} from './types'

export const API_URL = (import.meta.env.VITE_API_URL as string | undefined) || 'http://localhost:8000'

export class ApiError extends Error {
  status: number
  detail: string
  requestId?: string

  constructor(status: number, detail: string, requestId?: string) {
    super(detail)
    this.status = status
    this.detail = detail
    this.requestId = requestId
  }
}

function authHeaders(token: string, extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...(extra || {}) }
  // Task 3 - Add Authorization header with Bearer token
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  return headers
}

// reads the JSON body and throws a nice ApiError when the status is not ok
async function handle<T>(response: Response): Promise<T> {
  let data: unknown = null
  const text = await response.text()
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }

  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    if (data && typeof data === 'object' && 'detail' in data) {
      const d = (data as { detail: unknown }).detail
      detail = typeof d === 'string' ? d : JSON.stringify(d)
    } else if (typeof data === 'string' && data) {
      detail = data
    }
    const requestId = response.headers.get('X-Request-ID') || undefined
    throw new ApiError(response.status, detail, requestId)
  }

  return data as T
}

// ---------------- health / auth ----------------

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_URL}/health`)
  return handle<HealthResponse>(response)
}

export async function getMe(token: string): Promise<MeResponse> {
  const response = await fetch(`${API_URL}/auth/me`, { headers: authHeaders(token) })
  return handle<MeResponse>(response)
}

export async function sendAzureCredentials(
  token: string,
  creds: AzureCredentials,
  verify = true,
): Promise<AzureCredentialsResponse> {
  const response = await fetch(`${API_URL}/auth/azure`, {
    method: 'POST',
    headers: authHeaders(token, { 'Content-Type': 'application/json' }),
    body: JSON.stringify({ ...creds, verify }),
  })
  return handle<AzureCredentialsResponse>(response)
}

// ---------------- ask ----------------

export interface AskResult {
  httpStatus: number // 200 or 202
  body: AskResponse
  requestId?: string
}

export async function askQuestion(token: string, body: AskRequest, idempotencyKey: string): Promise<AskResult> {
  const response = await fetch(`${API_URL}/ask`, {
    method: 'POST',
    headers: authHeaders(token, {
      'Content-Type': 'application/json',
      'Idempotency-Key': idempotencyKey,
    }),
    body: JSON.stringify(body),
  })
  const data = await handle<AskResponse>(response)
  return {
    httpStatus: response.status,
    body: data,
    requestId: response.headers.get('X-Request-ID') || undefined,
  }
}

export async function getRequestStatus(token: string, requestId: string): Promise<RequestStatus> {
  const response = await fetch(`${API_URL}/requests/${encodeURIComponent(requestId)}`, {
    headers: authHeaders(token),
  })
  return handle<RequestStatus>(response)
}

// ---------------- "streaming" ----------------
// The backend /ask returns the whole certified answer in one JSON (it has to
// run the guardrails + validation before it can release anything). To keep the
// chat feeling live we reveal that answer chunk by chunk, like an SSE stream.
// onChunk gets the text piece, the promise resolves when everything is out.

export function streamText(
  text: string,
  onChunk: (piece: string) => void,
  options?: { chunkSize?: number; delayMs?: number; signal?: { cancelled: boolean } },
): Promise<void> {
  const chunkSize = options?.chunkSize ?? 3
  const delayMs = options?.delayMs ?? 14
  const signal = options?.signal

  return new Promise((resolve) => {
    let i = 0
    function tick() {
      if (signal && signal.cancelled) {
        // dump the rest at once
        onChunk(text.slice(i))
        resolve()
        return
      }
      if (i >= text.length) {
        resolve()
        return
      }
      // try to cut on a word boundary so markdown does not look broken for long
      let end = Math.min(text.length, i + chunkSize)
      const nextSpace = text.indexOf(' ', end)
      if (nextSpace !== -1 && nextSpace - end < 6) end = nextSpace + 1
      onChunk(text.slice(i, end))
      i = end
      setTimeout(tick, delayMs)
    }
    tick()
  })
}

// ---------------- review ----------------

export async function getReviewQueue(token: string): Promise<QueueItem[]> {
  const response = await fetch(`${API_URL}/review/queue`, { headers: authHeaders(token) })
  return handle<QueueItem[]>(response)
}

export async function getReviewPackage(token: string, requestId: string): Promise<ReviewPackage> {
  const response = await fetch(`${API_URL}/review/${encodeURIComponent(requestId)}`, {
    headers: authHeaders(token),
  })
  return handle<ReviewPackage>(response)
}

export async function submitReview(
  token: string,
  requestId: string,
  decision: ReviewDecision,
  editedAnswer: string,
  notes: string,
): Promise<ReviewOutcome> {
  const response = await fetch(`${API_URL}/review/${encodeURIComponent(requestId)}`, {
    method: 'POST',
    headers: authHeaders(token, { 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      decision,
      edited_answer: decision === 'edit' ? editedAnswer : null,
      reviewer_notes: notes,
    }),
  })
  return handle<ReviewOutcome>(response)
}

// ---------------- ops ----------------

export async function getSloReport(token: string, hours?: number): Promise<SloReport> {
  const params = hours ? `?hours=${hours}` : ''
  const response = await fetch(`${API_URL}/metrics/slo${params}`, { headers: authHeaders(token) })
  return handle<SloReport>(response)
}

export async function getOptimizationReport(token: string): Promise<OptimizationReport> {
  const response = await fetch(`${API_URL}/metrics/optimization`, { headers: authHeaders(token) })
  return handle<OptimizationReport>(response)
}

export async function invalidateCaches(token: string, reason: string): Promise<CacheInvalidationResponse> {
  const response = await fetch(`${API_URL}/cache/invalidate?reason=${encodeURIComponent(reason)}`, {
    method: 'POST',
    headers: authHeaders(token),
  })
  return handle<CacheInvalidationResponse>(response)
}

// ---------------- corpus ----------------

export async function getCorpusStatus(token: string): Promise<CorpusStatusResponse> {
  const response = await fetch(`${API_URL}/ingest/status`, { headers: authHeaders(token) })
  return handle<CorpusStatusResponse>(response)
}

export async function ingestFile(token: string, file: File, force: boolean): Promise<IngestResponse> {
  const form = new FormData()
  form.append('file', file)
  const response = await fetch(`${API_URL}/ingest?force=${force ? 'true' : 'false'}`, {
    method: 'POST',
    headers: authHeaders(token), // no content-type, the browser sets the multipart boundary
    body: form,
  })
  return handle<IngestResponse>(response)
}

export async function rebuildCorpus(token: string): Promise<RebuildResponse> {
  const response = await fetch(`${API_URL}/ingest/rebuild?confirm=true`, {
    method: 'POST',
    headers: authHeaders(token),
  })
  return handle<RebuildResponse>(response)
}
