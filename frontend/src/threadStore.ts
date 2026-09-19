// src/threadStore.ts
// Chat threads are kept in localStorage per user (the backend keeps the real
// conversation history by thread_id, this is only what the UI shows).

import { v4 as uuidv4 } from 'uuid'
import type { AskResponse, RequestEvaluation, RequestStatus } from './types'

export type MsgKind = 'answer' | 'pending' | 'refused' | 'clarification' | 'error'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  status: 'thinking' | 'streaming' | 'done' | 'error'
  kind?: MsgKind
  data?: AskResponse
  requestId?: string
  poll?: RequestStatus
  polledAt?: number // when the user last pressed "Check status"
  releasedAnswer?: string
  evaluation?: RequestEvaluation // RAGAS quality scores, fetched after the answer
  evaluationDone?: boolean // stop showing the "scoring…" shimmer (scored, or gave up)
}

export interface Thread {
  id: string // local id
  thread_id: string | null // backend thread id (comes back with the first answer)
  title: string
  createdAt: number
  messages: Message[]
}

export function storageKey(userId: string) {
  return `rpids.threads.${userId}`
}

export function loadThreads(userId: string): Thread[] {
  try {
    const raw = localStorage.getItem(storageKey(userId))
    return raw ? (JSON.parse(raw) as Thread[]) : []
  } catch {
    return []
  }
}

export function saveThreads(userId: string, threads: Thread[]) {
  try {
    // don't keep half-streamed states
    const clean = threads.map((t) => ({
      ...t,
      messages: t.messages.map((m) =>
        m.status === 'streaming' || m.status === 'thinking' ? { ...m, status: 'done' as const } : m,
      ),
    }))
    localStorage.setItem(storageKey(userId), JSON.stringify(clean.slice(0, 30)))
  } catch (e) {
    console.log('could not save threads', e)
  }
}

export function newThread(): Thread {
  return { id: uuidv4(), thread_id: null, title: 'New conversation', createdAt: Date.now(), messages: [] }
}
