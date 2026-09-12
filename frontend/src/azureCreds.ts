// src/azureCreds.ts
// The Azure OpenAI credentials are typed on the login page ONLY.
// We keep them in sessionStorage (cleared when the tab closes) and send them
// to the backend (POST /auth/azure) once the Auth0 login is done.
// They are never put in the .env file.

import type { AzureCredentials } from './types'

const KEY = 'rpids.azureCredentials'

// the two embedding models the login page offers, with the vector width each one
// writes. The backend derives the same width from the deployment name, so this is
// only here to show the consequence of the choice before it is submitted.
export const EMBEDDING_CHOICES = [
  { deployment: 'text-embedding-3-large', dimensions: 3072, label: 'large - 3072 dims, best recall' },
  { deployment: 'text-embedding-3-small', dimensions: 1536, label: 'small - 1536 dims, cheaper' },
] as const

export function dimensionsFor(deployment: string): number {
  const name = deployment.trim().toLowerCase()
  const hit = EMBEDDING_CHOICES.find((c) => name.includes(c.deployment) || name.includes(c.deployment.slice(-7)))
  return hit ? hit.dimensions : 0
}

export const DEFAULT_AZURE: AzureCredentials = {
  endpoint: '',
  api_key: '',
  api_version: '2024-10-21',
  small_deployment: 'gpt-4o-mini',
  strong_deployment: 'gpt-4o-mini',
  embedding_deployment: 'text-embedding-3-large',
  llamaparse_api_key: '',
}

export function loadAzureCreds(): AzureCredentials | null {
  try {
    const raw = sessionStorage.getItem(KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (!parsed.endpoint || !parsed.api_key) return null
    return { ...DEFAULT_AZURE, ...parsed }
  } catch (e) {
    console.log('could not read azure creds from sessionStorage', e)
    return null
  }
}

export function saveAzureCreds(creds: AzureCredentials) {
  sessionStorage.setItem(KEY, JSON.stringify(creds))
}

export function clearAzureCreds() {
  sessionStorage.removeItem(KEY)
}

// small helper so we never show the full key on screen
export function maskKey(key: string) {
  if (!key) return ''
  if (key.length <= 8) return '••••••••'
  return key.slice(0, 4) + '••••••••' + key.slice(-4)
}
