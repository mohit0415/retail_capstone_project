// src/pages/CorpusPage.tsx
// Admin only. GET /ingest/status, POST /ingest (upload), POST /ingest/rebuild, POST /cache/invalidate

import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, getCorpusStatus, ingestFile, invalidateCaches, rebuildCorpus } from '../apiService'
import { useAuth } from '../hooks/useAuth'
import type { CorpusStatusResponse, IngestResponse, RebuildResponse } from '../types'

export default function CorpusPage() {
  const auth = useAuth()
  const [status, setStatus] = useState<CorpusStatusResponse | null>(null)
  const [error, setError] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [force, setForce] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState<IngestResponse | null>(null)
  const [uploadError, setUploadError] = useState('')
  const [rebuilding, setRebuilding] = useState(false)
  const [rebuildResult, setRebuildResult] = useState<RebuildResponse | null>(null)
  const [confirmRebuild, setConfirmRebuild] = useState(false)
  const [cacheMsg, setCacheMsg] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)

  const load = useCallback(async () => {
    setError('')
    try {
      const token = await auth.getToken()
      setStatus(await getCorpusStatus(token))
    } catch (e) {
      setError(e instanceof ApiError ? `${e.status}: ${e.detail}` : String(e))
    }
  }, [auth])

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function upload() {
    if (!file) return
    setUploading(true)
    setUploadError('')
    setUploadResult(null)
    try {
      const token = await auth.getToken()
      const result = await ingestFile(token, file, force)
      setUploadResult(result)
      setFile(null)
      if (fileInput.current) fileInput.current.value = ''
      await load()
    } catch (e) {
      setUploadError(e instanceof ApiError ? `${e.status}: ${e.detail}` : String(e))
    } finally {
      setUploading(false)
    }
  }

  async function rebuild() {
    setRebuilding(true)
    setRebuildResult(null)
    setError('')
    try {
      const token = await auth.getToken()
      const result = await rebuildCorpus(token)
      setRebuildResult(result)
      setConfirmRebuild(false)
      await load()
    } catch (e) {
      setError(e instanceof ApiError ? `${e.status}: ${e.detail}` : String(e))
    } finally {
      setRebuilding(false)
    }
  }

  async function clearCaches() {
    setCacheMsg('')
    try {
      const token = await auth.getToken()
      const result = await invalidateCaches(token, 'admin console')
      setCacheMsg(
        'cleared ' +
          Object.entries(result.cleared)
            .map(([k, v]) => `${k}: ${v}`)
            .join(', '),
      )
    } catch (e) {
      setCacheMsg(e instanceof ApiError ? `${e.status}: ${e.detail}` : String(e))
    }
  }

  return (
    <>
      <div className="topbar">
        <div>
          <h1 className="page-title">Corpus Admin</h1>
          <p className="page-sub">Policy documents in the vector index. Uploading a document changes what every role can retrieve.</p>
        </div>
        <button className="btn" onClick={load}>
          ↻ Refresh
        </button>
      </div>

      {error && <div className="alert error" style={{ marginBottom: 14 }}>{error}</div>}

      {status && (
        <div className="grid-4">
          <div className="stat">
            <div className="stat-label">Index</div>
            <div className="stat-value">{status.indexed ? 'ready' : 'empty'}</div>
            <div className="stat-sub">{status.documents.length} document(s)</div>
          </div>
          <div className="stat">
            <div className="stat-label">Embedding model</div>
            <div className="stat-value" style={{ fontSize: 16 }}>
              {status.embed_model}
            </div>
            <div className="stat-sub">{status.embed_model_compatible ? '✓ compatible with stored chunks' : '⚠ mismatch: ' + status.message}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Corpus directory</div>
            <div className="stat-value mono" style={{ fontSize: 13 }}>
              {status.corpus_dir}
            </div>
            <div className="stat-sub">{status.unindexed_files.length} unindexed file(s)</div>
          </div>
          <div className="stat">
            <div className="stat-label">Caches</div>
            <button className="btn sm" style={{ marginTop: 8 }} onClick={clearCaches}>
              Invalidate all caches
            </button>
            <div className="stat-sub">{cacheMsg || 'answer, retrieval and LLM caches'}</div>
          </div>
        </div>
      )}

      <div className="grid-2" style={{ marginTop: 14 }}>
        <div className="card">
          <h3 className="card-title">Upload a policy document</h3>
          <div className="upload-zone" onClick={() => fileInput.current?.click()}>
            <input ref={fileInput} type="file" accept=".pdf,.md,.docx,.txt" style={{ display: 'none' }} onChange={(e) => setFile(e.target.files?.[0] || null)} />
            {file ? (
              <span>
                <b>{file.name}</b> · {(file.size / 1024).toFixed(0)} KB
              </span>
            ) : (
              <span>click to choose a .pdf / .md / .docx / .txt (max 20 MB)</span>
            )}
          </div>
          <div className="row" style={{ marginTop: 10 }}>
            <label className="chip clickable">
              <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} /> force re-index if the same file exists
            </label>
            <button className="btn primary" onClick={upload} disabled={!file || uploading}>
              {uploading ? (
                <>
                  <span className="spinner" /> ingesting…
                </>
              ) : (
                'Ingest'
              )}
            </button>
          </div>
          {uploadError && <div className="alert error" style={{ marginTop: 10 }}>{uploadError}</div>}
          {uploadResult && (
            <div className={'alert ' + (uploadResult.status === 'indexed' ? 'ok' : 'warn')} style={{ marginTop: 10 }}>
              <b>{uploadResult.status}</b> {uploadResult.file_name}
              {uploadResult.status === 'indexed' ? (
                <div className="small" style={{ marginTop: 4 }}>
                  {uploadResult.doc_type} v{uploadResult.version} · parsed with {uploadResult.parsed_with} · {uploadResult.total_nodes} nodes ({uploadResult.text_nodes} text, {uploadResult.table_nodes} table, {uploadResult.image_nodes} image) · {uploadResult.superseded_nodes} superseded
                </div>
              ) : (
                <div className="small" style={{ marginTop: 4 }}>{uploadResult.reason}</div>
              )}
            </div>
          )}
        </div>

        <div className="card">
          <h3 className="card-title">Rebuild the whole index</h3>
          <p className="muted small" style={{ marginTop: 0 }}>
            Empties the vector table and re-ingests every file in the corpus directory. Costs one embedding pass over the whole corpus.
          </p>
          {!confirmRebuild ? (
            <button className="btn danger" onClick={() => setConfirmRebuild(true)}>
              Rebuild corpus…
            </button>
          ) : (
            <div className="row">
              <span className="chip rose">are you sure?</span>
              <button className="btn danger" onClick={rebuild} disabled={rebuilding}>
                {rebuilding ? (
                  <>
                    <span className="spinner" /> rebuilding…
                  </>
                ) : (
                  'Yes, rebuild now'
                )}
              </button>
              <button className="btn ghost" onClick={() => setConfirmRebuild(false)} disabled={rebuilding}>
                cancel
              </button>
            </div>
          )}
          {rebuildResult && (
            <div className="alert ok" style={{ marginTop: 10 }}>
              cleared {rebuildResult.cleared_chunks} chunks · indexed {rebuildResult.indexed_files.length} file(s) · {rebuildResult.total_nodes} nodes
              {rebuildResult.skipped_files.length > 0 && (
                <div className="small" style={{ marginTop: 4 }}>
                  skipped: {rebuildResult.skipped_files.map((s) => `${s.file_name} (${s.reason})`).join(', ')}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {status && (
        <div className="card" style={{ marginTop: 14 }}>
          <h3 className="card-title">Indexed documents</h3>
          {status.documents.length === 0 ? (
            <p className="muted small">nothing indexed yet</p>
          ) : (
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    {Object.keys(status.documents[0]).map((k) => (
                      <th key={k}>{k.replace(/_/g, ' ')}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {status.documents.map((doc, i) => (
                    <tr key={i}>
                      {Object.values(doc).map((v, j) => (
                        <td key={j} className={typeof v === 'number' ? 'mono' : ''}>
                          {typeof v === 'object' ? JSON.stringify(v) : String(v ?? '')}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {status.unindexed_files.length > 0 && (
            <div className="alert warn" style={{ marginTop: 10 }}>
              Files in the corpus directory that are not indexed: {status.unindexed_files.join(', ')}
            </div>
          )}
        </div>
      )}
    </>
  )
}
