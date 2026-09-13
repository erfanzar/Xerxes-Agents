// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ReactElement } from 'react'
import { text, type RpcRecord } from './desktopRpc.js'

export interface RunHistoryPage {
  job: RpcRecord
  runs: RpcRecord[]
  hasMore: boolean
}

function startedAt(value: unknown): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'Start time unavailable'
  return new Date(value).toLocaleString()
}

/** Run summaries stay readable; full output is fetched only for the selected result. */
export function RunHistory({ page, result, busy, onClose, onMore, onInspect }: {
  page: RunHistoryPage
  result: RpcRecord | null
  busy: boolean
  onClose(): void
  onMore(): void
  onInspect(run: RpcRecord): void
}): ReactElement {
  return <section className="run-history" aria-label="Scheduled run history">
    <button disabled={busy} onClick={onClose}>Back to schedules</button>
    <h2>Run history</h2>
    <p>{text(page.job.prompt)}</p>
    {!page.runs.length && <p role="status">This schedule has not run yet.</p>}
    {page.runs.map(run => <div className="run-history__entry" key={text(run.id)}>
      <div className="studio-item">
        <div><strong>{startedAt(run.startedAt)}</strong><p>{text(run.state) || 'Status unavailable'}</p>
          {text(run.error) && <p className="studio-error">{text(run.error)}</p>}
        </div>
        <button disabled={busy} aria-expanded={result?.id === run.id} onClick={() => onInspect(run)}>
          {result?.id === run.id ? 'Refresh result' : 'View result'}
        </button>
      </div>
      {result && result.id === run.id && <div className="run-history__result" aria-label="Run result">
        {typeof result.exitCode === 'number' && <p>Exit {result.exitCode}</p>}
        {text(result.error) && result.error !== run.error && <p className="studio-error" role="alert">{text(result.error)}</p>}
        {text(result.output) ? <pre className="studio-source run-history__output" tabIndex={0} role="region" aria-label="Run output">{text(result.output)}</pre> : <p>No output recorded.</p>}
        {result.outputTruncated === true && <p className="studio-muted">Only the retained output is shown; earlier output was truncated.</p>}
      </div>}
    </div>)}
    {page.hasMore && <button disabled={busy} onClick={onMore}>Load earlier runs</button>}
  </section>
}
