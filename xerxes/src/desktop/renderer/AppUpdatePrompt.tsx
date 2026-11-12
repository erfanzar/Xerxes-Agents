// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Asks before the app updates itself, in a slim banner at the top of the
 * window that never blocks the task. A new GitHub release shows its version
 * with Install and restart / Later / Skip this version, and its notes behind
 * "What's new"; nothing is downloaded until the person agrees. While
 * installing it shows progress in the same bar; if it cannot install (a
 * development build, no installer for this computer, an unwritable
 * Applications folder) it says why and offers the release page.
 */

import { useState, type KeyboardEvent, type ReactElement } from 'react'

import { Icon } from './Icon.js'
import { Markdown } from './markdown.js'
import { store, type AppUpdateView } from './store.js'

function megabytes(bytes: number | undefined): string {
  return bytes === undefined ? '' : `${(bytes / 1_048_576).toFixed(0)} MB`
}

export function AppUpdatePrompt({ update }: { update: AppUpdateView | null }): ReactElement | null {
  const [notesOpen, setNotesOpen] = useState(false)
  const busy = update?.phase === 'downloading' || update?.phase === 'restarting'
  const onEscape = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key !== 'Escape' || (!notesOpen && busy)) return
    event.stopPropagation()
    event.preventDefault()
    if (notesOpen) setNotesOpen(false)
    else void store.appUpdateAction('dismiss')
  }
  if (!update || update.phase === 'checking') return null
  if (update.phase === 'current') {
    return (
      <div className="appbanner" role="status" aria-live="polite" onKeyDown={onEscape}>
        <span className="appbanner__icon" aria-hidden="true"><Icon name="check" size={13} /></span>
        <span className="appbanner__text"><strong>Xerxes Agents is up to date.</strong> {update.version ? `${update.version} is the latest version.` : 'You have the latest version.'}</span>
        <button className="btn btn--ghost" onClick={() => void store.appUpdateAction('dismiss')}>OK</button>
      </div>
    )
  }
  const version = update.version ? `Xerxes Agents ${update.version}` : 'A new version'
  const percent = update.total ? Math.min(100, Math.round(((update.received ?? 0) / update.total) * 100)) : 0
  const title = update.phase === 'error' ? 'The update did not install.' : update.phase === 'restarting' ? 'Restarting to finish the update…' : `${version} is available.`
  const detail = update.phase === 'available' ? (update.installable ? 'It installs in the background, then restarts the app; your tasks carry on.' : update.reason ?? 'It cannot be installed from here.')
    : update.phase === 'downloading' ? `Downloading… ${percent}% (${megabytes(update.received)} of ${megabytes(update.total)})`
    : update.phase === 'restarting' ? 'The new version opens in a moment.'
    : update.message
  return (
    <div className="appbanner" role="status" aria-live="polite" data-phase={update.phase} onKeyDown={onEscape}>
      <span className="appbanner__icon" aria-hidden="true"><Icon name={update.phase === 'error' ? 'close' : 'arrowUp'} size={13} /></span>
      <span className="appbanner__text"><strong>{title}</strong> {detail}</span>
      {update.phase === 'downloading' && <span className="appbanner__progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}><span style={{ width: `${percent}%` }} /></span>}
      <span className="appbanner__actions">
        {update.notes && update.phase !== 'restarting' && <button className="btn btn--ghost" aria-expanded={notesOpen} onClick={() => setNotesOpen(open => !open)}>What&apos;s new</button>}
        {update.phase === 'available' && update.installable && <button className="btn btn--solid" onClick={() => void store.appUpdateAction('install')}>Install and restart</button>}
        {(update.phase === 'error' || (update.phase === 'available' && !update.installable)) && <button className="btn" onClick={() => void store.appUpdateAction('open-release')}>Open release page</button>}
        {update.phase === 'error' && <button className="btn" onClick={() => void store.appUpdateAction('check')}>Try again</button>}
        {!busy && <button className="btn btn--ghost" onClick={() => void store.appUpdateAction('dismiss')}>Later</button>}
        {update.phase === 'available' && <button className="btn btn--ghost" onClick={() => void store.appUpdateAction('skip')}>Skip this version</button>}
      </span>
      {/* In the page (and read by screen readers) even while folded, so the notes are never lost. */}
      {update.notes && update.phase !== 'restarting' && <div className="appbanner__notes" hidden={!notesOpen}><Markdown text={update.notes} /></div>}
    </div>
  )
}
