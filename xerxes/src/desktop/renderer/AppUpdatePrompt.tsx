// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Asks before the app updates itself. A new GitHub release shows its version
 * and notes with Install and restart / Not now / Skip this version; nothing is
 * downloaded until the person agrees. While installing it shows progress; if
 * it cannot install (a development build, no installer for this computer, an
 * unwritable Applications folder) it says why and offers the release page.
 */

import type { ReactElement } from 'react'

import { Icon } from './Icon.js'
import { Markdown } from './markdown.js'
import { store, type AppUpdateView } from './store.js'

function megabytes(bytes: number | undefined): string {
  return bytes === undefined ? '' : `${(bytes / 1_048_576).toFixed(0)} MB`
}

export function AppUpdatePrompt({ update }: { update: AppUpdateView | null }): ReactElement | null {
  if (!update || update.phase === 'checking') return null
  if (update.phase === 'current') {
    return (
      <div className="appupdate" role="dialog" aria-modal="true" aria-labelledby="appupdate-title">
        <div className="appupdate__scrim" onClick={() => void store.appUpdateAction('dismiss')} />
        <section className="appupdate__card">
          <header className="appupdate__head">
            <span className="appupdate__icon" aria-hidden="true"><Icon name="check" size={16} /></span>
            <div>
              <h2 id="appupdate-title">Xerxes Agents is up to date</h2>
              <p>{update.version ? `${update.version} is the latest version.` : 'You have the latest version.'} It checks again every five minutes.</p>
            </div>
          </header>
          <footer className="appupdate__actions">
            <button className="btn btn--solid" autoFocus onClick={() => void store.appUpdateAction('dismiss')}>OK</button>
          </footer>
        </section>
      </div>
    )
  }
  const version = update.version ? `Xerxes Agents ${update.version}` : 'A new version'
  const busy = update.phase === 'downloading' || update.phase === 'restarting'
  const percent = update.total ? Math.min(100, Math.round(((update.received ?? 0) / update.total) * 100)) : 0
  return (
    <div className="appupdate" role="dialog" aria-modal="true" aria-labelledby="appupdate-title">
      <div className="appupdate__scrim" onClick={() => { if (!busy) void store.appUpdateAction('dismiss') }} />
      <section className="appupdate__card">
        <header className="appupdate__head">
          <span className="appupdate__icon" aria-hidden="true"><Icon name={update.phase === 'error' ? 'close' : 'arrowUp'} size={16} /></span>
          <div>
            <h2 id="appupdate-title">{update.phase === 'error' ? 'The update did not install' : update.phase === 'restarting' ? 'Restarting to finish the update' : `${version} is available`}</h2>
            <p>{update.phase === 'available' ? (update.installable ? 'May I download and install it? The app restarts when it is done; your runtime and conversations carry on.' : update.reason ?? 'It cannot be installed from here.')
              : update.phase === 'downloading' ? `Downloading… ${percent}% (${megabytes(update.received)} of ${megabytes(update.total)})`
              : update.phase === 'restarting' ? 'The new version opens in a moment.'
              : update.message}</p>
          </div>
        </header>
        {update.phase === 'downloading' && <div className="appupdate__progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}><span style={{ width: `${percent}%` }} /></div>}
        {update.notes && update.phase !== 'restarting' && <div className="appupdate__notes"><h3>What's new</h3><Markdown text={update.notes} /></div>}
        <footer className="appupdate__actions">
          {update.phase === 'available' && update.installable && <button className="btn btn--solid" onClick={() => void store.appUpdateAction('install')}>Install and restart</button>}
          {(update.phase === 'error' || (update.phase === 'available' && !update.installable)) && <button className="btn" onClick={() => void store.appUpdateAction('open-release')}>Open release page</button>}
          {update.phase === 'error' && <button className="btn" onClick={() => void store.appUpdateAction('check')}>Try again</button>}
          {!busy && <button className="btn" onClick={() => void store.appUpdateAction('dismiss')}>Not now</button>}
          {update.phase === 'available' && <button className="btn btn--ghost" onClick={() => void store.appUpdateAction('skip')}>Skip this version</button>}
        </footer>
      </section>
    </div>
  )
}
