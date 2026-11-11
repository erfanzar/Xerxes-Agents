// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useState, type ReactElement } from 'react'

import { Icon, type IconName } from './Icon.js'
import { RUN_MODE_CHOICES, gigabytes, memoryGigabytes } from './runModeView.js'
import { store, type Snapshot } from './store.js'
import type { RunModeName } from './types.js'

const MODE_ICONS: Record<RunModeName, IconName> = { private: 'shield', hybrid: 'spark', cloud: 'cloud' }

/**
 * How new tasks run: Private (Bonsai on this Mac, nothing leaves), Hybrid (a
 * cloud model leads and Bonsai does the file work here) or Cloud. Below the
 * choice, what the chosen mode needs: Bonsai set up on this Mac, and for Hybrid
 * and Cloud the cloud model that leads or does the work. The provider profiles
 * stay listed beneath.
 */
export function RunModeSection({ snap }: { snap: Snapshot }): ReactElement | null {
  /** A mode the person picked that still needs a step (setup, or a cloud provider) before it applies. */
  const [pending, setPending] = useState<RunModeName | null>(null)
  const view = snap.runMode
  // Once the mode it waited for is in use (setup finished, a provider chosen), nothing is pending.
  useEffect(() => { if (pending !== null && view?.mode === pending) setPending(null) }, [pending, view?.mode])
  // A runtime without run modes (or a snapshot from before them) shows nothing here.
  if (!view && !snap.runModeError) return null
  const shown = pending ?? view?.mode ?? null
  const busy = snap.runModeSaving || snap.turnActive
  const cloudProfiles = snap.providers.filter(provider => provider.local === false)
  const local = snap.localSetup ?? { status: null, job: null, error: '', loading: false }
  const job = local.job
  const settingUp = job?.state === 'running'

  const choose = async (mode: RunModeName): Promise<void> => {
    if (mode === view?.mode && !view.restartPending) { setPending(null); return }
    const result = await store.setRunMode(mode)
    setPending(result.ok ? null : result.code === 'needs_local_setup' || result.code === 'needs_cloud_profile' ? mode : null)
    if (result.code === 'needs_local_setup') void store.loadLocalSetup()
  }

  return (
    <section className="runmode" aria-label="How Xerxes runs new tasks">
      <div className="row__t">How Xerxes runs new tasks</div>
      <div className="modecards" role="radiogroup" aria-label="Mode">
        {RUN_MODE_CHOICES.map(choice => {
          const current = view?.mode === choice.mode
          return (
            <button
              key={choice.mode}
              role="radio"
              aria-checked={shown === choice.mode}
              className={`modecard${shown === choice.mode ? ' is-current' : ''}`}
              disabled={busy}
              onClick={() => void choose(choice.mode)}
            >
              <span className="modecard__head">
                <Icon name={MODE_ICONS[choice.mode]} size={15} />
                <span className="modecard__name">{choice.label}</span>
                {choice.mode === 'hybrid' && !view?.mode && <span className="provcard__badge provcard__badge--quiet">Recommended</span>}
                {current && <span className="provcard__badge provcard__badge--accent">In use</span>}
              </span>
              <span className="modecard__summary">{choice.summary}</span>
              <span className="modecard__boundary">{choice.boundary}</span>
            </button>
          )
        })}
      </div>
      {view?.mode === null && !pending && <p className="row__s">{view.worker ? 'Your current settings match none of these modes. Choose one to apply it.' : 'Nothing runs until you choose a mode.'}</p>}
      {view && view.lockedBy.length > 0 && (
        <p className="row__s">{view.lockedBy.join(' and ')} {view.lockedBy.length === 1 ? 'is' : 'are'} set in the environment and override{view.lockedBy.length === 1 ? 's' : ''} this choice.</p>
      )}
      {view?.restartPending && snap.runtimeUpdate === undefined && (
        <div className="runmode__notice" role="status">
          <span>Saved. The runtime applies it after a restart.</span>
          <button className="btn" disabled={snap.turnActive} onClick={() => void store.restartDaemon(false, false, 'mode')}>Restart now</button>
        </div>
      )}
      {snap.runModeError && <p role="alert" className="studio-error">{snap.runModeError}</p>}

      {(shown === 'private' || shown === 'hybrid') && (
        <div className="localpanel">
          <div className="localpanel__head">
            <Icon name="laptop" size={15} />
            <strong>On this Mac</strong>
            <span className="row__s">{shown === 'hybrid' ? 'does the file work' : 'does the work'}</span>
            {local.status?.server.running
              ? <span className="provcard__badge provcard__badge--accent">Running</span>
              : local.status ? <span className="provcard__badge provcard__badge--quiet">Stopped</span> : null}
          </div>
          {local.status === null && !local.error && <p className="row__s" role="status">Checking this Mac…</p>}
          {local.error && <p role="alert" className="studio-error">{local.error}</p>}
          {local.status && (
            <dl className="localpanel__facts">
              <div><dt>Model</dt><dd>{local.status.model.label ?? local.status.model.id ?? 'None fits'}</dd></div>
              <div><dt>Engine</dt><dd>{local.status.engine.found ? `${local.status.engine.label}${local.status.engine.version ? ` ${local.status.engine.version}` : ''}` : 'Not installed'}</dd></div>
              <div><dt>Window</dt><dd>{local.status.server.contextLength ? `${local.status.server.contextLength.toLocaleString()} tokens` : local.status.server.running ? 'Unknown' : 'Sized when it starts'}</dd></div>
              <div><dt>Memory</dt><dd>{local.status.memoryBytes ? memoryGigabytes(local.status.memoryBytes) : 'Unknown'}</dd></div>
            </dl>
          )}
          {settingUp && (
            <div className="localpanel__progress" role="status">
              <div className="progressbar" aria-hidden="true"><span style={{ width: `${job.percent ?? 0}%` }} /></div>
              <span className="row__s">{job.step}</span>
              <button className="btn" onClick={() => store.cancelLocalSetup()}>Cancel</button>
            </div>
          )}
          {job?.state === 'failed' && <p role="alert" className="studio-error">{job.error ?? 'Setup failed.'}</p>}
          {job?.state === 'cancelled' && <p className="row__s">Setup cancelled. A partial download is kept and resumes next time.</p>}
          {local.status?.blocker && !settingUp && (
            <div className="runmode__notice">
              <span>{local.status.blocker}</span>
              <button className="btn" onClick={() => void store.loadLocalSetup()}>Check again</button>
            </div>
          )}
          {local.status && !local.status.blocker && !settingUp && !(local.status.server.running && view?.mode === shown) && (
            <div className="localpanel__actions">
              <button className="btn btn--solid" disabled={busy} onClick={() => void store.startLocalSetup(shown)}>
                {local.status.model.downloaded
                  ? `Start Bonsai and use ${shown === 'private' ? 'Private' : 'Hybrid'}`
                  : `Download${local.status.model.bytes ? ` ${gigabytes(local.status.model.bytes)}` : ''} and set up Bonsai`}
              </button>
              {!local.status.model.downloaded && <span className="row__s">From Prism's Hugging Face repository, checked by SHA-256. It runs on this Mac.</span>}
            </div>
          )}
        </div>
      )}

      {shown === 'hybrid' && (
        <div className="localpanel">
          <div className="localpanel__head"><Icon name="cloud" size={15} /><strong>Cloud lead</strong><span className="row__s">plans, splits the work and checks it; cannot read your files</span></div>
          {cloudProfiles.length === 0
            ? <p className="row__s">Add a provider below: an API key or a sign-in.</p>
            : (
              <label className="field">
                <span>Lead</span>
                <select value={view?.mode === 'hybrid' ? view.worker?.profile ?? '' : ''} disabled={busy}
                  onChange={event => { if (event.target.value) void store.setRunMode('hybrid', { workerProfile: event.target.value }).then(result => { if (result.ok) setPending(null) }) }}>
                  <option value="" disabled>Choose the cloud model that leads</option>
                  {cloudProfiles.map(profile => <option key={profile.name} value={profile.name}>{profile.label}{profile.model ? ` · ${profile.model}` : ''}</option>)}
                </select>
              </label>
            )}
          <p className="row__s">Light and balanced agents run on Bonsai and answer directly; smart agents run on the lead. Change this under Agent intelligence.</p>
        </div>
      )}

      {shown === 'cloud' && (
        <div className="localpanel">
          <div className="localpanel__head"><Icon name="cloud" size={15} /><strong>Cloud provider</strong><span className="row__s">does the work</span></div>
          {cloudProfiles.length === 0
            ? <p className="row__s">Add a provider below: an API key or a sign-in.</p>
            : (
              <label className="field">
                <span>Provider</span>
                <select value={view?.mode === 'cloud' ? view.worker?.profile ?? '' : ''} disabled={busy}
                  onChange={event => { if (event.target.value) void store.setRunMode('cloud', { workerProfile: event.target.value }).then(result => { if (result.ok) setPending(null) }) }}>
                  <option value="" disabled>Choose a provider</option>
                  {cloudProfiles.map(profile => <option key={profile.name} value={profile.name}>{profile.label}{profile.model ? ` · ${profile.model}` : ''}</option>)}
                </select>
              </label>
            )}
        </div>
      )}
    </section>
  )
}
