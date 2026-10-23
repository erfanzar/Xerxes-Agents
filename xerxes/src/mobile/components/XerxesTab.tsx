// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useState, type ReactElement } from 'react'

import { relativeTime } from '../../client/format.js'
import type { ConfirmationState, SupervisorConfirmation } from '../../client/hubParsers.js'
import { useHubState, useNow, useServices } from '../context.js'
import { supervisorHeading } from '../viewModel.js'
import { ModelChooser, ModelSheet } from './ModelChooser.js'
import { TalkButton } from './TalkButton.js'
import { VoiceStatus } from './VoiceStatus.js'

export const SUPERVISOR_UNAVAILABLE = 'Supervisor not available on this hub'

const SETTLED: Readonly<Record<ConfirmationState, string | undefined>> = {
  pending: undefined,
  confirming: undefined,
  confirmed: 'Confirmed.',
  cancelled: 'Cancelled.',
  expired: 'Expired. Ask XERXES again if you still want this.',
  failed: undefined,
}

function ConfirmationCard({ item, canAct, onRespond }: { item: SupervisorConfirmation; canAct: boolean; onRespond: (decision: 'confirm' | 'cancel') => void }): ReactElement {
  const settled = SETTLED[item.state]
  return (
    <li className="card card-attention">
      <h3 className="card-title">XERXES proposes: {item.tool}</h3>
      <p>{item.summary}</p>
      {item.targets.length === 0 ? null : <ul className="notes">{item.targets.map(target => <li key={target}>{target}</li>)}</ul>}
      {item.requestedBy === undefined ? null : <p className="block-meta">Requested by {item.requestedBy}. Nothing runs until you confirm.</p>}
      {item.state === 'pending' ? (
        <div className="approval-actions" role="group" aria-label="Confirm or cancel this action">
          <button type="button" className="btn btn-primary" disabled={!canAct} onClick={() => onRespond('confirm')}>Confirm</button>
          <button type="button" className="btn" disabled={!canAct} onClick={() => onRespond('cancel')}>Cancel</button>
        </div>
      ) : null}
      {item.state === 'confirming' ? <p role="status">Sending…</p> : null}
      {settled === undefined ? null : <p role="status">{settled}</p>}
      {item.state === 'failed' ? <p role="alert" className="field-error">{item.message ?? 'That did not go through.'}</p> : null}
    </li>
  )
}

export function XerxesTab(): ReactElement {
  const state = useHubState()
  const now = useNow()
  const { controller, drafts, voice } = useServices()
  const supervisor = state.supervisor
  const config = state.supervisorConfig
  const unconfigured = config.status === 'unconfigured'
  const [sheet, setSheet] = useState(false)
  const [text, setText] = useState(() => drafts.get('supervisor'))
  const [fromVoice, setFromVoice] = useState(false)

  useEffect(() => {
    if (state.connection.state === 'live') void controller.supervisorOpen()
  }, [controller, state.connection.state])

  const live = state.connection.state === 'live'
  const update = (value: string): void => {
    setText(value)
    drafts.set('supervisor', value)
  }
  const send = async (): Promise<void> => {
    if (text.trim() === '' || supervisor.busy || !live) return
    // Ask the Hub to speak the reply on this device before sending, so it is listening when the reply lands.
    if (fromVoice) await voice?.armReply?.(true)
    const ok = await controller.supervisorSend(text.trim(), fromVoice ? 'voice' : 'text')
    if (ok) {
      setText('')
      drafts.clear('supervisor')
      setFromVoice(false)
    } else if (fromVoice) void voice?.armReply?.(false)
  }

  return (
    <section aria-labelledby="xerxes-title">
      <h1 id="xerxes-title" className="page-title">{supervisorHeading(config.config)}</h1>
      {supervisor.phase === 'unsupported' ? (
        <div role="status" className="empty-state">
          <h2>{SUPERVISOR_UNAVAILABLE}</h2>
          <p>XERXES watches every workspace for you and proposes actions you confirm. This Hub does not offer it.</p>
          <p className="block-meta">The Inbox, Task and Usage tabs work without it.</p>
        </div>
      ) : supervisor.phase === 'error' ? (
        <div role="alert" className="notice is-error">
          <p>Could not open the supervisor: {supervisor.error ?? 'unknown error'}</p>
          <button type="button" className="btn" onClick={() => { void controller.supervisorOpen() }}>Try again</button>
        </div>
      ) : (
        <>
          <div className="config-row">
            <button type="button" className="btn" aria-haspopup="dialog" onClick={() => setSheet(true)}>{config.canEdit ? 'Model' : 'Model (view)'}</button>
          </div>
          {sheet ? <ModelSheet onClose={() => setSheet(false)} /> : null}
          {unconfigured ? (
            <div className="empty-state" role="status">
              <h2>Choose XERXES's model</h2>
              <p>XERXES needs a model before it can answer. {config.canEdit ? 'Pick a machine, provider profile, model and thinking effort.' : 'The Hub owner chooses it from the Xerxes tab.'}</p>
              <ModelChooser />
            </div>
          ) : null}
          <p className="summary">Ask XERXES about your tasks. It can look across workspaces, and it proposes actions that wait for your confirmation. It cannot approve anything for you.</p>
          {supervisor.phase === 'loading' || supervisor.phase === 'idle' ? <p aria-busy="true">{live ? 'Opening…' : 'Waiting for connection…'}</p> : null}
          {supervisor.messages.length === 0 && supervisor.phase === 'ready' ? <p className="empty">No messages yet. Try “What needs me right now?”</p> : null}
          <ol className="blocks" aria-label="Conversation">
            {supervisor.messages.map(message => (
              <li key={message.id} className={`block block-${message.role === 'user' ? 'user' : 'assistant'}`}>
                <p className="block-who">
                  {message.role === 'xerxes' ? 'XERXES' : message.author ?? 'You'}
                  {message.source === 'voice' ? ' · voice' : ''}
                  <span className="block-meta"> · <time dateTime={message.at}>{relativeTime(message.at, now)}</time></span>
                </p>
                <p className="block-text">{message.text}</p>
              </li>
            ))}
          </ol>
          {supervisor.confirmations.length === 0 ? null : (
            <ul className="pending" aria-label="Waiting for your confirmation">
              {supervisor.confirmations.map(item => (
                <ConfirmationCard key={item.id} item={item} canAct={live} onRespond={decision => { void controller.supervisorRespond(item.id, decision) }} />
              ))}
            </ul>
          )}
          {supervisor.error !== undefined && supervisor.phase === 'ready' ? <p role="alert" className="field-error">{supervisor.error}</p> : null}
          <VoiceStatus voice={voice} />
          <form className="composer" aria-label="Message to XERXES" onSubmit={event => { event.preventDefault(); void send() }}>
            <label className="field">
              <span className="visually-hidden">Message to XERXES</span>
              <textarea rows={2} value={text} placeholder="Ask XERXES…" disabled={supervisor.phase !== 'ready'} onChange={event => { update(event.target.value); setFromVoice(false) }} />
            </label>
            <div className="composer-row">
              <TalkButton voice={voice} onTranscript={spoken => { update(text === '' ? spoken : `${text} ${spoken}`); setFromVoice(true) }} />
              <button type="submit" className="btn btn-primary" disabled={supervisor.phase !== 'ready' || !live || supervisor.busy || text.trim() === ''}>{supervisor.busy ? 'Sending…' : 'Send'}</button>
            </div>
          </form>
        </>
      )}
    </section>
  )
}
