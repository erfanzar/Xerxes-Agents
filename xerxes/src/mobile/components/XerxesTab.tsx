// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useRef, useState, type ReactElement } from 'react'

import { AUTOPILOT_COMPOSER_HINT, AUTOPILOT_THREAD_NOTE, autoCardNote, autopilotChipText, pendingBadgeLabel } from '../../client/autopilot.js'
import { newCommandId } from '../../client/bridge.js'
import { relativeTime } from '../../client/format.js'
import { watchChipText } from '../../client/watch.js'
import type { ConfirmationState, SupervisorConfirmation, SupervisorMessage } from '../../client/hubParsers.js'
import { activeAutopilotRows, autopilotPendingCount, autopilotThreadAvailable } from '../../client/hubStore.js'
import { useHubState, useNow, useServices } from '../context.js'
import { supervisorHeading } from '../viewModel.js'
import { remainingAfterSend, withTranscript } from './Composer.js'
import { JumpToLatest, useFollowScroll } from './TaskTranscript.js'
import { AutopilotSheet } from './AutopilotSheet.js'
import { WatchCards, WatchSheet } from './WatchSheet.js'
import { ModelChooser, ModelSheet } from './ModelChooser.js'
import { Markdown } from './Markdown.js'
import { TalkButton } from './TalkButton.js'
import { VoiceStatus } from './VoiceStatus.js'
import { CallPill, ComposerVoiceButton } from './CallControls.js'

export const SUPERVISOR_UNAVAILABLE = 'Supervisor not available on this hub'

const SETTLED: Readonly<Record<ConfirmationState, string | undefined>> = {
  pending: undefined,
  confirming: undefined,
  confirmed: 'Confirmed.',
  cancelled: 'Cancelled.',
  expired: 'Expired. Ask XERXES again if you still want this.',
  failed: undefined,
}

/** Who a thread message is from. Hub notices (Autopilot, Watch, permission answers) are never "You". */
export function supervisorAuthor(message: Pick<SupervisorMessage, 'role' | 'text' | 'author'>): string {
  if (message.role === 'xerxes') return 'XERXES'
  if (message.role === 'system') return /^autopilot\b/i.test(message.text) ? 'Autopilot' : /^watch\b/i.test(message.text) ? 'Watch' : 'Notice'
  return message.author ?? 'You'
}

function blockClass(message: SupervisorMessage): string {
  return message.role === 'user' ? 'block block-user' : message.role === 'system' ? 'block block-system' : 'block block-assistant'
}

/** One thread message; system notices get their own muted row. */
function ThreadMessage({ message, now }: { message: SupervisorMessage; now: number }): ReactElement {
  return (
    <li data-block-id={message.id} className={blockClass(message)}>
      <p className="block-who">
        {supervisorAuthor(message)}
        {message.source === 'voice' ? ' · voice' : ''}
        <span className="block-meta"> · <time dateTime={message.at}>{relativeTime(message.at, now)}</time></span>
      </p>
      {message.role === 'user' ? <p className="block-text">{message.text}</p> : <Markdown text={message.text} className="block-md" />}
    </li>
  )
}

export function ConfirmationCard({ item, canAct, onRespond }: { item: SupervisorConfirmation; canAct: boolean; onRespond: (decision: 'confirm' | 'cancel') => void }): ReactElement {
  const settled = SETTLED[item.state]
  if (item.auto === true) {
    // Autopilot ran this without a Confirm tap: it is a record, never a question.
    const running = item.state === 'pending' || item.state === 'confirming'
    return (
      <li className="card" data-auto="true">
        <h3 className="card-title">{running ? 'XERXES is doing' : 'XERXES did'}: {item.tool}</h3>
        <p>{item.summary}</p>
        {item.targets.length === 0 ? null : <ul className="notes">{item.targets.map(target => <li key={target}>{target}</li>)}</ul>}
        {item.state === 'failed'
          ? <p role="alert" className="field-error">{item.message ?? 'Autopilot tried this and it did not go through.'}</p>
          : running
            ? <p role="status" className="block-meta">Working…</p>
            : item.state === 'confirmed'
              ? <p role="status" className="block-meta">{autoCardNote(item.confirmedBy)}</p>
              : <p role="status" className="block-meta">{settled}</p>}
      </li>
    )
  }
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
      {item.state === 'failed' ? (
        <>
          <p role="alert" className="field-error">{item.message ?? 'That did not go through.'}</p>
          {/* The Hub only acts on a pending card, so asking again never runs it twice. */}
          <div className="approval-actions" role="group" aria-label="Try this action again">
            <button type="button" className="btn" disabled={!canAct} onClick={() => onRespond('confirm')}>Try again</button>
          </div>
        </>
      ) : null}
    </li>
  )
}

/** The owner's private Autopilot thread: wake-ups, replies, cards waiting for Confirm, and what XERXES did alone. */
export function AutopilotThreadBody(): ReactElement {
  const state = useHubState()
  const now = useNow()
  const { controller } = useServices()
  const thread = state.autopilotThread
  const live = state.connection.state === 'live'
  const list = useRef<HTMLDivElement>(null)
  const follow = useFollowScroll(list, thread.phase === 'ready', [thread.messages, thread.confirmations, thread.working])
  return (
    <div ref={list} aria-label="Autopilot thread" role="region">
      <p className="summary">{AUTOPILOT_THREAD_NOTE}</p>
      {thread.phase === 'loading' || thread.phase === 'idle' ? <p role="status" aria-busy="true">Opening the Autopilot thread…</p> : null}
      {thread.phase === 'error' ? (
        <div role="alert" className="notice is-error">
          <p>Could not open the Autopilot thread: {thread.error ?? 'unknown error'}</p>
          <button type="button" className="btn" disabled={!live} onClick={() => { void controller.autopilotThreadOpen() }}>Try again</button>
        </div>
      ) : null}
      {thread.phase === 'ready' && thread.messages.length === 0 && thread.confirmations.length === 0 ? <p className="empty">Nothing yet.</p> : null}
      <ol className="blocks" aria-label="Autopilot conversation">
        {thread.messages.map(message => <ThreadMessage key={message.id} message={message} now={now} />)}
      </ol>
      {thread.confirmations.length === 0 ? null : (
        <ul className="pending" aria-label="Autopilot actions">
          {thread.confirmations.map(item => (
            <ConfirmationCard key={item.id} item={item} canAct={live} onRespond={decision => { void controller.autopilotThreadRespond(item.id, decision) }} />
          ))}
        </ul>
      )}
      {thread.working === true ? <p role="status" aria-live="polite" className="working-line">Autopilot is working…</p> : null}
      <JumpToLatest visible={follow.away} onJump={follow.toBottom} />
    </div>
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
  const [autopilotSheet, setAutopilotSheet] = useState(false)
  const [threadView, setThreadView] = useState(false)
  const [watchSheet, setWatchSheet] = useState(false)
  const autopilotChip = autopilotChipText(activeAutopilotRows(state))
  const watchChip = state.watch.unsupported ? undefined : watchChipText(state.watch.rows)
  const threadAvailable = autopilotThreadAvailable(state)
  const showThread = threadAvailable && threadView
  const threadPending = autopilotPendingCount(state)
  const [text, setText] = useState(() => drafts.get('supervisor'))
  const [fromVoice, setFromVoice] = useState(false)

  useEffect(() => {
    if (state.connection.state === 'live') void controller.supervisorOpen()
  }, [controller, state.connection.state])

  // Open on the newest message and keep following it while the reader is at the bottom,
  // exactly like the task transcript; reading older messages is never yanked away.
  const conversation = useRef<HTMLDivElement>(null)
  const follow = useFollowScroll(conversation, supervisor.phase === 'ready' && !showThread, [supervisor.messages, supervisor.confirmations, supervisor.working, showThread])

  const live = state.connection.state === 'live'
  const latest = useRef(text)
  const sending = useRef(false)
  // One message id per draft text: retrying after an uncertain failure resends the same id, so the Hub dedupes it.
  const pending = useRef<{ readonly text: string; readonly id: string } | undefined>(undefined)
  // An error the person has moved past (edited the draft or sent again) is not shown again.
  const [dismissedError, setDismissedError] = useState<string | undefined>(undefined)
  const update = (value: string): void => {
    latest.current = value
    setText(value)
    drafts.set('supervisor', value)
    if (pending.current !== undefined && pending.current.text !== value.trim()) pending.current = undefined
    if (supervisor.error !== undefined) setDismissedError(supervisor.error)
  }
  const send = async (): Promise<void> => {
    const raw = latest.current
    const sent = raw.trim()
    if (sending.current || sent === '' || supervisor.busy || !live) return
    sending.current = true
    setDismissedError(undefined)
    try {
      // Ask the Hub to speak the reply on this device before sending, so it is listening when the reply lands.
      if (fromVoice) await voice?.armReply?.(true)
      if (pending.current === undefined || pending.current.text !== sent) pending.current = { text: sent, id: newCommandId() }
      const ok = await controller.supervisorSend(sent, fromVoice ? 'voice' : 'text', pending.current.id)
      if (ok) {
        pending.current = undefined
        const rest = remainingAfterSend(latest.current, raw)
        latest.current = rest
        setText(rest)
        if (rest === '') drafts.clear('supervisor')
        else drafts.set('supervisor', rest)
        setFromVoice(false)
      } else if (fromVoice) void voice?.armReply?.(false)
    } finally {
      sending.current = false
    }
  }
  const shownError = supervisor.error !== undefined && supervisor.error !== dismissedError && !supervisor.busy ? supervisor.error : undefined

  return (
    <section aria-labelledby="xerxes-title">
      <CallPill />
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
            {state.autopilot.unsupported ? null : (
              <button type="button" className="btn" aria-haspopup="dialog" onClick={() => setAutopilotSheet(true)}>{state.autopilot.canEdit || state.autopilot.phase !== 'ready' ? 'Autopilot' : 'Autopilot (view)'}</button>
            )}
            {state.watch.unsupported ? null : (
              <button type="button" className="btn" aria-haspopup="dialog" onClick={() => setWatchSheet(true)}>{state.watch.canEdit || state.watch.phase !== 'ready' ? 'Watch' : 'Watch (view)'}</button>
            )}
            {autopilotChip === undefined ? null : <span className="chip" role="status" aria-label="Autopilot status">{autopilotChip}</span>}
            {watchChip === undefined ? null : <span className="chip" role="status" aria-label="Watch status">{watchChip}</span>}
            {threadAvailable ? (
              <button
                type="button"
                className="btn"
                aria-pressed={showThread}
                onClick={() => { const next = !threadView; setThreadView(next); if (next) void controller.autopilotThreadOpen() }}
              >
                {showThread ? 'Back to chat' : 'Autopilot thread'}
                {threadPending > 0 ? <span className="chip" aria-label={pendingBadgeLabel(threadPending)}> {threadPending}</span> : null}
              </button>
            ) : null}
          </div>
          {sheet ? <ModelSheet onClose={() => setSheet(false)} /> : null}
          {autopilotSheet ? <AutopilotSheet onClose={() => setAutopilotSheet(false)} /> : null}
          {watchSheet ? <WatchSheet onClose={() => setWatchSheet(false)} /> : null}
          <WatchCards />
          {unconfigured ? (
            <div className="empty-state" role="status">
              <h2>Choose XERXES's model</h2>
              <p>XERXES needs a model before it can answer. {config.canEdit ? 'Pick a machine, provider profile, model and thinking effort.' : 'The Hub owner chooses it from the Xerxes tab.'}</p>
              <ModelChooser />
            </div>
          ) : null}
          <p className="summary">Ask XERXES about your tasks. It looks across every workspace and acts for you at once: it starts, steers and stops tasks, answers their questions and decides permission requests. With anyone else in the thread, its actions wait for your Confirm.</p>
          {showThread ? <AutopilotThreadBody /> : null}
          {showThread ? null : <>
          {supervisor.phase === 'loading' || supervisor.phase === 'idle' ? <p aria-busy="true">{live ? 'Opening…' : 'Waiting for connection…'}</p> : null}
          {supervisor.messages.length === 0 && supervisor.phase === 'ready' ? <p className="empty">No messages yet. Try “What needs me right now?”</p> : null}
          <div ref={conversation}>
            <ol className="blocks" aria-label="Conversation">
              {supervisor.messages.map(message => <ThreadMessage key={message.id} message={message} now={now} />)}
            </ol>
            {supervisor.confirmations.length === 0 ? null : (
              <ul className="pending" aria-label="Waiting for your confirmation">
                {supervisor.confirmations.map(item => (
                  <ConfirmationCard key={item.id} item={item} canAct={live} onRespond={decision => { void controller.supervisorRespond(item.id, decision) }} />
                ))}
              </ul>
            )}
            {supervisor.working === true && supervisor.phase === 'ready' ? <p role="status" aria-live="polite" className="working-line">XERXES is thinking…</p> : null}
            <JumpToLatest visible={follow.away} onJump={follow.toBottom} />
          </div>
          {shownError !== undefined && supervisor.phase === 'ready' ? <p role="alert" className="field-error">{shownError}</p> : null}
          <VoiceStatus voice={voice} />
          <form className="composer" aria-label="Message to XERXES" onSubmit={event => { event.preventDefault(); void send() }}>
            <label className="field">
              <span className="visually-hidden">Message to XERXES</span>
              <textarea rows={2} value={text} placeholder="Ask XERXES…" disabled={supervisor.phase !== 'ready'} onChange={event => { update(event.target.value); setFromVoice(false) }} />
            </label>
            <div className="composer-row">
              <TalkButton voice={voice} onTranscript={spoken => { update(withTranscript(latest.current, spoken)); setFromVoice(true) }} />
              <ComposerVoiceButton />
              <button type="submit" className="btn btn-primary" disabled={supervisor.phase !== 'ready' || !live || supervisor.busy || text.trim() === ''}>{supervisor.busy ? 'Sending…' : 'Send'}</button>
            </div>
          </form>
          </>}
        </>
      )}
    </section>
  )
}
