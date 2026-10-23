// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactElement } from 'react'

import type { ConnectionStatus } from '../../client/bridge.js'
import { relativeTime } from '../../client/format.js'
import { HubConnection } from '../../client/hubConnection.js'
import { createHubController, type HubController, type SupervisorConfigInput, type SupervisorOptionsInput } from '../../client/hubController.js'
import { createHubStore, type HubState, type HubStore, type SupervisorConfigState } from '../../client/hubStore.js'
import { ActionConfirmationCard } from './ActionConfirmationCard.js'
import { AgentOrb } from './AgentOrb.js'
import { elapsedOf } from './duration.js'
import { Icon } from './Icon.js'
import { Markdown } from './markdown.js'
import { ModelChooser } from './ModelChooser.js'
import {
  HUB_CHANGED_EVENT, cockpitCall, cockpitEnsureSession, createCockpitSockets, windowCockpitApi,
  type CockpitApi,
} from './hubClient.js'
import { clearLegacyHubUrl, hubLabelFor } from './hubPrefs.js'
import { store } from './store.js'

/**
 * The XERXES supervisor tab. It talks to the Hub over the Hub WebSocket and to
 * nothing else. The desktop's main process saves the Hub address, signs this
 * app in and owns the socket; this page shows "Connect a Hub" until a Hub is
 * saved in Settings → Hub & phone. Nothing here is simulated.
 */

export const CONNECT_HUB_TITLE = 'Connect a Hub'
export const SUPERVISOR_OLD_HUB = 'This Hub is too old to run XERXES. Update the Hub, then reopen this tab.'

function connectionNote(status: ConnectionStatus): string | undefined {
  switch (status.state) {
    case 'live': return undefined
    case 'connecting': return 'Connecting to the Hub…'
    case 'replaying': return 'Catching up…'
    case 'reconnecting': return 'Reconnecting — what you see may be outdated.'
    case 'revoked': return status.reason
    case 'closed': return 'Disconnected from the Hub.'
  }
}

function useNow(intervalMs = 30_000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), intervalMs)
    return () => clearInterval(timer)
  }, [intervalMs])
  return now
}

/** The text of the model chip in the tab header. */
export function modelChipText(config: SupervisorConfigState): string {
  const choice = config.config
  const label = choice === undefined ? undefined : choice.effort === null ? choice.model : `${choice.model} · ${choice.effort}`
  switch (config.status) {
    case 'configured': return label ?? 'Model chosen'
    case 'machine_offline': return label === undefined ? 'Machine offline' : `${label} · machine offline`
    case 'unconfigured': return 'No model chosen'
    case 'invalid': return 'Model unavailable'
    case 'unknown': return config.phase === 'error' ? 'Model unavailable' : 'Model…'
  }
}

export function ConnectHub({ unavailable = false, onOpenSettings }: {
  readonly unavailable?: boolean
  readonly onOpenSettings: () => void
}): ReactElement {
  return (
    <section className="supervisor supervisor--connect" aria-labelledby="supervisor-connect-title">
      <h3 id="supervisor-connect-title">{CONNECT_HUB_TITLE}</h3>
      {unavailable ? (
        <p className="studio-muted">XERXES is only available in the Xerxes desktop app.</p>
      ) : (
        <>
          <p className="studio-muted">XERXES lives on your Hub. Enter the Hub&apos;s SSH address once in Settings and this app signs in by itself. Nothing is shown until a Hub is connected.</p>
          <button type="button" className="btn btn--solid" onClick={onOpenSettings}>Open Hub &amp; phone settings</button>
        </>
      )}
    </section>
  )
}

export function SignInFailed({ message, denied, onRetry, onSignInAgain, onOpenSettings }: {
  readonly message: string
  /** The Hub refused this desktop, so only a fresh pairing helps. */
  readonly denied: boolean
  readonly onRetry: () => void
  readonly onSignInAgain: () => void
  readonly onOpenSettings: () => void
}): ReactElement {
  return (
    <section className="supervisor supervisor--connect" aria-labelledby="supervisor-signin-title">
      <h3 id="supervisor-signin-title">Could not sign in to the Hub</h3>
      <p role="alert" className="action-card__error">{message}</p>
      <div className="hubphone__row">
        {denied
          ? <button type="button" className="btn btn--solid" onClick={onSignInAgain}>Pair this desktop again</button>
          : <button type="button" className="btn btn--solid" onClick={onRetry}>Try again</button>}
        <button type="button" className="btn" onClick={onOpenSettings}>Hub &amp; phone settings</button>
      </div>
    </section>
  )
}

export interface SupervisorViewProps {
  readonly state: HubState
  readonly hubLabel: string
  readonly now: number
  readonly onSend: (text: string) => Promise<boolean>
  readonly onRespond: (confirmationId: string, decision: 'confirm' | 'cancel') => void
  readonly onReopen: () => void
  readonly onOpenSettings: () => void
  readonly chooserOpen: boolean
  readonly onChooserChange: (open: boolean) => void
  /** Refresh machines and the saved choice when the chooser appears. */
  readonly onChooserShown: () => void
  readonly onOptions: (input: SupervisorOptionsInput) => void
  readonly onSaveModel: (input: SupervisorConfigInput) => Promise<boolean>
}

/**
 * One conversation page, laid out like a task: a slim toolbar, the messages
 * in a scrolling reading column (XERXES replies rendered as Markdown, your
 * own as the task view's user bubble), and the task view's composer docked
 * at the bottom.
 */
export function SupervisorView(props: SupervisorViewProps): ReactElement {
  const { state, hubLabel, now, onSend, onRespond, onReopen, onOpenSettings } = props
  const supervisor = state.supervisor
  const config = state.supervisorConfig
  const live = state.connection.state === 'live'
  const note = connectionNote(state.connection)
  const [text, setText] = useState('')
  const unconfigured = config.status === 'unconfigured'
  const showChooser = (props.chooserOpen || unconfigured) && supervisor.phase !== 'unsupported'
  const ready = supervisor.phase === 'ready'
  const conversing = supervisor.phase !== 'unsupported' && supervisor.phase !== 'error'
  const canType = ready && !unconfigured
  // What you just sent, shown at once: the Hub only returns the thread after
  // XERXES has answered, so without this nothing moved for the whole wait.
  const [pending, setPending] = useState<{ readonly text: string; readonly at: string } | null>(null)
  const thinking = supervisor.busy || supervisor.working === true || pending !== null
  const canSend = canType && live && !thinking && text.trim() !== ''
  // The task view's "Thinking… 3s" clock.
  const [startedAt, setStartedAt] = useState<number | null>(null)
  const [, setTick] = useState(0)
  useEffect(() => { setStartedAt(current => (thinking ? current ?? Date.now() : null)) }, [thinking])
  useEffect(() => {
    if (!thinking) return
    const timer = setInterval(() => setTick(value => value + 1), 1_000)
    return () => clearInterval(timer)
  }, [thinking])
  const thinkingSeconds = startedAt === null ? 0 : Math.max(0, Math.floor((Date.now() - startedAt) / 1_000))
  const lastUser = [...supervisor.messages].reverse().find(message => message.role === 'user')
  const showPending = pending !== null && !(supervisor.messages.at(-1)?.role !== 'xerxes' && lastUser?.text === pending.text)
  const streamRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const items = supervisor.messages.length + supervisor.confirmations.length
  // Follow the newest message, as the task conversation does.
  useEffect(() => {
    const element = streamRef.current
    if (element) element.scrollTop = element.scrollHeight
  }, [items, thinking, showPending])
  const grow = (): void => {
    const element = inputRef.current
    if (!element) return
    element.style.height = 'auto'
    element.style.height = `${Math.min(element.scrollHeight, 220)}px`
  }
  const submit = async (): Promise<void> => {
    const message = text.trim()
    if (message === '' || thinking || !live || unconfigured) return
    // A chat clears the box and shows the message straight away.
    setPending({ text: message, at: new Date().toISOString() })
    setText('')
    requestAnimationFrame(grow)
    const sent = await onSend(message)
    setPending(null)
    // A failed send gives the draft back instead of losing it.
    if (!sent) setText(current => current || message)
  }
  return (
    <section className="supervisor" aria-label="XERXES">
      <header className="supervisor__bar">
        <span className="supervisor__hub" title={`Hub: ${hubLabel}`}><span className={`supervisor__dot${live ? ' is-live' : ''}`} aria-hidden="true" />{hubLabel}</span>
        <span className="supervisor__flex" />
        <span className="supervisor__model" aria-label="XERXES model">{modelChipText(config)}</span>
        <button type="button" className="btn" aria-expanded={showChooser} onClick={() => props.onChooserChange(!props.chooserOpen)}>Model</button>
        <button type="button" className="btn" onClick={onOpenSettings}>Hub settings</button>
      </header>
      {note || showChooser ? (
        <div className="supervisor__panel">
          {note ? <p role="status" className="notice">{note}</p> : null}
          {showChooser ? (
            <ModelChooser
              machines={state.machines}
              config={config}
              onOptions={props.onOptions}
              onSave={props.onSaveModel}
              onOpen={props.onChooserShown}
              {...(props.chooserOpen && !unconfigured ? { onClose: () => props.onChooserChange(false) } : {})}
            />
          ) : null}
        </div>
      ) : null}
      <div className="supervisor__stream" ref={streamRef}>
        <div className="supervisor__col">
          {supervisor.phase === 'unsupported' ? (
            <p role="status" className="notice">{SUPERVISOR_OLD_HUB}</p>
          ) : supervisor.phase === 'error' ? (
            <div role="alert" className="notice is-error">
              <p>Could not open the supervisor: {supervisor.error ?? 'unknown error'}</p>
              <button type="button" className="btn" onClick={onReopen}>Try again</button>
            </div>
          ) : (
            <>
              {!ready ? <p className="supervisor__empty" aria-busy="true">{live ? 'Opening XERXES…' : 'Waiting for the Hub…'}</p> : null}
              {ready && supervisor.messages.length === 0 && !pending && !unconfigured ? (
                <div className="supervisor__empty">
                  <strong>Ask XERXES about your work</strong>
                  <p>It sees every task on your Hub, proposes actions and waits for a person to confirm them. It cannot approve anything for you.</p>
                  <p className="supervisor__try">Try “What needs me right now?”</p>
                </div>
              ) : null}
              {supervisor.messages.length > 0 || showPending ? (
                <ol className="supervisor__messages" aria-label="Conversation">
                  {supervisor.messages.map(message => (
                    <li key={message.id} className={`supervisor__message supervisor__message--${message.role}`}>
                      <p className="supervisor__who">
                        {message.role === 'xerxes' ? 'XERXES' : message.role === 'system' ? 'Notice' : message.author ?? 'You'}
                        {message.source === 'voice' ? ' · voice' : ''}
                        <span> · <time dateTime={message.at}>{relativeTime(message.at, now)}</time></span>
                      </p>
                      {message.role === 'xerxes'
                        ? <div className="msg msg--agent"><Markdown text={message.text} className="msg__text" /></div>
                        : message.role === 'user'
                          ? <div className="msg msg--user"><div className="msg__text">{message.text}</div></div>
                          : <p className="supervisor__notice">{message.text}</p>}
                    </li>
                  ))}
                  {showPending && pending ? (
                    <li className="supervisor__message supervisor__message--user supervisor__message--pending">
                      <p className="supervisor__who">You<span> · <time dateTime={pending.at}>just now</time></span></p>
                      <div className="msg msg--user"><div className="msg__text">{pending.text}</div></div>
                    </li>
                  ) : null}
                </ol>
              ) : null}
              {supervisor.confirmations.length === 0 ? null : (
                <ul className="supervisor__cards" aria-label="Actions XERXES proposes">
                  {supervisor.confirmations.map(item => (
                    <li key={item.id}><ActionConfirmationCard item={item} canAct={live} onRespond={decision => onRespond(item.id, decision)} /></li>
                  ))}
                </ul>
              )}
              {thinking ? (
                <div className="streamstatus supervisor__thinking" role="status" aria-live="polite">
                  <AgentOrb size={20} state="working" />
                  <span className="streamstatus__phrase">Thinking…</span>
                  <span className="streamstatus__clock">{elapsedOf(thinkingSeconds)}</span>
                </div>
              ) : null}
              {supervisor.error !== undefined && ready && !unconfigured ? <p role="alert" className="action-card__error">{supervisor.error}</p> : null}
            </>
          )}
        </div>
      </div>
      {conversing ? (
        <form className="supervisor__dock" aria-label="Message to XERXES" onSubmit={event => { event.preventDefault(); void submit() }}>
          <div className="composer-dock">
            <div className="composer">
              <label className="visually-hidden" htmlFor="supervisor-input">Message to XERXES</label>
              <textarea
                id="supervisor-input"
                ref={inputRef}
                className="composer__input"
                rows={1}
                value={text}
                placeholder={unconfigured ? 'Choose a model first' : 'Ask XERXES…'}
                disabled={!canType}
                onChange={event => { setText(event.target.value); grow() }}
                onKeyDown={event => {
                  if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void submit() }
                }}
              />
              <button type="submit" className="composer__send" aria-label={thinking ? 'Sending' : 'Send to XERXES'} title="Send (⏎ · ⇧⏎ newline)" disabled={!canSend}><Icon name="arrowUp" size={16} /></button>
            </div>
          </div>
          <p className="supervisor__hint">XERXES proposes; nothing runs until you confirm. ⏎ send · ⇧⏎ new line</p>
        </form>
      ) : null}
    </section>
  )
}

interface Session {
  readonly store: HubStore
  readonly controller: HubController
  disposeSockets(): void
}

function createSession(api: CockpitApi, wsUrl: string): Session {
  const hubStore = createHubStore(undefined, { coalesceMs: 50 })
  const sockets = createCockpitSockets(api)
  // The socket lives in the main process: it opens the saved Hub with its own
  // Origin and cookies, whatever URL is named here.
  const connection = new HubConnection({ url: wsUrl, socketFactory: sockets.factory, ensureSession: cockpitEnsureSession(api) })
  return { store: hubStore, controller: createHubController({ connection, store: hubStore }), disposeSockets: sockets.dispose }
}

function Connected({ api, origin, wsUrl, onOpenSettings }: {
  readonly api: CockpitApi
  readonly origin: string
  readonly wsUrl: string
  readonly onOpenSettings: () => void
}): ReactElement {
  const session = useMemo(() => createSession(api, wsUrl), [api, wsUrl])
  const state = useSyncExternalStore(session.store.subscribe, session.store.getSnapshot, session.store.getSnapshot)
  const now = useNow()
  const [chooserOpen, setChooserOpen] = useState(false)
  useEffect(() => {
    session.controller.start()
    return () => {
      session.controller.dispose()
      session.disposeSockets()
    }
  }, [session])
  useEffect(() => {
    if (state.connection.state === 'live') void session.controller.supervisorOpen()
  }, [session, state.connection.state])
  return (
    <SupervisorView
      state={state}
      hubLabel={hubLabelFor(origin)}
      now={now}
      onSend={text => session.controller.supervisorSend(text, 'text')}
      onRespond={(id, decision) => { void session.controller.supervisorRespond(id, decision) }}
      onReopen={() => { void session.controller.supervisorOpen() }}
      onOpenSettings={onOpenSettings}
      chooserOpen={chooserOpen}
      onChooserChange={setChooserOpen}
      onChooserShown={() => { void session.controller.machinesList(); void session.controller.supervisorConfigGet() }}
      onOptions={input => { void session.controller.supervisorConfigOptions(input) }}
      onSaveModel={input => session.controller.supervisorConfigSet(input)}
    />
  )
}

type Gate =
  | { readonly phase: 'unavailable' }
  | { readonly phase: 'reading' }
  | { readonly phase: 'no_hub' }
  | { readonly phase: 'signing_in' }
  | { readonly phase: 'failed'; readonly message: string; readonly denied: boolean }
  | { readonly phase: 'ready'; readonly origin: string; readonly wsUrl: string }

export function SupervisorPage(): ReactElement {
  const api = useMemo(() => windowCockpitApi(), [])
  const [gate, setGate] = useState<Gate>(() => (api === undefined ? { phase: 'unavailable' } : { phase: 'reading' }))
  const run = useRef(0)
  const openSettings = useCallback(() => store.openSettings('hub'), [])

  const load = useCallback(async (fresh: boolean): Promise<void> => {
    if (api === undefined) return
    const token = (run.current += 1)
    setGate({ phase: 'reading' })
    const status = await cockpitCall(api, 'status')
    if (token !== run.current) return
    if (!status.ok) { setGate({ phase: 'failed', message: status.message, denied: false }); return }
    if (status.status.target === null) { setGate({ phase: 'no_hub' }); return }
    setGate({ phase: 'signing_in' })
    const result = await cockpitCall(api, 'signin', fresh ? { fresh: true } : {})
    if (token !== run.current) return
    if (!result.ok) {
      if (result.code === 'no_target') setGate({ phase: 'no_hub' })
      else setGate({ phase: 'failed', message: result.message, denied: result.denied === true })
      return
    }
    setGate({ phase: 'ready', origin: result.origin, wsUrl: result.wsUrl })
  }, [api])

  useEffect(() => {
    clearLegacyHubUrl()
    void load(false)
    const changed = (): void => { void load(false) }
    window.addEventListener(HUB_CHANGED_EVENT, changed)
    return () => {
      run.current += 1
      window.removeEventListener(HUB_CHANGED_EVENT, changed)
    }
  }, [load])

  if (api === undefined || gate.phase === 'unavailable') return <ConnectHub unavailable onOpenSettings={openSettings} />
  switch (gate.phase) {
    case 'reading':
    case 'signing_in':
      return <section className="supervisor supervisor--connect"><p aria-busy="true">{gate.phase === 'reading' ? 'Reading the saved Hub…' : 'Signing in to the Hub…'}</p></section>
    case 'no_hub':
      return <ConnectHub onOpenSettings={openSettings} />
    case 'failed':
      return <SignInFailed message={gate.message} denied={gate.denied} onRetry={() => { void load(false) }} onSignInAgain={() => { void load(true) }} onOpenSettings={openSettings} />
    case 'ready':
      return <Connected api={api} origin={gate.origin} wsUrl={gate.wsUrl} onOpenSettings={openSettings} />
  }
}
