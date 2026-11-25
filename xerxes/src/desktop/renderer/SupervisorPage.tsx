// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactElement } from 'react'
import { pageIsInactive, subscribePerfPaused } from './pageVisibility.js'

import type { ConnectionStatus } from '../../client/bridge.js'
import { HubConnection } from '../../client/hubConnection.js'
import { createSupervisorSlackClient, type SupervisorSlackClient } from '../../client/supervisorSlack.js'
import { SupervisorSlack } from './SupervisorSlack.js'
import { AUTOPILOT_COMPOSER_HINT, autopilotChipText, pendingBadgeLabel } from '../../client/autopilot.js'
import { createHubController, type AutopilotInput, type HubController, type SupervisorConfigInput, type SupervisorOptionsInput, type WatchInput } from '../../client/hubController.js'
import { activeAutopilotRows, autopilotPendingCount, autopilotThreadAvailable, createHubStore, isHubOwner, orderTasks, type HubState, type HubStore, type SupervisorConfigState } from '../../client/hubStore.js'
import { SuperagentControlPanel } from '../../client/SuperagentControlPanel.js'
import type { SuperagentControlClient } from '../../client/superagentControl.js'
import { watchChipText } from '../../client/watch.js'
import { ActionConfirmationCard } from './ActionConfirmationCard.js'
import { AgentOrb } from './AgentOrb.js'
import { AutopilotPanel } from './AutopilotPanel.js'
import { AutopilotThreadView } from './AutopilotThreadView.js'
import { XerxesLogoMark } from './BrandMark.js'
import { WatchPanel, WatchStrip } from './WatchPanel.js'
import { elapsedOf } from './duration.js'
import { Icon } from './Icon.js'
import { ModelChooser } from './ModelChooser.js'
import { SupervisorControls, type ControlsTab, type ControlsTabEntry } from './SupervisorControls.js'
import { LiveSteps, ThreadMessages } from './SupervisorThread.js'
import { isOpenCard } from '../../client/supervisorRecords.js'
import { useTranscriptScroll } from './transcriptScroll.js'
import { liveSteps } from '../../client/supervisorSteps.js'
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
    case 'reconnecting': return 'Reconnecting. What you see may be outdated.'
    case 'revoked': return status.reason
    case 'closed': return 'Disconnected from the Hub.'
  }
}

function useNow(intervalMs = 30_000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = setInterval(() => { if (!pageIsInactive()) setNow(Date.now()) }, intervalMs)
    const unsub = subscribePerfPaused(() => { if (!pageIsInactive()) setNow(Date.now()) })
    return () => { clearInterval(timer); unsub() }
  }, [intervalMs])
  return now
}

const DEFAULT_COMPOSER_HINT = 'XERXES acts for the Hub owner at once; with anyone else in the thread its actions wait for Confirm.'

/** One-tap starting questions in the empty conversation. */
const SUGGESTIONS: readonly string[] = ['What needs me right now?', 'What is every task doing?', 'Did anything fail today?']

type PillTone = 'on' | 'off' | 'warn'

/** A header control: its name and live state; opens the controls dialog on its tab. */
function ControlPill({ name, value, tone, expanded, disabled = false, title, onClick }: {
  readonly name: string
  readonly value: string
  readonly tone: PillTone
  readonly expanded: boolean
  readonly disabled?: boolean
  readonly title?: string
  readonly onClick: () => void
}): ReactElement {
  return (
    <button
      type="button"
      className="xpill"
      data-tone={tone}
      aria-haspopup="dialog"
      aria-expanded={expanded}
      disabled={disabled}
      title={title ?? `${name}: ${value}`}
      onClick={onClick}
    >
      <span className="xpill__dot" aria-hidden="true" />
      <span className="xpill__name">{name}</span>
      <span className="xpill__value">{value}</span>
    </button>
  )
}

function modelTone(config: SupervisorConfigState): PillTone {
  switch (config.status) {
    case 'configured': return 'on'
    case 'unknown': return config.phase === 'error' ? 'warn' : 'off'
    default: return 'warn'
  }
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
      <span className="xempty__mark" aria-hidden="true"><XerxesLogoMark size={52} className="xempty__logo" decorative /></span>
      <h2 id="supervisor-connect-title">{CONNECT_HUB_TITLE}</h2>
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
      <span className="xempty__mark" aria-hidden="true"><AgentOrb size={40} state="failed" live={false} /></span>
      <h2 id="supervisor-signin-title">Could not sign in to the Hub</h2>
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
  readonly controlClient?: SuperagentControlClient
  readonly slack?: { readonly client: SupervisorSlackClient; readonly openOAuth: (url: string) => Promise<boolean> }
  readonly hubLabel: string
  readonly now: number
  readonly onSend: (text: string) => Promise<boolean>
  /** Stops XERXES's running turn (Stop, and the first half of Steer). Absent: no stop or steer. */
  readonly onCancel?: () => Promise<boolean>
  readonly onRespond: (confirmationId: string, decision: 'confirm' | 'cancel') => void
  readonly onReopen: () => void
  readonly onOpenSettings: () => void
  readonly chooserOpen: boolean
  readonly onChooserChange: (open: boolean) => void
  /** Refresh machines and the saved choice when the chooser appears. */
  readonly onChooserShown: () => void
  readonly onOptions: (input: SupervisorOptionsInput) => void
  readonly onSaveModel: (input: SupervisorConfigInput) => Promise<boolean>
  /** Autopilot panel open state. Absent: the panel is closed and cannot be opened. */
  readonly autopilotOpen?: boolean
  readonly onAutopilotChange?: (open: boolean) => void
  readonly onSaveAutopilot?: (input: AutopilotInput) => Promise<boolean>
  readonly onRetryAutopilot?: () => void
  /** The owner's private Autopilot thread view. Absent: the view cannot be opened. */
  readonly autopilotThreadOpen?: boolean
  readonly onAutopilotThreadChange?: (open: boolean) => void
  readonly onRespondAutopilot?: (confirmationId: string, decision: 'confirm' | 'cancel') => void
  /** Watch panel open state. Absent: the panel cannot be opened (the live cards still show). */
  readonly watchOpen?: boolean
  readonly onWatchChange?: (open: boolean) => void
  readonly onSaveWatch?: (input: WatchInput) => Promise<boolean>
  readonly onRetryWatch?: () => void
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
  const autopilot = state.autopilot
  const autopilotChip = autopilotChipText(activeAutopilotRows(state), state.autopilot.all)
  const autopilotAvailable = props.onAutopilotChange !== undefined && !autopilot.unsupported
  const showAutopilot = autopilotAvailable && props.autopilotOpen === true && supervisor.phase !== 'unsupported'
  const threadAvailable = props.onAutopilotThreadChange !== undefined && autopilotThreadAvailable(state)
  const showThread = threadAvailable && props.autopilotThreadOpen === true && supervisor.phase !== 'unsupported'
  const threadPending = autopilotPendingCount(state)
  const watch = state.watch
  const watchAvailable = props.onWatchChange !== undefined && !watch.unsupported
  const showWatch = watchAvailable && props.watchOpen === true && supervisor.phase !== 'unsupported'
  const watchChip = watch.unsupported ? undefined : watchChipText(watch.rows)
  const watchTasks = useMemo(() => orderTasks(Object.values(state.tasks), state.workspaces), [state.tasks, state.workspaces])
  const live = state.connection.state === 'live'
  const note = connectionNote(state.connection)
  const [text, setText] = useState('')
  const [safetyOpen, setSafetyOpen] = useState(false)
  const [slackOpen, setSlackOpen] = useState(false)
  const slack = isHubOwner(state) && supervisor.phase !== 'unsupported' ? props.slack : undefined
  const showSlack = slackOpen && slack !== undefined
  useEffect(() => { if (slack === undefined) setSlackOpen(false) }, [slack])
  const safetyAvailable = props.controlClient !== undefined && isHubOwner(state)
  const unconfigured = config.status === 'unconfigured'
  // An unconfigured XERXES keeps its chooser in the page; everything else opens the controls dialog.
  const inlineChooser = unconfigured && supervisor.phase !== 'unsupported'
  const inlineChooserRef = useRef<HTMLDivElement>(null)
  const controlsTab: ControlsTab | undefined = safetyOpen && safetyAvailable ? 'safety' : showAutopilot ? 'autopilot' : showWatch ? 'watch'
    : props.chooserOpen && !inlineChooser && supervisor.phase !== 'unsupported' ? 'model' : undefined
  const controlTabs: ControlsTabEntry[] = [
    ...(autopilotAvailable ? [{ id: 'autopilot' as const, label: 'Autopilot', hint: autopilotChip?.replace(/^Autopilot /, '') ?? 'Off' }] : []),
    ...(watchAvailable ? [{ id: 'watch' as const, label: 'Watch', hint: watchChip ?? 'Nothing watched' }] : []),
    { id: 'model', label: 'Model', hint: modelChipText(config) },
    ...(safetyAvailable ? [{ id: 'safety' as const, label: 'Safety', hint: 'Owner controls' }] : []),
  ]
  const closeControls = (): void => {
    setSafetyOpen(false)
    if (props.autopilotOpen) props.onAutopilotChange?.(false)
    if (props.watchOpen) props.onWatchChange?.(false)
    if (props.chooserOpen) props.onChooserChange(false)
  }
  const focusInlineChooser = (): void => {
    closeControls()
    setSlackOpen(false)
    props.onChooserShown()
    const focus = (): void => {
      const container = inlineChooserRef.current
      if (!container) return
      container.scrollIntoView?.({ behavior: 'smooth', block: 'nearest' })
      const target = container.querySelector<HTMLElement>('button:not(:disabled), input:not(:disabled), [tabindex="0"]') ?? container
      target.focus?.()
    }
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(focus)
    else focus()
  }
  const openTab = (tab: ControlsTab): void => {
    if (tab === 'model' && inlineChooser) {
      focusInlineChooser()
      return
    }
    if (tab === 'model' && supervisor.phase === 'unsupported') return
    setSafetyOpen(tab === 'safety')
    if (tab !== 'autopilot' && props.autopilotOpen) props.onAutopilotChange?.(false)
    if (tab !== 'watch' && props.watchOpen) props.onWatchChange?.(false)
    if (tab !== 'model' && props.chooserOpen) props.onChooserChange(false)
    if (tab === 'autopilot' && !props.autopilotOpen) props.onAutopilotChange?.(true)
    if (tab === 'watch' && !props.watchOpen) props.onWatchChange?.(true)
    if (tab === 'model' && !props.chooserOpen) props.onChooserChange(true)
  }
  const ready = supervisor.phase === 'ready'
  const conversing = supervisor.phase !== 'unsupported' && supervisor.phase !== 'error'
  const canType = ready && !unconfigured
  // What you just sent, shown at once: the Hub only returns the thread after
  // XERXES has answered, so without this nothing moved for the whole wait.
  const [pending, setPending] = useState<{ readonly text: string; readonly at: string } | null>(null)
  const thinking = supervisor.busy || supervisor.working === true || pending !== null
  // The running turn's steps; a finished turn's steps fold above its reply instead.
  const runningSteps = liveSteps(supervisor.steps, supervisor.messages)
  // While XERXES works, sending queues the message behind the running turn (the Hub runs them in order).
  const canSend = canType && live && text.trim() !== ''
  // The task view's "Thinking… 3s" clock.
  const [startedAt, setStartedAt] = useState<number | null>(null)
  const [, setTick] = useState(0)
  useEffect(() => { setStartedAt(current => (thinking ? current ?? Date.now() : null)) }, [thinking])
  useEffect(() => {
    if (!thinking) return
    const timer = setInterval(() => { if (!pageIsInactive()) setTick(value => value + 1) }, 1_000)
    const unsub = subscribePerfPaused(() => { if (!pageIsInactive()) setTick(value => value + 1) })
    return () => { clearInterval(timer); unsub() }
  }, [thinking])
  const thinkingSeconds = startedAt === null ? 0 : Math.max(0, Math.floor((Date.now() - startedAt) / 1_000))
  const lastUser = [...supervisor.messages].reverse().find(message => message.role === 'user')
  const showPending = pending !== null && !(supervisor.messages.at(-1)?.role !== 'xerxes' && lastUser?.text === pending.text)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  // Follow the tail while it grows (replies, live steps, cards), as the task view does; scrolling up pauses it.
  const scroll = useTranscriptScroll(showSlack ? 'slack' : showThread ? 'autopilot' : 'shared', { more: false, automatic: false, load: async () => undefined })
  const openCards = supervisor.confirmations.filter(isOpenCard)
  const records = supervisor.confirmations.filter(card => !isOpenCard(card))
  const grow = (): void => {
    const element = inputRef.current
    if (!element) return
    element.style.height = 'auto'
    element.style.height = `${Math.min(element.scrollHeight, 220)}px`
  }
  /** `queue`: behind the running turn (default). `steer`: stop the running turn, then send. */
  const submit = async (mode: 'queue' | 'steer' = 'queue'): Promise<void> => {
    const message = text.trim()
    if (message === '' || !live || unconfigured) return
    const busy = thinking
    // A chat clears the box and shows the message straight away; a queued one
    // appears in the thread as soon as the Hub stores it.
    if (!busy) setPending({ text: message, at: new Date().toISOString() })
    // Sending brings you back to the tail, even if you had scrolled up to read.
    scroll.scrollToLatest()
    setText('')
    requestAnimationFrame(grow)
    if (busy && mode === 'steer') await props.onCancel?.()
    const sent = await onSend(message)
    if (!busy) setPending(null)
    // A failed send gives the draft back instead of losing it.
    if (!sent) setText(current => current || message)
  }
  const canSteer = thinking && canSend && props.onCancel !== undefined
  return (
    <section className="supervisor" aria-label="XERXES">
      <header className="supervisor__bar">
        {threadAvailable || slack !== undefined ? (
          <div className="xseg" role="group" aria-label="XERXES view">
            <button type="button" aria-pressed={!showThread && !showSlack} onClick={() => {
              setSlackOpen(false)
              if (showThread) props.onAutopilotThreadChange?.(false)
            }}>Chat</button>
            {threadAvailable ? <button type="button" aria-pressed={showThread && !showSlack} onClick={() => {
              setSlackOpen(false)
              if (!showThread) props.onAutopilotThreadChange?.(true)
            }}>
              Activity
              {threadPending > 0 ? <span className="supervisor__badge" aria-label={pendingBadgeLabel(threadPending)}>{threadPending}</span> : null}
            </button> : null}
            {slack !== undefined ? <button type="button" aria-pressed={showSlack} onClick={() => { closeControls(); setSlackOpen(true) }}>Slack</button> : null}
          </div>
        ) : null}
        <span className="supervisor__flex" />
        <div className="xpills" role="group" aria-label="XERXES status">
          {autopilotAvailable ? (
            <ControlPill name="Autopilot" value={autopilotChip === undefined ? 'Off' : autopilotChip.replace(/^Autopilot on · /, '')}
              tone={autopilotChip === undefined ? 'off' : 'on'} expanded={controlsTab === 'autopilot'} onClick={() => openTab('autopilot')} />
          ) : null}
          {watchAvailable ? (
            <ControlPill name="Watch" value={watchChip === undefined ? 'Off' : watchChip.replace(/^Watching · /, '')}
              tone={watchChip === undefined ? 'off' : watch.rows.some(row => row.active && (row.stats?.anomalies.length ?? 0) > 0) ? 'warn' : 'on'}
              expanded={controlsTab === 'watch'} onClick={() => openTab('watch')} />
          ) : null}
          <ControlPill
            name="Model"
            value={modelChipText(config)}
            tone={modelTone(config)}
            expanded={controlsTab === 'model' || inlineChooser}
            disabled={supervisor.phase === 'unsupported'}
            title={inlineChooser ? 'Model: No model chosen · jump to model setup' : `Model: ${modelChipText(config)}`}
            onClick={() => openTab('model')}
          />
          {safetyAvailable ? <ControlPill name="Safety" value="Owner controls" tone="off" expanded={controlsTab === 'safety'} onClick={() => openTab('safety')} /> : null}
        </div>
        <span className="xbar__sep" aria-hidden="true" />
        <span className="supervisor__hub" title={`Hub: ${hubLabel}${live ? '' : ' (not connected)'}`}>
          <span className={`supervisor__dot${live ? ' is-live' : ''}`} aria-hidden="true" />
          <span className="supervisor__hubname">{hubLabel}</span>
        </span>
        <button type="button" className="xicon" aria-label="Hub settings" title="Hub settings" onClick={onOpenSettings}><Icon name="settings" size={15} /></button>
      </header>
      {controlsTab === undefined ? null : (
        <SupervisorControls
          tab={controlsTab}
          tabs={controlTabs}
          onTab={openTab}
          onClose={closeControls}
        >
          {controlsTab === 'safety' && props.controlClient !== undefined ? <SuperagentControlPanel client={props.controlClient} online={live} /> : controlsTab === 'autopilot' ? (
            <AutopilotPanel
              autopilot={autopilot}
              live={live}
              onSave={props.onSaveAutopilot ?? (async () => false)}
              onRetry={props.onRetryAutopilot ?? (() => undefined)}
            />
          ) : controlsTab === 'watch' ? (
            <WatchPanel
              watch={watch}
              tasks={watchTasks}
              workspaces={state.workspaces}
              live={live}
              onSave={props.onSaveWatch ?? (async () => false)}
              onRetry={props.onRetryWatch ?? (() => undefined)}
            />
          ) : (
            <ModelChooser
              machines={state.machines}
              config={config}
              onOptions={props.onOptions}
              onSave={props.onSaveModel}
              onOpen={props.onChooserShown}
              onClose={closeControls}
              embedded
            />
          )}
        </SupervisorControls>
      )}
      {!showSlack && (note || inlineChooser || (!watch.unsupported && watch.rows.some(row => row.enabled))) ? (
        <div className="supervisor__panel">
          {note ? <p role="status" className="notice">{note}</p> : null}
          {watch.unsupported || supervisor.phase === 'unsupported' ? null : <WatchStrip rows={watch.rows} now={now} />}
          {inlineChooser ? (
            // No model yet: the chat cannot work without one, so the chooser stays in the page.
            <div ref={inlineChooserRef} tabIndex={-1}>
              <ModelChooser
                machines={state.machines}
                config={config}
                onOptions={props.onOptions}
                onSave={props.onSaveModel}
                onOpen={props.onChooserShown}
              />
            </div>
          ) : null}
        </div>
      ) : null}
      <div className="supervisor__stage">
        <div className="t3-topbar-fade" aria-hidden="true" />
        <div key={showSlack ? 'slack' : 'thread'} className="supervisor__stream" ref={showSlack ? undefined : scroll.ref}>
          <div className="supervisor__col">
            {showSlack ? (live
              ? <SupervisorSlack key={state.selfUserId} client={slack.client} openOAuth={slack.openOAuth} />
              : <p role="status" className="notice">{note ?? 'Reconnect to the Hub to open Slack.'}</p>
            ) : showThread ? (
              <AutopilotThreadView thread={state.autopilotThread} live={live} now={now} onRespond={props.onRespondAutopilot ?? (() => undefined)} />
            ) : supervisor.phase === 'unsupported' ? (
              <p role="status" className="notice">{SUPERVISOR_OLD_HUB}</p>
            ) : supervisor.phase === 'error' ? (
              <div role="alert" className="notice is-error">
                <p>Could not open XERXES: {supervisor.error ?? 'unknown error'}</p>
                <button type="button" className="btn" onClick={onReopen}>Try again</button>
              </div>
            ) : (
              <>
                {!ready ? (
                  <p className="xempty xempty--wait" role="status" aria-busy="true">
                    <AgentOrb size={36} state="connecting" live />
                    <span>{live ? 'Opening XERXES…' : 'Waiting for the Hub…'}</span>
                  </p>
                ) : null}
                {ready && supervisor.messages.length === 0 && !pending && !unconfigured ? (
                  <div className="xempty">
                    <span className="xempty__mark" aria-hidden="true"><XerxesLogoMark size={52} className="xempty__logo" decorative /></span>
                    <h2>Ask XERXES about your work</h2>
                    <p>It sees every task on your Hub. It can start, steer and stop them and answer their questions for you.</p>
                    <div className="xempty__try" role="group" aria-label="Try asking">
                      {SUGGESTIONS.map(suggestion => (
                        <button key={suggestion} type="button" className="xsuggest" disabled={!canType}
                          onClick={() => { setText(suggestion); inputRef.current?.focus(); requestAnimationFrame(grow) }}>
                          <span>{suggestion}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                ) : null}
                <ThreadMessages messages={supervisor.messages} {...(showPending && pending ? { pending } : {})} now={now} label="Conversation" records={records} {...(supervisor.steps ? { steps: supervisor.steps } : {})} />
                {openCards.length === 0 ? null : (
                  <ul className="supervisor__cards" aria-label="Actions XERXES proposes">
                    {openCards.map(item => (
                      <li key={item.id}><ActionConfirmationCard item={item} canAct={live} onRespond={decision => onRespond(item.id, decision)} /></li>
                    ))}
                  </ul>
                )}
                {thinking || runningSteps.length > 0 ? <LiveSteps steps={runningSteps} seconds={thinkingSeconds} /> : null}
                {supervisor.error !== undefined && ready && !unconfigured ? <p role="alert" className="xevent" data-tone="error"><Icon name="error" size={13} /><span className="xevent__text">{supervisor.error}</span></p> : null}
              </>
            )}
          </div>
        </div>
        {!showSlack && scroll.showJumpLatest ? (
          <div className="jump-latest-dock" data-scroll-ignore="true">
            <button
              type="button"
              className="jump-latest"
              data-working={thinking ? '' : undefined}
              aria-label="Jump to latest"
              title="Jump to latest"
              onClick={scroll.scrollToLatest}
            >
              <Icon name="arrowDown" size={14} />
              <span className="jump-latest__label">Jump to latest</span>
            </button>
          </div>
        ) : null}
      </div>
      {conversing && !showThread && !showSlack ? (
        <form className="supervisor__dock composer-wrap" aria-label="Message to XERXES" onSubmit={event => { event.preventDefault(); void submit() }}>
          <div className="composer-dock">
            <div className="composer">
              <label className="visually-hidden" htmlFor="supervisor-input">Message to XERXES</label>
              <textarea
                id="supervisor-input"
                ref={inputRef}
                className="composer__input"
                rows={1}
                value={text}
                placeholder={unconfigured ? 'Choose a model first' : thinking ? 'Queue a message, or steer XERXES…' : 'Ask XERXES…'}
                disabled={!canType}
                onChange={event => { setText(event.target.value); grow() }}
                onKeyDown={event => {
                  if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing) return
                  event.preventDefault()
                  // ⌘⏎ steers: stop the running turn and answer this instead.
                  void submit(event.metaKey || event.ctrlKey ? 'steer' : 'queue')
                }}
              />
              {thinking && !text.trim() && props.onCancel !== undefined
                ? <button type="button" className="composer__send composer__send--stop" aria-label="Stop XERXES" title="Stop this turn" onClick={() => { void props.onCancel?.() }}><Icon name="stop" size={18} /></button>
                : <button type="submit" className="composer__send" aria-label={thinking ? 'Queue message' : 'Send to XERXES'} title={thinking ? 'Queue after this turn (⏎) · ⌘⏎ steer' : 'Send (⏎ · ⇧⏎ newline)'} disabled={!canSend}><Icon name="arrowUp" size={16} /></button>}
            </div>
          </div>
          <div className="composer__toolbar">
            {canSteer ? <button type="button" className="cchip composer__text supervisor__steer" title="Stop the running turn and answer this message now (⌘⏎)" onClick={() => { void submit('steer') }}>Steer now</button> : null}
            <span className="supervisor__hint" title={autopilotChip === undefined ? DEFAULT_COMPOSER_HINT : AUTOPILOT_COMPOSER_HINT}>
              {autopilotChip === undefined ? 'Asks first when others are in the thread' : 'Autopilot on · acts at once'}
            </span>
            <span className="composer__flex" />
            <button
              type="button"
              className={`cchip composer__text supervisor__model-btn${unconfigured ? ' is-custom' : ''}`}
              title={inlineChooser ? 'Choose a model for XERXES in the setup panel above' : "Change XERXES's model"}
              onClick={() => openTab('model')}
            >
              <span>{modelChipText(config)}</span>
              <Icon name="caretDown" size={10} />
            </button>
          </div>
        </form>
      ) : null}
    </section>
  )
}

interface Session {
  readonly store: HubStore
  readonly controller: HubController
  readonly slack: SupervisorSlackClient
  disposeSockets(): void
}

// Optional trusted origin baked into the Desktop build by the release operator.
// An absent setting deliberately does not grant trust to a Hub-supplied origin.
declare const __XERXES_SLACK_BROKER_ORIGIN__: string | undefined

function createSession(api: CockpitApi, wsUrl: string): Session {
  const hubStore = createHubStore(undefined, { coalesceMs: 50 })
  const sockets = createCockpitSockets(api)
  // The socket lives in the main process: it opens the saved Hub with its own
  // Origin and cookies, whatever URL is named here.
  const connection = new HubConnection({ url: wsUrl, socketFactory: sockets.factory, ensureSession: cockpitEnsureSession(api) })
  const brokerOrigin = typeof __XERXES_SLACK_BROKER_ORIGIN__ === 'string' ? __XERXES_SLACK_BROKER_ORIGIN__ : undefined
  const slack = createSupervisorSlackClient({
    ...(brokerOrigin === undefined ? {} : { trustedBrokerOrigin: brokerOrigin }),
    allowed: () => connection.status().state === 'live' && isHubOwner(hubStore.getSnapshot()),
    request: (method, params) => connection.request(method, params),
  })
  return { store: hubStore, controller: createHubController({ connection, store: hubStore }), slack, disposeSockets: sockets.dispose }
}

async function openSlackOAuth(url: string): Promise<boolean> {
  return (await window.xerxes?.openExternal?.(url)) === true
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
  const [autopilotOpen, setAutopilotOpen] = useState(false)
  const [autopilotThreadOpen, setAutopilotThreadOpen] = useState(false)
  const [watchOpen, setWatchOpen] = useState(false)
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
      controlClient={session.controller.superagentControl}
      slack={{ client: session.slack, openOAuth: openSlackOAuth }}
      hubLabel={hubLabelFor(origin)}
      now={now}
      onSend={text => session.controller.supervisorSend(text, 'text')}
      onCancel={() => session.controller.supervisorCancel()}
      onRespond={(id, decision) => { void session.controller.supervisorRespond(id, decision) }}
      onReopen={() => { void session.controller.supervisorOpen() }}
      onOpenSettings={onOpenSettings}
      chooserOpen={chooserOpen}
      onChooserChange={setChooserOpen}
      onChooserShown={() => { void session.controller.machinesList(); void session.controller.supervisorConfigGet() }}
      onOptions={input => { void session.controller.supervisorConfigOptions(input) }}
      onSaveModel={input => session.controller.supervisorConfigSet(input)}
      autopilotOpen={autopilotOpen}
      onAutopilotChange={open => { setAutopilotOpen(open); if (open) void session.controller.loadAutopilot() }}
      onSaveAutopilot={input => session.controller.setAutopilot(input)}
      onRetryAutopilot={() => { void session.controller.loadAutopilot() }}
      autopilotThreadOpen={autopilotThreadOpen}
      onAutopilotThreadChange={open => { setAutopilotThreadOpen(open); if (open) void session.controller.autopilotThreadOpen() }}
      onRespondAutopilot={(id, decision) => { void session.controller.autopilotThreadRespond(id, decision) }}
      watchOpen={watchOpen}
      onWatchChange={open => { setWatchOpen(open); if (open) void session.controller.loadWatches() }}
      onSaveWatch={input => session.controller.setWatch(input)}
      onRetryWatch={() => { void session.controller.loadWatches() }}
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
      return (
        <section className="supervisor supervisor--connect">
          <p className="xempty xempty--wait" role="status" aria-busy="true">
            <AgentOrb size={36} state="connecting" live />
            <span>{gate.phase === 'reading' ? 'Reading the saved Hub…' : 'Signing in to the Hub…'}</span>
          </p>
        </section>
      )
    case 'no_hub':
      return <ConnectHub onOpenSettings={openSettings} />
    case 'failed':
      return <SignInFailed message={gate.message} denied={gate.denied} onRetry={() => { void load(false) }} onSignInAgain={() => { void load(true) }} onOpenSettings={openSettings} />
    case 'ready':
      return <Connected api={api} origin={gate.origin} wsUrl={gate.wsUrl} onOpenSettings={openSettings} />
  }
}
