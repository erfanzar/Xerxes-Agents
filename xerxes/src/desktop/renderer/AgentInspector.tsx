// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactElement } from 'react'
import { pageIsBackground } from './pageVisibility.js'
import { agentState } from './AgentRoster.js'
import { desktopCall, desktopError, record, text, type RpcRecord } from './desktopRpc.js'
import { OutputViewer, readableOutput } from './OutputViewer.js'
import type { Block, SessionRow } from './types.js'
import { Markdown } from './markdown.js'
import { agentResult, resultLabel } from './agentResult.js'
import { ActivityGroup, BlockView } from './App.js'
import { isDisclosedActivity, keyedActivityGroups } from './activityGroups.js'
import { useTranscriptScroll } from './transcriptScroll.js'
import { Icon } from './Icon.js'
import { store } from './store.js'
import {
  AGENT_HISTORY_PAGE, agentHistoryBlocks, agentLiveEventOf, agentTranscriptBlocks, appendLiveEvent, liveAgentBlocks,
  prependAgentHistory, readAgentHistory, refreshAgentHistory, retainLiveEvents, settleSavedTools, toolIdsOf,
  type AgentHistoryState, type AgentLiveEvent,
} from './agentTranscript.js'

/**
 * An agent's own prose for display: markup-like tags it wrote to itself
 * alone on a line (`<diagnostics>`, `</thinking>`) removed — never a `<T>` in prose —, and runs of blank lines
 * collapsed — streamed summaries arrive padded with them.
 */
export function agentProse(text: string): string {
  return text.replace(/^[ \t]*<\/?[a-z_][\w-]*>[ \t]*$/gim, '').replace(/\n{3,}/g, '\n\n').trim()
}

/** How often a working agent's saved conversation is re-read; live events fill the gaps. */
const LIVE_POLL_MS = 2000

/** What the runtime said about the agent's own saved conversation. */
type HistoryView =
  | { readonly kind: 'unknown' }
  | { readonly kind: 'unsupported' }
  | { readonly kind: 'unavailable'; readonly note: string }
  | { readonly kind: 'page'; readonly state: AgentHistoryState }

/** A live event with its arrival order, so a fresh saved page can retire what it already covers. */
interface ReceivedEvent extends AgentLiveEvent { readonly seq: number }

// ── The last `subagent.inspect` reply per open agent ─────────────────────
// The pane polls; the rail's details card reads the same reply instead of
// polling a second time.
const inspected = new Map<string, RpcRecord>()
const inspectedListeners = new Set<() => void>()
function publishAgentDetail(key: string, detail: RpcRecord): void {
  inspected.delete(key)
  inspected.set(key, detail)
  // Bounded: the agents opened most recently in this window.
  while (inspected.size > 32) {
    const oldest = inspected.keys().next().value
    if (oldest === undefined) break
    inspected.delete(oldest)
  }
  for (const listener of inspectedListeners) listener()
}
const subscribeInspected = (listener: () => void): (() => void) => { inspectedListeners.add(listener); return () => { inspectedListeners.delete(listener) } }
function useAgentDetail(key: string): RpcRecord | null {
  return useSyncExternalStore(subscribeInspected, () => inspected.get(key) ?? null, () => null)
}

/** What is known about an agent, from its fleet row and (once read) its inspect reply. */
function agentFacts(row: SessionRow, detail: RpcRecord | null) {
  const info = row.agentDetails
  const rawOutput = text(detail?.output)
  // A schema answer (or, before its output loads, the summary excerpt of one)
  // reads as labelled fields rather than a line of JSON.
  const fromOutput = agentResult(rawOutput)
  return {
    base: info?.baseAgent || text(detail?.agent_id),
    parent: info?.parentId || text(detail?.parent_id) || text(detail?.creator_id),
    prompt: info?.goal || text(detail?.prompt),
    model: info?.model || text(detail?.model),
    profile: info?.providerProfile || text(detail?.provider_profile),
    effort: info?.reasoningEffort || text(detail?.reasoning_effort),
    result: fromOutput ?? (info?.summary ? agentResult(info.summary) : null),
    output: fromOutput?.complete ? JSON.stringify(fromOutput.fields, null, 2) : readableOutput(rawOutput),
    outputTrimmed: detail?.output_truncated === true || fromOutput?.complete === false,
  }
}

/**
 * The conversation an older runtime can still give: it keeps no per-step
 * history, only the task, recent notes, tool calls and a summary. Drawn as
 * the same chat the main conversation is — the task as the person's
 * message, notes as replies, calls folded into one work group.
 */
export function retainedAgentBlocks(info: SessionRow['agentDetails'], prompt: string, reply: string): Block[] {
  const blocks: Block[] = []
  if (prompt) blocks.push({ kind: 'user', id: -1, text: prompt })
  const thinking = (info?.thinking ?? []).map(agentProse).filter(Boolean)
  if (thinking.length) blocks.push({ kind: 'thinking', id: -2, text: thinking.join('\n\n'), streaming: false })
  ;(info?.notes ?? []).map(agentProse).filter(Boolean).forEach((note, index) => blocks.push({ kind: 'agent', id: -10 - index, text: note, streaming: false }))
  const calls = info?.toolCalls ?? []
  if (calls.length) blocks.push({ kind: 'tools', id: -3, items: calls, running: calls.some(call => call.state === 'working') })
  if (reply) blocks.push({ kind: 'agent', id: -4, text: reply, streaming: false })
  return blocks
}

/**
 * A sub-agent opened in place of the task's conversation (DeepSeek Harness's
 * layout). The pane is only its chat — the task as the opening message, its
 * replies and work groups — and the composer's place holds a dock that
 * messages, stops or retries it. Who it is and what it produced live in the
 * rail ({@link AgentDetailsCard}).
 */
export function AgentInspector({ row, sessionKey, online }: {row: SessionRow; rows?: readonly SessionRow[]; sessionKey: string; online: boolean}): ReactElement {
  const info = row.agentDetails
  const [detail, setDetail] = useState<RpcRecord | null>(null)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  const [history, setHistory] = useState<HistoryView>({ kind: 'unknown' })
  const [live, setLive] = useState<readonly ReceivedEvent[]>([])
  const [older, setOlder] = useState({ loading: false, error: '' })
  const received = useRef(0)
  const blockIds = useRef(new Map<string, number>())
  const state = agentState(row.status)
  const active = state.priority < 2
  const detailKey = `${sessionKey}:${row.id}`
  useEffect(() => {
    if (!online || info?.provisional) return
    let current = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const load = async () => {
      try {
        if (pageIsBackground()) { if (current && active) timer = setTimeout(() => void load(), 3000); return }
        const sent = received.current
        const result = await desktopCall(window.xerxes, sessionKey, 'subagent.inspect', { task: row.id, history_limit: AGENT_HISTORY_PAGE })
        const agent = record(result.agent)
        if (text(agent.id) !== row.id) throw new Error('The daemon returned a different agent. Refresh to try again.')
        const read = readAgentHistory(result)
        if (current) {
          setDetail(agent); setError('')
          publishAgentDetail(detailKey, agent)
          // A conversation already on screen stays when one read misses it.
          setHistory(previous => read.kind === 'page'
            ? { kind: 'page', state: refreshAgentHistory(previous.kind === 'page' ? previous.state : null, read.page, read.clipped) }
            : previous.kind === 'page' ? previous : read)
          // Events that arrived before this read are part of the saved page now.
          if (read.kind === 'page') setLive(events => retainLiveEvents(events, sent, read.page))
        }
      } catch (failure) {if (current) setError(desktopError(failure))}
      if (current && active) timer = setTimeout(() => void load(), LIVE_POLL_MS)
    }
    void load()
    return () => {current=false;if(timer)clearTimeout(timer)}
    // `active` follows `state.priority`; any status class change reads once more.
  }, [row.id, sessionKey, online, info?.provisional, state.priority, retry])
  // The child's own stream: the same events the main conversation folds,
  // wrapped in `subagent_event` envelopes for this agent.
  useEffect(() => {
    if (info?.provisional) return
    return window.xerxes.onEvent(event => {
      if (event.type !== 'subagent_event') return
      const next = agentLiveEventOf(event.payload, row.id)
      if (!next) return
      received.current += 1
      const item: ReceivedEvent = { ...next, seq: received.current }
      setLive(events => appendLiveEvent(events, item))
    })
  }, [row.id, info?.provisional])
  const page = history.kind === 'page' ? history.state : null
  const loadOlder = async (): Promise<void> => {
    if (!page?.more || !page.before || older.loading) return
    const before = page.before
    setOlder({ loading: true, error: '' })
    try {
      const result = await desktopCall(window.xerxes, sessionKey, 'subagent.inspect', { task: row.id, history_limit: AGENT_HISTORY_PAGE, before })
      const read = readAgentHistory(result)
      if (read.kind !== 'page') throw new Error('Earlier activity is not available from this runtime.')
      setHistory(previous => previous.kind === 'page' && previous.state.before === before ? { kind: 'page', state: prependAgentHistory(previous.state, read.page, read.clipped) } : previous)
      setOlder({ loading: false, error: '' })
    } catch (failure) { setOlder({ loading: false, error: desktopError(failure) }) }
  }
  const scroll = useTranscriptScroll(`agent:${row.id}`, { more: Boolean(page?.more), loading: older.loading, automatic: !older.error, load: loadOlder })
  const facts = agentFacts(row, detail)
  // Without a saved page the reply is the agent's summary, or its output.
  const reply = info?.provisional || facts.result?.complete ? '' : agentProse(info?.summary || '') || facts.output
  // Until the first read answers, the summary is not yet known to be all there is:
  // show only the task, so a summary never flashes in as if it were a reply.
  const loading = history.kind === 'unknown' && online && !info?.provisional && !error
  const base = useMemo((): Block[] => {
    if (loading) return facts.prompt ? [{ kind: 'user', id: -1, text: facts.prompt }] : []
    if (page) {
      const saved = agentHistoryBlocks(page.actions, blockIds.current, active)
      // The task opens the chat; a page that starts later already shows a pager.
      if (page.more || !facts.prompt || saved.some(block => block.kind === 'user')) return saved
      const opening: Block = { kind: 'user', id: -1, text: facts.prompt }
      return [opening, ...saved]
    }
    return retainedAgentBlocks(info, facts.prompt, reply)
  }, [loading, page, active, info, facts.prompt, reply])
  const blocks = useMemo(() => agentTranscriptBlocks(settleSavedTools(base, live), liveAgentBlocks(live, toolIdsOf(base), active), active), [base, live, active])

  return <section className="agent-inspector agent-pane" aria-label="Agent inspector" data-state={state.tone}>
    <div className="stream agent-transcript" ref={scroll.ref} tabIndex={0} role="region" aria-label="Agent conversation">
      <div className="stream__col">
        {!online && <p role="status" className="agent-transcript__note">Disconnected. Showing what was loaded so far; reconnect for updates and controls.</p>}
        {history.kind === 'unsupported' && detail && <p className="agent-transcript__note">This runtime keeps only a summary of this agent’s work. Update the runtime to follow every step.</p>}
        {history.kind === 'unavailable' && <p className="agent-transcript__note">{history.note}</p>}
        {info?.provisional && <p role="status" className="agent-transcript__note">{state.tone === 'failed' && info.error ? 'The spawn request failed before an agent identity was confirmed. No running agent is reported for this request.' : state.priority < 2 ? 'Waiting for the runtime to identify this spawn request.' : 'This request has finished. Its runtime identity was not recorded.'}</p>}
        {page && (page.more || older.error) && <div className="history-pager">
          <button className="btn" disabled={older.loading} onClick={() => void scroll.loadOlder()}>{older.loading ? 'Loading earlier activity…' : older.error ? 'Retry loading earlier activity' : 'Show earlier activity'}</button>
          {older.error && <p role="alert">{older.error}</p>}
        </div>}
        <AgentTranscriptFeed blocks={blocks} active={active} />
        {!page && !loading && facts.result && <section className="msg msg--agent"><AgentResultFields result={facts.result} /></section>}
        {loading && <p className="agent-transcript__note">Loading the agent’s conversation…</p>}
        {page?.clipped && <p className="agent-transcript__note">Some long text in this conversation is shortened.</p>}
        {info?.error && <p className="studio-error">{info.error}</p>}
        {error && <div role="alert" className="studio-error"><p>{error}</p><button disabled={!online} onClick={() => setRetry(value => value+1)}>Retry agent details</button></div>}
      </div>
      {!scroll.following && blocks.length > 0 && <button className="jump-latest" onClick={scroll.scrollToLatest}>
        {active ? 'Still working — jump to latest' : 'Jump to latest'} <Icon name="chevron" size={12} />
      </button>}
    </div>
    {info?.provisional
      ? <div className="agent-dock agent-dock--readonly composer-wrap" role="note"><div className="composer-dock"><p>
          <strong>{state.tone === 'failed' ? 'This request did not start an agent.' : state.priority < 2 ? 'Not controllable yet.' : 'Read-only.'}</strong>{' '}
          {state.tone !== 'failed' && state.priority < 2 ? 'Message and stop appear once the runtime confirms this agent.' : 'No runtime identity was recorded, so it cannot be messaged or retried.'}
        </p></div></div>
      : <AgentDock key={row.id} id={row.id} title={row.title} active={active} online={online} />}
  </section>
}

/**
 * The open agent's card at the top of the Activity rail: who it is, what it
 * was asked, what it produced and what it touched — everything that is not
 * its conversation.
 */
export function AgentDetailsCard({ row, rows, sessionKey }: { row: SessionRow; rows: readonly SessionRow[]; sessionKey: string }): ReactElement {
  const info = row.agentDetails
  const detail = useAgentDetail(`${sessionKey}:${row.id}`)
  const state = agentState(row.status)
  const facts = agentFacts(row, detail)
  // Rows from older snapshots may omit either list.
  const files = info ? ([['Files read', info.filesRead], ['Files changed', info.filesWritten]] as const).flatMap(([label, paths]) => Array.isArray(paths) && paths.length > 0 ? [[label, paths] as const] : []) : []
  return <section className="railcard agent-card" aria-label="Agent details" data-state={state.tone}>
    <header className="railcard__head"><span className="railcard__title" title={row.title}>{row.title}</span><span className="railcard__meta" data-tone={state.tone}>{state.label}</span></header>
    <dl className="agent-card__facts">
      <dt>Base agent</dt><dd>{facts.base || 'Not reported'}</dd>
      <dt>{info?.provisional ? 'Requested model' : 'Model'}</dt><dd>{facts.model || 'Not reported'}</dd>
      {facts.profile && <><dt>{info?.provisional ? 'Requested provider profile' : 'Provider profile'}</dt><dd>{facts.profile}</dd></>}
      {facts.effort && <><dt>Reasoning effort</dt><dd>{facts.effort}</dd></>}
      <dt>Created by</dt><dd>{facts.parent ? rows.find(candidate => candidate.id === facts.parent)?.title || facts.parent : 'Parent task'}</dd>
    </dl>
    {facts.prompt && <details className="agent-card__fold"><summary>Assigned task</summary><p className="agent-card__prompt">{facts.prompt}</p></details>}
    {facts.result?.complete && <details className="agent-card__fold" open><summary>Result</summary><AgentResultFields result={facts.result} /></details>}
    {facts.output && !facts.result?.complete && <details className="agent-card__fold"><summary>Output</summary><OutputViewer text={facts.output} label="Agent output" />{facts.outputTrimmed && <p className="agent-card__note">Only the beginning of this output was saved.</p>}</details>}
    {files.map(([label, paths]) => <details key={label} className="agent-card__fold"><summary>{label} <span>{paths.length}</span></summary><ul>{paths.map(path => <li key={path}><code>{path}</code></li>)}</ul></details>)}
    <details className="agent-card__fold"><summary>Runtime details</summary><dl className="agent-card__facts">
      <dt>Agent ID</dt><dd>{info?.provisional ? state.priority < 2 ? 'Not assigned yet' : 'Not recorded' : row.id}</dd>
      {info?.toolCount !== undefined && <><dt>Tools</dt><dd>{info.toolCount}</dd></>}
      {info?.inputTokens !== undefined && <><dt>Input tokens</dt><dd>{info.inputTokens.toLocaleString()}</dd></>}
      {info?.outputTokens !== undefined && <><dt>Output tokens</dt><dd>{info.outputTokens.toLocaleString()}</dd></>}
    </dl></details>
  </section>
}

/**
 * The composer's stand-in while an agent is open. While it runs, text steers
 * it at its next step and Stop ends only this agent; once it settled, the
 * text becomes the opening instruction of a retry.
 */
function AgentDock({ id, title, active, online }: { id: string; title: string; active: boolean; online: boolean }): ReactElement {
  const [message, setMessage] = useState('')
  const [busy, setBusy] = useState<'' | 'send' | 'stop' | 'retry'>('')
  const [feedback, setFeedback] = useState<{ readonly text: string; readonly failed: boolean } | null>(null)
  const input = useRef<HTMLTextAreaElement>(null)
  const grow = (): void => {
    const node = input.current
    if (!node) return
    node.style.height = 'auto'
    node.style.height = `${Math.min(node.scrollHeight, 200)}px`
  }
  useEffect(grow, [message])
  const run = async (action: 'send' | 'stop' | 'retry'): Promise<void> => {
    if (busy) return
    setBusy(action); setFeedback(null)
    try {
      const reply = action === 'send' ? await store.steerAgent(id, message) : await store.controlAgent(id, action, message)
      setFeedback({ text: reply, failed: false })
      if (action !== 'stop') setMessage('')
    } catch (error) { setFeedback({ text: desktopError(error), failed: true }) }
    finally { setBusy('') }
  }
  return <fieldset className="agent-dock composer-wrap" disabled={!online}>
    <legend className="visually-hidden">Control {title}</legend>
    <div className="composer-dock"><div className="composer">
      <textarea
        ref={input}
        className="composer__input"
        rows={1}
        aria-label={active ? 'Message this agent' : 'Follow-up for retry'}
        placeholder={active ? 'Message this agent — it reads it at its next step' : 'Optional instruction for a new attempt'}
        value={message}
        spellCheck={false}
        onChange={event => setMessage(event.target.value)}
        onKeyDown={event => {
          // Enter sends while it runs; a settled agent is only retried by the button.
          if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); if (active && message.trim()) void run('send') }
          // The window's Escape arms stopping the task's turn; here it only leaves the box.
          else if (event.key === 'Escape') { event.stopPropagation(); event.currentTarget.blur() }
        }}
      />
      <div className="agent-dock__actions">
        {active ? <>
          <button className="composer__send composer__send--stop" aria-label="Stop agent" title="Stop this agent — the task and its other agents keep running" disabled={busy !== ''} onClick={() => void run('stop')}><Icon name="stop" size={16} /></button>
          <button className="composer__send" aria-label="Send to agent" title="Send (⏎ · ⇧⏎ newline)" disabled={busy !== '' || !message.trim()} onClick={() => void run('send')}><Icon name="arrowUp" size={16} /></button>
        </> : <button className="composer__send agent-dock__retry" aria-label="Retry agent" title="Start a new attempt of this agent" disabled={busy !== ''} onClick={() => void run('retry')}><Icon name="retry" size={14} /><span>{busy === 'retry' ? 'Retrying…' : 'Retry'}</span></button>}
      </div>
    </div></div>
    <div className="agent-dock__foot">{feedback
      ? <p role="status" data-error={feedback.failed || undefined}>{feedback.text}</p>
      : <span>{active ? 'Stopping ends only this agent. The task and other agents keep running.' : 'This agent has finished. Retry starts a new attempt with your instruction.'}</span>}</div>
  </fieldset>
}

/**
 * A sub-agent's blocks grouped and drawn by the main conversation's own
 * components: prose as replies, reasoning and tool calls folded into
 * "Ran N commands" groups whose rows open to the command, output and diff.
 */
export function AgentTranscriptFeed({ blocks, active }: { blocks: readonly Block[]; active: boolean }): ReactElement {
  if (!blocks.length) return <p className="agent-transcript__note">{active ? 'Waiting for the agent’s first step…' : 'This agent recorded no activity.'}</p>
  return <>{keyedActivityGroups(blocks).map(({ key, blocks: group }, index, groups) => (
    <div key={key} data-history-anchor={group[0]!.id}>
      {isDisclosedActivity(group) ? <ActivityGroup blocks={group} active={active && index === groups.length - 1} turnActive={active} /> : group.map(block => <BlockView key={block.id} block={block} />)}
    </div>
  ))}</>
}

/** A structured result, one labelled section per field. */
function AgentResultFields({ result }: { result: NonNullable<ReturnType<typeof agentResult>> }): ReactElement {
  return <section className="agent-inspector__result">
    <dl>{Object.entries(result.fields).map(([key, value]) => <div key={key}>
      <dt>{resultLabel(key)}</dt>
      <dd>{typeof value === 'string' ? <Markdown text={agentProse(value)} className="md--compact" />
        : Array.isArray(value) && value.every(item => typeof item === 'string' || typeof item === 'number') ? <ul>{value.map((item, index) => <li key={index}>{String(item)}</li>)}</ul>
        : value !== null && typeof value === 'object' ? <pre>{JSON.stringify(value, null, 2)}</pre>
        : <p>{String(value)}</p>}</dd>
    </div>)}</dl>
    {!result.complete && <p className="studio-muted">The saved result was cut short; fields after these were not kept.</p>}
  </section>
}
