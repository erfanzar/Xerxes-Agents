// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'
import type { Block, SessionRow } from './types.js'
import { Icon } from './Icon.js'
import { ToolCallRow } from './Execution.js'
import { store } from './store.js'
import { desktopError } from './desktopRpc.js'
import { toolFailureText } from './blocks.js'

/**
 * What an agent runs on, as `profile/model[effort]` — e.g.
 * `openai/gpt-5[xhigh]`. Only reported facts: a child that sets no effort
 * runs at the provider's default, so the bracket is omitted rather than
 * guessed. Empty when the runtime has not reported a model yet.
 */
export function agentKindLabel(details: { readonly model?: string; readonly providerProfile?: string; readonly reasoningEffort?: string } | undefined): string {
  const model = details?.model?.trim() ?? ''
  if (!model) return ''
  const profile = details?.providerProfile?.trim() ?? ''
  const route = profile && !model.startsWith(`${profile}/`) ? `${profile}/${model}` : model
  const effort = details?.reasoningEffort?.trim() ?? ''
  return effort ? `${route}[${effort}]` : route
}

export function agentState(status: string): { label: string; priority: number; tone: string } {
  if (['failed', 'error', 'timeout'].includes(status)) return { label: 'Failed', priority: 2, tone: 'failed' }
  if (['waiting', 'blocked', 'needs_input'].includes(status)) return { label: 'Needs attention', priority: 1, tone: 'waiting' }
  if (['running', 'working', 'acting', 'queued', 'pending', 'starting'].includes(status)) return { label: ['queued', 'pending'].includes(status) ? 'Queued' : status === 'starting' ? 'Awaiting runtime status' : 'Working', priority: 0, tone: 'working' }
  if (['done', 'completed', 'succeeded'].includes(status)) return { label: 'Completed', priority: 3, tone: 'done' }
  if (['cancelled', 'canceled', 'interrupted'].includes(status)) return { label: 'Stopped', priority: 2, tone: 'stopped' }
  return { label: status || 'Unknown', priority: 2, tone: 'idle' }
}

/** Keep requests visible while a daemon snapshot is pending; never invent a controllable agent id. */
export function activityFleetRows(rows: readonly SessionRow[], blocks: readonly Block[]): readonly SessionRow[] {
  const result = [...rows]
  const failures = new Map<string, string>()
  for (const block of blocks) {
    if (block.kind === 'user') failures.clear()
    // Pass `output` explicitly: the default parameter re-runs detailOf,
    // which JSON.parses and re-stringifies — and tool.output is ALREADY a
    // detailOf result (blocks.ts:242), so it was an idempotent re-parse of
    // every tool output in the transcript on every store notification.
    if (block.kind === 'tools') for (const tool of block.items) failures.set(tool.id, toolFailureText({error:tool.error,result:tool.output}, tool.output))
    if (block.kind !== 'agents') continue
    for (const original of block.members) {
    const error = !original.runtimeId ? failures.get(original.key.slice(0, original.key.lastIndexOf(':'))) : ''
    const member = error && original.status === 'working' ? {...original,status:'failed',error} : original
    const matched = result.findIndex(row => member.runtimeId ? row.id === member.runtimeId : row.id === member.key || row.agentDetails?.requestKey === member.key || row.title === member.title)
    if (matched >= 0) {
      const row = result[matched]!
      result[matched] = {...row, agentDetails: {summary:'',error:'',model:'',filesRead:[],filesWritten:[],...row.agentDetails,requestKey:member.key,baseAgent:row.agentDetails?.baseAgent || member.baseAgent || "",goal:row.agentDetails?.goal || member.prompt || ""}}
      continue
    }
    result.push({id:member.key,key:member.key,title:member.title,status:member.status === 'working' ? 'starting' : member.status,age:'',current:false,kind:'subagent',turns:0,messages:0,cwd:'',untitled:false,
      agentDetails:{provisional:true,baseAgent:member.baseAgent || "",goal:member.prompt || "",summary:agentState(member.status).priority < 2 ? 'The spawn request is visible in the conversation. Waiting for the runtime to report this agent’s identity and state.' : 'This request has finished. Its runtime details were not recorded; the original request is retained.',error:member.error || '',model:member.model || '',providerProfile:member.providerProfile || '',filesRead:[],filesWritten:[]}})
    }
  }
  return result
}

export function AgentRoster({ rows, onInspect }: { rows: readonly SessionRow[]; onInspect?: (id: string) => void }): ReactElement {
  const ordered = [...rows].sort((a, b) => agentState(a.status).priority - agentState(b.status).priority)
  const current = ordered.filter(row => agentState(row.status).priority < 2)
  const completed = ordered.filter(row => agentState(row.status).priority >= 2)
  const failures = completed.filter(row => agentState(row.status).tone === 'failed').length
  const renderRows = (members: readonly SessionRow[]) => members.map(row => {
    const state = agentState(row.status)
    const info = row.agentDetails
    const latest = info?.toolCalls?.findLast(call => call.state === 'working')
    const preview = info?.error || (state.tone === 'working' ? info?.notes?.at(-1) || latest?.arg || info?.thinking?.at(-1) : '') || info?.summary
    return <details className="agent-record" key={row.id} data-state={state.tone}>
      <summary onClick={event => { if (onInspect) { event.preventDefault(); onInspect(row.id) } }}>
        <Icon name="chevron" size={13} />
        <span className="agent-record__title">{row.title}</span>
        <span className="agent-record__state">{state.label}</span>
        {preview && <span className="agent-record__preview">{preview}</span>}
        {(info?.model || info?.toolCount !== undefined) && <span className="agent-record__meta">{info.model && <span>{info.model}</span>}{info.toolCount !== undefined && info.toolCount > 0 && <span>{info.toolCount} tools</span>}</span>}
      </summary>
      <div className="agent-record__details">
        {info?.goal && info.goal !== row.title && <p>{info.goal}</p>}
        {info?.parentId && <p>Created by {rows.find(parent=>parent.id===info.parentId)?.title || info.parentId}</p>}
        {preview ? <p>{preview}</p> : <p>{state.tone === 'working' ? 'The agent is working. No summary has been reported yet.' : 'No summary was reported for this agent.'}</p>}
        {info?.error && info.summary && <p>{info.summary}</p>}
        {info && (info.inputTokens !== undefined || info.outputTokens !== undefined) && <dl>{info.inputTokens !== undefined && <><dt>Input tokens</dt><dd>{info.inputTokens.toLocaleString()}</dd></>}{info.outputTokens !== undefined && <><dt>Output tokens</dt><dd>{info.outputTokens.toLocaleString()}</dd></>}</dl>}
        {info && [['Files read', info.filesRead], ['Files changed', info.filesWritten]].map(([label, paths]) => Array.isArray(paths) && paths.length > 0 ? <div key={String(label)}><strong>{label}</strong><ul>{paths.map(path => <li key={path}><code>{path}</code></li>)}</ul></div> : null)}
        {info?.toolCalls && info.toolCalls.length > 0 && <section aria-label={`${row.title} tool calls`}><strong>Tool calls</strong>{info.toolCalls.map(item=><ToolCallRow key={item.id} item={item} label={item.name.replaceAll('_',' ')} />)}</section>}
        {info?.thinking && info.thinking.length > 0 && <details><summary>Reasoning</summary>{info.thinking.map((line,index)=><p key={index}>{line}</p>)}</details>}
        {info?.notes && info.notes.length > 0 && <details><summary>Progress</summary>{info.notes.map((line,index)=><p key={index}>{line}</p>)}</details>}
        {!info?.provisional && <AgentControls id={row.id} active={state.tone === 'working'} />}
      </div>
    </details>
  })
  return <div className="agent-roster">
    {current.length > 0 && <div className="agent-roster__current">{renderRows(current)}</div>}
    {completed.length > 0 && <details className="agent-roster__history">
      <summary><Icon name="chevron" size={13}/><span>Past agents</span><span>{completed.length}{failures ? ` · ${failures} failed` : ''}</span></summary>
      {renderRows(completed)}
    </details>}
  </div>
}

export function AgentControls({ id, active }: { id: string; active: boolean }): ReactElement {
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [feedback, setFeedback] = useState('')
  const [failed, setFailed] = useState(false)
  /**
   * One text box, two meanings, because the agent's state decides what a
   * message can possibly do: while it runs, text steers it mid-flight; once
   * it is dead, the only way to say anything is to start a new attempt with
   * the text as the opening instruction. Splitting these into two controls
   * would show a disabled one most of the time.
   */
  const run = async (action: 'send' | 'control'): Promise<void> => {
    if (busy) return
    setBusy(true); setFeedback(''); setFailed(false)
    try {
      setFeedback(action === 'send'
        ? await store.steerAgent(id, message)
        : await store.controlAgent(id, active ? 'stop' : 'retry', message))
      if (action === 'send') setMessage('')
    }
    catch (error) { setFailed(true); setFeedback(desktopError(error)) }
    finally { setBusy(false) }
  }
  const send = (): void => { void run('send') }
  return <div className="agent-controls">
    <label>
      {active ? 'Message this agent' : 'Follow-up for retry'}
      <input
        value={message}
        onChange={event => setMessage(event.target.value)}
        placeholder={active ? 'Redirect it — read at its next step' : 'Optional instruction'}
        // Enter sends while it runs; with no running agent there is nothing
        // to send to, so the key does nothing rather than silently retrying.
        onKeyDown={event => { if (event.key === 'Enter' && active && message.trim()) { event.preventDefault(); send() } }}
      />
    </label>
    <div className="agent-controls__row">
      {active && <button disabled={busy || !message.trim()} onClick={send}>{busy ? 'Sending…' : 'Send'}</button>}
      <button disabled={busy} onClick={() => void run('control')}>{busy ? 'Requesting…' : active ? 'Stop agent' : 'Retry agent'}</button>
    </div>
    {feedback && <p role="status" data-error={failed || undefined}>{feedback}</p>}
  </div>
}
