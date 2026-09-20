// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useEffect, useState, type ReactElement } from 'react'
import { AgentControls, agentState } from './AgentRoster.js'
import { desktopCall, desktopError, record, text, type RpcRecord } from './desktopRpc.js'
import { OutputViewer, readableOutput } from './OutputViewer.js'
import { ToolCallRow } from './Execution.js'
import type { SessionRow } from './types.js'

export function AgentInspector({ row, rows, sessionKey, online }: {row: SessionRow; rows: readonly SessionRow[]; sessionKey: string; online: boolean}): ReactElement {
  const info = row.agentDetails
  const [detail, setDetail] = useState<RpcRecord | null>(null)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  const state = agentState(row.status)
  useEffect(() => {
    if (!online || info?.provisional) return
    let current = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const load = async () => {
      try {
        const result = await desktopCall(window.xerxes, sessionKey, 'subagent.inspect', {task:row.id})
        const agent = record(result.agent)
        if (text(agent.id) !== row.id) throw new Error('The daemon returned a different agent. Refresh to try again.')
        if (current) {setDetail(agent);setError('')}
      } catch (failure) {if (current) setError(desktopError(failure))}
      if (current && state.priority < 2) timer = setTimeout(() => void load(), 3000)
    }
    void load()
    return () => {current=false;if(timer)clearTimeout(timer)}
  }, [row.id, sessionKey, online, info?.provisional, state.priority, retry])
  const base = info?.baseAgent || text(detail?.agent_id)
  const parent = info?.parentId || text(detail?.parent_id) || text(detail?.creator_id)
  const prompt = info?.goal || text(detail?.prompt)
  const model = info?.model || text(detail?.model)
  const output = readableOutput(text(detail?.output))
  return <article aria-label="Agent inspector" data-state={state.tone}>
    <header className="agent-inspector__heading"><h2>{row.title}</h2><span>{state.label}</span></header>
    <dl className="agent-inspector__identity">
      <dt>Base agent</dt><dd>{base || 'Not reported'}</dd>
      <dt>Model</dt><dd>{model || 'Not reported'}</dd>
      {Boolean(info?.providerProfile || detail?.provider_profile) && <><dt>Provider profile</dt><dd>{info?.providerProfile || text(detail?.provider_profile)}</dd></>}
      {Boolean(info?.reasoningEffort || detail?.reasoning_effort) && <><dt>Reasoning effort</dt><dd>{info?.reasoningEffort || text(detail?.reasoning_effort)}</dd></>}
      <dt>Created by</dt><dd>{parent ? rows.find(candidate => candidate.id === parent)?.title || parent : 'Parent task'}</dd>
    </dl>
    {prompt && <section><h3>Assigned task</h3><p className="agent-inspector__prompt">{prompt}</p></section>}
    {!online && <p role="status">Disconnected. Showing retained activity; reconnect for updates and controls.</p>}
    {info?.provisional && <p role="status">{state.priority < 2 ? 'Waiting for the runtime to identify this spawn request. Its assigned task is available here; controls will appear when its identity is confirmed.' : 'This request has finished. Its runtime identity was not recorded, so live details and controls are unavailable. The original task remains inspectable.'}</p>}
    {error && <div role="alert" className="studio-error"><p>{error}</p><button disabled={!online} onClick={() => setRetry(value => value+1)}>Retry agent details</button></div>}
    {info?.error && <p className="studio-error">{info.error}</p>}
    {info?.summary && !info.provisional && <section><h3>Latest summary</h3><p>{info.summary}</p></section>}
    {Boolean(info?.notes?.length) && <section><h3>Recent activity</h3>{info!.notes!.map((note, index) => <p key={index}>{note}</p>)}</section>}
    {Boolean(info?.toolCalls?.length) && <section><h3>Tool calls <span>{info!.toolCalls!.length}</span></h3>{info!.toolCalls!.map(call => <ToolCallRow key={call.id} item={call} label={call.name.replaceAll('_',' ')} />)}</section>}
    {output && <><OutputViewer text={output} label="Agent output"/><p className="studio-muted">Retained output from the latest saved agent state.</p></>}
    {!output && !info?.notes?.length && !info?.toolCalls?.length && !info?.provisional && <p className="studio-muted">{detail || error ? 'No detailed activity has been recorded yet.' : 'Loading recorded activity…'}</p>}
    {Boolean(info?.thinking?.length) && <details><summary>Reasoning</summary>{info!.thinking!.map((line,index)=><p key={index}>{line}</p>)}</details>}
    {info && [['Files read', info.filesRead], ['Files changed', info.filesWritten]].map(([label, paths]) => Array.isArray(paths) && paths.length > 0 ? <section key={String(label)}><h3>{label}</h3><ul>{paths.map(path => <li key={path}><code>{path}</code></li>)}</ul></section> : null)}
    <details className="agent-inspector__metadata"><summary>Runtime details</summary><dl><dt>Agent ID</dt><dd>{info?.provisional ? 'Not assigned yet' : row.id}</dd>{info?.toolCount !== undefined && <><dt>Tools</dt><dd>{info.toolCount}</dd></>}{info?.inputTokens !== undefined && <><dt>Input tokens</dt><dd>{info.inputTokens.toLocaleString()}</dd></>}{info?.outputTokens !== undefined && <><dt>Output tokens</dt><dd>{info.outputTokens.toLocaleString()}</dd></>}</dl></details>
    {!info?.provisional && <fieldset disabled={!online} className="agent-inspector__controls"><AgentControls id={row.id} active={state.priority < 2}/></fieldset>}
  </article>
}
