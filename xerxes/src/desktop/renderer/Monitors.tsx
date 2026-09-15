// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useEffect, useRef, useState, type ReactElement } from 'react'
import { createMonitor, listMonitors, listMonitorWebhooks, monitorAction, type MonitorSettings, type MonitorView } from '../../ui/lib/monitors.js'
import type { GatewayRpc } from '../../ui/app/interfaces.js'
import { desktopCall, desktopError } from './desktopRpc.js'
import { store, type Snapshot } from './store.js'

export function monitorFormSettings(data: FormData): MonitorSettings {
  const source=String(data.get('source')),target=String(data.get('target')??'').trim(),react=data.get('react')==='on'
  const trigger=source==='file'?'change':source==='terminal'&&data.get('trigger')==='completion'?'completion':'output'
  const settings:MonitorSettings={duration_seconds:Number(data.get('duration')),react,max_reactions:Number(data.get('attempts')),reaction_timeout_seconds:Number(data.get('timeout')),trigger,
    ...(source==='file'?{source_kind:'file',file_path:target}:source==='websocket'?{source_kind:'websocket',websocket_url:target}:source==='webhook'?{source_kind:'webhook',webhook_name:target}:{terminal_id:target}),
    ...(trigger==='output'?{match:String(data.get('match')??'').trim()}:{}),
    ...(react&&String(data.get('tokens')??'').trim()?{max_total_tokens:Number(data.get('tokens'))}:{})}
  if(!target)throw Error('Choose a source to watch.')
  if(trigger==='output'&&!settings.match)throw Error('Enter text to match.')
  if(!Number.isSafeInteger(settings.duration_seconds)||settings.duration_seconds<1||settings.duration_seconds>86400)throw Error('Watch duration must be 1–86400 seconds.')
  return settings
}
export function Monitors({snap}:{snap:Snapshot}):ReactElement {
  const [rows,setRows]=useState<MonitorView[]>([]),[detail,setDetail]=useState<MonitorView|null>(null),[creating,setCreating]=useState(false),[editing,setEditing]=useState(false)
  const [source,setSource]=useState('terminal'),[trigger,setTrigger]=useState('output'),[react,setReact]=useState(false),[webhooks,setWebhooks]=useState<string[]>([])
  const [error,setError]=useState(''),[busy,setBusy]=useState(false)
  const pending=useRef(false),epoch=useRef(0)
  const rpc:GatewayRpc=(method,params)=>window.xerxes.call(method,{...params,session_key:snap.sessionKey})
  const run=async(work:()=>Promise<void>)=>{if(pending.current)return;pending.current=true;setBusy(true);setError('');try{await work()}catch(failure){setError(desktopError(failure))}finally{pending.current=false;setBusy(false)}}
  const load=async()=>{const version=epoch.current,next=await listMonitors(rpc);if(version===epoch.current)setRows(next)}
  useEffect(()=>{epoch.current++;void run(load);return()=>{epoch.current++}},[snap.sessionKey])
  return <section className="monitor-manager" aria-label="Monitors">
    <div className="lsp-actions"><button disabled={busy} onClick={()=>void run(load)}>Refresh monitors</button><button disabled={busy} onClick={()=>{setCreating(true);setEditing(false);setReact(false);setSource("terminal");setTrigger("output");storeTerminals();}}>Create monitor</button></div>
    {error&&<p className="studio-error" role="alert">{error}</p>}{busy&&<p role="status">Updating monitors…</p>}
    {!busy&&!error&&!rows.length&&<p>No monitors in this session.</p>}
    {rows.map(row=><div className="goal-criterion" key={row.id}><strong>{row.source?.kind==='file'?row.source.path:row.source?.kind==='websocket'?row.source.url:row.source?.kind==='webhook'?row.source.name:row.terminalId}</strong><p>{row.state} · {row.trigger} · {row.reaction}</p><button disabled={busy} onClick={()=>void run(async()=>{setDetail(await monitorAction(rpc,row.id,'inspect'));setEditing(false)})}>Inspect monitor</button></div>)}
    {detail&&<div className="monitor-detail"><h4>Monitor details</h4><p>{detail.sourceStatus} · {detail.state}</p><p>{detail.reaction}</p>{detail.error&&<p className="studio-error">{detail.error}</p>}{detail.events.map((event,i)=><pre key={i}>{event}</pre>)}{detail.omittedEvents>0&&<p>{detail.omittedEvents} earlier events omitted.</p>}
      <div className="lsp-actions">{detail.stopAction&&<button disabled={busy} onClick={()=>void run(async()=>{setDetail(await monitorAction(rpc,detail.id,'stop'));await load()})}>{detail.stopAction==='cancel-reactions'?'Cancel reactions':'Stop watching'}</button>}{detail.policy&&<button disabled={busy} onClick={()=>{setEditing(true);setCreating(false)}}>Edit reaction limits</button>}</div>
      {editing&&detail.policy&&<form key={detail.policy.revision} onSubmit={e=>{e.preventDefault();const data=new FormData(e.currentTarget);void run(async()=>{await desktopCall(window.xerxes,snap.sessionKey,'monitor.update',{monitor_id:detail.id,revision:detail.policy!.revision,max_reactions:Number(data.get('attempts')),reaction_timeout_seconds:Number(data.get('timeout')),max_total_tokens:String(data.get('tokens')??'').trim()?Number(data.get('tokens')):null});setDetail(await monitorAction(rpc,detail.id,'inspect'));setEditing(false);await load()})}}><ReactionFields policy={detail.policy}/><div className="lsp-actions"><button disabled={busy}>Save limits</button><button type="button" disabled={busy} onClick={()=>setEditing(false)}>Cancel</button></div></form>}
    </div>}
    {creating&&<form className="monitor-create" onSubmit={e=>{e.preventDefault();try{const settings=monitorFormSettings(new FormData(e.currentTarget));void run(async()=>{setDetail(await createMonitor(rpc,settings));setCreating(false);await load()})}catch(failure){setError(desktopError(failure))}}}>
      <h4>Create monitor</h4><label className="field">Source<select name="source" aria-label="Monitor source" value={source} onChange={e=>{setSource(e.target.value);if(e.target.value==='webhook')void run(async()=>setWebhooks(await listMonitorWebhooks(rpc)))}}><option value="terminal">Terminal</option><option value="file">Workspace file</option><option value="websocket">WebSocket</option><option value="webhook">Configured webhook</option></select></label>
      <label className="field">{source==='terminal'?'Terminal':source==='file'?'Workspace-relative file path':source==='websocket'?'WebSocket URL':'Webhook'}{source==='terminal'?<select name="target" aria-label="Monitor target" required>{snap.terminals.filter(t=>t.running).map(t=><option value={t.id} key={t.id}>{t.label||t.command||t.id}</option>)}</select>:source==='webhook'?<select name="target" aria-label="Monitor target" required>{webhooks.map(name=><option key={name}>{name}</option>)}</select>:<input name="target" aria-label="Monitor target" required placeholder={source==='file'?'reports/results.json':'wss://example.com/events'}/>}</label>
      {source==='terminal'&&<label className="field">Trigger<select name="trigger" value={trigger} onChange={e=>setTrigger(e.target.value)}><option value="output">Output contains text</option><option value="completion">Command finishes</option></select></label>}
      {source!=='file'&&(source!=='terminal'||trigger==='output')&&<label className="field">Text to match<input name="match" required/></label>}
      <label className="field">Watch duration (seconds)<input name="duration" type="number" min={1} max={86400} defaultValue={3600} required/></label>
      <label className="choice-row"><input name="react" type="checkbox" checked={react} onChange={e=>setReact(e.target.checked)}/> Let the agent react automatically</label><p>{react?'Each reaction can use tools and model tokens within these limits.':'Only notify this session when the monitor matches.'}</p>
      <div hidden={!react}><ReactionFields/></div><div className="lsp-actions"><button disabled={busy}>Start monitor</button><button type="button" disabled={busy} onClick={()=>setCreating(false)}>Cancel</button></div>
    </form>}
  </section>
  function storeTerminals(){store.loadTerminals()}
}
function ReactionFields({policy}:{policy?:NonNullable<MonitorView['policy']>}):ReactElement{return <><label className="field">Maximum reactions<input name="attempts" type="number" min={1} max={10} defaultValue={policy?.maxReactions??3} required/></label><label className="field">Reaction timeout (seconds)<input name="timeout" type="number" min={1} max={120} defaultValue={(policy?.maxDurationMs??60000)/1000} required/></label><label className="field">Total token limit (optional)<input name="tokens" type="number" min={1} defaultValue={policy?.maxTotalTokens??''}/></label></>}
export function MonitorsDisclosure({snap}:{snap:Snapshot}):ReactElement {const [open,setOpen]=useState(false);return <details className="activity-context" onToggle={e=>setOpen(e.currentTarget.open)}><summary>Monitors</summary>{open&&<Monitors key={snap.sessionKey} snap={snap}/>}</details>}
