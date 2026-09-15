// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useEffect, useRef, useState, type ReactElement } from 'react'
import { acknowledgeRun, cancelRun, inspectRun, listRunPage, type RunDetail } from '../../ui/lib/runs.js'
import type { GatewayRpc } from '../../ui/app/interfaces.js'
import { desktopError, scheduleTime } from './desktopRpc.js'
import type { Snapshot } from './store.js'

type Page = Awaited<ReturnType<typeof listRunPage>>
export function UnifiedRuns({snap,close}:{snap:Snapshot;close():void}):ReactElement {
  const [scope,setScope]=useState<'workspace'|'session'>('workspace'),[kind,setKind]=useState(''),[state,setState]=useState(''),[unread,setUnread]=useState(false)
  const [cursors,setCursors]=useState<{startedAt:number;id:string}[]>([]),[page,setPage]=useState<Page|null>(null),[detail,setDetail]=useState<RunDetail|null>(null)
  const [error,setError]=useState(''),[notice,setNotice]=useState(''),[busy,setBusy]=useState(false),[refresh,setRefresh]=useState(0)
  const epoch=useRef(0),pending=useRef(false)
  const rpc:GatewayRpc=(method,params)=>window.xerxes.call(method,{...params,session_key:snap.sessionKey})
  const before=cursors.at(-1)
  useEffect(()=>{
    let current=true;const version=++epoch.current;let timer:ReturnType<typeof setTimeout>|undefined
    setPage(null);setDetail(null);setBusy(true);setError('')
    const load=async()=>{
      if(pending.current){if(current)timer=setTimeout(load,2000);return}
      try{const next=await listRunPage(rpc,unread,scope,undefined,before,{...(kind?{kind}:{}),...(state?{state}:{})});if(current&&version===epoch.current){setPage(next);setError('')}}catch(failure){if(current)setError(desktopError(failure))}
      finally{if(current){setBusy(false);timer=setTimeout(load,2000)}}
    };void load();return()=>{current=false;epoch.current++;if(timer)clearTimeout(timer)}
  },[snap.sessionKey,scope,kind,state,unread,before,refresh])
  const inspect=async(id:string)=>{if(pending.current)return;pending.current=true;setBusy(true);setError('');const version=epoch.current
    try{const next=await inspectRun(rpc,id,scope);if(version===epoch.current)setDetail(next)}catch(failure){if(version===epoch.current)setError(desktopError(failure))}finally{pending.current=false;if(version===epoch.current)setBusy(false)}}
  const act=async(action:'ack'|'cancel')=>{if(!detail||pending.current)return;pending.current=true;setBusy(true);setError('');setNotice('');const version=epoch.current
    try{if(action==='ack')await acknowledgeRun(rpc,detail,scope);else await cancelRun(rpc,detail,scope);if(version===epoch.current){setNotice(action==='ack'?'Run marked as read.':'Cancellation requested.');const updated=await inspectRun(rpc,detail.id,scope);if(version===epoch.current){setDetail(updated);setPage(current=>current?{...current,runs:current.runs.flatMap(row=>row.id!==updated.id?[row]:((unread&&!updated.unread)||(state&&updated.state!==state))?[]:[updated])}:current)}}}
    catch(failure){if(version===epoch.current)setError(desktopError(failure))}finally{pending.current=false;if(version===epoch.current)setBusy(false)}}
  const reset=()=>{setCursors([]);setNotice('')}
  return <section className="unified-runs" aria-label="All run history">
    <button onClick={close} disabled={busy}>Back to schedules</button><h2>Run history</h2><p>Review background work across agents, terminals, schedules and monitors.</p>
    <div className="run-filters">
      <label>Scope<select aria-label="Scope" value={scope} disabled={busy} onChange={e=>{setScope(e.target.value as typeof scope);reset()}}><option value="workspace">Workspace</option><option value="session">This session</option></select></label>
      <label>Kind<select aria-label="Kind" value={kind} disabled={busy} onChange={e=>{setKind(e.target.value);reset()}}>{['','agent','terminal','schedule','monitor'].map(v=><option key={v} value={v}>{v||'All kinds'}</option>)}</select></label>
      <label>Status<select aria-label="Status" value={state} disabled={busy} onChange={e=>{setState(e.target.value);reset()}}>{['','running','failed','interrupted','cancelled','succeeded'].map(v=><option key={v} value={v}>{v||'All states'}</option>)}</select></label>
    </div>
    <label className="choice-row"><input type="checkbox" checked={unread} disabled={busy} onChange={e=>{setUnread(e.target.checked);reset()}}/> Unread only</label>
    <button disabled={busy} onClick={()=>setRefresh(v=>v+1)}>Refresh runs</button>
    {busy&&<p role="status">Loading run history…</p>}{error&&<p className="studio-error" role="alert">{error}</p>}{notice&&<p role="status">{notice}</p>}
    {page?.attentionTotal? <details><summary>{page.attentionTotal} requests need attention</summary>{page.attention.map(row=><p key={row.id}>{row.kind}: {row.title}</p>)}<p>Open the owning conversation to respond.</p></details>:null}
    {page?.upcomingTotal? <details><summary>{page.upcomingTotal} upcoming schedules</summary>{page.upcoming.map(row=><p key={row.id}>{row.title}<br/>{scheduleTime(row.nextRunAt,row.timezone)} · {row.executionState}</p>)}</details>:null}
    {page&&!page.runs.length&&<p>No runs match these filters.</p>}
    {page?.runs.map(run=><div className="run-history__entry" key={run.id}>
      <div className="studio-item"><div><strong>{run.title}</strong><p>{run.kind} · {run.state}{run.unread?' · Unread':''}<br/>{new Date(run.startedAt).toLocaleString()}</p></div><button disabled={busy} aria-expanded={detail?.id===run.id} onClick={()=>void inspect(run.id)}>{detail?.id===run.id?'Refresh result':'View result'}</button></div>
      {detail?.id===run.id&&<div className="run-history__result">
        {detail.error&&<p className={detail.state === 'failed' ? 'studio-error' : 'studio-muted'}>{detail.error}</p>}
        {detail.tokenUsage&&<p>{(detail.tokenUsage.inputTokens+detail.tokenUsage.outputTokens).toLocaleString()} tokens{!detail.tokenUsage.complete?' · incomplete measurement':''}</p>}
        {detail.reactionHealth&&<p>Reaction: {detail.reactionHealth.state} · {detail.reactionHealth.attempts}/{detail.reactionHealth.maxReactions} attempts · {detail.reactionHealth.pendingEvents} pending events{detail.reactionHealth.lastError?` · ${detail.reactionHealth.lastError}`:''}</p>}
        {detail.output?<pre className="studio-source run-history__output" tabIndex={0} aria-label="Run output">{detail.output}</pre>:<p>No output recorded.</p>}{detail.outputTruncated&&<p>Only retained output is shown.</p>}
        <div className="lsp-actions">{detail.unread&&<button disabled={busy||!!error} onClick={()=>void act('ack')}>Mark as read</button>}{detail.cancelLabel&&<button disabled={busy||!!error} onClick={()=>void act('cancel')}>{detail.cancelLabel}</button>}</div>
      </div>}
    </div>)}
    <div className="lsp-actions"><button disabled={busy||!cursors.length} onClick={()=>setCursors(v=>v.slice(0,-1))}>Newer runs</button><button disabled={busy||!page?.hasMore||!page.runs.length} onClick={()=>{const last=page!.runs.at(-1)!;setCursors(v=>[...v,{id:last.id,startedAt:last.startedAt}])}}>Older runs</button></div>
  </section>
}
