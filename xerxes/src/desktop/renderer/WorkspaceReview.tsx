// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import {useEffect,useRef,useState,type ReactElement} from 'react'
import {listWorkspaces,inspectWorkspace,checkWorkspaceApply,applyWorkspaceReview,listWorkspaceIntegrations,inspectWorkspaceIntegration,recoverWorkspaceIntegration,type WorkspaceRecord,type WorkspaceReview as Review,type WorkspaceApplyCheck,type WorkspaceIntegration} from '../../ui/lib/workspaces.js'
import {DiffPreview} from './DiffPreview.js'
import type {GatewayRpc} from '../../ui/app/interfaces.js'
import {desktopError} from './desktopRpc.js'
import type {Snapshot} from './store.js'

export function WorkspaceReview({snap,close}:{snap:Snapshot;close():void}):ReactElement {
  const[rows,setRows]=useState<WorkspaceRecord[]>([]),[next,setNext]=useState<string|undefined>(),[review,setReview]=useState<Review|null>(null),[check,setCheck]=useState<WorkspaceApplyCheck|null>(null)
  const[integrations,setIntegrations]=useState<WorkspaceIntegration[]|null>(null),[recoveryNext,setRecoveryNext]=useState<string|undefined>(),[recovery,setRecovery]=useState<{id:string;files:Awaited<ReturnType<typeof inspectWorkspaceIntegration>>}|null>(null)
  const[busy,setBusy]=useState(false),[error,setError]=useState(''),[notice,setNotice]=useState(''),[confirmation,setConfirmation]=useState<'apply'|'recover'|null>(null)
  const pending=useRef(false),alive=useRef(true)
  const rpc:GatewayRpc=(method,params)=>window.xerxes.call(method,{...params,session_key:snap.sessionKey})
  const run=async(work:()=>Promise<void>)=>{if(pending.current)return;pending.current=true;setBusy(true);setError('');try{await work()}catch(failure){if(alive.current){setError(desktopError(failure));setCheck(null);setConfirmation(null)}}finally{pending.current=false;if(alive.current)setBusy(false)}}
  const load=async(append=false)=>{const page=await listWorkspaces(rpc,append?next:undefined);if(alive.current){setRows(old=>append?[...old,...page.records]:page.records);setNext(page.next)}}
  useEffect(()=>{alive.current=true;void run(()=>load());return()=>{alive.current=false}},[])
  const recoveries=async(append=false)=>{const page=await listWorkspaceIntegrations(rpc,append?recoveryNext:undefined);if(alive.current){setIntegrations(old=>append?[...old??[],...page.records]:page.records);setRecoveryNext(page.next)}}
  return <section className="managed-review" aria-label="Managed workspace review">
    <button disabled={busy} onClick={close}>Back to workspaces</button><h2>Review isolated work</h2><p>Inspect a task workspace, check its patch against the destination, then apply the reviewed changes.</p>
    <div className="lsp-actions"><button disabled={busy} onClick={()=>void run(async()=>{setReview(null);setCheck(null);setConfirmation(null);setIntegrations(null);await load()})}>Refresh workspaces</button><button disabled={busy} onClick={()=>void run(()=>recoveries())}>Integration recovery</button></div>
    {busy&&<p role="status">Loading workspace review…</p>}{error&&<p className="studio-error" role="alert">{error}</p>}{notice&&<p role="status">{notice}</p>}
    {integrations===null?<>
      {!rows.length&&!busy&&!error&&<p>No managed task workspaces recorded.</p>}
      {!review&&<>{rows.map(row=><div className="studio-item" key={row.id}><div><strong>{row.branch}</strong><p>{row.path}</p>{row.error&&<p className="studio-error">{row.error}</p>}</div><button disabled={busy} onClick={()=>void run(async()=>{setCheck(null);setConfirmation(null);setNotice('');const value=await inspectWorkspace(rpc,row.id);if(alive.current)setReview(value)})}>Review changes</button></div>)}
      {next&&<button disabled={busy} onClick={()=>void run(()=>load(true))}>Load more workspaces</button>}</>}
      {review&&<><button disabled={busy} onClick={()=>{setReview(null);setCheck(null);setConfirmation(null)}}>Back to workspace list</button><h3>{review.branch}</h3><p>{review.status} · {review.path}</p>{review.setup&&<p>{review.setup}</p>}<p>Base: {review.base}<br/>Head: {review.head}</p>
        <DiffPreview diff={review.diff} label="Managed workspace diff" />
        <div className="lsp-actions"><button disabled={busy||!review.reviewId} onClick={()=>void run(async()=>{setCheck(null);setConfirmation(null);const value=await checkWorkspaceApply(rpc,review);if(alive.current)setCheck(value)})}>Check integration</button>{check?.destinationState&&<button disabled={busy} onClick={()=>setConfirmation('apply')}>Apply reviewed changes</button>}</div>{check&&<p>{check.message}</p>}
        {confirmation==='apply'&&check&&<div role="group" aria-label="Confirm workspace apply"><p>Apply this reviewed patch to {check.destination}? The runtime creates a recovery backup and rejects changes made since the check.</p><button disabled={busy} onClick={()=>void run(async()=>{const message=await applyWorkspaceReview(rpc,review,check);if(alive.current){setNotice(message);setCheck(null);setConfirmation(null)}})}>Confirm apply</button><button disabled={busy} onClick={()=>setConfirmation(null)}>Cancel</button></div>}
      </>}
    </>:<><h3>Integration recovery</h3>{!integrations.length&&<p>No integration records.</p>}{integrations.map(row=><div className="goal-criterion" key={row.id}><strong>{row.destination}</strong><p>{row.status} · {row.backupPath}</p>{row.error&&<p className="studio-error">{row.error}</p>}<button disabled={busy} onClick={()=>void run(async()=>{setConfirmation(null);const files=await inspectWorkspaceIntegration(rpc,row.id);if(alive.current)setRecovery({id:row.id,files})})}>Inspect recovery</button></div>)}{recoveryNext&&<button disabled={busy} onClick={()=>void run(()=>recoveries(true))}>Load earlier integrations</button>}
      {recovery&&<><h4>Proposed recovery</h4>{recovery.files.map(file=><p key={file.path}><strong>{file.path}</strong><br/>{file.action} · {file.reason}</p>)}<button disabled={busy} onClick={()=>setConfirmation('recover')}>Recover integration</button>{confirmation==='recover'&&<div role="group" aria-label="Confirm workspace recovery"><p>Recover this integration using the recorded backup? Concurrent edits are preserved and conflicts remain visible.</p><button disabled={busy} onClick={()=>void run(async()=>{const message=await recoverWorkspaceIntegration(rpc,recovery.id);if(alive.current){setNotice(message);setRecovery(null);setConfirmation(null)}await recoveries()})}>Confirm recovery</button><button disabled={busy} onClick={()=>setConfirmation(null)}>Cancel</button></div>}</>}
    </>}
  </section>
}
