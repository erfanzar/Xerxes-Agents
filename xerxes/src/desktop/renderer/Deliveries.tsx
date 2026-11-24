// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import {useEffect,useRef,useState,type ReactElement} from 'react'
import {desktopCall,desktopError,record,records} from './desktopRpc.js'
import type {Snapshot} from './store.js'
import {ActionDialog} from './ActionDialog.js'
type Delivery={id:string;platform:string;recipient:string;state:string;attempts:number;content:string;error:string}
export function parseDelivery(value:unknown):Delivery{const row=record(value);if(typeof row.id!=='string'||typeof row.platform!=='string'||typeof row.recipient!=='string'||!Number.isSafeInteger(row.attempts)||Number(row.attempts)<0||!['pending','sending','sent','uncertain'].includes(String(row.state)))throw Error('Invalid delivery record');return{id:row.id,platform:row.platform,recipient:row.recipient,state:String(row.state),attempts:Number(row.attempts),content:typeof row.content==='string'?row.content:'',error:typeof row.error==='string'?row.error:''}}
export function Deliveries({snap,scheduleId,close}:{snap:Snapshot;scheduleId:string;close():void}):ReactElement{
  const[rows,setRows]=useState<Delivery[]>([]),[detail,setDetail]=useState<Delivery|null>(null),[error,setError]=useState(''),[busy,setBusy]=useState(false),[confirm,setConfirm]=useState<'sent'|'retry'|'send'|null>(null)
  const pending=useRef(false),alive=useRef(true)
  const[selected,setSelected]=useState<string|null>(null)
  const call=(method:string,params:Record<string,unknown>={})=>desktopCall(window.xerxes,snap.sessionKey,method,{schedule_id:scheduleId,...params})
  const run=async(work:()=>Promise<void>)=>{if(pending.current)return;pending.current=true;setBusy(true);setError('');try{await work()}catch(failure){if(alive.current)setError(desktopError(failure))}finally{pending.current=false;if(alive.current)setBusy(false)}}
  const load=async()=>{const value=await call('schedule.deliveries');const next=records(value.deliveries).map(parseDelivery);if(alive.current)setRows(next)}
  useEffect(()=>{alive.current=true;void run(load);return()=>{alive.current=false}},[])
  const inspect=async(id:string)=>{const value=parseDelivery((await call('schedule.delivery.inspect',{delivery_id:id})).delivery);if(value.id!==id)throw Error('Delivery changed. Refresh before acting.');if(alive.current){setDetail(value);setConfirm(null)}}
  const submit=()=>{if(!detail||!confirm)return;const action=confirm;void run(async()=>{await call(action==='send'?'schedule.delivery.send':'schedule.delivery.resolve',{delivery_id:detail.id,...(action==='send'?{}:{decision:action,attempts:detail.attempts})});if(alive.current)setConfirm(null);await inspect(detail.id);await load()})}
  const closeDialog=()=>{setSelected(null);setDetail(null);setConfirm(null);setError('')}
  const dialogTitle=confirm==='send'?'Send saved output?':confirm==='retry'?'Allow retry?':confirm==='sent'?'Mark delivery as sent?':'Delivery details'
  const dialogConfirmLabel=confirm==='send'?'Confirm send':confirm==='retry'?'Confirm retry':confirm==='sent'?'Confirm mark as sent':'Close'
  const dialogBusyLabel=confirm==='send'?'Sending…':confirm?'Updating…':'Loading…'
  return <section className="deliveries" aria-label="Schedule deliveries"><div className="lsp-actions"><button type="button" className="btn btn--ghost" disabled={busy} onClick={close}>Back to schedules</button><button type="button" className="btn btn--ghost" disabled={busy} onClick={()=>void run(async()=>{await load();if(detail)await inspect(detail.id)})}>Refresh deliveries</button></div><h2>Deliveries</h2><p>Inspect saved output and its destination before sending or resolving an uncertain delivery.</p>{busy&&selected===null&&<p role="status">Loading deliveries…</p>}{error&&selected===null&&<p className="studio-error" role="alert">{error}</p>}{!rows.length&&!busy&&!error&&<p>No deliveries recorded.</p>}
    {rows.map(row=><div className="studio-item" key={row.id}><div><strong>{row.platform} · {row.recipient||'Default destination'}</strong><p>{row.state} · {row.attempts} attempts</p></div><button type="button" className="btn btn--ghost" disabled={busy} onClick={()=>{setSelected(row.id);setDetail(null);setConfirm(null);void run(()=>inspect(row.id))}}>Inspect delivery</button></div>)}
    {selected!==null&&<ActionDialog title={dialogTitle} done={confirm===null} confirmLabel={dialogConfirmLabel} busyLabel={dialogBusyLabel} busy={busy} error={error} onCancel={()=>confirm!==null?setConfirm(null):closeDialog()} onConfirm={()=>confirm!==null?submit():closeDialog()}>
    {busy&&<p role="status">Loading delivery…</p>}
    {!detail&&!busy&&error&&<button type="button" className="btn" onClick={()=>void run(()=>inspect(selected))}>Retry inspection</button>}
    {detail&&<div>
      <div className="row__t">{detail.platform} · {detail.recipient||'Default destination'}</div>
      <p className="row__s">{detail.state} · {detail.attempts} attempts</p>
      {detail.error&&<p className="studio-error">{detail.error}</p>}
      {confirm ? (
        <p role="status">{confirm==='sent'?'Confirm you checked the destination and this message arrived.':confirm==='retry'?'Check the destination first. Allowing retry may send a duplicate message.':`Send this saved output to ${detail.recipient||'the default destination'} on ${detail.platform}?`}</p>
      ) : (
        <>
          <pre className="studio-source" tabIndex={0}>{detail.content||'No saved output.'}</pre>
          {(detail.state==='pending'||detail.state==='uncertain')&&<div className="lsp-actions">{detail.state==='pending'&&<button type="button" className="btn" disabled={busy} onClick={()=>setConfirm('send')}>Send saved output</button>}{detail.state==='uncertain'&&<><button type="button" className="btn" disabled={busy} onClick={()=>setConfirm('sent')}>Mark as sent</button><button type="button" className="btn" disabled={busy} onClick={()=>setConfirm('retry')}>Allow retry</button></>}</div>}
        </>
      )}
    </div>}
    </ActionDialog>}
  </section>
}
