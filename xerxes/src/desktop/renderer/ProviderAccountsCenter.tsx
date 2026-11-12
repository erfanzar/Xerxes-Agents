// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useRef, useState } from 'react'

import { desktopError } from './desktopRpc.js'
import { CopyButton } from './CopyButton.js'
import { ActionDialog } from './ActionDialog.js'
import type { XerxesBridge } from './types.js'

interface AccountRow {id:string;kind:string;label:string;profileName:string;provider:string;model:string;available:boolean}
interface AccountList {accounts:readonly AccountRow[];can_sync:boolean;destination:string|null}
interface UsageWindow {label:string;used_percent:number;resets_at:string|null;detail:string|null}
interface SyncReview {review:string;account_id:string;destination:string}
interface AccountAction {kind:'sync'|'remove';account:AccountRow;phase:'reviewing'|'ready'|'working'|'done';review?:SyncReview;error:string}

export function formatAccountUsageWindow(window: UsageWindow): string {
  if (!Number.isFinite(window.used_percent)) return `${window.label}: usage unavailable`
  const reset = window.resets_at && Number.isFinite(Date.parse(window.resets_at)) ? ` · resets ${new Date(window.resets_at).toLocaleString()}` : ''
  return `${window.label}: ${Math.round(window.used_percent)}% used${reset}`
}

export function ProviderLoginLink({url,busy,write}:{url:string;busy:boolean;write?:(text:string)=>Promise<unknown>}) {
  return <div className="provider-accounts__signin" data-state={busy?'waiting':'ready'}>
    <strong>{busy?'Finish sign-in in your browser':'Sign-in link'}</strong>
    <p>Nothing is registered until sign-in completes. If the browser did not open, copy the link and open it yourself.</p>
    {url
      ? <div className="provider-accounts__signin-link"><input aria-label="Sign-in link" readOnly value={url} spellCheck={false} onFocus={event=>event.currentTarget.select()}/><CopyButton text={url} label="Copy sign-in link" {...(write?{write}:{})}/></div>
      : <p className="studio-muted">Preparing the secure sign-in link…</p>}
  </div>
}

interface RecordedUsage {
  status:'available'|'unavailable';profile:string;since?:number|null;calls?:number;input_tokens?:number;output_tokens?:number;cache_read_tokens?:number;cache_write_tokens?:number;cost_usd?:number
}

export function AccountRecordedUsage({report}:{report:RecordedUsage}) {
  if(report.status!=='available')return <span>Recorded usage unavailable on this workspace. No consumption or price estimate was made.</span>
  const since=typeof report.since==='number'&&Number.isFinite(report.since)?` since ${new Date(report.since).toLocaleString()}`:''
  return <span className="provider-account__recorded">Recorded on this workspace for profile {report.profile} (by profile name, not credential){since}: {report.calls} calls · {report.input_tokens} input · {report.output_tokens} output · {report.cache_read_tokens} cache read · {report.cache_write_tokens} cache write tokens
    <span>{typeof report.cost_usd==='number'&&Number.isFinite(report.cost_usd)?`Published-price cost: $${report.cost_usd.toFixed(4)} (not a provider bill)`:'Price unavailable; no dollar estimate.'}</span>
  </span>
}

/** The only key in the renderer is this user's unsaved input; clear on every attempt. */
export function GeminiAccountForm({busy,save}:{busy:boolean;save:(label:string,apiKey:string,model:string)=>Promise<void>}) {
  const [label,setLabel]=useState(''),[apiKey,setApiKey]=useState(''),[model,setModel]=useState(''),[pending,setPending]=useState(false),[error,setError]=useState('')
  const submit=async()=>{
    if(busy||pending)return
    const key=apiKey
    setApiKey('');setPending(true);setError('')
    try {await save(label,key,model);setLabel('')}
    catch(error){setError(desktopError(error).split(key||'\0').join('[redacted]'))}
    finally{setPending(false)}
  }
  return <form className="provider-accounts__gemini" aria-label="Add Gemini API-key account" onSubmit={event=>{event.preventDefault();void submit()}}>
    <p>Stored only on this computer in owner-only provider profile storage. No validation or billable model call is made when saving. To transfer a key, review and confirm Sync to SSH separately.</p>
    <input className="palette__in" aria-label="Gemini account label" autoComplete="off" value={label} onChange={event=>setLabel(event.target.value)} placeholder="Account label (work, personal…)" maxLength={80} disabled={busy||pending} required/>
    <input className="palette__in" type="password" aria-label="Gemini API key" autoComplete="off" spellCheck={false} autoCapitalize="none" value={apiKey} onChange={event=>setApiKey(event.target.value)} placeholder="Gemini API key" disabled={busy||pending} required/>
    <input className="palette__in" aria-label="Gemini model id" autoComplete="off" spellCheck={false} value={model} onChange={event=>setModel(event.target.value)} placeholder="Model id, e.g. gemini-2.5-flash" maxLength={512} disabled={busy||pending} required/>
    <button className="btn" type="submit" disabled={busy||pending||!label.trim()||!apiKey.trim()||!model.trim()}>{pending?'Saving Gemini account…':'Save Gemini account'}</button>
    {error&&<p role="alert" className="studio-error">{error}</p>}
  </form>
}

export function ProviderAccountsCenter({accounts,onAccountLogin,sessionKey,turnActive,changed}:{accounts?:XerxesBridge['accounts'];onAccountLogin?:XerxesBridge['onAccountLogin'];sessionKey:string;turnActive:boolean;changed:()=>void}) {
  const alive=useRef(true), [list,setList]=useState<AccountList|null>(null), [usage,setUsage]=useState<Record<string,unknown>>({})
  const [kind,setKind]=useState<'claude'|'codex'|'gemini'>('claude'), [label,setLabel]=useState(''), [busy,setBusy]=useState(false), [error,setError]=useState('')
  const [action,setAction]=useState<AccountAction|null>(null), actionRequest=useRef(0)
  const closeAction=()=>{actionRequest.current+=1;setAction(null)}
  const loginId=useRef(''), [loginUrl,setLoginUrl]=useState(''), [signingIn,setSigningIn]=useState(false)
  useEffect(()=>{alive.current=true;return()=>{alive.current=false}},[])
  useEffect(()=>onAccountLogin?.(event=>{
    if(alive.current&&event.login_id===loginId.current)setLoginUrl(event.url)
  }),[onAccountLogin])
  const load=async()=>{
    if(!accounts)return
    const result=await accounts('list',{}) as AccountList & {ok?:boolean;error?:string}
    if(result.ok===false)throw new Error(result.error||'Could not load provider accounts')
    if(!alive.current)return
    setList(result)
    void Promise.all(result.accounts.map(async account=>{
      try{const report=await accounts('usage',{accountId:account.id});if(alive.current)setUsage(current=>({...current,[account.id]:report}))}
      catch(error){if(alive.current)setUsage(current=>({...current,[account.id]:{status:'unknown',reason:desktopError(error)}}))}
    }))
  }
  useEffect(()=>{void load().catch(error=>{if(alive.current)setError(desktopError(error))})},[accounts])
  const run=async(work:()=>Promise<void>)=>{setBusy(true);setError('');try{await work()}catch(error){if(alive.current)setError(desktopError(error))}finally{if(alive.current)setBusy(false)}}
  const login=()=>run(async()=>{
    setSigningIn(true)
    loginId.current=globalThis.crypto.randomUUID()
    setLoginUrl('')
    try {
      const result=await accounts!('login',{kind,label,loginId:loginId.current}) as {ok?:boolean;error?:string}
      if(result.ok===false)throw new Error(result.error||'Sign-in failed')
      setLabel('');setLoginUrl('');loginId.current='';await load();changed()
    } catch(error) {
      // A failed callback listener makes its authorize URL unusable. Never
      // leave a copied link that can only land on "cannot connect".
      setLoginUrl('');loginId.current=''
      throw error
    } finally {if(alive.current)setSigningIn(false)}
  })
  const saveGemini=async(label:string,apiKey:string,model:string)=>{
    setBusy(true);setError('')
    try {
      const result=await accounts!('create-gemini',{label,apiKey,model}) as {ok?:boolean;error?:string}
      if(result.ok===false)throw new Error(result.error||'Could not save Gemini account')
      await load();changed()
    } finally {if(alive.current)setBusy(false)}
  }
  const activate=(id:string)=>run(async()=>{
    const result=await accounts!('activate',{accountId:id,sessionKey}) as {ok?:boolean;error?:string}
    if(result.ok===false)throw new Error(result.error||'Could not switch account')
    await load();changed()
  })
  const reviewSync=async(account:AccountRow)=>{
    const request=++actionRequest.current
    setAction({kind:'sync',account,phase:'reviewing',error:''})
    try {
      const result=await accounts!('sync-review',{accountId:account.id}) as SyncReview & {ok?:boolean;error?:string}
      if(result.ok===false)throw new Error(result.error||'Could not review sync')
      if(alive.current&&actionRequest.current===request)setAction({kind:'sync',account,phase:'ready',review:result,error:''})
    } catch(error) {
      if(alive.current&&actionRequest.current===request)setAction({kind:'sync',account,phase:'ready',error:desktopError(error)})
    }
  }
  const confirmAction=async()=>{
    if(!action)return
    if(action.phase==='done'){closeAction();return}
    if(action.phase!=='ready')return
    if(action.kind==='sync'&&!action.review){void reviewSync(action.account);return}
    const request=actionRequest.current
    setAction({...action,phase:'working',error:''})
    try {
      const result=action.kind==='sync'
        ? await accounts!('sync',{review:action.review!.review,confirm:true}) as {ok?:boolean;error?:string}
        : await accounts!('remove',{accountId:action.account.id}) as {ok?:boolean;error?:string}
      if(result.ok===false)throw new Error(result.error||`Could not ${action.kind==='sync'?'synchronize':'remove'} account`)
      if(alive.current&&actionRequest.current===request)setAction({...action,phase:'done',error:''})
      await load();changed()
    } catch(error) {
      // Transfer reviews are one-use. A failed/uncertain transfer needs a fresh
      // human review, never an automatic replay of the previous credential write.
      if(alive.current&&actionRequest.current===request)setAction({kind:action.kind,account:action.account,phase:'ready',error:desktopError(error)})
    }
  }
  return <section className="provider-accounts" aria-label="Provider accounts">
    <div className="row__t">Provider accounts</div>
    <p className="modal__sub">One account center for this Mac and every SSH workspace. Each Claude/Codex login has isolated credentials. Gemini is API-key only; no Gemini OAuth or subscription quota readback is supported.</p>
    <div className="provider-accounts__login">
      <select className="palette__in" aria-label="Account provider" value={kind} onChange={event=>{const value=event.target.value;if(value==='claude'||value==='codex'||value==='gemini')setKind(value)}} disabled={busy}>
        <option value="claude">Claude</option><option value="codex">Codex</option><option value="gemini">Gemini (API key)</option>
      </select>
      {kind!=='gemini'&&<><input className="palette__in" aria-label="OAuth account label" value={label} onChange={event=>setLabel(event.target.value)} placeholder="Account label (work, personal…)" disabled={busy}/>
      <button className="btn" disabled={busy||!accounts||!label.trim()} onClick={login}>Sign in</button></>}
    </div>
    {kind==='gemini'&&<GeminiAccountForm busy={busy||!accounts} save={saveGemini}/>}
    {(signingIn||loginUrl)&&<ProviderLoginLink url={loginUrl} busy={signingIn} write={text=>accounts!('copy-login-link',{url:text})}/>}
    {error&&<p role="alert" className="studio-error">{error}</p>}
    {list?.accounts.map(account=>{
      const report=usage[account.id] as {status?:string;reason?:string;windows?:UsageWindow[];recorded?:RecordedUsage}|undefined
      return <div className="provcard provider-account" key={account.id}>
        <div className="provcard__main">
          <span className="provcard__text">
            <span className="provcard__title"><span className="provcard__name">{account.label}</span><span className="provcard__badge provcard__badge--quiet">{account.kind}</span></span>
            <span className="provcard__model">{account.profileName} · {account.model||'model chosen when used'}</span>
            <span className="provcard__meta">{account.provider} · {account.available?'available on this workspace':'not available on this workspace yet'}</span>
          </span>
          <span className="provcard__tools">
            <button className="btn" disabled={busy||turnActive||!account.available} onClick={()=>void activate(account.id)}>Use</button>
            {list.can_sync&&<button className="btn" disabled={busy} onClick={()=>void reviewSync(account)}>Sync to SSH</button>}
            {(account.kind!=='api_key'||account.provider==='gemini')&&<button className="btn" disabled={busy} onClick={()=>{actionRequest.current+=1;setAction({kind:'remove',account,phase:'ready',error:''})}}>Remove</button>}
          </span>
        </div>
        <div className="provider-account__usage">
          {report?.windows?.map(window=><span key={window.label}>{formatAccountUsageWindow(window)}</span>)}
          {(!report||report.status==='unknown')&&<span>{report?.reason||'Loading usage…'}</span>}
          {report?.recorded&&<AccountRecordedUsage report={report.recorded}/>}
        </div>
      </div>
    })}
    {list&&!list.accounts.length&&<p className="studio-muted">No accounts yet. Sign in above or choose Gemini to save a named API-key account.</p>}
    {action&&<ActionDialog
      title={action.phase==='done'?(action.kind==='sync'?'Account synced':'Account removed'):(action.kind==='sync'?`Sync ${action.account.label} to SSH`:`Remove ${action.account.label}?`)}
      confirmLabel={action.phase==='done'?'Done':action.kind==='remove'?'Remove account':!action.review&&action.phase==='ready'?'Review sync again':'Confirm sync'}
      busyLabel={action.kind==='sync'?'Syncing account…':'Removing account…'}
      busy={action.phase==='working'} disabled={action.phase==='reviewing'} done={action.phase==='done'} error={action.error}
      onConfirm={()=>void confirmAction()} onCancel={closeAction}>
      {action.phase==='reviewing'?<p role="status">Preparing transfer review for <strong>{action.account.label}</strong>…</p>
        :action.phase==='done'?<p role="status">{action.kind==='sync'?<><strong>{action.account.label}</strong> synced to <strong>{action.review?.destination||list?.destination||'this SSH host'}</strong>.</>:<><strong>{action.account.label}</strong> was removed. Other accounts were not changed.</>}</p>
        :<><p>{action.kind==='sync'?<>Transfer <strong>{action.account.label}</strong> to <strong>{action.review?.destination||list?.destination||'this SSH host'}</strong>?</>:<>Remove <strong>{action.account.label}</strong> and its saved {action.account.kind==='api_key'?'API key':'sign-in'} from this Mac?</>}</p>
          <p className="studio-muted">{action.kind==='sync'?'Only this account’s credential and linked profile are sent over SSH. A same-named remote profile will be replaced on confirmation. Other accounts are not transferred.':'Its local provider profile is also removed. This does not sign out other accounts or delete copies previously synchronized to SSH hosts.'}</p></>}
    </ActionDialog>}
  </section>
}
