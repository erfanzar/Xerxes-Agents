// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useRef, useState, type ReactNode } from 'react'

import { desktopError } from './desktopRpc.js'
import { CopyButton } from './CopyButton.js'
import { ActionDialog } from './ActionDialog.js'
import type { XerxesBridge } from './types.js'

export interface AccountRow {id:string;kind:string;label:string;profileName:string;provider:string;model:string;available:boolean}
export type AgyCredentialMode = 'adc' | 'gemini_api_key' | 'keyring'
export interface AgySourceView {readonly mode:AgyCredentialMode;readonly label:string;readonly available:boolean;readonly detail:string;readonly headlessSafe:boolean;readonly accountId?:string}
export interface AgyStatusView {
  readonly agyInstalled:boolean
  readonly sources:readonly AgySourceView[]
  readonly recommendedMode:AgyCredentialMode|null
  readonly canSync:boolean
  readonly destination:string|null
  readonly syncedDestination:string|null
  readonly syncedMode:AgyCredentialMode|null
}
/** What a provider row borrows from its signed-in account: usage, Sync and Remove (which also clears the credential). */
export interface AccountSlot {readonly account:AccountRow;readonly usage:ReactNode;readonly busy:boolean;readonly remove?:()=>void;readonly sync?:()=>void
  /** Renew a broken or expired Claude/Codex sign-in in place (and refresh the open SSH host's copy). */
  readonly relogin?:()=>void}
interface AccountList {accounts:readonly AccountRow[];can_sync:boolean;destination:string|null}
interface UsageWindow {label:string;used_percent:number;resets_at:string|null;detail:string|null}
interface SyncReview {review:string;account_id:string;destination:string}
interface AgySyncReview {review:string;mode:AgyCredentialMode;label:string;headless_safe:boolean;destination:string}
interface AccountAction {kind:'sync'|'remove';account:AccountRow;phase:'reviewing'|'ready'|'working'|'done';review?:SyncReview;error:string}
interface AgySyncAction {mode:AgyCredentialMode;label:string;phase:'reviewing'|'ready'|'working'|'done';review?:AgySyncReview;error:string}

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

export function AgyCredentialCard({
  status,
  selectedMode,
  onSelectMode,
  busy,
  onSync,
}: {
  readonly status: AgyStatusView
  readonly selectedMode: AgyCredentialMode | ''
  readonly onSelectMode: (mode: AgyCredentialMode) => void
  readonly busy: boolean
  readonly onSync: () => void
}) {
  const availableSources = status.sources.filter(source => source.available)
  const chosen = availableSources.find(source => source.mode === selectedMode) ?? availableSources[0]
  return <div className="provcard provider-account provider-account--agy" aria-label="Antigravity CLI credentials">
    <div className="provcard__main">
      <span className="provcard__text">
        <span className="provcard__title">
          <span className="provcard__name">Antigravity (agy)</span>
          <span className="provcard__badge provcard__badge--quiet">{status.agyInstalled ? 'CLI ready' : 'auto-installs'}</span>
          {status.syncedDestination && status.syncedMode && <span className="provcard__badge provcard__badge--quiet">synced · {status.syncedMode}</span>}
        </span>
        <span className="provcard__model">
          {availableSources.length > 0
            ? availableSources.map(source => source.label).join(' · ')
            : 'No local agy credentials detected yet'}
        </span>
        <span className="provcard__meta">
          {chosen
            ? chosen.detail
            : 'Sign in with `agy acp --login`, run `gcloud auth application-default login`, or save a Gemini API-key account above.'}
        </span>
      </span>
      {status.canSync && availableSources.length > 0 && <span className="provcard__tools">
        {availableSources.length > 1 && (
          <select
            className="palette__in"
            aria-label="Antigravity credential source"
            value={chosen?.mode ?? ''}
            disabled={busy}
            onChange={event => {
              const value = event.target.value
              if (value === 'adc' || value === 'gemini_api_key' || value === 'keyring') onSelectMode(value)
            }}
          >
            {availableSources.map(source => (
              <option key={source.mode} value={source.mode}>
                {source.label}
              </option>
            ))}
          </select>
        )}
        <button className="btn" disabled={busy || !chosen} onClick={onSync}>Sync to SSH</button>
      </span>}
    </div>
  </div>
}

export function ProviderAccountsCenter({accounts,onAccountLogin,sessionKey,turnActive,changed,rows,shownProfiles}:{accounts?:XerxesBridge['accounts'];onAccountLogin?:XerxesBridge['onAccountLogin'];sessionKey:string;turnActive:boolean;changed:()=>void
  /** The one Providers list: the caller renders each profile row with its account slot. */
  rows?:(slots:ReadonlyMap<string,AccountSlot>)=>ReactNode
  /** Profiles the caller already shows; their accounts are not listed a second time. */
  shownProfiles?:ReadonlySet<string>}) {
  const alive=useRef(true), [list,setList]=useState<AccountList|null>(null), [usage,setUsage]=useState<Record<string,unknown>>({})
  const [agyStatus,setAgyStatus]=useState<AgyStatusView|null>(null), [agyMode,setAgyMode]=useState<AgyCredentialMode|''>(''), [agyAction,setAgyAction]=useState<AgySyncAction|null>(null), agyRequest=useRef(0)
  const [agyLogin,setAgyLogin]=useState<{loginId:string;url:string;code:string;submitting:boolean;error:string}|null>(null)
  const [kind,setKind]=useState<'claude'|'codex'|'gemini_subscription'|'gemini'>('claude'), [label,setLabel]=useState(''), [busy,setBusy]=useState(false), [error,setError]=useState('')
  const [action,setAction]=useState<AccountAction|null>(null), actionRequest=useRef(0)
  const closeAction=()=>{actionRequest.current+=1;setAction(null)}
  const closeAgyAction=()=>{agyRequest.current+=1;setAgyAction(null)}
  const loginId=useRef(''), [loginUrl,setLoginUrl]=useState(''), [signingIn,setSigningIn]=useState(false)
  useEffect(()=>{alive.current=true;return()=>{alive.current=false}},[])
  useEffect(()=>onAccountLogin?.(event=>{
    if(alive.current&&event.login_id===loginId.current)setLoginUrl(event.url)
  }),[onAccountLogin])
  const load=async()=>{
    if(!accounts)return
    const [result, agy] = await Promise.all([
      accounts('list',{}) as Promise<AccountList & {ok?:boolean;error?:string}>,
      accounts('agy-status',{}).catch(()=>null) as Promise<(AgyStatusView & {ok?:boolean})|null>,
    ])
    if(result.ok===false)throw new Error(result.error||'Could not load provider accounts')
    if(!alive.current)return
    setList(result)
    if(agy&&Array.isArray(agy.sources)){
      setAgyStatus(agy)
      setAgyMode(current=>{
        if(current&&agy.sources.some(source=>source.mode===current&&source.available))return current
        return agy.recommendedMode??''
      })
    }
    void Promise.all(result.accounts.map(async account=>{
      try{const report=await accounts('usage',{accountId:account.id});if(alive.current)setUsage(current=>({...current,[account.id]:report}))}
      catch(error){if(alive.current)setUsage(current=>({...current,[account.id]:{status:'unknown',reason:desktopError(error)}}))}
    }))
  }
  const reviewAgySync=async(mode:AgyCredentialMode)=>{
    if(!accounts)return
    const source=agyStatus?.sources.find(item=>item.mode===mode)
    const label=source?.label??mode
    const request=++agyRequest.current
    setAgyAction({mode,label,phase:'reviewing',error:''})
    try {
      const result=await accounts('agy-sync-review',{mode,...(source?.accountId?{accountId:source.accountId}:{})}) as AgySyncReview & {ok?:boolean;error?:string}
      if(result.ok===false||typeof result.review!=='string')throw new Error(result.error||'Could not review Antigravity transfer')
      if(alive.current&&agyRequest.current===request)setAgyAction({mode,label:result.label||label,phase:'ready',review:result,error:''})
    } catch(error) {
      if(alive.current&&agyRequest.current===request)setAgyAction({mode,label,phase:'ready',error:desktopError(error)})
    }
  }
  const confirmAgySync=async()=>{
    if(!agyAction||!accounts)return
    if(agyAction.phase==='done'){closeAgyAction();return}
    if(agyAction.phase!=='ready')return
    if(!agyAction.review){void reviewAgySync(agyAction.mode);return}
    const request=agyRequest.current
    setAgyAction({...agyAction,phase:'working',error:''})
    try {
      const result=await accounts('agy-sync',{review:agyAction.review.review,confirm:true}) as {ok?:boolean;error?:string}
      if(result.ok===false)throw new Error(result.error||'Could not transfer Antigravity credentials')
      if(alive.current&&agyRequest.current===request)setAgyAction({...agyAction,phase:'done',error:''})
      await load();changed()
    } catch(error) {
      if(alive.current&&agyRequest.current===request)setAgyAction({mode:agyAction.mode,label:agyAction.label,phase:'ready',error:desktopError(error)})
    }
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
  const [notice,setNotice]=useState('')
  const relogin=(account:AccountRow)=>run(async()=>{
    setSigningIn(true);setNotice('')
    loginId.current=globalThis.crypto.randomUUID()
    setLoginUrl('')
    try {
      const result=await accounts!('relogin',{accountId:account.id,loginId:loginId.current}) as {ok?:boolean;error?:string;synced?:string|null;sync_error?:string}
      if(result.ok===false)throw new Error(result.error||'Sign-in failed')
      setLoginUrl('');loginId.current=''
      if(alive.current)setNotice(result.synced?`${account.label} signed in again and synced to ${result.synced}.`
        :result.sync_error?`${account.label} signed in again on this Mac. Sync to SSH failed: ${result.sync_error}`
        :`${account.label} signed in again.`)
      await load();changed()
    } catch(error) {
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
  /** Usage lines for one account. API-key profiles with no quota adapter say nothing rather than explain why. */
  const usageOf=(account:AccountRow):ReactNode=>{
    const report=usage[account.id] as {status?:string;reason?:string;windows?:UsageWindow[];recorded?:RecordedUsage}|undefined
    const quiet=account.kind==='api_key'&&(!report||report.status==='unknown')&&!report?.recorded
    if(quiet)return null
    return <div className="provider-account__usage">
      {report?.windows?.map(window=><span key={window.label}>{formatAccountUsageWindow(window)}</span>)}
      {(!report||report.status==='unknown')&&<span>{report?.reason||'Loading usage…'}</span>}
      {report?.recorded&&<AccountRecordedUsage report={report.recorded}/>}
    </div>
  }
  const removable=(account:AccountRow)=>account.kind!=='api_key'||account.provider==='gemini'
  const slots=new Map<string,AccountSlot>((list?.accounts??[]).map(account=>[account.profileName,{
    account, usage:usageOf(account), busy,
    ...(removable(account)?{remove:()=>{actionRequest.current+=1;setAction({kind:'remove',account,phase:'ready',error:''})}}:{}),
    ...(list?.can_sync?{sync:()=>void reviewSync(account)}:{}),
    ...(account.kind!=='api_key'?{relogin:()=>void relogin(account)}:{}),
  }]))
  const merged=Boolean(rows)
  const unlisted=(list?.accounts??[]).filter(account=>!shownProfiles?.has(account.profileName))
  const startAgyLogin=()=>run(async()=>{
    if(!accounts)return
    setAgyLogin(null)
    const started=await accounts('agy-login-start',{}) as {ok?:boolean;login_id?:string;url?:string;error?:string}
    if(started.ok===false||typeof started.login_id!=='string'||typeof started.url!=='string'){
      throw new Error(started.error||'Could not start Antigravity Google sign-in')
    }
    if(alive.current)setAgyLogin({loginId:started.login_id,url:started.url,code:'',submitting:false,error:''})
  })
  const completeAgyLogin=async()=>{
    if(!accounts||!agyLogin||agyLogin.submitting)return
    const code=agyLogin.code.trim()
    if(!code)return
    setAgyLogin(current=>current?{...current,submitting:true,error:''}:null)
    try{
      const res=await accounts('agy-login-complete',{loginId:agyLogin.loginId,code}) as {ok?:boolean;error?:string}
      if(res.ok===false)throw new Error(res.error||'Could not complete Antigravity sign-in')
      if(alive.current){
        setAgyLogin(null)
        setNotice('Signed in to Gemini Subscription (Antigravity) on this computer.')
      }
      await load();changed()
    }catch(error){
      if(alive.current)setAgyLogin(current=>current?{...current,submitting:false,error:desktopError(error)}:null)
    }
  }
  const cancelAgyLogin=()=>{
    if(!agyLogin)return
    const id=agyLogin.loginId
    setAgyLogin(null)
    void accounts?.('agy-login-cancel',{loginId:id}).catch(()=>{})
  }
  const agySubscriptionSignedIn=Boolean(agyStatus?.sources.some(source=>source.mode==='keyring'&&source.available))
  return <section className="provider-accounts" aria-label={merged?'Providers':'Provider accounts'}>
    <div className="row__t">{merged?'Providers':'Provider accounts'}</div>
    <p className="modal__sub">{merged
      ? 'Sign in to Claude, Codex, or Gemini Subscription (Antigravity), or save a Gemini API key. Other API keys and endpoints use Add provider below.'
      : 'One account center for this Mac and every SSH workspace. Each Claude/Codex login has isolated credentials. Gemini Subscription signs in via Antigravity (agy) and syncs to SSH hosts; native Gemini is API-key only (no Gemini OAuth quota readback).'}</p>
    <div className="provider-accounts__login">
      <select className="palette__in" aria-label="Account provider" value={kind} onChange={event=>{const value=event.target.value;if(value==='claude'||value==='codex'||value==='gemini_subscription'||value==='gemini')setKind(value)}} disabled={busy}>
        <option value="claude">Claude</option>
        <option value="codex">Codex</option>
        <option value="gemini_subscription">Gemini Subscription (Antigravity)</option>
        <option value="gemini">Gemini (API key)</option>
      </select>
      {(kind==='claude'||kind==='codex')&&<><input className="palette__in" aria-label="OAuth account label" value={label} onChange={event=>setLabel(event.target.value)} placeholder="Account label (work, personal…)" disabled={busy}/>
      <button className="btn" disabled={busy||!accounts||!label.trim()} onClick={login}>Sign in</button></>}
      {kind==='gemini_subscription'&&<>
        <button className="btn" disabled={busy||!accounts||Boolean(agyLogin?.submitting)} onClick={startAgyLogin}>
          {agySubscriptionSignedIn?'Re-sign in with Google':'Sign in with Google'}
        </button>
        {agyStatus?.canSync&&agySubscriptionSignedIn&&(
          <button className="btn" disabled={busy||!accounts} onClick={()=>void reviewAgySync('keyring')}>Sync to SSH</button>
        )}
      </>}
    </div>
    {kind==='gemini_subscription'&&<div className="provider-accounts__gemini" aria-label="Gemini Subscription (Antigravity)">
      <p>{agySubscriptionSignedIn
        ? 'Signed in to Gemini Subscription (Antigravity) on this computer. Transfer it to any SSH workspace below or from the Antigravity runtime banner.'
        : 'Sign in with Google via Antigravity CLI (`agy`) to use Gemini Subscription models locally and transfer credentials securely to SSH hosts.'}</p>
      {agyLogin&&<>
        <ProviderLoginLink url={agyLogin.url} busy={agyLogin.submitting} write={text=>accounts!('copy-login-link',{url:text})}/>
        <form onSubmit={event=>{event.preventDefault();void completeAgyLogin()}} style={{display:'flex',gap:'8px',marginTop:'8px',alignItems:'center'}}>
          <input
            className="palette__in"
            aria-label="Antigravity authorization code"
            autoComplete="off"
            spellCheck={false}
            placeholder="Paste authorization code or callback URL from Google…"
            value={agyLogin.code}
            disabled={agyLogin.submitting}
            onChange={event=>setAgyLogin(current=>current?{...current,code:event.target.value}:null)}
            style={{flex:1}}
          />
          <button className="btn" type="submit" disabled={agyLogin.submitting||!agyLogin.code.trim()}>
            {agyLogin.submitting?'Completing…':'Complete sign-in'}
          </button>
          <button className="btn" type="button" disabled={agyLogin.submitting} onClick={cancelAgyLogin}>Cancel</button>
        </form>
        {agyLogin.error&&<p role="alert" className="studio-error">{agyLogin.error}</p>}
      </>}
    </div>}
    {kind==='gemini'&&<GeminiAccountForm busy={busy||!accounts} save={saveGemini}/>}
    {(signingIn||loginUrl)&&<ProviderLoginLink url={loginUrl} busy={signingIn} write={text=>accounts!('copy-login-link',{url:text})}/>}
    {error&&<p role="alert" className="studio-error">{error}</p>}
    {notice&&<p role="status" className="studio-muted">{notice}</p>}
    {rows?.(slots)}
    {unlisted.map(account=>{
      return <div className="provcard provider-account" key={account.id}>
        <div className="provcard__main">
          <span className="provcard__text">
            <span className="provcard__title"><span className="provcard__name">{account.label}</span><span className="provcard__badge provcard__badge--quiet">{account.kind}</span></span>
            <span className="provcard__model">{account.profileName} · {account.model||'model chosen when used'}</span>
            <span className="provcard__meta">{account.provider} · {account.available?'available on this workspace':'not available on this workspace yet'}</span>
          </span>
          <span className="provcard__tools">
            <button className="btn" disabled={busy||turnActive||!account.available} onClick={()=>void activate(account.id)}>Use</button>
            {list?.can_sync&&<button className="btn" disabled={busy} onClick={()=>void reviewSync(account)}>Sync to SSH</button>}
            {removable(account)&&<button className="btn" disabled={busy} onClick={()=>slots.get(account.profileName)?.remove?.()}>Remove</button>}
          </span>
        </div>
        {usageOf(account)}
      </div>
    })}
    {!merged&&list&&!list.accounts.length&&<p className="studio-muted">No accounts yet. Sign in above or choose Gemini to save a named API-key account.</p>}
    {agyStatus && <AgyCredentialCard
      status={agyStatus}
      selectedMode={agyMode}
      onSelectMode={setAgyMode}
      busy={busy}
      onSync={()=>{
        const chosen=agyMode||agyStatus.recommendedMode
        if(chosen)void reviewAgySync(chosen)
      }}
    />}
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
    {agyAction&&<ActionDialog
      title={agyAction.phase==='done'?'Antigravity (agy) synced':`Sync Antigravity (${agyAction.label}) to SSH`}
      confirmLabel={agyAction.phase==='done'?'Done':!agyAction.review&&agyAction.phase==='ready'?'Review transfer again':'Confirm sync'}
      busyLabel="Transferring agy credentials…"
      busy={agyAction.phase==='working'} disabled={agyAction.phase==='reviewing'} done={agyAction.phase==='done'} error={agyAction.error}
      onConfirm={()=>void confirmAgySync()} onCancel={closeAgyAction}>
      {agyAction.phase==='reviewing'?<p role="status">Preparing transfer review for <strong>Antigravity ({agyAction.label})</strong>…</p>
        :agyAction.phase==='done'?<p role="status"><strong>Antigravity ({agyAction.label})</strong> credentials transferred to <strong>{agyAction.review?.destination||agyStatus?.destination||'this SSH host'}</strong>.</p>
        :<><p>Transfer <strong>Antigravity ({agyAction.label})</strong> credentials to <strong>{agyAction.review?.destination||agyStatus?.destination||'this SSH host'}</strong>?</p>
          <p className="studio-muted">{agyAction.mode==='adc'
            ?'Transfers ~/.config/gcloud/application_default_credentials.json (mode 0600), enables ADC in ~/.gemini/config/config.json, and writes owner-only agy runtime env settings over SSH stdin.'
            :agyAction.mode==='gemini_api_key'
              ?'Transfers the Gemini API key into owner-only ~/.xerxes/provider-accounts/agy/env.json (mode 0600) and sets modelProvider: "gemini" for agy over SSH stdin.'
              :'Transfers your local Gemini Subscription (Antigravity) OAuth token over SSH stdin into ~/.gemini/antigravity-cli/antigravity-oauth-token (mode 0600) and the host keyring when available. Works on headless Linux/TPU SSH hosts.'}</p></>}
    </ActionDialog>}
  </section>
}
