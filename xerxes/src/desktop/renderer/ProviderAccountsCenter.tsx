// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useId, useRef, useState, type ReactNode } from 'react'

import { desktopError } from './desktopRpc.js'
import { CopyButton } from './CopyButton.js'
import { SelectField } from './SelectField.js'
import { ActionDialog } from './ActionDialog.js'
import { localMachineLabel } from './runModeView.js'
import type { XerxesBridge } from './types.js'

export interface AccountRow {id:string;kind:string;label:string;profileName:string;provider:string;model:string;available:boolean}
export type AgyCredentialMode = 'adc' | 'gemini_api_key' | 'keyring'

/** Human-readable badge label for a provider account kind token. */
export function accountKindLabel(kind: string): string {
  switch (kind) {
    case 'api_key': return 'API key'
    case 'claude': return 'Claude sign-in'
    case 'claude-code': return 'Claude Code sign-in'
    case 'codex': return 'Codex sign-in'
    case 'sign-in': return 'Sign-in'
    default: return kind.replace(/[_-]+/g, ' ')
  }
}

/** Human-readable label for an Antigravity credential mode token. */
export function agyModeLabel(mode: string): string {
  switch (mode) {
    case 'adc': return 'Google Cloud ADC'
    case 'gemini_api_key': return 'Gemini API key'
    case 'keyring': return 'Antigravity sign-in'
    default: return mode.replace(/[_-]+/g, ' ')
  }
}
/** Sources that may be copied to an SSH host. Antigravity's own sign-in (`keyring`) never leaves the computer it was made on. */
export type AgyTransferMode = Exclude<AgyCredentialMode, 'keyring'>
export interface AgySourceView {readonly mode:AgyCredentialMode;readonly label:string;readonly available:boolean;readonly detail:string;readonly headlessSafe:boolean;readonly transferable?:boolean;readonly accountId?:string}
/** Narrow a source to one Xerxes may transfer; guards against an older main process still offering `keyring`. */
export function isTransferableAgySource(source:AgySourceView):source is AgySourceView & {readonly mode:AgyTransferMode} {
  return source.available && source.mode!=='keyring' && source.transferable!==false
}
/** The source a one-click "transfer to SSH" uses, or null when nothing transferable is available. */
export function agyQuickTransferSource(status:AgyStatusView|null|undefined):(AgySourceView & {readonly mode:AgyTransferMode})|null {
  if(!status?.canSync)return null
  const sources=status.sources.filter(isTransferableAgySource)
  return sources.find(source=>source.mode===status.recommendedMode)??sources[0]??null
}
export interface AgyStatusView {
  readonly agyInstalled:boolean
  readonly sources:readonly AgySourceView[]
  readonly recommendedMode:AgyTransferMode|null
  readonly canSync:boolean
  readonly destination:string|null
  readonly syncedDestination:string|null
  readonly syncedMode:AgyTransferMode|null
}
/** What an agy credential transfer writes on the SSH host, for the confirmation dialog. */
export function agyTransferDetail(mode:AgyTransferMode):string {
  return mode==='adc'
    ?'Transfers ~/.config/gcloud/application_default_credentials.json (mode 0600), enables ADC in ~/.gemini/config/config.json, and writes owner-only agy runtime env settings over SSH stdin.'
    :'Transfers the Gemini API key into owner-only ~/.xerxes/provider-accounts/agy/env.json (mode 0600) and sets modelProvider: "gemini" for agy over SSH stdin.'
}
/** What a provider row borrows from its signed-in account: usage, Sync and Remove (which also clears the credential). */
export interface AccountSlot {readonly account:AccountRow;readonly usage:ReactNode;readonly busy:boolean;readonly remove?:()=>void;readonly sync?:()=>void
  /** Renew a broken or expired Claude/Codex sign-in in place (and refresh the open SSH host's copy). */
  readonly relogin?:()=>void}
interface AccountList {accounts:readonly AccountRow[];can_sync:boolean;destination:string|null}
interface UsageWindow {label:string;used_percent:number;resets_at:string|null;detail:string|null}
interface SyncReview {review:string;account_id:string;destination:string}
interface AgySyncReview {review:string;mode:AgyTransferMode;label:string;headless_safe:boolean;destination:string}
interface AccountAction {kind:'sync'|'remove';account:AccountRow;phase:'reviewing'|'ready'|'working'|'done';review?:SyncReview;error:string}
interface AgySyncAction {mode:AgyTransferMode;label:string;phase:'reviewing'|'ready'|'working'|'done';review?:AgySyncReview;error:string}

export function formatAccountUsageWindow(window: UsageWindow): string {
  if (!Number.isFinite(window.used_percent)) return `${window.label}: usage unavailable`
  const reset = window.resets_at && Number.isFinite(Date.parse(window.resets_at)) ? ` · resets ${new Date(window.resets_at).toLocaleString()}` : ''
  return `${window.label}: ${Math.round(window.used_percent)}% used${reset}`
}

export const ACCOUNT_PROVIDER_OPTIONS = [
  { value: 'claude', label: 'Claude' },
  { value: 'codex', label: 'Codex' },
  { value: 'gemini_subscription', label: 'Gemini Subscription (Antigravity)' },
  { value: 'gemini', label: 'Gemini (API key)' },
] as const

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
  const baseId = useId()
  const errId = `${baseId}-err`
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
    <div className="field">
      <label htmlFor={`${baseId}-label`}>Account label</label>
      <input id={`${baseId}-label`} className="palette__in" aria-label="Gemini account label" autoComplete="off" value={label} onChange={event=>setLabel(event.target.value)} placeholder="Account label (work, personal…)" maxLength={80} disabled={busy||pending} aria-invalid={error ? true : undefined} {...(error ? { 'aria-describedby': errId } : {})} required/>
    </div>
    <div className="field">
      <label htmlFor={`${baseId}-key`}>Gemini API key</label>
      <input id={`${baseId}-key`} className="palette__in" type="password" aria-label="Gemini API key" autoComplete="off" spellCheck={false} autoCapitalize="none" value={apiKey} onChange={event=>setApiKey(event.target.value)} placeholder="Gemini API key" disabled={busy||pending} aria-invalid={error ? true : undefined} {...(error ? { 'aria-describedby': errId } : {})} required/>
    </div>
    <div className="field">
      <label htmlFor={`${baseId}-model`}>Model ID</label>
      <input id={`${baseId}-model`} className="palette__in" aria-label="Gemini model id" autoComplete="off" spellCheck={false} value={model} onChange={event=>setModel(event.target.value)} placeholder="Model id, e.g. gemini-2.5-flash" maxLength={512} disabled={busy||pending} aria-invalid={error ? true : undefined} {...(error ? { 'aria-describedby': errId } : {})} required/>
    </div>
    <button className="btn btn--solid" type="submit" disabled={busy||pending||!label.trim()||!apiKey.trim()||!model.trim()}>{pending?'Saving Gemini account…':'Save Gemini account'}</button>
    {error&&<p id={errId} role="alert" className="studio-error">{error}</p>}
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
  readonly selectedMode: AgyTransferMode | ''
  readonly onSelectMode: (mode: AgyTransferMode) => void
  readonly busy: boolean
  readonly onSync: () => void
}) {
  const availableSources = status.sources.filter(source => source.available)
  const transferSources = status.sources.filter(isTransferableAgySource)
  const chosen = transferSources.find(source => source.mode === selectedMode) ?? transferSources[0]
  const shown = chosen ?? availableSources[0]
  return <div className="provcard provider-account provider-account--agy" aria-label="Antigravity CLI credentials">
    <div className="provcard__main">
      <span className="provcard__text">
        <span className="provcard__title">
          <span className="provcard__name">Antigravity (agy)</span>
          <span className="provcard__badge provcard__badge--quiet">{status.agyInstalled ? 'CLI ready' : 'Auto-installs'}</span>
          {status.syncedDestination && status.syncedMode && <span className="provcard__badge provcard__badge--quiet">Synced · {agyModeLabel(status.syncedMode)}</span>}
        </span>
        <span className="provcard__model">
          {availableSources.length > 0
            ? availableSources.map(source => source.label).join(' · ')
            : 'No local agy credentials detected yet'}
        </span>
        <span className="provcard__meta">
          {shown
            ? shown.detail
            : <>Sign in with <code>agy acp --login</code>, run <code>gcloud auth application-default login</code>, or save a Gemini API-key account above.</>}
        </span>
      </span>
      {status.canSync && transferSources.length > 0 && <span className="provcard__tools">
        {transferSources.length > 1 && (
          <SelectField
            ariaLabel="Antigravity credential source"
            listLabel="Antigravity credential source"
            value={chosen?.mode ?? ''}
            disabled={busy}
            options={transferSources.map(source => ({ value: source.mode, label: source.label }))}
            onChange={value => {
              if (value === 'adc' || value === 'gemini_api_key') onSelectMode(value)
            }}
          />
        )}
        <button type="button" className="btn" disabled={busy || !chosen} onClick={onSync}>Sync to SSH</button>
      </span>}
    </div>
  </div>
}

export function ProviderAccountsCenter({accounts,onAccountLogin,sessionKey,turnActive,changed,rows,shownProfiles}:{accounts?:XerxesBridge['accounts'];onAccountLogin?:XerxesBridge['onAccountLogin'];sessionKey:string;turnActive:boolean;changed:()=>void
  /** The one Providers list: the caller renders each profile row with its account slot. */
  rows?:(slots:ReadonlyMap<string,AccountSlot>)=>ReactNode
  /** Profiles the caller already shows; their accounts are not listed a second time. */
  shownProfiles?:ReadonlySet<string>}) {
  const alive=useRef(true), loadEpoch=useRef(0), [list,setList]=useState<AccountList|null>(null), [usage,setUsage]=useState<Record<string,unknown>>({})
  const [agyStatus,setAgyStatus]=useState<AgyStatusView|null>(null), [agyMode,setAgyMode]=useState<AgyTransferMode|''>(''), [agyAction,setAgyAction]=useState<AgySyncAction|null>(null), agyRequest=useRef(0)
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
    const epoch=++loadEpoch.current
    const [result, agy] = await Promise.all([
      accounts('list',{}) as Promise<AccountList & {ok?:boolean;error?:string}>,
      accounts('agy-status',{}).catch(()=>null) as Promise<(AgyStatusView & {ok?:boolean})|null>,
    ])
    if(result.ok===false)throw new Error(result.error||'Could not load provider accounts')
    if(!alive.current||loadEpoch.current!==epoch)return
    setList(result)
    if(agy&&Array.isArray(agy.sources)){
      setAgyStatus(agy)
      setAgyMode(current=>{
        if(current&&agy.sources.some(source=>source.mode===current&&source.available))return current
        return agy.recommendedMode??''
      })
    } else {
      setAgyStatus(null)
    }
    void Promise.all(result.accounts.map(async account=>{
      try{const report=await accounts('usage',{accountId:account.id});if(alive.current&&loadEpoch.current===epoch)setUsage(current=>({...current,[account.id]:report}))}
      catch(error){if(alive.current&&loadEpoch.current===epoch)setUsage(current=>({...current,[account.id]:{status:'unknown',reason:desktopError(error)}}))}
    }))
  }
  const reviewAgySync=async(mode:AgyTransferMode)=>{
    if(!accounts)return
    const source=agyStatus?.sources.find(item=>item.mode===mode)
    const label=source?.label??agyModeLabel(mode)
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
  useEffect(()=>{
    setList(null);setUsage({});setAgyStatus(null);setError('')
    void load().catch(error=>{if(alive.current)setError(desktopError(error))})
  },[accounts,sessionKey])
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
  const machine=localMachineLabel()
  const relogin=(account:AccountRow)=>run(async()=>{
    setSigningIn(true);setNotice('')
    loginId.current=globalThis.crypto.randomUUID()
    setLoginUrl('')
    try {
      const result=await accounts!('relogin',{accountId:account.id,loginId:loginId.current}) as {ok?:boolean;error?:string;synced?:string|null;sync_error?:string}
      if(result.ok===false)throw new Error(result.error||'Sign-in failed')
      setLoginUrl('');loginId.current=''
      if(alive.current)setNotice(result.synced?`${account.label} signed in again and synced to ${result.synced}.`
        :result.sync_error?`${account.label} signed in again on ${machine}. Sync to SSH failed: ${result.sync_error}`
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
  const recheckAgySignIn=()=>run(async()=>{await load();changed()})
  const agySubscriptionSignedIn=Boolean(agyStatus?.sources.some(source=>source.mode==='keyring'&&source.available))
  return <section className="provider-accounts" aria-label={merged?'Providers':'Provider accounts'}>
    <div className="row__t">{merged?'Providers':'Provider accounts'}</div>
    <p className="modal__sub">{merged
      ? 'Sign in to Claude, Codex, or Gemini Subscription (Antigravity), or save a Gemini API key. Other API keys and endpoints use Add provider below.'
      : `One account center for ${machine} and every SSH workspace. Each Claude/Codex login has isolated credentials. Gemini Subscription signs in via Antigravity (agy) on each computer that runs it; native Gemini is API-key only (no Gemini OAuth quota readback).`}</p>
    <form className="provider-accounts__login" onSubmit={event=>{event.preventDefault();if((kind==='claude'||kind==='codex')&&!busy&&accounts&&label.trim())void login()}}>
      <SelectField
        className="provider-accounts__select"
        ariaLabel="Account provider"
        listLabel="Account provider"
        value={kind}
        disabled={busy}
        options={ACCOUNT_PROVIDER_OPTIONS}
        onChange={value => {
          if (value === 'claude' || value === 'codex' || value === 'gemini_subscription' || value === 'gemini') setKind(value)
        }}
      />
      {(kind==='claude'||kind==='codex')&&<><input className="palette__in" aria-label="OAuth account label" value={label} onChange={event=>setLabel(event.target.value)} placeholder="Account label (work, personal…)" disabled={busy}/>
      <button type="submit" className="btn btn--solid" disabled={busy||!accounts||!label.trim()}>Sign in</button></>}
      {kind==='gemini_subscription'&&<>
        <button type="button" className="btn" disabled={busy||!accounts} onClick={recheckAgySignIn}>Check again</button>
      </>}
    </form>
    {kind==='gemini_subscription'&&<div className="provider-accounts__gemini" aria-label="Gemini Subscription (Antigravity)">
      <p>{agySubscriptionSignedIn
        ? 'Signed in to Gemini Subscription (Antigravity) on this computer. The sign-in stays here: on an SSH host, run the command below there. For headless or third-party harness usage, Google recommends a Gemini API key or Google Cloud ADC.'
        : 'Antigravity signs in on its own. Run this in a terminal, finish the Google sign-in in your browser, then select Check again. Xerxes never sees your Google sign-in. For headless or third-party harness usage, Google recommends a Gemini API key or Google Cloud ADC.'}</p>
      <div className="provider-accounts__signin-link"><input aria-label="Antigravity sign-in command" readOnly value="agy acp --login" spellCheck={false} onFocus={event=>event.currentTarget.select()}/><CopyButton text="agy acp --login" label="Copy command"/></div>
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
            <span className="provcard__title"><span className="provcard__name">{account.label}</span><span className="provcard__badge provcard__badge--quiet">{accountKindLabel(account.kind)}</span></span>
            <span className="provcard__model">{account.profileName} · {account.model||'model chosen when used'}</span>
            <span className="provcard__meta">{account.provider} · {account.available?'available on this workspace':'not available on this workspace yet'}</span>
          </span>
          <span className="provcard__tools">
            <button type="button" className="btn" disabled={busy||turnActive||!account.available} onClick={()=>void activate(account.id)}>Use</button>
            {list?.can_sync&&<button type="button" className="btn" disabled={busy} onClick={()=>void reviewSync(account)}>Sync to SSH</button>}
            {removable(account)&&<button type="button" className="btn" disabled={busy} onClick={()=>slots.get(account.profileName)?.remove?.()}>Remove</button>}
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
      title={action.phase==='done'?(action.kind==='sync'?'Account synced':'Account removed'):(action.kind==='sync'?`Sync ${action.account.label} to SSH`:`Remove ${action.account.label}`)}
      confirmLabel={action.phase==='done'?'Done':action.kind==='remove'?'Remove account':!action.review&&action.phase==='ready'?'Review sync again':'Confirm sync'}
      busyLabel={action.kind==='sync'?'Syncing account…':'Removing account…'}
      busy={action.phase==='working'} disabled={action.phase==='reviewing'} done={action.phase==='done'} destructive={action.kind==='remove'&&action.phase!=='done'} error={action.error}
      onConfirm={()=>void confirmAction()} onCancel={closeAction}>
      {action.phase==='reviewing'?<p role="status">Preparing transfer review for <strong>{action.account.label}</strong>…</p>
        :action.phase==='done'?<p role="status">{action.kind==='sync'?<><strong>{action.account.label}</strong> synced to <strong>{action.review?.destination||list?.destination||'this SSH host'}</strong>.</>:<><strong>{action.account.label}</strong> was removed. Other accounts were not changed.</>}</p>
        :<><p>{action.kind==='sync'?<>Transfer <strong>{action.account.label}</strong> to <strong>{action.review?.destination||list?.destination||'this SSH host'}</strong>?</>:<>Remove <strong>{action.account.label}</strong> and its saved {action.account.kind==='api_key'?'API key':'sign-in'} from {machine}?</>}</p>
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
          <p className="studio-muted">{agyTransferDetail(agyAction.mode)}</p></>}
    </ActionDialog>}
  </section>
}
