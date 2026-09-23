// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useEffect, useRef, useState } from 'react'
import type { ShareableLocalProfile } from '../../protocol/shareableLocalProfiles.js'
import type { XerxesBridge } from './types.js'
import { desktopError } from './desktopRpc.js'

interface Review {
  review: string; destination: string; workspace: string; sessionKey: string;
  profiles: readonly ShareableLocalProfile[]; running: boolean; remoteProfile: string; model: string;
  source: string; expiresAt?: number; sharedProfiles: string[];
}
export function RemoteProviders({remote, changed}: {remote?: XerxesBridge['remote']; changed?: () => void}) {
  const [review,setReview] = useState<Review | null>(null)
  const [error,setError] = useState(''), [busy,setBusy] = useState(false), [editing,setEditing] = useState(false)
  const [selected,setSelected] = useState<string[]>([]), [primary,setPrimary] = useState('')
  const [minutes,setMinutes] = useState(480), [controlled,setControlled] = useState(false)
  const alive = useRef(false)
  const read = async (reset = false) => {
    if (!remote) throw new Error('Update the desktop app to use local providers over SSH.')
    const result = await remote('provider-review', {}) as Review & {ok?:boolean;error?:string}
    if (result.ok !== true || !Array.isArray(result.profiles)) throw new Error(result.error || 'Provider setup is unavailable. Reconnect and retry.')
    if (!alive.current) return
    setReview(result)
    if (reset) {
      const supported=result.profiles.filter(profile=>profile.supported).slice(0,32)
      setSelected(supported.map(profile=>profile.name))
      setPrimary(supported.find(profile=>profile.name===result.remoteProfile)?.name ?? supported[0]?.name ?? '')
      setControlled(false)
    }
  }
  useEffect(()=>{alive.current=true; void read(true).catch(error=>{if(alive.current)setError(desktopError(error))}); return()=>{alive.current=false}},[remote])
  const act = async (action: () => Promise<unknown>) => {
    setBusy(true);setError('')
    try {await action(); if(alive.current){await read();changed?.()}}
    catch(error){if(alive.current)setError(desktopError(error))}
    finally{if(alive.current)setBusy(false)}
  }
  const needsControlled = review?.profiles.some(profile=>selected.includes(profile.name)&&profile.providerControlledOutput)
  return <section className="remote-providers" aria-label="SSH provider source">
    <h3>Provider source for this conversation</h3>
    {error && <p className="studio-error" role="alert">{error} <button className="btn" disabled={busy} onClick={()=>void act(()=>read(true))}>Refresh setup</button></p>}
    {!review ? <p role="status">{error ? 'Provider setup could not be loaded.' : 'Reading local provider setup…'}</p> : <>
      <button className="btn" disabled={busy} onClick={()=>void act(()=>read())}>Refresh provider status</button>
      <p><strong>{review.source==='remote' ? 'Remote credentials' : review.source==='local' && (review.expiresAt ?? 0)>Date.now() ? 'Local credentials · access enabled' : 'Local credentials · access needs renewal'}</strong><br/>
        Code and tools: {review.destination} · {review.workspace}</p>
      <p className="studio-muted">{review.source==='remote' ? `This task uses the remote host’s saved setup (${review.remoteProfile || 'no profile reported'}). Local key changes do not update it. Choose local access below, or use the remote provider controls to keep that setup.` : `Provider requests and authentication run on this computer. Access ends ${review.expiresAt ? new Date(review.expiresAt).toLocaleString() : 'when disconnected'}, or when this workspace view closes. Reconnect requires review; remote credentials are never used as a fallback.`}</p>
      {!editing ? <div className="row__actions">
        <button className="btn" disabled={busy} onClick={()=>void act(async()=>{await read(true);setEditing(true)})}>Review local provider access</button>
        {review.source==='local' && <button className="btn" disabled={busy} onClick={()=>void act(()=>remote!('provider-revoke',{sessionKey:review.sessionKey}))}>Revoke local access</button>}
      </div> : <fieldset disabled={busy}>
        <legend>Share selected local providers with this task and its agents</legend>
        <p>Keys stay on this computer. Updated saved keys apply to new requests for the same provider and endpoint. Access lasts up to eight hours and is not saved across app restarts.</p>
        <div className="remote-providers__list">{review.profiles.map(profile=><label key={profile.name} className="remote-providers__choice">
          <input type="checkbox" checked={selected.includes(profile.name)} disabled={!profile.supported} onChange={event=>{const names=event.target.checked?[...selected,profile.name]:selected.filter(name=>name!==profile.name);setSelected(names);if(!names.includes(primary))setPrimary(names[0]??'')}}/>
          <span><strong>{profile.name}</strong> · {profile.model}<small>{profile.credentialSource}{!profile.supported ? ` · ${profile.setup}` : ''}</small></span>
        </label>)}</div>
        {!review.profiles.length && <p>No local profiles. Open a local workspace’s Models & Providers settings to add one.</p>}
        <label className="field">Primary profile<select value={primary} onChange={event=>setPrimary(event.target.value)}>{selected.map(name=><option key={name}>{name}</option>)}</select></label>
        <label className="field">Access duration<select value={minutes} onChange={event=>setMinutes(Number(event.target.value))}><option value={60}>1 hour</option><option value={240}>4 hours</option><option value={480}>8 hours</option></select></label>
        <p className="studio-muted">Per provider: 10,000 requests, 16 concurrent requests, and 32,768 output tokens per request where supported. Readiness is configured, not verified by a live provider call.</p>
        {needsControlled && <label className="remote-providers__choice"><input type="checkbox" checked={controlled} onChange={event=>setControlled(event.target.checked)}/><span>Allow subscription providers to control output length; the token cap does not apply to them.</span></label>}
        {review.running && <p role="status">This task or its agents are working. Wait until they finish before changing its provider source.</p>}
        <div className="row__actions"><button className="btn" disabled={review.running || !selected.length || !primary || selected.length>32 || Boolean(needsControlled&&!controlled)} onClick={()=>void act(async()=>{
          const result=await remote!('provider-share',{consent:true,review:review.review,profiles:selected,profile:primary,minutes,providerControlledOutput:controlled}) as {ok?:boolean;error?:string}
          if(result.ok!==true)throw new Error(result.error || 'Could not share local providers.')
          if(alive.current)setEditing(false)
        })}>Enable local access for this task</button><button className="btn" onClick={()=>setEditing(false)}>Cancel</button></div>
      </fieldset>}
    </>}
  </section>
}
