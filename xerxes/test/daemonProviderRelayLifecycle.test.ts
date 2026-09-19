// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import {expect,test} from 'bun:test'
import {mkdtemp,rm} from 'node:fs/promises'
import {tmpdir} from 'node:os'
import {join} from 'node:path'
import {ProfileStore} from '../src/bridge/profiles.js'
import {DaemonProviderRelays} from '../src/daemon/providerRelays.js'
import type {LlmClient} from '../src/llms/client.js'
const policy=()=>({consent:true,destination:'fixture',workspace:'/fixture',profile:'local',model:'gpt-4o',expires_at:Date.now()+60_000,max_requests:1,max_output_tokens:1024,max_concurrent:1})
const frame={op:'next',id:'first',request:{model:'gpt-4o',messages:[{role:'user',content:'Synthetic request'}]}}
async function fixture(client:LlmClient){
  const dir=await mkdtemp(join(tmpdir(),'xr-relay-lifecycle-'))
  const profiles=new ProfileStore(join(dir,'profiles.json'))
  profiles.save({name:'local',provider:'openai',baseUrl:'https://provider.invalid/v1',apiKey:'synthetic',model:'gpt-4o'})
  const relays=new DaemonProviderRelays(profiles,()=>client)
  return {relays,async close(){relays.close();await rm(dir,{recursive:true,force:true})}}
}

test('revocation retains noncooperative work until settlement while unrelated grants can be retired',async()=>{
  const started=Promise.withResolvers<void>(),finish=Promise.withResolvers<void>()
  const f=await fixture({async *stream(){started.resolve();await finish.promise;yield {content:'late data must not escape'}}})
  const owner={}
  try{
    const grant=f.relays.authorize(owner,policy())
    const pending=f.relays.next(owner,grant.id,frame)
    await started.promise
    f.relays.revoke(owner,grant.id)
    expect(await pending).toEqual({error:'grant_revoked'})
    expect(f.relays.status(owner,grant.id)).toMatchObject({status:'revoked',activeRequests:1})
    expect(f.relays.hasLiveGrants()).toBe(true)
    for(let i=0;i<160;i++){const other=f.relays.authorize(owner,policy());f.relays.revoke(owner,other.id)}
    expect(f.relays.status(owner,grant.id)).toMatchObject({status:'revoked',activeRequests:1})
    finish.resolve();await Bun.sleep(5)
    expect(f.relays.hasLiveGrants()).toBe(false)
    expect(f.relays.status(owner,grant.id)).toMatchObject({status:'revoked',activeRequests:0})
    expect(await f.relays.next(owner,grant.id,frame)).toEqual({error:'grant_revoked'})
  }finally{finish.resolve();await f.close()}
})

test('expired grants free their slots while preserving expiration diagnostics and owner isolation',async()=>{
  const f=await fixture({async *stream(){throw new Error('Must not execute expired work')}})
  const owner={},other={}
  try{
    const grants=Array.from({length:128},()=>f.relays.authorize(owner,{...policy(),expires_at:Date.now()+1000}))
    await Bun.sleep(1010)
    expect(f.relays.hasLiveGrants()).toBe(false)
    const fresh=f.relays.authorize(owner,policy())
    expect(f.relays.status(owner,fresh.id).status).toBe('active')
    const last=grants.at(-1)!
    expect(f.relays.status(owner,last.id)).toMatchObject({status:'expired',activeRequests:0})
    expect(await f.relays.next(owner,last.id,frame)).toEqual({error:'grant_expired'})
    expect(()=>f.relays.status(other,last.id)).toThrow('unavailable')
    f.relays.disconnect(owner)
    expect(()=>f.relays.status(owner,last.id)).toThrow('unavailable')
  }finally{await f.close()}
})

test('an exhausted grant can deliver its final buffered output before explicit revocation',async()=>{
  const f=await fixture({async *stream(){for(let i=0;i<300;i++)yield {content:String(i)}}})
  const owner={}
  try{
    const grant=f.relays.authorize(owner,policy())
    const first=await f.relays.next(owner,grant.id,frame)
    expect(first).toHaveProperty('done',false)
    expect(f.relays.status(owner,grant.id).status).toBe('exhausted')
    let output='deltas' in first?first.deltas?.map(d=>d.content??'').join('')??'':''
    for(let i=0;i<20;i++){
      const part=await f.relays.next(owner,grant.id,{op:'next',id:'first'})
      expect(part).not.toHaveProperty('error')
      if('deltas' in part)output+=part.deltas?.map(d=>d.content??'').join('')??''
      if('done' in part&&part.done)break
    }
    expect(output).toBe(Array.from({length:300},(_,i)=>String(i)).join(''))
    f.relays.revoke(owner,grant.id)
    expect(f.relays.status(owner,grant.id)).toMatchObject({status:'revoked',activeRequests:0})
  }finally{await f.close()}
})
