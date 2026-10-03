// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, realpath, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { Socket } from 'node:net'
import { JobStore } from '../src/cron/jobs.js'
import { DesktopProviderForwarding } from '../src/desktop/main/providerForwarding.js'
import { DaemonRpc } from '../src/desktop/main/daemon.js'
import { ProfileStore } from '../src/bridge/profiles.js'
import { DaemonServer } from '../src/daemon/server.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { RemoteProviderBindings } from '../src/daemon/remoteProviderBindings.js'
import type { CompletionRequest } from '../src/llms/client.js'
import { seedModelsDev } from './fixtures/modelsDev.js'

// Model capabilities come from models.dev at runtime; tests use its fixture.
seedModelsDev()

async function until(check:()=>boolean) { for(let i=0;i<300;i++){if(check())return;await Bun.sleep(10)}throw new Error('Timed out') }
async function fixture(preferred?: () => Promise<string | undefined>) {
  const dir=await realpath(await mkdtemp(join(tmpdir(),'xpf-'))), bindings=new RemoteProviderBindings(2000)
  const profiles=new ProfileStore(join(dir,'local-profiles.json')), seen:string[]=[], events:unknown[]=[]
  let fallback=0, hold=false, aborted=false
  const save=(key:string,baseUrl='https://fixture.invalid')=>profiles.save({name:'local-api',provider:'openai',model:'gpt-4o',apiKey:key,baseUrl})
  save('fixture-old-key')
  const localRuntime=new InMemoryDaemonRuntime(undefined,{currentProjectDirectory:dir,sessionDirectory:join(dir,'local-sessions')})
  const localServer=new DaemonServer({cronLeasePath:join(dir,'local-cron.lease'),cronStoreFactory:()=>new JobStore(join(dir,'local-jobs.json')),runtime:localRuntime,socketPath:join(dir,'l.sock'),projectDirectory:dir,profileStore:profiles,
    relayClientFactory:(_model,profile)=>({async *stream(_request:CompletionRequest,signal?:AbortSignal){
      seen.push(profile.api_key)
      if(hold){await new Promise<void>(resolve=>{if(signal?.aborted)resolve();else signal?.addEventListener('abort',()=>resolve(),{once:true})});aborted=true;return}
      yield {content:'LOCAL_REPLY'}
    }})})
  const runner=new AgentTurnRunner({model:'gpt-4o',tools:[],remoteProviderBindings:bindings,llm:{async *stream(){fallback++;yield {content:'UNAPPROVED'}}}})
  const remoteRuntime=new InMemoryDaemonRuntime(runner,{currentProjectDirectory:dir,sessionDirectory:join(dir,'remote-sessions'),model:'gpt-4o'})
  const remoteServer=new DaemonServer({cronLeasePath:join(dir,'remote-cron.lease'),cronStoreFactory:()=>new JobStore(join(dir,'remote-jobs.json')),runtime:remoteRuntime,socketPath:join(dir,'r.sock'),projectDirectory:dir,remoteProviderBindings:bindings,profileStore:new ProfileStore(join(dir,'remote-profiles.json'))})
  await localServer.start();await remoteServer.start()
  const local=new DaemonRpc({socketPath:join(dir,'l.sock'),projectDir:dir,deadlineMs:3000})
  let forwarding:DesktopProviderForwarding
  const remote=new DaemonRpc({socketPath:join(dir,'r.sock'),projectDir:dir,deadlineMs:3000,providerRelay:(binding,frame,signal)=>forwarding.relay(binding,frame,signal)})
  forwarding=new DesktopProviderForwarding((method,params)=>local.call(method,params),(method,params)=>remote.call(method,params),'verified-host',dir,preferred)
  remote.onConnection(online=>{if(!online)void forwarding.disconnect()});local.onConnection(online=>{if(!online)void forwarding.disconnect()})
  remote.onEvent((type,payload)=>events.push({type,payload}));remote.on('protocol_error',value=>events.push(value))
  const initialized=await remote.call('initialize',{})
  const created={session_id:(initialized.session as {id:string}).id}
  const authorize=async()=>{const review=await forwarding.inspect();await forwarding.authorize({review:review.review,consent:true,profiles:['local-api'],profile:'local-api',minutes:60});return review}
  const turn=async()=>{const count=remoteRuntime.listSessions().find(s=>s.id===created.session_id)!.messages.length;const submitted=await remote.call('turn.submit',{text:'Reply locally',submission_id:crypto.randomUUID()});expect(submitted).toMatchObject({ok:true});await until(()=>{const s=remoteRuntime.listSessions().find(s=>s.id===created.session_id)!;return s.messages.length>count&&!s.activeTurnId});for(let i=0;i<300;i++){if((await remote.call('session.status',{history_limit:0})).provider_binding_busy===false)return;await Bun.sleep(10)}throw new Error('Turn did not settle')}
  return {dir,profiles,save,seen,events,local,remote,forwarding,remoteRuntime,created,authorize,turn,hold:()=>{hold=true},aborted:()=>aborted,fallback:()=>fallback,
    async close(){await forwarding.disconnect();local.dispose();remote.dispose();await localServer.stop();await remoteServer.stop();await localRuntime.shutdown();await remoteRuntime.shutdown();bindings.close();await rm(dir,{recursive:true,force:true})}}
}

test('desktop forwards through real daemons, rotates keys per request, revokes and retains a fail-closed saved requirement',async()=>{
  const f=await fixture()
  try {
    const review=await f.authorize()
    expect(review.source).toBe('remote')
    await f.turn()
    f.save('fixture-new-key')
    await f.turn()
    expect(f.seen).toContain('fixture-old-key');expect(f.seen.at(-1)).toBe('fixture-new-key')
    expect((await f.forwarding.inspect()).source).toBe('local')
    const models=await f.remote.call('fetch_models',{for_model_selection:true})
    expect(models.models).toEqual(['gpt-4o'])
    await f.forwarding.revoke(review.sessionKey)
    expect((await f.forwarding.inspect()).source).toBe('local-unavailable')
    const count=f.seen.length
    await f.turn();expect(f.seen.length).toBe(count);expect(f.fallback()).toBe(0)
    await f.remoteRuntime.flushSessions()
    const saved=await Bun.file(join(f.dir,'remote-sessions',String(f.created.session_id)+'.json')).text()
    expect(saved).toContain('local_provider_binding')
    for(const serialized of [JSON.stringify(f.events),saved,JSON.stringify(review)]){
      expect(serialized).not.toContain('fixture-old-key');expect(serialized).not.toContain('fixture-new-key');expect(serialized).not.toContain('provider.remote.request')
    }
    await f.authorize();await f.turn();expect(f.seen.length).toBeGreaterThan(count)
  }finally{await f.close()}
},20000)

test('disconnect cancels pending local work; reconnect retains identity but needs fresh consent',async()=>{
  const f=await fixture()
  try {
    await f.authorize();f.hold();await f.remote.call('turn.submit',{text:'Wait'})
    await until(()=>f.seen.length>0)
    ;(f.remote as unknown as {socket:Socket}).socket.destroy()
    await until(f.aborted)
    await until(()=>!f.remoteRuntime.listSessions().find(s=>s.id===f.created.session_id)?.activeTurnId)
    await f.remote.call('initialize',{resume_session_id:f.created.session_id})
    const review=await f.forwarding.inspect()
    expect(review.source).toBe('local-unavailable');expect(f.fallback()).toBe(0)
    expect(review.sessionKey).toBe(f.remoteRuntime.listSessions().find(s=>s.id===f.created.session_id)!.sessionKey)
  }finally{await f.close()}
},15000)

test('review is bound to the session and refuses working tasks, stale selections and changed routes',async()=>{
  const f=await fixture()
  try {
    const review=await f.forwarding.inspect()
    await f.remote.call('session.open',{session_key:'another'})
    await expect(f.forwarding.authorize({consent:true,review:review.review,profiles:['local-api'],profile:'local-api',minutes:60})).rejects.toThrow('conversation changed')
    expect(await f.remote.call('provider.remote.bind',{consent:true,session_key:review.sessionKey,source:'test',profile:'local-api',model:'gpt-4o'})).toMatchObject({ok:false})
    await f.remote.call('initialize',{resume_session_id:f.created.session_id})
    await f.authorize()
    f.save('fixture-new-key','https://changed.invalid')
    await f.turn();expect(f.seen).toHaveLength(0);expect(f.fallback()).toBe(0)
    f.save('fixture-new-key');await f.authorize();f.hold();await f.remote.call('turn.submit',{text:'Hold'})
    await until(()=>f.seen.length>0)
    const busy=await f.forwarding.inspect();expect(busy.running).toBe(true)
    await expect(f.forwarding.authorize({consent:true,review:busy.review,profiles:['local-api'],profile:'local-api',minutes:60})).rejects.toThrow('working')
  }finally{await f.close()}
},15000)

test('partial authorization failure and cancellation revoke every prepared grant without binding',async()=>{
  for(const cancel of [false,true]){
    const revoked:string[]=[],calls:string[]=[]
    const profile=(name:string)=>({name,model:'gpt-4o',supported:true,credential_source:'local saved profile',output_limit_mode:'request-bound'})
    let count=0, forwarding:DesktopProviderForwarding
    forwarding=new DesktopProviderForwarding(async(method,params)=>{
      if(method==='provider.relay.inventory')return {ok:true,profiles:[profile('one'),profile('two')]}
      if(method==='provider.relay.revoke'){revoked.push(String(params.id));return {ok:true}}
      if(method==='provider.relay.authorize'){
        count++
        if(count===2){if(cancel)await forwarding.disconnect();else return {ok:false,error:'private diagnostic'}}
        return {ok:true,grant:{id:String(count).repeat(32)}}
      }
      return {ok:false}
    },async(method)=>{calls.push(method);return {ok:true,provider_binding_session_guard_supported:true,session:{key:'task',cwd:'/project',status:'idle'}}},'host','/project')
    try{
      const review=await forwarding.inspect()
      await expect(forwarding.authorize({review:review.review,consent:true,profile:'one',profiles:['one','two'],minutes:60})).rejects.toThrow()
      expect(calls).not.toContain('provider.remote.bind');expect(revoked).toContain('1'.repeat(32))
      if(cancel)expect(revoked).toContain('2'.repeat(32))
    }finally{await forwarding.disconnect()}
  }
})

test('consent and old runtime checks reject before creating grants',async()=>{
  let grants=0
  const local=async(method:string)=>{if(method==='provider.relay.authorize')grants++;return {ok:true,profiles:[{name:'subscription',model:'gpt-4o',supported:true,output_limit_mode:'provider-controlled'}]}}
  const remote=async()=>({ok:true,provider_binding_session_guard_supported:true,session:{key:'task',cwd:'/project',status:'idle'}})
  const forwarding=new DesktopProviderForwarding(local,remote,'host','/project')
  try{
    const review=await forwarding.inspect()
    const input={review:review.review,consent:true,profile:'subscription',profiles:['subscription'],minutes:60}
    await expect(forwarding.authorize({...input,consent:false})).rejects.toThrow('Review')
    await expect(forwarding.authorize(input)).rejects.toThrow('provider-controlled')
    await expect(forwarding.authorize({...input,providerControlledOutput:true,minutes:999})).rejects.toThrow('duration')
    expect(grants).toBe(0)
    const old=new DesktopProviderForwarding(local,async()=>({ok:true}), 'host','/project')
    await expect(old.inspect()).rejects.toThrow('Update the SSH runtime')
  }finally{await forwarding.disconnect()}
})

test('expired access is reported unavailable and cannot fall back to remote credentials',async()=>{
  const originalNow=Date.now
  let offset=0
  Date.now=()=>originalNow()+offset
  const f=await fixture()
  try{
    await f.authorize()
    const access=await f.forwarding.inspect()
    offset=access.expiresAt!-originalNow()+1
    expect((await f.forwarding.inspect()).source).toBe('local-unavailable')
    await f.turn();expect(f.seen).toHaveLength(0);expect(f.fallback()).toBe(0)
  }finally{Date.now=originalNow;await f.close()}
})

// A sign-in profile (ChatGPT, Claude Code) has no key to copy; the fixture's
// stand-in is a keyless profile, which the local daemon reports the same way.
const signIn = (profiles: ProfileStore) => profiles.save({name:'local-signin',provider:'openai',model:'gpt-4o',apiKey:'',baseUrl:'https://fixture.invalid',setActive:false})

test('a new SSH conversation follows this computer\'s sign-in provider with no review, and renews after a reconnect',async()=>{
  let active='local-signin'
  const f=await fixture(async()=>active)
  try {
    signIn(f.profiles)
    let changes=0
    f.forwarding.onChange(()=>{changes++})
    expect(await f.forwarding.current()).toMatchObject({bound:false})
    expect(await f.forwarding.follow()).toEqual({status:'bound'})
    expect(changes).toBeGreaterThan(0)
    const bound=await f.forwarding.inspect()
    expect(bound.source).toBe('local')
    expect(bound.sharedProfiles).toEqual(['local-signin','local-api'])
    await f.turn()
    expect(f.seen.at(-1)).toBe('')
    expect(f.fallback()).toBe(0)
    // What the composer badge and the workspace panel show.
    const shown=await f.forwarding.current()
    expect(shown).toMatchObject({bound:true,live:true,destination:'verified-host',profile:'local-signin',model:'gpt-4o'})
    // The turn, plus the session title the host asks for afterwards: both run here.
    expect((shown as {requests:number}).requests).toBeGreaterThanOrEqual(1)
    expect(typeof (shown as {lastAt?:number}).lastAt).toBe('number')
    expect(await f.forwarding.follow()).toEqual({status:'skipped',reason:'already following'})
    // Transport loss ends access; the saved requirement makes the next pass renew it.
    await f.forwarding.disconnect()
    expect((await f.forwarding.inspect()).source).toBe('local-unavailable')
    // Still bound on the host, no longer carried here: the badge turns into a warning.
    const ended=await f.forwarding.current() as {bound:boolean;live:boolean;requests:number;inFlight:number}
    expect(ended).toMatchObject({bound:true,live:false,inFlight:0})
    expect(ended.requests).toBeGreaterThanOrEqual(1)
    expect(await f.forwarding.follow()).toEqual({status:'bound'})
    await f.turn()
    expect(f.fallback()).toBe(0)
    // A keyed active profile is copied to the host instead.
    active='local-api'
    expect((await f.forwarding.follow()).status).toBe('skipped')
  }finally{await f.close()}
},20000)

test('a conversation already run on the host\'s providers is not taken over',async()=>{
  const f=await fixture(async()=>'local-signin')
  try {
    signIn(f.profiles)
    await f.turn()
    expect(f.fallback()).toBe(1)
    expect(await f.forwarding.follow()).toEqual({status:'skipped',reason:'this conversation uses the host\'s providers'})
    expect((await f.forwarding.inspect()).source).toBe('remote')
  }finally{await f.close()}
},20000)

test('without a preferred profile nothing is followed automatically',async()=>{
  const f=await fixture()
  try {
    signIn(f.profiles)
    expect(await f.forwarding.follow()).toEqual({status:'skipped',reason:'automatic provider access is off'})
    expect((await f.forwarding.inspect()).source).toBe('remote')
  }finally{await f.close()}
},20000)

test('switching an SSH task to a provider on this computer sticks through renewals, even for a task that ran on the host',async()=>{
  const f=await fixture(async()=>'local-signin')
  try {
    signIn(f.profiles)
    await f.turn()
    expect(f.fallback()).toBe(1)
    // Settings → switch: the person's explicit choice applies to a used conversation.
    expect(await f.forwarding.useLocal('local-api')).toEqual({ok:true})
    expect((await f.forwarding.inspect()).sharedProfiles[0]).toBe('local-api')
    await f.turn()
    expect(f.seen.at(-1)).toBe('fixture-old-key')
    // After transport loss, renewal keeps local-api rather than the Mac's active local-signin.
    await f.forwarding.disconnect()
    expect(await f.forwarding.follow()).toEqual({status:'bound'})
    expect((await f.forwarding.inspect()).sharedProfiles[0]).toBe('local-api')
    await expect(f.forwarding.useLocal('missing')).rejects.toThrow('no missing profile')
  }finally{await f.close()}
},20000)
