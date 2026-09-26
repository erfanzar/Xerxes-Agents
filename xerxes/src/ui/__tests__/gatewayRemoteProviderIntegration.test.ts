// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdtemp, rm, realpath } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { Socket } from 'node:net'
import { expect, it, vi } from 'vitest'
import { AgentTurnRunner } from '../../daemon/turnRunner.js'
import { InMemoryDaemonRuntime } from '../../daemon/runtime.js'
import { DaemonServer } from '../../daemon/server.js'
import { RemoteProviderBindings, LOCAL_PROVIDER_BINDING } from '../../daemon/remoteProviderBindings.js'
import { LocalProviderRelay } from '../../security/localProviderRelay.js'
import { LocalProviderEndpoint } from '../../security/localProviderEndpoint.js'
import type { LlmClient } from '../../llms/client.js'
import { GatewayClient } from '../gatewayClient.js'
import { ProfileStore } from '../../bridge/profiles.js'
import { DaemonTranscriptStore } from '../../session/daemonTranscript.js'
import { localCapabilitySnapshot } from '../../daemon/localReasoningCapabilities.js'
import { catalogReasoningLevels, fallbackReasoningLevels } from '../../llms/reasoningLevels.js'
import { seedModelsDev } from '../../../test/fixtures/modelsDev.js'

// Model capabilities come from models.dev at runtime; tests use its fixture.
seedModelsDev()

async function fixture(provider: LlmClient, model = 'gpt-4o') {
  const dir = await realpath(await mkdtemp(join(tmpdir(), 'xr-bind-')))
  const bindings = new RemoteProviderBindings(1000)
  const profiles = new ProfileStore(join(dir, 'profiles.json'))
  const fallback = vi.fn()
  const remote: LlmClient = { async *stream() { fallback(); yield { content: 'unapproved remote provider' } } }
  const auxiliaryFallback = vi.fn(() => remote)
  const runner = new AgentTurnRunner({ model, llm: remote, fallbackModel: 'other-model', createLlmForModel: () => remote, remoteProviderBindings: bindings, tools: [] })
  const runtime = new InMemoryDaemonRuntime(runner, { model, currentProjectDirectory: dir, sessionDirectory: join(dir, 'sessions') })
  const server = new DaemonServer({ runtime, socketPath: join(dir, 'rpc.sock'), projectDirectory: dir, remoteProviderBindings: bindings,
    titleClientFactory: auxiliaryFallback, projectAgentClientFactory: auxiliaryFallback, profileStore: profiles })
  const authority = new LocalProviderRelay(), peer = { destination: 'isolated-fixture', workspace: dir }
  const grant = authority.authorize(peer, { profile: 'local-profile', model, expiresAt: Date.now()+60_000, maxRequests: 8, maxOutputTokens: 8192, maxConcurrent: 1 }, () => ({ client: provider, routeIdentity: 'fixed' }), 'fixed')
  const endpoint = new LocalProviderEndpoint(authority, peer, grant.token)
  let expectedBinding = ''
  const privateRequests: unknown[] = [], publicEvents: unknown[] = []
  const client = new GatewayClient({ externalSocketPath: join(dir, 'rpc.sock'), projectDir: dir,
    providerRelay: async (binding, frame) => { expect(binding).toBe(expectedBinding); privateRequests.push(frame); return endpoint.handle(frame) } })
  client.on('event', event => publicEvents.push(event))
  await server.start(); await client.start();
  const created = await client.request<{ session_id: string }>('session.create')
  const bind = async () => {
    const capabilities=localCapabilitySnapshot(model,catalogReasoningLevels(model,'openai')??fallbackReasoningLevels('openai'))
    const result = await client.request<{ ok: boolean; binding: { binding: string } }>('provider.remote.bind', { consent: true, source: 'local workstation', profile: 'local-profile', model, capabilities })
    expect(result.ok).toBe(true); expectedBinding = result.binding.binding
    return result
  }
  return { dir, client, runtime, bindings, runner, server, authority, peer, grant, endpoint, fallback, auxiliaryFallback, profiles, created, privateRequests, publicEvents, bind,
    async close() { client.close(); endpoint.close(); authority.close(); await server.stop(); await runtime.shutdown(); await rm(dir, {recursive:true,force:true}) } }
}

it('real socket binding runs the native agent through private frames and persists only the local requirement', async () => {
  const f = await fixture({ async *stream() { yield { content: 'Approved local reply.' } } })
  try {
    expect(await f.client.request('provider.remote.bind', { consent: false })).toMatchObject({ ok: false, code: 'invalid_request' })
    await f.bind()
    await f.client.request('turn.submit', { text: 'Use the approved provider.' })
    await vi.waitFor(() => expect(JSON.stringify(f.runtime.listSessions())).toContain('Approved local reply.'))
    await vi.waitFor(() => expect(f.runtime.listSessions().every(s=>!s.activeTurnId)).toBe(true))
    expect(f.privateRequests.length).toBeGreaterThan(0)
    expect(JSON.stringify(f.publicEvents)).not.toContain('provider.remote.request')
    expect(f.fallback).not.toHaveBeenCalled()
    await f.runtime.flushSessions()
    const saved = await Bun.file(join(f.dir,'sessions',f.created.session_id+'.json')).text()
    expect(saved).toContain(LOCAL_PROVIDER_BINDING)
    expect(saved).not.toContain(f.grant.token)
    const restarted = new InMemoryDaemonRuntime(f.runner, { model: 'gpt-4o', currentProjectDirectory: f.dir, sessionDirectory: join(f.dir,'sessions') })
    f.bindings.close()
    try {
      const resumed = await restarted.openSession(f.created.session_id, 'default', { resume:true, cwd:f.dir })
      const events: unknown[]=[]
      await restarted.submitTurn(resumed.sessionKey,'Continue.',event=>events.push(event))
      expect(JSON.stringify(events)).toContain('Local provider grant is unavailable')
      expect(f.fallback).not.toHaveBeenCalled()
    } finally { await restarted.shutdown() }
  } finally { await f.close() }
})

it('revocation returns an explicit failure instead of using the remote provider or fallback model', async () => {
  const f = await fixture({ async *stream() { yield {content:'must not run'} } })
  try {
    await f.bind(); f.authority.revoke(f.peer,f.grant.token)
    await f.client.request('turn.submit',{text:'After revocation.'})
    await vi.waitFor(()=>expect(JSON.stringify(f.publicEvents)).toContain('revoked'))
    expect(f.fallback).not.toHaveBeenCalled()
  } finally { await f.close() }
})

it('disconnect rejects pending context requests and reconnect requires an explicit fresh binding', async () => {
  let requested=false
  const f=await fixture({async *stream(_request,signal){requested=true;await new Promise<void>(resolve=>signal!.addEventListener('abort',()=>resolve(),{once:true}));yield {content:'late'} }})
  try {
    await f.bind(); await f.client.request('turn.submit',{text:'Wait for provider.'})
    await vi.waitFor(()=>expect(requested).toBe(true))
    ;(f.client as unknown as {socket:Socket}).socket.destroy()
    await Bun.sleep(40)
    const session=f.runtime.listSessions().find(s=>s.id===f.created.session_id)!
    await vi.waitFor(()=>expect(session.activeTurnId).toBe(''))
    expect(()=>f.bindings.client(session,'gpt-4o')).toThrow('unavailable')
    await f.client.start()
    await f.client.request('session.resume',{session_id:f.created.session_id,preserve_view:true})
    expect(()=>f.bindings.client(session,'gpt-4o')).toThrow('unavailable')
    expect(f.fallback).not.toHaveBeenCalled()
    // Even a socket that reclaims the session cannot restore spent authority.
    expect(JSON.stringify(f.publicEvents)).not.toContain('provider.remote.request')
  } finally { await f.close() }
})

it('automatic titles use only the authorized model and local binding', async () => {
  const models:string[]=[]
  const f=await fixture({async *stream(request){models.push(request.model);yield {content:String(request.messages[0]?.content).startsWith('Write a very short title')?'Local Bound Title':'First local answer.'} }})
  try {
    await f.bind(); await f.client.request('turn.submit',{text:'Title this first exchange.'})
    await vi.waitFor(()=>expect(f.runtime.listSessions().find(s=>s.id===f.created.session_id)?.metadata.title).toBe('Local Bound Title'))
    expect(models).toEqual(['gpt-4o','gpt-4o'])
    expect(f.auxiliaryFallback).not.toHaveBeenCalled()
    expect(f.fallback).not.toHaveBeenCalled()
  } finally {await f.close()}
})

it('agent draft generation receives private replies while its RPC is still pending', async () => {
  const f=await fixture({async *stream(){yield {content:'---\nname: local-reviewer\ndescription: Review code\n---\nFind defects.'}}})
  try {
    await f.bind()
    const result=await f.client.request('agentPreset.projectGenerate',{description:'Review code through the approved local model.'})
    expect(result).toMatchObject({ok:true,id:'local-reviewer',revision:null})
    expect(f.auxiliaryFallback).not.toHaveBeenCalled()
    expect(f.fallback).not.toHaveBeenCalled()
  } finally {await f.close()}
})

it('manual compaction uses the local binding and retains history on released authority', async () => {
  let calls=0
  const f=await fixture({async *stream(){calls++;yield {content:'Keep the tested workspace decisions and continue the implementation.'}}})
  try {
    await f.bind()
    const session=f.runtime.listSessions().find(s=>s.id===f.created.session_id)!
    session.messages.push(...Array.from({length:24},(_,i)=>({role:i%2?'assistant':'user',content:`Retained exchange ${i}: `+'details '.repeat(500)})))
    const compacted=await f.client.request('slash.exec',{command:'compact'})
    expect(compacted).toMatchObject({ok:true})
    expect(calls).toBeGreaterThan(0)
    expect(JSON.stringify(session.messages)).toContain('Keep the tested workspace decisions')
    session.messages.push(...Array.from({length:24},(_,i)=>({role:i%2?'assistant':'user',content:`More exchange ${i}: `+'retained '.repeat(500)})))
    const before=JSON.stringify(session.messages),previous=calls
    await f.client.request('provider.remote.release')
    expect(await f.client.request('slash.exec',{command:'compact'})).toMatchObject({ok:false})
    expect(JSON.stringify(session.messages)).toBe(before)
    expect(calls).toBe(previous)
    expect(f.fallback).not.toHaveBeenCalled()
  } finally {await f.close()}
})

it('ordinary session titles use their pinned profile, not the globally active profile', async () => {
  const f=await fixture({async *stream(){yield {content:'unused relay'}}})
  try {
    f.profiles.save({name:'pinned',provider:'openai',model:'gpt-4o',baseUrl:'https://unused.invalid/v1',apiKey:'synthetic-pinned',setActive:false})
    f.profiles.save({name:'global',provider:'anthropic',model:'claude-fixture',baseUrl:'https://unused.invalid/v1',apiKey:'synthetic-global',setActive:true})
    f.runtime.listSessions().find(s=>s.id===f.created.session_id)!.metadata.provider_profile='pinned'
    await f.client.request('turn.submit',{text:'Use my session profile for the title.'})
    await vi.waitFor(()=>expect(f.auxiliaryFallback).toHaveBeenCalled())
    expect(f.auxiliaryFallback).toHaveBeenCalledWith('gpt-4o-mini',expect.objectContaining({name:'pinned'}))
    expect(f.profiles.active()?.name).toBe('global')
  } finally {await f.close()}
})

it('missing pinned credentials do not select global credentials for an automatic title', async () => {
  const f=await fixture({async *stream(){yield {content:'unused relay'}}})
  try {
    f.profiles.save({name:'global',provider:'openai',model:'gpt-4o',baseUrl:'https://unused.invalid/v1',apiKey:'synthetic-global'})
    const session=f.runtime.listSessions().find(s=>s.id===f.created.session_id)!
    session.metadata.provider_profile='removed-profile'
    await f.client.request('turn.submit',{text:'Keep the provisional title.'})
    await vi.waitFor(()=>expect(session.turnCount).toBe(1))
    await Bun.sleep(30)
    expect(f.auxiliaryFallback).not.toHaveBeenCalled()
    expect(session.metadata.title_derived).toBe(true)
  } finally {await f.close()}
})

it('release during compaction cancels the local provider and retains the original history', async () => {
  let started=false,aborted=false
  const f=await fixture({async *stream(_request,signal){started=true;await new Promise<void>(resolve=>signal!.addEventListener('abort',()=>{aborted=true;resolve()},{once:true}));yield {content:'late summary'}}})
  try {
    await f.bind()
    const session=f.runtime.listSessions().find(s=>s.id===f.created.session_id)!
    session.messages.push(...Array.from({length:24},(_,i)=>({role:i%2?'assistant':'user',content:`Exchange ${i}: `+'retained '.repeat(500)})))
    const before=JSON.stringify(session.messages)
    const pending=f.client.request('slash.exec',{command:'compact'})
    await vi.waitFor(()=>expect(started).toBe(true))
    expect(await f.client.request('provider.remote.release')).toMatchObject({ok:true})
    expect(await pending).toMatchObject({ok:false})
    await vi.waitFor(()=>expect(aborted).toBe(true))
    expect(JSON.stringify(session.messages)).toBe(before)
    expect(f.fallback).not.toHaveBeenCalled()
  } finally {await f.close()}
})

it('optional titles never create a provider pin for an unpinned session', async () => {
  const f=await fixture({async *stream(){yield {content:'unused relay'}}})
  try {
    f.profiles.save({name:'global',provider:'openai',model:'gpt-4o',baseUrl:'https://unused.invalid/v1',apiKey:'synthetic-global'})
    const session=f.runtime.listSessions().find(s=>s.id===f.created.session_id)!
    expect(session.metadata.provider_profile).toBeUndefined()
    await f.client.request('turn.submit',{text:'Keep the inherited route.'})
    await vi.waitFor(()=>expect(f.auxiliaryFallback).toHaveBeenCalled())
    expect(session.metadata.provider_profile).toBeUndefined()
    expect(f.auxiliaryFallback).toHaveBeenCalledWith('gpt-4o',undefined)
  } finally {await f.close()}
})

for (const method of ['provider_select', 'set_model']) it(`${method} explicitly switches an idle local task to remote credentials and persists the choice`, async () => {
  const f = await fixture({ async *stream() { yield {content:'local before override'} } })
  try {
    f.profiles.save({name:'remote-choice',provider:'openai',baseUrl:'https://provider.invalid/v1',model:'gpt-4o',apiKey:'synthetic-remote-key'})
    await f.bind()
    const session = f.runtime.listSessions().find(s=>s.id===f.created.session_id)!
    session.messages.push({role:'user',content:'Persist this choice'},{role:'assistant',content:'Existing history'})
    const before = await f.client.request<{session:{local_provider_label:string}}>('session.status',{structured:true,history_limit:0})
    expect(before.session.local_provider_label).toContain('Local provider: local-profile')
    const params = method === 'provider_select' ? {name:'remote-choice'} : {model:'gpt-4o',provider_profile:'remote-choice'}
    expect(await f.client.request(method,params)).toMatchObject({ok:true})
    expect(Object.hasOwn(session.metadata,LOCAL_PROVIDER_BINDING)).toBe(false)
    expect(f.bindings.client(session,session.model)).toBeUndefined()
    const after = await f.client.request<{session:{local_provider_label:string}}>('session.status',{structured:true,history_limit:0})
    expect(after.session.local_provider_label).toBe('')
    const saved = await Bun.file(join(f.dir,'sessions',session.id+'.json')).json()
    expect(JSON.stringify(saved)).not.toContain(LOCAL_PROVIDER_BINDING)
    const restarted = new InMemoryDaemonRuntime(f.runner,{currentProjectDirectory:f.dir,sessionDirectory:join(f.dir,'sessions'),model:'gpt-4o'})
    try {
      const resumed = await restarted.openSession(session.id,'default',{resume:true,cwd:f.dir})
      expect(resumed.metadata.provider_profile).toBe('remote-choice')
      await restarted.submitTurn(resumed.sessionKey,'Use the explicit remote choice.',()=>{})
      expect(f.fallback).toHaveBeenCalled()
      expect(f.privateRequests).toHaveLength(0)
    } finally {await restarted.shutdown()}
  } finally {await f.close()}
})

it('implicit model selection and busy remote overrides leave local authority unchanged', async () => {
  const f = await fixture({async *stream(){yield {content:'local'}}})
  try {
    f.profiles.save({name:'remote-choice',provider:'openai',baseUrl:'https://provider.invalid/v1',model:'gpt-4o',apiKey:'synthetic'})
    await f.bind()
    const session=f.runtime.listSessions().find(s=>s.id===f.created.session_id)!
    const marker=session.metadata[LOCAL_PROVIDER_BINDING]
    expect(await f.client.request('set_model',{model:'gpt-4o'})).toMatchObject({ok:true,model:'gpt-4o'})
    session.activeTurnId='active-fixture'
    try {expect(await f.client.request('provider_select',{name:'remote-choice'})).toMatchObject({ok:false,error:expect.stringContaining('Stop the active turn')})}
    finally {session.activeTurnId=''}
    expect(session.metadata[LOCAL_PROVIDER_BINDING]).toBe(marker)
    expect(f.bindings.client(session,session.model)).toBeDefined()
  } finally {await f.close()}
})

for (const route of ['model','provider']) for (const marker of [undefined, {version:1,source:'unavailable'}]) it(`failed remote ${route} selection restores even a malformed local requirement (${String(marker)})`, async () => {
  const f = await fixture({async *stream(){yield {content:'local'}}})
  try {
    f.profiles.save({name:'remote-choice',provider:'openai',baseUrl:'https://provider.invalid/v1',model:'gpt-4o',apiKey:'synthetic'})
    const session=f.runtime.listSessions().find(s=>s.id===f.created.session_id)!
    session.metadata[LOCAL_PROVIDER_BINDING]=marker
    session.metadata.local_provider_profile='second-local'
    const failure=vi.spyOn(f.runtime,'setSessionModel').mockRejectedValueOnce(new Error('fixture persistence unavailable'))
    try {await expect(route === 'model' ? f.client.request('set_model',{model:'gpt-4o',provider_profile:'remote-choice'}) : f.client.request('provider_select',{name:'remote-choice'})).rejects.toThrow('fixture persistence unavailable')}
    finally {failure.mockRestore()}
    expect(Object.hasOwn(session.metadata,LOCAL_PROVIDER_BINDING)).toBe(true)
    expect(session.metadata[LOCAL_PROVIDER_BINDING]).toBe(marker)
    expect(session.metadata.local_provider_profile).toBe('second-local')
    expect(()=>f.bindings.client(session,session.model)).toThrow('unavailable')
    expect(f.fallback).not.toHaveBeenCalled()
  } finally {await f.close()}
})

for(const route of ['rpc','slash'])it(`${route} reasoning changes remain on the local-bound task and preserve remote profile defaults`,async()=>{
  const f=await fixture({async *stream(){yield {content:'local answer'}}},'gpt-5')
  try{
    f.profiles.save({name:'remote-choice',provider:'openai',baseUrl:'https://provider.invalid/v1',apiKey:'synthetic',model:'gpt-5',sampling:{reasoning_effort:'low',thinking:true}})
    await f.bind()
    const session=f.runtime.listSessions().find(s=>s.id===f.created.session_id)!
    session.metadata.provider_profile='remote-choice'
    session.messages.push({role:'user',content:'Persist the effort choice'},{role:'assistant',content:'Existing conversation'})
    const before=f.profiles.get('remote-choice')
    const response=route==='rpc'?await f.client.request('set_reasoning',{reasoning_effort:'high'}):await f.client.request('slash.exec',{command:'thinking high'})
    expect(response).toMatchObject({ok:true})
    expect(session.reasoningEffort).toBe('high')
    expect(session.reasoningPinned).toBe(true)
    expect(f.profiles.get('remote-choice')).toEqual(before)
    const saved=await Bun.file(join(f.dir,'sessions',session.id+'.json')).json()
    expect(saved.metadata.reasoning_effort).toBe('high')
    expect(saved.metadata).toHaveProperty(LOCAL_PROVIDER_BINDING)
    const restarted=new InMemoryDaemonRuntime(f.runner,{model:'gpt-5',currentProjectDirectory:f.dir,sessionDirectory:join(f.dir,'sessions')})
    try {
      const resumed=await restarted.openSession(session.id,'default',{resume:true,cwd:f.dir})
      expect(resumed.reasoningEffort).toBe('high')
      expect(resumed.reasoningPinned).toBe(true)
      expect(resumed.metadata).toHaveProperty(LOCAL_PROVIDER_BINDING)
    } finally {await restarted.shutdown()}
    expect(f.fallback).not.toHaveBeenCalled()
  }finally{await f.close()}
})

it('a local reasoning persistence failure is reported without changing remote defaults or revoking the route',async()=>{
  const f=await fixture({async *stream(){yield {content:'local answer'}}},'gpt-5')
  try {
    f.profiles.save({name:'remote-choice',provider:'openai',baseUrl:'https://provider.invalid/v1',apiKey:'synthetic',model:'gpt-5',sampling:{reasoning_effort:'low'}})
    await f.bind()
    const session=f.runtime.listSessions().find(s=>s.id===f.created.session_id)!
    session.metadata.provider_profile='remote-choice'
    session.messages.push({role:'user',content:'Existing task'},{role:'assistant',content:'Existing reply'})
    const marker=session.metadata[LOCAL_PROVIDER_BINDING], before=f.profiles.get('remote-choice')
    const failure=vi.spyOn(DaemonTranscriptStore.prototype,'save').mockRejectedValueOnce(new Error('fixture disk unavailable'))
    try {await expect(f.client.request('set_reasoning',{reasoning_effort:'high'})).rejects.toThrow('previous effort is unchanged')}
    finally {failure.mockRestore()}
    expect(f.profiles.get('remote-choice')).toEqual(before)
    expect(session.reasoningEffort).toBeUndefined()
    expect(session.reasoningPinned).not.toBe(true)
    expect(session.metadata[LOCAL_PROVIDER_BINDING]).toBe(marker)
    expect(f.bindings.client(session,session.model)).toBeDefined()
    expect(await f.client.request('set_reasoning',{reasoning_effort:'high'})).toMatchObject({ok:true})
    expect(await Bun.file(join(f.dir,'sessions',session.id+'.json')).json()).toMatchObject({metadata:{reasoning_effort:'high'}})
    expect(f.privateRequests).toHaveLength(0)
  } finally {await f.close()}
})

it('a runtime without task reasoning support cannot change global defaults through local /thinking',async()=>{
  const f=await fixture({async *stream(){yield {content:'local answer'}}},'gpt-5')
  try {
    f.profiles.save({name:'remote-choice',provider:'openai',baseUrl:'https://provider.invalid/v1',apiKey:'synthetic',model:'gpt-5',sampling:{reasoning_effort:'low'}})
    await f.bind()
    const session=f.runtime.listSessions().find(s=>s.id===f.created.session_id)!
    session.metadata.provider_profile='remote-choice'
    const before=f.profiles.get('remote-choice'), reload=vi.spyOn(f.runtime,'reload')
    Object.defineProperty(f.runtime,'setSessionReasoning',{value:undefined,configurable:true})
    try {
      expect(await f.client.request('slash.exec',{command:'thinking high'})).toMatchObject({ok:false,error:expect.stringContaining('cannot save reasoning')})
      expect(await f.client.request('set_reasoning',{reasoning_effort:'high'})).toMatchObject({ok:false})
      expect(f.profiles.get('remote-choice')).toEqual(before)
      expect(reload).not.toHaveBeenCalled()
      expect(session.reasoningEffort).toBeUndefined()
    } finally {delete (f.runtime as Partial<InMemoryDaemonRuntime>).setSessionReasoning;reload.mockRestore()}
  } finally {await f.close()}
})

it('local control choices ignore a conflicting remote profile and their source reaches the TUI gateway',async()=>{
  const f=await fixture({async *stream(){yield {content:'unused'}}},'gpt-5')
  try {
    f.profiles.save({name:'remote-choice',provider:'deepseek',baseUrl:'https://provider.invalid',apiKey:'synthetic',model:'gpt-5'})
    const capabilities={version:1,model:'gpt-5',reasoning:{shape:'effort',efforts:['low','ultra'],canDisable:false,provenance:'provider_reported'}}
    expect(await f.client.request('provider.remote.bind',{consent:true,source:'fixture workstation',profile:'local-profile',model:'gpt-5',capabilities})).toMatchObject({ok:true})
    const session=f.runtime.listSessions().find(s=>s.id===f.created.session_id)!
    session.metadata.provider_profile='remote-choice'
    expect(await f.client.request('reasoning.levels')).toMatchObject({shape:'effort',source:'provider',note:expect.stringContaining('reported by the local provider'),levels:[{effort:'low'},{effort:'ultra'}]})
    expect(await f.client.request('set_reasoning',{reasoning_effort:'ultra'})).toMatchObject({ok:true,reasoning_effort:'ultra'})
    expect(await f.client.request('set_reasoning',{reasoning_effort:'off'})).toMatchObject({ok:false})
    expect(f.profiles.get('remote-choice')?.sampling.reasoning_effort).toBeUndefined()
    await f.client.request('provider.remote.release')
    expect(await f.client.request('reasoning.levels')).toMatchObject({levels:[{effort:'low'},{effort:'ultra'}]})
    expect(f.privateRequests).toHaveLength(0)
  } finally {await f.close()}
})

for(const metadata of [{profile:'legacy'}, {model:'gpt-5',capabilities:{version:1,model:'other'}}])it('missing or damaged local capabilities expose recovery instructions and never guess remote controls',async()=>{
  const f=await fixture({async *stream(){yield {content:'unused'}}},'gpt-5')
  try {
    const session=f.runtime.listSessions().find(s=>s.id===f.created.session_id)!
    session.metadata[LOCAL_PROVIDER_BINDING]=metadata
    expect(await f.client.request('reasoning.levels')).toMatchObject({levels:[],shape:'unknown',source:'unavailable',note:expect.stringContaining('Reopen this SSH task')})
    expect(await f.client.request('set_reasoning',{reasoning_effort:'high'})).toMatchObject({ok:false})
    expect(await f.client.request('slash.exec',{command:'thinking high'})).toMatchObject({ok:false})
    expect(session.reasoningEffort).toBeUndefined()
    expect(f.fallback).not.toHaveBeenCalled()
  } finally {await f.close()}
})

it('a nonreasoning local model exposes an explanation rather than an unusable effort menu',async()=>{
  const f=await fixture({async *stream(){yield {content:'unused'}}})
  try {
    await f.bind()
    expect(await f.client.request('reasoning.levels')).toMatchObject({levels:[],shape:'inherent',note:expect.stringContaining('no selectable reasoning controls')})
    expect(await f.client.request('set_reasoning',{reasoning_effort:'high'})).toMatchObject({ok:false})
  } finally {await f.close()}
})
