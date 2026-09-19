// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdtemp, realpath, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { connect, createServer, type Socket } from 'node:net'
import { expect, it, vi } from 'vitest'
import { ProfileStore } from '../../bridge/profiles.js'
import { JobStore } from '../../cron/jobs.js'
import { DaemonServer } from '../../daemon/server.js'
import { InMemoryDaemonRuntime } from '../../daemon/runtime.js'
import { AgentTurnRunner } from '../../daemon/turnRunner.js'
import { RemoteProviderBindings } from '../../daemon/remoteProviderBindings.js'
import { GatewayClient } from '../gatewayClient.js'
import { prepareRemoteTask, shareableLocalProfiles, type RemoteTaskDecision, type RemoteTaskReview } from '../lib/remoteTaskSetup.js'
const storage = (dir: string) => ({cronLeasePath: join(dir,'cron.lease'), cronStoreFactory: () => new JobStore(join(dir,'jobs.json')), cronArchiveDirectory: join(dir,'cron-archive'), legacyScheduleDirectory: join(dir,'scheduler')})
const approve: RemoteTaskDecision = {kind:'local',profile:'local',durationMinutes:15,maxRequests:50,maxOutputTokens:4096,maxConcurrent:1,consentProviderControlledOutput:false}
async function fixture() {
  const dir = await realpath(await mkdtemp(join(tmpdir(),'xr-task-setup-')))
  let localCalls = 0, remoteCalls = 0
  const profiles = new ProfileStore(join(dir,'local-profiles.json'))
  profiles.save({name:'local',provider:'openai',baseUrl:'https://provider.invalid/v1',model:'gpt-4o',apiKey:'synthetic-private-key'})
  const localRuntime = new InMemoryDaemonRuntime(undefined,{currentProjectDirectory:dir,sessionDirectory:join(dir,'local-sessions')})
  const localServer = new DaemonServer({...storage(join(dir,'local')),runtime:localRuntime,projectDirectory:dir,socketPath:join(dir,'local.sock'),profileStore:profiles,
    relayClientFactory:()=>({async *stream(){localCalls++;yield {content:'LOCAL REPLY'}}})})
  const bindings = new RemoteProviderBindings()
  const remoteProfiles = new ProfileStore(join(dir,'remote-profiles.json'))
  remoteProfiles.save({name:'remote',provider:'openai',baseUrl:'https://provider.invalid/v1',model:'gpt-4o',apiKey:'synthetic-remote-key'})
  const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({model:'gpt-4o',tools:[],remoteProviderBindings:bindings,llm:{async *stream(){remoteCalls++;yield {content:'REMOTE REPLY'}}}}),{model:'gpt-4o',currentProjectDirectory:dir,sessionDirectory:join(dir,'remote-sessions')})
  const server = new DaemonServer({...storage(join(dir,'remote')),runtime,projectDirectory:dir,socketPath:join(dir,'remote.sock'),remoteProviderBindings:bindings,profileStore:remoteProfiles})
  await localServer.start();await server.start()
  const local = new GatewayClient({externalSocketPath:join(dir,'local.sock'),projectDir:dir})
  let renderer = new GatewayClient({externalSocketPath:join(dir,'remote.sock'),projectDir:dir})
  await local.start();await renderer.start()
  const rpc = vi.fn((method:string,params:Record<string,unknown>)=>local.request(method,params))
  const controller = new AbortController()
  return {dir,server,local,renderer,rpc,runtime,controller,get localCalls(){return localCalls},get remoteCalls(){return remoteCalls},
    async attach(prepared: Awaited<ReturnType<typeof prepareRemoteTask>>) {
      renderer.close()
      renderer = new GatewayClient({externalSocketPath:join(dir,'remote.sock'),projectDir:dir,
        preparedSession:{id:prepared.sessionId,key:prepared.sessionKey!}})
      await renderer.start()
      await renderer.request('session.resume',{session_id:prepared.sessionId})
      return renderer
    },
    remote:{machine:{alias:'test',target:'fixture-host',workspacePath:dir},projectDir:dir,socketPath:join(dir,'remote.sock'),signal:controller.signal},
    async close(){controller.abort();renderer.close();local.close();await server.stop();await localServer.stop();await runtime.shutdown();await localRuntime.shutdown();bindings.close();await rm(dir,{recursive:true,force:true})}}
}

for (const decision of [{kind:'remote'} as const, approve]) it(`reopens an untouched ${decision.kind} task on a new preparation connection without saving a phantom conversation`,async()=>{
  const f=await fixture()
  let first:Awaited<ReturnType<typeof prepareRemoteTask>>|undefined,second:typeof first
  try {
    first=await prepareRemoteTask(f.remote,f.rpc,async()=>decision)
    const renderer=await f.attach(first)
    const session=f.runtime.listSessions().find(s=>s.id===first!.sessionId)!
    expect(session.messages).toHaveLength(0)
    renderer.close();await first.close()
    const reviewed:RemoteTaskReview[]=[]
    second=await prepareRemoteTask({...f.remote,resumeSessionId:first.sessionId},f.rpc,async value=>{reviewed.push(value);return decision})
    expect(second.sessionId).toBe(first.sessionId)
    expect(second.sessionKey).toBe(first.sessionKey)
    expect(f.runtime.listSessions().filter(s=>s.id===first!.sessionId)).toEqual([session])
    expect(reviewed[0]?.localRequirement.includes('Local provider')).toBe(decision.kind==='local')
    expect(await Bun.file(join(f.dir,'remote-sessions',session.id+'.json')).exists()).toBe(false)
    const restarted=new InMemoryDaemonRuntime(undefined,{currentProjectDirectory:f.dir,sessionDirectory:join(f.dir,'remote-sessions')})
    try {await expect(restarted.openSession(session.id,'default',{resume:true,cwd:f.dir})).rejects.toThrow('missing')}
    finally {await restarted.shutdown()}
    expect(f.localCalls).toBe(0);expect(f.remoteCalls).toBe(0)
  } finally {await second?.close();await first?.close();await f.close()}
})
it('prepares and reviews before authorizing, then serves the exact remote task without copying credentials',async()=>{
  const f=await fixture()
  let prepared:Awaited<ReturnType<typeof prepareRemoteTask>>|undefined
  try {
    let reviewed: RemoteTaskReview | undefined
    let before: string[] = []
    prepared=await prepareRemoteTask(f.remote,f.rpc,async value=>{reviewed=value;before=f.rpc.mock.calls.map(c=>c[0]);return approve})
    expect(before).toEqual(["provider.relay.inventory"])
    expect(reviewed?.localBindingSupported).toBe(true)
    expect(reviewed?.workspace).toBe(f.dir)
    expect(reviewed?.profiles.find(p => p.name === "local")).toMatchObject({name:"local",model:"gpt-4o"})
    expect(JSON.stringify(reviewed)).not.toContain("synthetic-private-key")
    const renderer = await f.attach(prepared)
    const session=f.runtime.listSessions().find(s=>s.id===prepared!.sessionId)!
    expect(f.runtime.listSessions().filter(s=>s.id===prepared!.sessionId)).toHaveLength(1)
    await renderer.request('turn.submit',{text:'Run the approved task.'})
    await vi.waitFor(()=>expect(session.messages.at(-1)?.content).toBe('LOCAL REPLY'))
    expect(f.localCalls).toBe(2);expect(f.remoteCalls).toBe(0)
    expect(session.messages.at(-1)?.content).toBe('LOCAL REPLY')
    expect(JSON.stringify(session)).not.toContain('synthetic-private-key')
    await prepared.close()
    await f.runtime.submitTurn(session.sessionKey,'Try after revocation.',()=>{})
    expect(f.localCalls).toBe(2);expect(f.remoteCalls).toBe(0)
    expect(session.messages.at(-1)?.turn_outcome).toMatchObject({reason:'turn_failed'})
  } finally {await prepared?.close();await f.close()}
})
it('keeping remote setup grants no local authority and preserves remote execution',async()=>{
  const f=await fixture()
  let prepared:Awaited<ReturnType<typeof prepareRemoteTask>>|undefined
  try {
    prepared=await prepareRemoteTask(f.remote,f.rpc,async()=>({kind:'remote'}))
    await f.attach(prepared)
    const session=f.runtime.listSessions().find(s=>s.id===prepared!.sessionId)!
    await f.runtime.submitTurn(session.sessionKey,'Use remote credentials.',()=>{})
    expect(f.localCalls).toBe(0);expect(f.remoteCalls).toBe(1)
    expect(f.rpc.mock.calls.some(c=>c[0]==='provider.relay.authorize')).toBe(false)
  } finally {await prepared?.close();await f.close()}
})
it('abort while reviewing prevents late approval from granting access',async()=>{
  const f=await fixture()
  try {
    await expect(prepareRemoteTask(f.remote,f.rpc,async()=>{f.controller.abort();return approve})).rejects.toThrow('Remote task setup failed')
    expect(f.rpc.mock.calls.some(c=>c[0]==='provider.relay.authorize')).toBe(false)
    expect(f.localCalls).toBe(0)
  } finally {await f.close()}
})
it('remote disconnect revokes the local grant and never falls back',async()=>{
  const f=await fixture()
  let prepared:Awaited<ReturnType<typeof prepareRemoteTask>>|undefined
  try {
    prepared=await prepareRemoteTask(f.remote,f.rpc,async()=>approve)
    await f.server.stop()
    await vi.waitFor(()=>expect(f.rpc.mock.calls.some(c=>c[0]==='provider.relay.revoke')).toBe(true))
    expect(f.localCalls).toBe(0);expect(f.remoteCalls).toBe(0)
  } finally {await prepared?.close();await f.close()}
})
it('a malformed inventory is rejected without echoing arbitrary data',()=>{
  expect(()=>shareableLocalProfiles({ok:true,profiles:[{name:'bad\nsecret',model:'m'}]})).toThrow('Remote task setup failed')
})

it('reopens saved tasks with the daemon-confirmed key and requires fresh approval after closure',async()=>{
  const f=await fixture()
  let first:Awaited<ReturnType<typeof prepareRemoteTask>>|undefined,second:Awaited<ReturnType<typeof prepareRemoteTask>>|undefined
  try {
    first=await prepareRemoteTask(f.remote,f.rpc,async()=>approve)
    const session=f.runtime.listSessions().find(s=>s.id===first!.sessionId)!
    await f.runtime.submitTurn(session.sessionKey,'Persist the first local task.',()=>{})
    await first.close()
    const before=f.localCalls
    let inspected:RemoteTaskReview|undefined
    second=await prepareRemoteTask({...f.remote,resumeSessionId:first.sessionId},f.rpc,async value=>{inspected=value;return approve})
    expect(inspected?.sessionId).toBe(first.sessionId)
    expect(inspected?.localRequirement).toContain('local')
    expect(f.localCalls).toBe(before)
    const renderer=await f.attach(second)
    const status=await renderer.request<{session:{id:string;key:string}}>('session.status',{structured:true,history_limit:0})
    expect(status.session.id).toBe(first.sessionId)
    expect(status.session.key).toBe(second.sessionKey)
    await renderer.request('turn.submit',{text:'Continue the same saved task.'})
    await vi.waitFor(()=>expect(f.localCalls).toBeGreaterThan(before))
    expect(f.remoteCalls).toBe(0)
  } finally {await second?.close();await first?.close();await f.close()}
})

it('preflight omits history on the actual wire while ordinary child resume still restores it',async()=>{
  const f=await fixture()
  const sockets=new Set<Socket>()
  let wire=''
  const proxy=createServer(client=>{
    const upstream=connect(f.remote.socketPath)
    sockets.add(client);sockets.add(upstream)
    upstream.on('data',data=>{wire+=data.toString()})
    client.on('error',()=>upstream.destroy());upstream.on('error',()=>client.destroy())
    client.on('close',()=>{upstream.destroy();sockets.delete(client)})
    upstream.on('close',()=>sockets.delete(upstream))
    client.pipe(upstream).pipe(client)
  })
  let prepared:Awaited<ReturnType<typeof prepareRemoteTask>>|undefined
  try {
    const seed=await f.runtime.openSession('history-seed','default',{cwd:f.dir})
    seed.messages.push({role:'user',content:'Existing conversation'})
    const marker='history-only-sentinel:'+'x'.repeat(4096)
    for(let i=0;i<100;i++)seed.messages.push({role:'assistant',content:marker})
    // The label includes surrounding explanatory text beyond the maximum
    // profile-name length. It must never disappear into a remote-default label.
    seed.metadata.local_provider_binding={profile:'p'.repeat(512)}
    await f.runtime.flushSessions()
    const socketPath=join(f.dir,'proxy.sock')
    await new Promise<void>(resolve=>proxy.listen(socketPath,resolve))
    let inspected:RemoteTaskReview|undefined
    prepared=await prepareRemoteTask({...f.remote,socketPath,resumeSessionId:seed.id},f.rpc,async value=>{inspected=value;return {kind:'remote'}})
    expect(inspected?.localRequirement).toContain('p'.repeat(512))
    expect(wire).not.toContain('history-only-sentinel')
    expect(wire).not.toContain('replay_assistant')
    expect(Buffer.byteLength(wire)).toBeLessThan(64*1024)
    expect(f.rpc.mock.calls.some(c=>c[0]==='provider.relay.authorize')).toBe(false)
    const renderer=await f.attach(prepared)
    const resumed=await renderer.request<{messages:{text?:string}[];message_count:number}>('session.resume',{session_id:seed.id})
    expect(resumed.message_count).toBe(101)
    expect(resumed.messages.filter(m=>m.text===marker)).toHaveLength(100)
  } finally {
    await prepared?.close()
    for(const socket of sockets)socket.destroy()
    await new Promise<void>(resolve=>proxy.close(()=>resolve()))
    await f.close()
  }
})
