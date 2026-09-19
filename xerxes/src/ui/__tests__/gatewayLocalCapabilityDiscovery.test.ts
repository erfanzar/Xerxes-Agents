// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import {expect,it,vi} from 'vitest'
import {mkdtemp,realpath,rm} from 'node:fs/promises'
import {tmpdir} from 'node:os'
import {join} from 'node:path'
import {ProfileStore} from '../../bridge/profiles.js'
import {DaemonServer} from '../../daemon/server.js'
import {InMemoryDaemonRuntime} from '../../daemon/runtime.js'
import {RemoteProviderBindings} from '../../daemon/remoteProviderBindings.js'
import {GatewayClient} from '../gatewayClient.js'

it('local binding suppresses stale remote discovery notices and resume does not query the remote provider',async()=>{
  const dir=await realpath(await mkdtemp(join(tmpdir(),'xr-local-discovery-')))
  const release=Promise.withResolvers<void>();let calls=0
  const http=Bun.serve({hostname:'127.0.0.1',port:0,idleTimeout:0,async fetch(){calls++;await release.promise;return new Response('fixture unavailable',{status:503})}})
  const profiles=new ProfileStore(join(dir,'profiles.json'))
  profiles.save({name:'remote',provider:'openai',model:'gpt-5',baseUrl:http.url.href+'v1',apiKey:'synthetic-only'})
  const runtime=new InMemoryDaemonRuntime(undefined,{model:'gpt-5',runtimeSettings:{provider:'openai',model:'gpt-5',base_url:http.url.href+'v1'},currentProjectDirectory:dir,sessionDirectory:join(dir,'sessions')})
  const bindings=new RemoteProviderBindings()
  const server=new DaemonServer({runtime,profileStore:profiles,socketPath:join(dir,'rpc.sock'),projectDirectory:dir,remoteProviderBindings:bindings,autoDiscoverModelCapabilities:true})
  const client=new GatewayClient({externalSocketPath:join(dir,'rpc.sock'),projectDir:dir})
  const events:unknown[]=[];client.on('event',event=>events.push(event))
  try {
    await server.start();await client.start()
    const created=await client.request<{session_id:string}>('session.create')
    await vi.waitFor(()=>expect(calls).toBe(1))
    expect(await client.request('provider.remote.bind',{consent:true,source:'local fixture',profile:'local',model:'gpt-4o'})).toMatchObject({ok:true})
    events.length=0;release.resolve();await Bun.sleep(50)
    expect(JSON.stringify(events)).not.toContain('Model capability discovery')
    await client.request('session.resume',{session_id:created.session_id,history_limit:0})
    await Bun.sleep(50)
    expect(calls).toBe(1)
    expect(JSON.stringify(events)).not.toContain('Model capability discovery')
  } finally {release.resolve();client.close();await server.stop();await runtime.shutdown();bindings.close();http.stop(true);await rm(dir,{recursive:true,force:true})}
})
