// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, it } from 'vitest'
import { mkdtemp, realpath, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { InMemoryDaemonRuntime } from '../../daemon/runtime.js'
import { DaemonServer } from '../../daemon/server.js'
import { ProfileStore } from '../../bridge/profiles.js'
import { GatewayClient } from '../gatewayClient.js'

it('permissions reports the selected session policy without changing another session', async () => {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'xr-permission-rpc-')))
  const runtime = new InMemoryDaemonRuntime(undefined, {currentProjectDirectory:directory, sessionDirectory:join(directory,'sessions'), permissionMode:'auto'})
  const server = new DaemonServer({runtime, socketPath:join(directory,'rpc.sock'), projectDirectory:directory, profileStore:new ProfileStore(join(directory,'profiles.json'))})
  const first = new GatewayClient({externalSocketPath:join(directory,'rpc.sock'), projectDir:directory})
  const second = new GatewayClient({externalSocketPath:join(directory,'rpc.sock'), projectDir:directory})
  try {
    await server.start(); await first.start(); await second.start()
    const created = await first.request<{session_id:string}>('session.create'); await second.request('session.create')
    expect(await first.request('slash.exec',{command:'permissions manual'})).toMatchObject({ok:true,permission_mode:'manual'})
    expect(await first.request('slash.exec',{command:'permissions'})).toMatchObject({ok:true,permission_mode:'manual'})
    expect(await second.request('slash.exec',{command:'permissions'})).toMatchObject({ok:true,permission_mode:'auto'})
    expect(await first.request('slash.exec',{command:'permissions nonsense'})).toMatchObject({ok:false})
    expect(await first.request('slash.exec',{command:'permissions'})).toMatchObject({permission_mode:'manual'})
    const policies: string[] = []
    second.on('event', event => {
      if (event.type === 'session.info' && event.payload.permission_mode) policies.push(event.payload.permission_mode)
    })
    await second.request('session.resume', {session_id:created.session_id})
    expect(policies.length).toBeGreaterThan(0)
    expect(new Set(policies)).toEqual(new Set(['manual']))
  } finally {first.close();second.close();await server.stop();await runtime.shutdown();await rm(directory,{recursive:true,force:true})}
})
