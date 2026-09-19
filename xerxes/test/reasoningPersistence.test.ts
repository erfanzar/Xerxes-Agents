// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import {expect,spyOn,test} from 'bun:test'
import {mkdtemp,rm} from 'node:fs/promises'
import {tmpdir} from 'node:os'
import {join} from 'node:path'
import {InMemoryDaemonRuntime} from '../src/daemon/runtime.js'
import {DaemonTranscriptStore} from '../src/session/daemonTranscript.js'

async function fixture(){
  const dir=await mkdtemp(join(tmpdir(),'xr-effort-save-'))
  const store=new DaemonTranscriptStore({directory:join(dir,'sessions'),currentProjectDirectory:dir})
  const runtime=new InMemoryDaemonRuntime(undefined,{currentProjectDirectory:dir,transcriptStore:store})
  const session=await runtime.openSession('task')
  session.messages.push({role:'user',content:'Existing task'},{role:'assistant',content:'Existing reply'})
  await runtime.setSessionReasoning('task','low')
  return {runtime,session,store,read:()=>Bun.file(join(dir,'sessions',session.id+'.json')).json(),
    async close(){await runtime.shutdown();await rm(dir,{recursive:true,force:true})}}
}

test('a failed effort save preserves live and durable settings and context deltas',async()=>{
  const f=await fixture()
  try {
    const before={effort:f.session.reasoningEffort,pinned:f.session.reasoningPinned,lastActive:f.session.lastActive,metadata:structuredClone(f.session.metadata)}
    const failure=spyOn(f.store,'save').mockRejectedValueOnce(new Error('fixture storage failed'))
    try {await expect(f.runtime.setSessionReasoning('task','high')).rejects.toThrow('previous effort is unchanged')}
    finally {failure.mockRestore()}
    expect({effort:f.session.reasoningEffort,pinned:f.session.reasoningPinned,lastActive:f.session.lastActive,metadata:f.session.metadata}).toEqual(before)
    expect((await f.read()).metadata.reasoning_effort).toBe('low')
    await f.runtime.flushSessions()
    expect((await f.read()).metadata.reasoning_effort).toBe('low')
  } finally {await f.close()}
})

test('a pending failed effort does not escape into another selection or a concurrent flush',async()=>{
  const f=await fixture(),started=Promise.withResolvers<void>(),finish=Promise.withResolvers<void>()
  try {
    const save=spyOn(f.store,'save').mockImplementationOnce(async()=>{started.resolve();await finish.promise;throw new Error('first save failed')})
    const first=f.runtime.setSessionReasoning('task','high').catch(error=>error.message)
    await started.promise
    const second=f.runtime.setSessionReasoning('task','medium'),flush=f.runtime.flushSessions()
    const visibleWhileSaving=f.session.reasoningEffort
    finish.resolve()
    expect(await first).toContain('previous effort is unchanged')
    expect(await first).not.toContain('first save failed')
    await second;await flush;save.mockRestore()
    expect(visibleWhileSaving).toBe('low')
    expect(f.session.reasoningEffort).toBe('medium')
    expect((await f.read()).metadata.reasoning_effort).toBe('medium')
    expect(JSON.stringify(f.session.metadata.context_deltas)).not.toContain('high')
  } finally {finish.resolve();await f.close()}
})
