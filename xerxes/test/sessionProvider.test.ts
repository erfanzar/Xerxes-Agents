// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { ProfileStore } from '../src/bridge/profiles.js'
import { sessionProvider, profileAcceptsModel } from '../src/daemon/sessionProvider.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { httpErrorBody } from '../src/llms/httpErrorBody.js'

test('a resumed GPT chat resolves Codex even with a polluted active Kimi model',async()=>{
  const root=await mkdtemp(join(tmpdir(),'xr-provider-pin-'))
  try{
    const profiles=new ProfileStore(join(root,'profiles.json'))
    profiles.save({name:'codex',provider:'openai-codex',model:'gpt-6-astra',apiKey:'',baseUrl:''})
    profiles.save({name:'kimi',provider:'kimi-code',model:'gpt-6-astra',apiKey:'fixture',baseUrl:'https://example.invalid'})
    profiles.setActive('kimi')
    const calls:string[]=[]
    const runner=new AgentTurnRunner({model:'gpt-6-astra',llm:{async *stream(){throw new Error('Wrong global Kimi transport');yield {content:''}}},
      resolveSessionProvider:(session,model)=>{const profile=sessionProvider(profiles,session,model)!;return {llm:{async *stream(){calls.push(profile.name);yield {content:'hello'}}},providerOverrides:{provider:profile.provider}}}})
    const runtime=new InMemoryDaemonRuntime(runner,{model:'gpt-6-astra',currentProjectDirectory:root,sessionDirectory:join(root,'sessions')})
    const chat=await runtime.openSession('gpt-chat')
    await runtime.submitTurn(chat.sessionKey,'hello',()=>{})
    expect(calls).toEqual(['codex'])
    expect(chat.metadata.provider_profile).toBe('codex')
    profiles.setActive('kimi')
    await runtime.submitTurn(chat.sessionKey,'again',()=>{})
    expect(calls).toEqual(['codex','codex'])
    const resumedRuntime=new InMemoryDaemonRuntime(runner,{model:'kimi-for-coding',currentProjectDirectory:root,sessionDirectory:join(root,'sessions')})
    const resumed=await resumedRuntime.openSession(chat.id,undefined,{resume:true})
    expect(resumed.metadata.provider_profile).toBe('codex')
    await resumedRuntime.submitTurn(resumed.sessionKey,'after restart',()=>{})
    expect(calls.at(-1)).toBe('codex')
    expect(()=>sessionProvider(profiles,{metadata:{provider_profile:'kimi'}},'gpt-6-astra')).toThrow('cannot serve')
    expect(profileAcceptsModel(profiles.get('kimi')!,'kimi-for-coding')).toBe(true)
    expect(profileAcceptsModel(profiles.get('codex')!,'codex/gpt-6-astra')).toBe(true)
  }finally{await rm(root,{recursive:true,force:true})}
})
test('HTML provider errors are concise and do not echo challenge page IPs or scripts',()=>{
  const message=httpErrorBody('<!DOCTYPE html><html>Cloudflare <script>secret()</script>Your IP address: 10.0.0.1</html>')
  expect(message).toContain('blocked by Cloudflare')
  expect(message).not.toContain('<')
  expect(message).not.toContain('10.0.0.1')
  expect(httpErrorBody('{"error":"quota"}')).toBe('{"error":"quota"}')
})

test('changing a profile during streaming survives the finishing turn and restart', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xr-live-provider-'))
  let finish!: () => void
  let started!: () => void
  const waiting = new Promise<void>(resolve => { finish = resolve })
  const ready = new Promise<void>(resolve => { started = resolve })
  try {
    const calls: unknown[] = []
    const runner = new AgentTurnRunner({ model: 'gpt-6-astra', llm: { async *stream() { throw new Error('Wrong default'); yield { content: '' } } },
      resolveSessionProvider: session => ({ llm: { async *stream() {
        calls.push(session.metadata.provider_profile)
        started()
        await waiting
        yield { content: 'done' }
      } } }),
    })
    const runtime = new InMemoryDaemonRuntime(runner, { model: 'gpt-6-astra', currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
    const chat = await runtime.openSession('chat')
    chat.metadata.provider_profile = 'first'
    const turn = runtime.submitTurn(chat.sessionKey, 'hello', () => {})
    await ready
    await runtime.setSessionModel(chat.sessionKey, 'gpt-6-astra', 'second')
    finish()
    await turn
    expect(chat.metadata.provider_profile).toBe('second')
    await runtime.submitTurn(chat.sessionKey, 'next', () => {})
    expect(calls).toEqual(['first', 'second'])
    const restarted = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
    const resumed = await restarted.openSession(chat.id, undefined, { resume: true })
    expect(resumed.metadata.provider_profile).toBe('second')
  } finally { finish(); await rm(root, { recursive: true, force: true }) }
})

test('a persisted provider pin cannot fall back after the last API profile disappears', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xr-missing-provider-'))
  try {
    let calls = 0
    const fallback = { async *stream() { calls++; yield { content: 'wrong provider' } } }
    const profiles = { list: () => [], get: () => undefined, active: () => undefined }
    const runner = new AgentTurnRunner({ model: 'gpt-4o', llm: fallback,
      resolveSessionProvider: (session, model) => { sessionProvider(profiles, session, model); return { llm: fallback } },
    })
    const options = { model: 'gpt-4o', currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') }
    const runtime = new InMemoryDaemonRuntime(runner, options)
    const original = await runtime.openSession('pinned')
    await runtime.submitTurn(original.sessionKey, 'first exchange', () => {})
    calls = 0
    await runtime.setSessionModel(original.sessionKey, 'gpt-4o', 'original-provider')
    const restarted = new InMemoryDaemonRuntime(runner, options)
    const resumed = await restarted.openSession(original.id, undefined, { resume: true })
    expect(resumed.metadata.provider_profile).toBe('original-provider')
    const events: unknown[] = []
    await restarted.submitTurn(resumed.sessionKey, 'continue', event => { events.push(event) })
    expect(calls).toBe(0)
    expect(JSON.stringify(events)).toContain('Use /model')
    expect(resumed.metadata.provider_profile).toBe('original-provider')
    expect(resumed.status).toBe('idle')
    // Unpinned setups can still use an explicitly configured runtime client.
    expect(sessionProvider(profiles, { metadata: {} }, 'gpt-4o')).toBeUndefined()
  } finally { await rm(root, { recursive: true, force: true }) }
})
