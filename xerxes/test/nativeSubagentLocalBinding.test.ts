// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { BUILTIN_AGENTS } from '../src/agents/definitions.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { createNativeSubagentHost, type NativeSubagentHostOptions } from '../src/daemon/subagentHost.js'
import { RemoteProviderBindings, type RemoteProviderRequest } from '../src/daemon/remoteProviderBindings.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'

function fixture(reply = true, bundle = false) {
  const bindings = new RemoteProviderBindings(1000)
  const owner = {}, session = { id: 'local-parent', cwd: process.cwd(), metadata: {} as Record<string, unknown> }
  const frames: RemoteProviderRequest[] = []
  let fallbackCalls = 0
  const send = (frame: RemoteProviderRequest) => {
    frames.push(frame)
    if (reply) queueMicrotask(() => {
      try { bindings.reply(owner, frame.binding, frame.request_id, { done: true, deltas: [{ content: 'Approved local child reply.' }] }) } catch { /* cancelled request */ }
    })
    return true
  }
  const bind = () => bindings.bind(owner, session, { source: 'local', profile: 'approved', model: 'gpt-4o', ...(bundle ? {alternatives:[{source:'local',profile:'second',model:'gpt-4o'}]} : {}) }, send)
  bind()
  const registry = new ToolRegistry()
  const options: NativeSubagentHostOptions = {
    agentDefinitions: BUILTIN_AGENTS, cwd: session.cwd, model: 'gpt-4o', permissionMode: 'accept-all',
    eventBus: new DaemonSubagentEventBus(), tools: registry.definitions(), toolExecutor: registry,
    llm: { async *stream() { fallbackCalls++; yield { content: 'WRONG remote fallback' } } },
    resolveSourceClient: (source, model, profile) => {
      if (source !== session.id) throw new Error('Unknown source session')
      return bindings.sourceClient(session, model, profile)
    },
    resolveSourceProvider: () => { throw new Error('Local source must not resolve remote profiles') },
    validateInheritedSelection: async () => { throw new Error('Local source must not validate a remote provider') },
  }
  return { bindings, owner, session, frames, bind, options, fallbackCalls: () => fallbackCalls }
}

const spawn = (host: ReturnType<typeof createNativeSubagentHost>) => host.managerPort.spawn({
  promptProfile: 'default', sourceAgentId: 'local-parent', message: 'Complete the delegated task.',
})

test('a delegated explicit approved provider survives allocation when two routes share a model', async () => {
  const f = fixture(true,true), host = createNativeSubagentHost(f.options)
  try {
    const task = await host.managerPort.spawn({promptProfile:'default',sourceAgentId:f.session.id,message:'Use the second local provider.',agent:{id:'default',model:'gpt-4o',providerProfile:'second'}})
    await host.managerPort.wait([task.id],5000)
    const snapshot = host.managerPort.listHandles().find(item => item.id === task.id)!
    expect(snapshot.status).toBe('completed')
    expect(snapshot.providerProfile).toBe('second')
    expect(snapshot.providerRoute).toBe(f.bindings.sourceClient(f.session,'gpt-4o','second')!.route)
    expect(snapshot.providerRoute).not.toBe(f.bindings.sourceClient(f.session,'gpt-4o','approved')!.route)
    expect(f.fallbackCalls()).toBe(0)
  } finally {f.bindings.close();await host.manager.shutdown()}
})

test('native children use the parent local binding and persist only a route fingerprint', async () => {
  const f = fixture(), host = createNativeSubagentHost({ ...f.options, reasoningEffort: 'high', temperature: 1.9, topK: 999, maxTokens: 65536, topP: 0.99 })
  try {
    const task = await spawn(host)
    await host.managerPort.wait([task.id], 5000)
    const snapshot = host.managerPort.listHandles().find(item => item.id === task.id)!
    expect(snapshot.status).toBe('completed')
    expect(snapshot.providerRoute).toMatch(/^[a-f0-9]{64}$/)
    expect(snapshot.providerProfile).toBe('approved')
    expect(f.frames.some(frame => frame.frame.op === 'next')).toBe(true)
    const request = f.frames.find(frame => frame.frame.op === 'next' && frame.frame.request)?.frame
    if (request?.op !== 'next') throw new Error('Missing provider request')
    for (const field of ['thinking', 'temperature', 'topK', 'maxTokens', 'topP']) expect(request.request).not.toHaveProperty(field)
    expect(snapshot.reasoningEffort).toBeUndefined()
    expect(f.fallbackCalls()).toBe(0)
    expect(JSON.stringify(snapshot)).not.toContain('WRONG remote fallback')
  } finally { f.bindings.close(); await host.manager.shutdown() }
})

test('unapproved child model and profile overrides are rejected before allocation', async () => {
  const f = fixture(), host = createNativeSubagentHost(f.options)
  try {
    for (const agent of [{ model: 'different' }, { providerProfile: 'remote' }]) {
      await expect(host.managerPort.spawn({ promptProfile: 'default', sourceAgentId: 'local-parent', message: 'work', agent: { id: 'default', ...agent } })).rejects.toThrow('not covered')
    }
    expect(host.managerPort.listHandles()).toHaveLength(0)
    expect(f.frames).toHaveLength(0)
    expect(f.fallbackCalls()).toBe(0)
  } finally { f.bindings.close(); await host.manager.shutdown() }
})

test('local child preserves an explicit agent effort instead of the remote fallback', async () => {
  const f = fixture(), host = createNativeSubagentHost({ ...f.options, reasoningEffort: 'high' })
  try {
    const task = await host.managerPort.spawn({ promptProfile: 'default', sourceAgentId: f.session.id, message: 'work', agent: { id: 'default', reasoningEffort: 'low' } })
    await host.managerPort.wait([task.id], 5000)
    const request = f.frames.find(frame => frame.frame.op === 'next' && frame.frame.request)?.frame
    if (request?.op !== 'next') throw new Error('Missing provider request')
    expect(request.request?.thinking).toEqual({ effort: 'low' })
    expect(host.managerPort.listHandles().find(item => item.id === task.id)?.reasoningEffort).toBe('low')
  } finally { f.bindings.close(); await host.manager.shutdown() }
})

test('disconnect interrupts an active local child and never uses the remote fallback', async () => {
  const f = fixture(false), host = createNativeSubagentHost(f.options)
  try {
    const task = await spawn(host)
    for (let i = 0; i < 100 && !f.frames.length; i++) await Bun.sleep(5)
    expect(f.frames.length).toBeGreaterThan(0)
    f.bindings.disconnect(f.owner)
    await host.managerPort.wait([task.id], 5000)
    const snapshot = host.managerPort.listHandles().find(item => item.id === task.id)!
    expect(snapshot.status).not.toBe('completed')
    expect(snapshot.error).toContain('Check the parent session local connection and authorization')
    expect(f.frames.some(frame => frame.frame.op === 'cancel')).toBe(true)
    expect(f.fallbackCalls()).toBe(0)
  } finally { f.bindings.close(); await host.manager.shutdown() }
})

test('recovered child cannot turn a replacement binding into its original authority', async () => {
  const f = fixture(), first = createNativeSubagentHost(f.options)
  const task = await spawn(first)
  await first.managerPort.wait([task.id], 5000)
  const snapshot = first.managerPort.listHandles().find(item => item.id === task.id)!
  await first.manager.shutdown()
  f.bindings.disconnect(f.owner)
  f.bind()
  const restarted = createNativeSubagentHost(f.options)
  try {
    restarted.turnCoordinator.restore?.(f.session.id, [snapshot])
    restarted.managerPort.resume(snapshot.id)
    await expect(restarted.retry(snapshot.id, { message: 'retry', sourceAgentId: f.session.id })).rejects.toThrow('provider route changed')
    expect(f.fallbackCalls()).toBe(0)
  } finally { f.bindings.close(); await restarted.manager.shutdown() }
})

test('cancelling a native child cancels its pending private provider stream', async () => {
  const f = fixture(false), host = createNativeSubagentHost(f.options)
  try {
    const task = await spawn(host)
    for (let i = 0; i < 100 && !f.frames.length; i++) await Bun.sleep(5)
    expect(f.frames.length).toBeGreaterThan(0)
    host.managerPort.close(task.id)
    await host.managerPort.wait([task.id], 5000)
    for (let i = 0; i < 100 && !f.frames.some(frame => frame.frame.op === 'cancel'); i++) await Bun.sleep(5)
    expect(f.frames.some(frame => frame.frame.op === 'cancel')).toBe(true)
    expect(host.managerPort.listHandles().find(item => item.id === task.id)?.status).not.toBe('completed')
    expect(f.fallbackCalls()).toBe(0)
  } finally { f.bindings.close(); await host.manager.shutdown() }
})
