// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { RemoteProviderBindings, LOCAL_PROVIDER_BINDING, type RemoteProviderRequest } from '../src/daemon/remoteProviderBindings.js'
import { ConnectionLeases } from '../src/daemon/connectionLease.js'

const selection = { source: 'local workstation', profile: 'chosen', model: 'gpt-4o' }
const completion = { model: 'gpt-4o', messages: [{ role: 'user' as const, content: 'Private model context' }] }
const session = () => ({ id: 'session-a', cwd: '/workspace/a', metadata: {} as Record<string, unknown> })

test('binding cannot be hijacked, moved to another workspace or changed to another model', async () => {
  const bindings = new RemoteProviderBindings(), owner = {}, other = {}, s = session()
  const frames: RemoteProviderRequest[] = []
  try {
    expect(bindings.client(s,selection.model)).toBeUndefined()
    bindings.bind(owner,s,selection,frame=>{frames.push(frame);return true})
    expect(()=>bindings.bind(other,s,selection,()=>true)).toThrow('unavailable')
    expect(()=>bindings.client({...s,cwd:'/workspace/b'},selection.model)).toThrow('unavailable')
    expect(()=>bindings.client(s,'different-model')).toThrow('not covered')
    const stream=bindings.client(s,selection.model)!.stream(completion)[Symbol.asyncIterator]()
    const pending=stream.next()
    const frame=frames[0]!
    expect(()=>bindings.reply(other,frame.binding,frame.request_id,{done:true})).toThrow('unavailable')
    bindings.reply(owner,frame.binding,frame.request_id,{done:true,deltas:[{content:'approved'}]})
    expect(await pending).toMatchObject({value:{content:'approved'}})
    expect(await stream.next()).toMatchObject({done:true})
    expect(()=>bindings.reply(owner,frame.binding,frame.request_id,{done:true})).toThrow('unavailable')
    bindings.disconnect(owner)
    expect(()=>bindings.client(s,selection.model)).toThrow('unavailable')
    expect(s.metadata).toHaveProperty(LOCAL_PROVIDER_BINDING)
  } finally {bindings.close()}
})

test('cancel and timeout settle pending pulls and late replies cannot revive them', async () => {
  const bindings=new RemoteProviderBindings(20),owner={},s=session(),frames:RemoteProviderRequest[]=[]
  try {
    bindings.bind(owner,s,selection,frame=>{frames.push(frame);return true})
    const abort=new AbortController()
    const next=bindings.client(s,selection.model)!.stream(completion,abort.signal)[Symbol.asyncIterator]().next().catch(error=>error)
    abort.abort()
    expect(await next).toHaveProperty('code','cancelled')
    expect(()=>bindings.reply(owner,frames[0]!.binding,frames[0]!.request_id,{done:true})).toThrow('unavailable')
    const timed=bindings.client(s,selection.model)!.stream(completion)[Symbol.asyncIterator]().next().catch(error=>error)
    await Bun.sleep(35)
    expect(await timed).toHaveProperty('code','grant_unavailable')
  } finally {bindings.close()}
})

test('private model context is never put in a reconnect journal', () => {
  const frames: object[]=[]
  const transport={activeSessionKey:'session',send:(frame:object)=>frames.push(frame)}
  const leases=new ConnectionLeases(()=>{},1000)
  try {
    const token=leases.enable(transport),owner=leases.owner(transport)
    expect(leases.sendPrivate(owner,{private:'before'})).toBe(true)
    expect(frames).toEqual([{private:'before'}])
    leases.disconnect(transport)
    expect(leases.sendPrivate(owner,{private:'offline context'})).toBe(false)
    const replacement={activeSessionKey:'session',send:(frame:object)=>frames.push(frame)}
    leases.resume(replacement,token)
    expect(leases.sendPrivate(owner,{private:'restoration context'})).toBe(false)
    expect(leases.takeReplay(owner)).toEqual([])
    expect(leases.sendPrivate(owner,{private:'after'})).toBe(true)
    expect(frames).toEqual([{private:'before'},{private:'after'}])
  } finally {leases.close()}
})

test('paused relay consumers remain bounded and release cancels every tracked stream', async () => {
  const bindings=new RemoteProviderBindings(),owner={},s=session(),frames:RemoteProviderRequest[]=[]
  try {
    bindings.bind(owner,s,selection,frame=>{frames.push(frame);return true})
    for(let i=0;i<16;i++){
      const next=bindings.client(s,selection.model)!.stream(completion)[Symbol.asyncIterator]().next()
      const frame=frames.at(-1)!
      bindings.reply(owner,frame.binding,frame.request_id,{done:false,deltas:[{content:'partial'}]})
      expect(await next).toMatchObject({value:{content:'partial'}})
    }
    const count=frames.filter(frame=>frame.frame.op==='next').length
    await expect(bindings.client(s,selection.model)!.stream(completion)[Symbol.asyncIterator]().next()).rejects.toHaveProperty('code','concurrency_limit')
    expect(frames.filter(frame=>frame.frame.op==='next')).toHaveLength(count)
    bindings.disconnect(owner)
    const cancelled=new Set(frames.filter(frame=>frame.frame.op==='cancel').map(frame=>frame.frame.id))
    for(const frame of frames.filter(frame=>frame.frame.op==='next'))expect(cancelled.has(frame.frame.id)).toBe(true)
  } finally {bindings.close()}
})

test('explicit remote override refuses pending and paused streams, then removes authority and requirement', async () => {
  const bindings = new RemoteProviderBindings(), owner = {}, s = session(), frames: RemoteProviderRequest[] = []
  try {
    bindings.bind(owner, s, selection, frame => { frames.push(frame); return true })
    const client = bindings.client(s, selection.model)!
    const stream = client.stream(completion)[Symbol.asyncIterator]()
    const pending = stream.next()
    expect(() => bindings.useRemote(s)).toThrow('concurrency')
    const first = frames.at(-1)!
    bindings.reply(owner, first.binding, first.request_id, { done: false, deltas: [{ content: 'partial' }] })
    await pending
    expect(() => bindings.useRemote(s)).toThrow('concurrency')
    const finished = stream.next()
    const last = frames.at(-1)!
    bindings.reply(owner, last.binding, last.request_id, { done: true })
    await finished
    bindings.useRemote(s)
    expect(Object.hasOwn(s.metadata, LOCAL_PROVIDER_BINDING)).toBe(false)
    expect(bindings.client(s, selection.model)).toBeUndefined()
    await expect(client.stream(completion)[Symbol.asyncIterator]().next()).rejects.toHaveProperty('code', 'grant_unavailable')
  } finally { bindings.close() }
})
