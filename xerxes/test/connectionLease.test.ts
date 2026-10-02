// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { ConnectionLeases } from '../src/daemon/connectionLease.js'
import type { DaemonTransportConnection } from '../src/daemon/transport.js'

function transport(key = 'original') {
  const frames: object[] = []
  const connection: DaemonTransportConnection = { activeSessionKey: key, send: frame => { frames.push(frame) } }
  return { connection, frames }
}

test('reconnect retains ownership, forwards to only the replacement, and follows session selection', () => {
  const expired: DaemonTransportConnection[] = []
  const leases = new ConnectionLeases(owner => { expired.push(owner) })
  const first = transport(), second = transport()
  try {
    const token = leases.enable(first.connection)
    const owner = leases.owner(first.connection)
    expect(token).toMatch(/^[a-f0-9]{64}$/)
    expect(leases.enable(first.connection)).toBe(token)
    owner.activeSessionKey = 'chosen'
    expect(first.connection.activeSessionKey).toBe('chosen')
    owner.send({ before: true })
    expect(leases.disconnect(first.connection)).toBe(true)
    const missed = { disconnected: true }
    owner.send(missed)
    missed.disconnected = false
    expect(leases.resume(second.connection, token)).toBe(owner)
    expect(leases.takeReplay(owner)).toEqual([{ disconnected: true }])
    owner.send({ after: true })
    expect(first.frames).toEqual([{ before: true }])
    expect(second.frames).toEqual([{ after: true }])
    expect(second.connection.activeSessionKey).toBe('chosen')
    expect(expired).toEqual([])
  } finally { leases.close() }
})

test('invalid tokens and attempts to take over a connected owner fail without changing either client', () => {
  const leases = new ConnectionLeases(() => {})
  const first = transport(), second = transport('other')
  try {
    const token = leases.enable(first.connection)
    expect(() => leases.resume(second.connection, 'invented')).toThrow('expired or invalid')
    expect(() => leases.resume(second.connection, token)).toThrow('already attached')
    expect(second.connection.activeSessionKey).toBe('other')
    expect(leases.owner(second.connection)).toBe(second.connection)
  } finally { leases.close() }
})

test('a disconnected lease expires once and cannot be reclaimed afterwards', async () => {
  const expired: DaemonTransportConnection[] = []
  const leases = new ConnectionLeases(owner => { expired.push(owner) }, 5)
  const first = transport()
  const token = leases.enable(first.connection)
  const owner = leases.owner(first.connection)
  leases.disconnect(first.connection)
  await Bun.sleep(20)
  expect(expired).toEqual([owner])
  expect(() => leases.resume(transport().connection, token)).toThrow('expired or invalid')
  leases.close()
  expect(expired).toHaveLength(1)
})

test('a reconnect that never initializes stops journaling without orphaning its transport; detached overflow expires', async () => {
  const expired: DaemonTransportConnection[] = []
  const leases = new ConnectionLeases(owner => { expired.push(owner) }, 5)
  const first = transport(), second = transport()
  const token = leases.enable(first.connection)
  const owner = leases.owner(first.connection)
  leases.disconnect(first.connection)
  leases.resume(second.connection, token)
  owner.send({ missed: true })
  await Bun.sleep(20)
  expect(expired).toEqual([])
  expect(leases.owner(second.connection)).toBe(owner)
  expect(leases.takeReplay(owner)).toBeUndefined()
  owner.send({ live: true })
  expect(second.frames).toEqual([{ live: true }])
  const next = transport()
  const nextToken = leases.enable(next.connection)
  const nextOwner = leases.owner(next.connection)
  leases.disconnect(next.connection)
  nextOwner.send({ text: 'x'.repeat(1024 * 1024) })
  expect(leases.has(nextToken)).toBe(false)
  expect(expired).toEqual([nextOwner])
  leases.close()
  expect(expired).toEqual([nextOwner, owner])
})

test('a reconnect late in the grace window keeps the owner past the original outage deadline', async () => {
  const expired: DaemonTransportConnection[] = []
  const leases = new ConnectionLeases(owner => { expired.push(owner) }, 60)
  const first = transport(), second = transport()
  try {
    const token = leases.enable(first.connection)
    const owner = leases.owner(first.connection)
    leases.disconnect(first.connection)
    owner.send({ missed: true })
    await Bun.sleep(45)
    leases.resume(second.connection, token)
    // initialize is still awaiting when the outage deadline passes.
    await Bun.sleep(30)
    expect(expired).toEqual([])
    expect(leases.owner(second.connection)).toBe(owner)
    expect(leases.takeReplay(owner)).toEqual([{ missed: true }])
    owner.send({ live: true })
    expect(second.frames).toEqual([{ live: true }])
  } finally { leases.close() }
})

test('journal overflow while a reconnected transport restores drops the replay, not the owner', () => {
  const expired: DaemonTransportConnection[] = []
  const leases = new ConnectionLeases(owner => { expired.push(owner) })
  const first = transport(), second = transport()
  try {
    const token = leases.enable(first.connection)
    const owner = leases.owner(first.connection)
    leases.disconnect(first.connection)
    owner.send({ text: 'x'.repeat(1000 * 1024) })
    leases.resume(second.connection, token)
    const overflow = { text: 'y'.repeat(100 * 1024) }
    owner.send(overflow)
    expect(expired).toEqual([])
    expect(leases.has(token)).toBe(true)
    expect(leases.owner(second.connection)).toBe(owner)
    expect(leases.takeReplay(owner)).toBeUndefined()
    expect(second.frames).toEqual([overflow])
  } finally { leases.close() }
})

test('reconnection clears expiry; shutdown releases connected and detached owners exactly once', async () => {
  const expired: DaemonTransportConnection[] = []
  const leases = new ConnectionLeases(owner => { expired.push(owner) }, 5)
  const first = transport(), second = transport(), other = transport('other')
  const token = leases.enable(first.connection)
  leases.disconnect(first.connection)
  leases.takeReplay(leases.resume(second.connection, token))
  const otherToken = leases.enable(other.connection)
  await Bun.sleep(20)
  expect(expired).toEqual([])
  leases.disconnect(other.connection)
  leases.close()
  await Bun.sleep(20)
  expect(expired).toHaveLength(2)
  expect(() => leases.resume(transport().connection, otherToken)).toThrow('expired or invalid')
  expect(leases.disconnect(transport().connection)).toBe(false)
})
