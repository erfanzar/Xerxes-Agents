// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  CapabilityRegistry,
  type CapabilityManifest,
} from '../src/runtime/capabilityRegistry.js'

test('empty registry denies every capability request', () => {
  const registry = new CapabilityRegistry()
  expect(registry.isAllowed('plugin-a', 'filesystem:read')).toBeFalse()
  expect(registry.isAllowed('plugin-a', 'network:outbound')).toBeFalse()
  expect(registry.isAllowed('plugin-a', 'subprocess:spawn')).toBeFalse()
})

test('registering a manifest grants only declared capabilities', () => {
  const registry = new CapabilityRegistry()
  const manifest: CapabilityManifest = {
    id: 'plugin-a',
    capabilities: [
      { scope: 'filesystem', action: 'read', resources: ['README.md', 'src/**'] },
      { scope: 'network', action: 'outbound', hosts: ['api.example.com'] },
    ],
  }

  registry.register(manifest)
  expect(registry.isAllowed('plugin-a', 'filesystem:read', 'README.md')).toBeTrue()
  expect(registry.isAllowed('plugin-a', 'filesystem:read', 'package.json')).toBeFalse()
  expect(registry.isAllowed('plugin-a', 'network:outbound', 'api.example.com')).toBeTrue()
  expect(registry.isAllowed('plugin-a', 'network:outbound', 'evil.com')).toBeFalse()
  expect(registry.isAllowed('plugin-a', 'subprocess:spawn')).toBeFalse()
})

test('transactional registration rolls back on failure and restores prior state', () => {
  const registry = new CapabilityRegistry()
  registry.register({ id: 'plugin-a', capabilities: [{ scope: 'filesystem', action: 'read' }] })

  let rolledBack = false
  try {
    registry.transaction(tx => {
      tx.register({ id: 'plugin-b', capabilities: [{ scope: 'network', action: 'outbound' }] })
      tx.register({ id: 'plugin-a', capabilities: [{ scope: 'filesystem', action: 'write' }] })
      throw new Error('planned failure')
    })
  } catch (error) {
    rolledBack = String(error).includes('planned failure')
  }

  expect(rolledBack).toBeTrue()
  expect(registry.isAllowed('plugin-b', 'network:outbound')).toBeFalse()
  expect(registry.isAllowed('plugin-a', 'filesystem:write')).toBeFalse()
  expect(registry.isAllowed('plugin-a', 'filesystem:read')).toBeTrue()
})

test('diff shows added, removed, and changed capabilities before commit', () => {
  const registry = new CapabilityRegistry()
  registry.register({ id: 'plugin-a', capabilities: [{ scope: 'filesystem', action: 'read' }] })

  const diff = registry.diff([
    { id: 'plugin-a', capabilities: [{ scope: 'filesystem', action: 'read' }, { scope: 'filesystem', action: 'write' }] },
    { id: 'plugin-b', capabilities: [{ scope: 'network', action: 'outbound' }] },
  ])

  expect(diff.added).toEqual([
    { pluginId: 'plugin-a', capability: { scope: 'filesystem', action: 'write' } },
    { pluginId: 'plugin-b', capability: { scope: 'network', action: 'outbound' } },
  ])
  expect(diff.removed).toEqual([])
})

test('wildcard capability grants all resources within a scope and action', () => {
  const registry = new CapabilityRegistry()
  registry.register({ id: 'plugin-a', capabilities: [{ scope: 'filesystem', action: 'read' }] })
  expect(registry.isAllowed('plugin-a', 'filesystem:read', 'anything')).toBeTrue()
})

test('unregister removes all capabilities for a plugin', () => {
  const registry = new CapabilityRegistry()
  registry.register({ id: 'plugin-a', capabilities: [{ scope: 'filesystem', action: 'read' }] })
  registry.unregister('plugin-a')
  expect(registry.isAllowed('plugin-a', 'filesystem:read')).toBeFalse()
})

test('an empty allow-list permits nothing, and a dual-axis grant is satisfiable', () => {
  const registry = new CapabilityRegistry()

  // "no resources permitted" is the natural reading of an empty list, and it
  // used to grant every path because matchesAny returned true when empty.
  registry.register({ id: 'locked', capabilities: [{ scope: 'fs', action: 'read', resources: [] }] })
  expect(registry.isAllowed('locked', 'fs:read', '/etc/passwd')).toBe(false)

  // Omitting the field is how "unscoped on this axis" is expressed.
  registry.register({ id: 'open', capabilities: [{ scope: 'fs', action: 'read' }] })
  expect(registry.isAllowed('open', 'fs:read', '/etc/passwd')).toBe(true)

  registry.register({ id: 'scoped', capabilities: [{ scope: 'fs', action: 'read', resources: ['/work/**'] }] })
  expect(registry.isAllowed('scoped', 'fs:read', '/work/a.txt')).toBe(true)
  expect(registry.isAllowed('scoped', 'fs:read', '/etc/passwd')).toBe(false)

  // A grant carrying both axes: the target is one string — a path or a host —
  // so requiring it to match BOTH lists made the grant unsatisfiable.
  registry.register({
    id: 'both',
    capabilities: [{ scope: 'net', action: 'fetch', resources: ['/work/**'], hosts: ['api.example.com'] }],
  })
  expect(registry.isAllowed('both', 'net:fetch', 'api.example.com')).toBe(true)
  expect(registry.isAllowed('both', 'net:fetch', '/work/a.txt')).toBe(true)
  expect(registry.isAllowed('both', 'net:fetch', 'evil.example.com')).toBe(false)
})

test('an async transaction rolls back when it rejects', async () => {
  const registry = new CapabilityRegistry()
  registry.register({ id: 'p', capabilities: [{ scope: 'fs', action: 'read', resources: ['/work/**'] }] })

  await expect(registry.transaction(async tx => {
    tx.register({ id: 'p', capabilities: [{ scope: 'fs', action: 'read' }] })
    throw new Error('half applied')
  })).rejects.toThrow('half applied')

  // The widened grant must not survive the rejection.
  expect(registry.isAllowed('p', 'fs:read', '/etc/passwd')).toBe(false)
  expect(registry.isAllowed('p', 'fs:read', '/work/a.txt')).toBe(true)
})
