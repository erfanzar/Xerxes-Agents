// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, spyOn, test } from 'bun:test'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  CircularDependencyError,
  DependencyResolver,
  HookRunner,
  PluginConflictError,
  PluginRegistry,
  PluginType,
  TOOL_PERMISSION_HOOK,
  VersionConstraint,
  parseDependency,
  resolveToolPermission,
} from '../src/index.js'

test('plugin dependency constraints and load order preserve compatible releases', () => {
  expect(new VersionConstraint('>=1.2,<2.0').satisfies('1.4.0')).toBe(true)
  expect(new VersionConstraint('~=1.2.3').satisfies('1.3.0')).toBe(false)
  expect(parseDependency('tools>=2.0')).toEqual({ name: 'tools', versionConstraint: '>=2.0' })
  expect(new DependencyResolver().topologicalSort({ app: ['tools'], tools: [] })).toEqual(['tools', 'app'])
  expect(() => new DependencyResolver().topologicalSort({ app: ['tools'], tools: ['app'] })).toThrow(CircularDependencyError)
})

test('hook runner threads mutation values and isolates observer failures', async () => {
  const hooks = new HookRunner()
  hooks.register('before_tool_call', payload => ({ ...(payload.arguments as Record<string, unknown>), safe: true }))
  hooks.register('on_turn_end', () => { throw new Error('isolated') })
  hooks.register('on_turn_end', async () => 'recorded')
  const payload: Record<string, unknown> = { arguments: { path: 'file.ts' } }
  expect(await hooks.run('before_tool_call', payload)).toEqual({ path: 'file.ts', safe: true })
  expect(payload.arguments).toEqual({ path: 'file.ts', safe: true })
  expect(await hooks.runAsync('on_turn_end')).toEqual(['recorded'])
})

test('plugin registry owns resources, validates dependencies, and removes all owned capabilities', () => {
  const registry = new PluginRegistry()
  registry.registerPlugin({ name: 'base', version: '1.2.0', pluginType: PluginType.TOOL })
  registry.registerTool('base_tool', () => 'ok', undefined, 'base')
  registry.registerPlugin({ name: 'dependent', dependencies: ['base>=2.0'] })
  expect(registry.validateDependencies()).toEqual(["Plugin 'dependent' has version conflict: base: requires >=2.0, found 1.2.0"])
  expect(registry.getLoadOrder()).toEqual(['base', 'dependent'])
  expect(() => registry.registerTool('base_tool', () => 'again')).toThrow(PluginConflictError)
  registry.unregisterPlugin('base')
  expect(registry.getTool('base_tool')).toBeUndefined()
})

test('hook runner awaits asynchronous hooks and resolves the complete result list', async () => {
  const hooks = new HookRunner()
  const order: string[] = []
  hooks.register('on_turn_end', async () => {
    await Bun.sleep(10)
    order.push('async')
    return 'async-result'
  })
  hooks.register('on_turn_end', () => {
    order.push('sync')
    return 'sync-result'
  })
  expect(await hooks.run('on_turn_end')).toEqual(['async-result', 'sync-result'])
  expect(order).toEqual(['async', 'sync'])

  const payload: Record<string, unknown> = { arguments: { path: 'a.ts' } }
  hooks.register('before_tool_call', async current => {
    await Bun.sleep(1)
    return { ...(current.arguments as Record<string, unknown>), checked: true }
  })
  expect(await hooks.run('before_tool_call', payload)).toEqual({ path: 'a.ts', checked: true })
  expect(payload.arguments).toEqual({ path: 'a.ts', checked: true })
})

test('hook runner surfaces failures without breaking dispatch', async () => {
  const errors: unknown[][] = []
  const spy = spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
    errors.push(args)
  })
  try {
    const hooks = new HookRunner()
    hooks.register('on_turn_end', () => { throw new Error('observer exploded') })
    hooks.register('on_turn_end', () => 'healthy')
    expect(await hooks.run('on_turn_end')).toEqual(['healthy'])
    expect(errors).toHaveLength(1)
    expect(String(errors[0]?.[0])).toContain("Hook 'on_turn_end' callback failed: observer exploded")
  } finally {
    spy.mockRestore()
  }
})

test('plugin discovery rolls back partial registrations and reports the failure', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-plugins-'))
  try {
    await writeFile(join(directory, 'broken.mjs'), `
export function register(registry) {
  registry.registerPlugin({ name: 'broken-plugin' })
  registry.registerTool('broken_tool', () => 'nope', undefined, 'broken-plugin')
  registry.registerTool('standalone_tool', () => 'nope')
  registry.registerHook('on_turn_end', () => 'nope')
  throw new Error('register exploded')
}
`, 'utf8')
    await writeFile(join(directory, 'counted.mjs'), `
globalThis.__xerxesPluginExecutions = (globalThis.__xerxesPluginExecutions ?? 0) + 1
export function register(registry) {
  registry.registerPlugin({ name: 'counted-plugin' })
}
`, 'utf8')
    await writeFile(join(directory, 'healthy.mjs'), `
export function register(registry) {
  registry.registerPlugin({ name: 'healthy-plugin' })
  registry.registerTool('healthy_tool', () => 'ok', undefined, 'healthy-plugin')
}
`, 'utf8')
    const globalCount = globalThis as unknown as Record<string, number | undefined>
    delete globalCount.__xerxesPluginExecutions

    const registry = new PluginRegistry()
    const errors: unknown[][] = []
    const spy = spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
      errors.push(args)
    })
    let discovered: string[] = []
    try {
      discovered = await registry.discover(directory)
    } finally {
      spy.mockRestore()
    }

    expect([...discovered].sort()).toEqual(['counted-plugin', 'healthy-plugin'])
    expect([...registry.pluginNames].sort()).toEqual(['counted-plugin', 'healthy-plugin'])
    expect(registry.getPlugin('broken-plugin')).toBeUndefined()
    expect(registry.getTool('broken_tool')).toBeUndefined()
    expect(registry.getTool('standalone_tool')).toBeUndefined()
    expect(registry.getHooks('on_turn_end')).toEqual([])
    expect(registry.loadErrors).toHaveLength(1)
    expect(registry.loadErrors[0]).toContain('broken.mjs')
    expect(registry.loadErrors[0]).toContain('register exploded')
    expect(errors.some(args => String(args[0]).includes('broken.mjs'))).toBeTrue()

    // A re-discovery skips already-loaded modules instead of re-executing them into conflicts.
    expect(await registry.discover(directory)).toEqual([])
    const executions: unknown = globalCount.__xerxesPluginExecutions
    expect(executions).toBe(1)
    expect([...registry.pluginNames].sort()).toEqual(['counted-plugin', 'healthy-plugin'])
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})

test('a permission hook can veto a tool call, and any deny beats every allow', async () => {
  const hooks = new HookRunner()
  const seen: Array<Record<string, unknown>> = []
  hooks.register(TOOL_PERMISSION_HOOK, payload => {
    seen.push({ ...payload })
    return { allow: true }
  })
  hooks.register(TOOL_PERMISSION_HOOK, async payload => {
    const args = payload.arguments as { readonly path?: string }
    await Bun.sleep(1)
    return args.path === '/etc/shadow'
      ? { allow: false, reason: 'system credential file', source: 'path-guard' }
      : { allow: true }
  })
  hooks.register(TOOL_PERMISSION_HOOK, () => ({ allow: true }))

  const allowed = await resolveToolPermission(hooks, { toolName: 'read_file', arguments: { path: 'src/a.ts' } })
  expect(allowed).toEqual({ allowed: true, denials: [], reason: '' })

  const denied = await resolveToolPermission(hooks, { toolName: 'read_file', arguments: { path: '/etc/shadow' } })
  expect(denied.allowed).toBeFalse()
  expect(denied.reason).toBe('system credential file')
  expect(denied.denials).toEqual([{ allow: false, reason: 'system credential file', source: 'path-guard' }])
  // The observing hook sees the real call, and the point is not a mutation hook,
  // so its return value never becomes the tool's arguments.
  expect(seen.at(-1)).toEqual({ toolName: 'read_file', arguments: { path: '/etc/shadow' } })
})

test('permission hooks fail closed: a throwing or malformed guard denies instead of consenting', async () => {
  const spy = spyOn(console, 'error').mockImplementation(() => undefined)
  try {
    const throwing = new HookRunner()
    throwing.register(TOOL_PERMISSION_HOOK, () => { throw new Error('policy database offline') })
    throwing.register(TOOL_PERMISSION_HOOK, () => ({ allow: true }))
    const failed = await resolveToolPermission(throwing, { toolName: 'exec_command' })
    expect(failed.allowed).toBeFalse()
    expect(failed.reason).toContain('policy database offline')
    expect(failed.denials[0]?.source).toBe(TOOL_PERMISSION_HOOK)

    const malformed = new HookRunner()
    malformed.register(TOOL_PERMISSION_HOOK, () => 'looks fine to me')
    const rejected = await resolveToolPermission(malformed, { toolName: 'exec_command' })
    expect(rejected.allowed).toBeFalse()
    expect(rejected.reason).toContain('not a permission verdict')

    // No guards installed is the ordinary case and must keep the agent working.
    expect(await resolveToolPermission(new HookRunner(), { toolName: 'exec_command' }))
      .toEqual({ allowed: true, denials: [], reason: '' })

    // A denial with no stated reason still explains itself to the model.
    const terse = new HookRunner()
    terse.register(TOOL_PERMISSION_HOOK, () => ({ allow: false }))
    expect((await resolveToolPermission(terse, { toolName: 'exec_command' })).reason)
      .toBe("tool 'exec_command' denied by a permission hook")
  } finally {
    spy.mockRestore()
  }
})
