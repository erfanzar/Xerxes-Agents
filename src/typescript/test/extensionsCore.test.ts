// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  CircularDependencyError,
  DependencyResolver,
  HookRunner,
  PluginConflictError,
  PluginRegistry,
  PluginType,
  VersionConstraint,
  parseDependency,
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
  expect(hooks.run('before_tool_call', payload)).toEqual({ path: 'file.ts', safe: true })
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
