// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  COMMAND_REGISTRY,
  SlashPluginConflictError,
  SlashPluginRegistry,
  defaultSlashPluginRegistry,
  getDefaultSlashPluginRegistry,
  registerSlash,
  registeredSlashes,
  resolveSlash,
  unregisterSlash,
} from '../src/index.js'

test('slash plugin registries resolve normalized aliases and list deterministic command metadata', () => {
  const registry = new SlashPluginRegistry()
  const zeta = registry.register('/Zeta', () => 'zeta', {
    aliases: ['/zed'],
    argsHint: '<value>',
    description: 'Run the zeta plugin.',
  })
  registry.register('alpha_plugin', () => 'alpha')

  expect(registry.resolve('/zeta argument')).toBe(zeta)
  expect(registry.resolve('/ZED@XerxesBot')).toBe(zeta)
  expect(registry.list().map(plugin => plugin.command.name)).toEqual(['alpha_plugin', 'zeta'])
  expect(registry.allCommands().slice(0, COMMAND_REGISTRY.length)).toEqual([...COMMAND_REGISTRY])
  expect(registry.allCommands().slice(COMMAND_REGISTRY.length).map(command => command.name)).toEqual([
    'alpha_plugin',
    'zeta',
  ])
  expect(zeta.command).toMatchObject({
    aliases: ['zed'],
    argsHint: '<value>',
    category: 'tools',
    cliOnly: false,
    gatewayOnly: false,
  })
})

test('slash plugins cannot shadow built-ins, built-in aliases, or plugin-owned tokens', () => {
  const registry = new SlashPluginRegistry()
  const handler = () => undefined

  expect(() => registry.register('', handler)).toThrow(TypeError)
  expect(() => registry.register('not a command', handler)).toThrow(TypeError)
  expect(() => registry.register('help', handler)).toThrow(SlashPluginConflictError)
  expect(() => registry.register('custom', handler, { aliases: ['reset'] })).toThrow(SlashPluginConflictError)
  expect(() => registry.register('custom', handler, { aliases: ['q'] })).toThrow(SlashPluginConflictError)

  registry.register('first_plugin', handler, { aliases: ['first_alias'] })
  expect(() => registry.register('first_alias', handler)).toThrow(SlashPluginConflictError)
  expect(() => registry.register('second_plugin', handler, { aliases: ['first_plugin'] })).toThrow(SlashPluginConflictError)
  expect(() => registry.register('third_plugin', handler, { aliases: ['first_alias'] })).toThrow(SlashPluginConflictError)
})

test('unregister removes only exact canonical plugin names and their aliases', () => {
  const registry = new SlashPluginRegistry()
  registry.register('remove_me', () => undefined, { aliases: ['remove_alias'] })

  expect(registry.unregister('/remove_alias')).toBe(false)
  expect(registry.resolve('/remove_alias')).toBeDefined()
  expect(registry.unregister('/remove_me')).toBe(true)
  expect(registry.resolve('/remove_me')).toBeUndefined()
  expect(registry.resolve('/remove_alias')).toBeUndefined()
  expect(registry.unregister('remove_me')).toBe(false)
  expect(registry.unregister('not a command')).toBe(false)
})

test('the root-exported default registry is shared only through explicit helpers', () => {
  const name = 'default_slash_plugin_test'
  unregisterSlash(name)
  try {
    const plugin = registerSlash(name, () => 'ok', { aliases: ['default_slash_alias_test'] })

    expect(defaultSlashPluginRegistry).toBe(getDefaultSlashPluginRegistry())
    expect(resolveSlash('/default_slash_alias_test')).toBe(plugin)
    expect(registeredSlashes()).toContain(plugin)
  } finally {
    unregisterSlash(name)
  }
})
