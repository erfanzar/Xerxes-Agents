// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  CATEGORIES,
  COMMAND_REGISTRY,
  commandAvailableOnSurface,
  listCommands,
  normalizeTelegramCommandName,
  resolveCommand,
  telegramBotCommands,
  type CommandDefinition,
} from '../src/bridge/commands.js'

test('bridge command registry preserves canonical metadata, aliases, categories, and registry order', () => {
  expect(CATEGORIES).toContain('session')
  expect(COMMAND_REGISTRY).toHaveLength(67)
  expect(resolveCommand('/compress now')).toMatchObject({ name: 'compact', aliases: ['compress'] })
  expect(resolveCommand('/thinking high')).toMatchObject({ name: 'reasoning', argsHint: '[level]' })
  expect(resolveCommand('/q')).toMatchObject({ name: 'exit' })
  expect(resolveCommand('/help@XerxesBot')).toMatchObject({ name: 'help' })
  expect(resolveCommand('unknown')).toBeUndefined()
  expect(listCommands('snapshots').map(commandDefinition => commandDefinition.name)).toEqual([
    'rollback',
    'snapshot',
    'snapshots',
  ])
})

test('surface filters keep CLI-only definitions out of gateways and honor gateway-only definitions', () => {
  expect(resolveCommand('/queue', { surface: 'gateway' })).toBeUndefined()
  expect(resolveCommand('/queue', { surface: 'cli' })?.name).toBe('queue')
  expect(listCommands({ surface: 'gateway' }).map(commandDefinition => commandDefinition.name)).not.toContain('statusbar')

  const gatewayOnly: CommandDefinition = {
    name: 'gateway-test',
    description: 'Gateway-only test command',
    category: 'messaging',
    aliases: [],
    argsHint: '',
    cliOnly: false,
    gatewayOnly: true,
    deprecated: false,
    examples: [],
  }
  expect(commandAvailableOnSurface(gatewayOnly, 'cli')).toBeFalse()
  expect(commandAvailableOnSurface(gatewayOnly, 'gateway')).toBeTrue()
})

test('Telegram rendering normalizes canonical names, filters CLI-only and invalid definitions, and deduplicates', () => {
  expect(normalizeTelegramCommandName('/retry-connection@XerxesBot')).toBe('retry_connection')
  expect(normalizeTelegramCommandName('not valid!')).toBeUndefined()
  expect(normalizeTelegramCommandName('a'.repeat(33))).toBeUndefined()

  const custom: CommandDefinition[] = [
    {
      name: 'skill-create',
      description: 'Create a skill',
      category: 'skills',
      aliases: [],
      argsHint: '',
      cliOnly: false,
      gatewayOnly: false,
      deprecated: false,
      examples: [],
    },
    {
      name: 'skill_create',
      description: 'Duplicate after Telegram normalization',
      category: 'skills',
      aliases: [],
      argsHint: '',
      cliOnly: false,
      gatewayOnly: false,
      deprecated: false,
      examples: [],
    },
    {
      name: 'local-only',
      description: 'Must not reach Telegram',
      category: 'info',
      aliases: [],
      argsHint: '',
      cliOnly: true,
      gatewayOnly: false,
      deprecated: false,
      examples: [],
    },
    {
      name: 'bad!',
      description: 'Invalid Telegram name',
      category: 'info',
      aliases: [],
      argsHint: '',
      cliOnly: false,
      gatewayOnly: false,
      deprecated: false,
      examples: [],
    },
  ]
  expect(telegramBotCommands(custom)).toEqual([{
    command: 'skill_create',
    description: 'Create a skill',
  }])
  expect(telegramBotCommands().find(commandDefinition => commandDefinition.command === 'retry_connection')).toEqual({
    command: 'retry_connection',
    description: 'Retry the last failed provider connection',
  })
})
