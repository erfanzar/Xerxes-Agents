// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ACP_HELP, parseAcpCommandOptions } from '../src/acp/command.js'

test('ACP command parser retains the native ACP launch contract', () => {
  expect(parseAcpCommandOptions([])).toEqual({
    help: false,
    permissionMode: 'accept-all',
    projectDirectory: undefined,
    writeRegistry: false,
  })
  expect(parseAcpCommandOptions([
    '--project-dir',
    '/workspace',
    '--permission-mode',
    'manual',
    '--write-registry',
  ])).toEqual({
    help: false,
    permissionMode: 'manual',
    projectDirectory: '/workspace',
    writeRegistry: true,
  })
  expect(parseAcpCommandOptions(['--help'])).toMatchObject({ help: true })
  expect(ACP_HELP).toContain('xerxes-acp')
})

test('ACP command parser rejects unknown and incomplete launch options', () => {
  expect(() => parseAcpCommandOptions(['--permission-mode', 'plan'])).toThrow('must be one of')
  expect(() => parseAcpCommandOptions(['--permission-mode'])).toThrow('requires a value')
  expect(() => parseAcpCommandOptions(['--unknown'])).toThrow('Unknown ACP option')
})
