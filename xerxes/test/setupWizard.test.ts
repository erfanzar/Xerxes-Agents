// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  runSetupWizard,
  writeSetupConfig,
  type SetupStep,
} from '../src/runtime/setupWizard.js'

test('setup wizard applies defaults, preserves supplied answers, and records skipped optional values', () => {
  const result = runSetupWizard({ provider: 'openai', api_key: '' })

  expect(result.answers).toMatchObject({
    provider: 'openai',
    model: 'claude-opus-4-6',
    permission_mode: 'accept-all',
  })
  expect(result.skipped).toEqual(['api_key'])
  expect(Object.isFrozen(result.answers)).toBe(true)
})

test('setup wizard validates required answers against caller-provided steps', () => {
  const steps: readonly SetupStep[] = [{
    key: 'port',
    prompt: 'Port',
    validator: value => typeof value === 'number' && value > 0,
  }]

  expect(() => runSetupWizard({ port: 0 }, { steps })).toThrow('invalid value for setup step')
  expect(runSetupWizard({ port: 11996 }, { steps }).answers).toEqual({ port: 11996 })
})

test('setup wizard writes safe YAML-compatible scalars without a Python/YAML runtime', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-setup-'))
  const target = join(directory, 'nested', 'config.yaml')
  try {
    await writeSetupConfig({
      enabled: true,
      model: 'gpt "quoted"',
      retries: 2,
    }, target)
    expect(await readFile(target, 'utf8')).toBe([
      'enabled: true',
      'model: "gpt \\"quoted\\""',
      'retries: 2',
      '',
    ].join('\n'))
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
