// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { extractAgentOption, parseValueOptions } from '../src/runtime/commandOptions.js'

const CLI = join(import.meta.dir, '../src/cli.ts')

async function runCli(args: readonly string[]): Promise<{
  readonly exitCode: number
  readonly stderr: string
  readonly stdout: string
}> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-bun-cli-flags-'))
  try {
    const child = Bun.spawn([process.execPath, CLI, ...args], {
      cwd: root,
      env: { ...process.env, XERXES_HOME: join(root, 'home') },
      stderr: 'pipe',
      stdout: 'pipe',
    })
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
      child.exited,
    ])
    return { exitCode, stderr, stdout }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

test('agent option extracts a name or path and keeps the remaining prompt words', () => {
  expect(extractAgentOption(['--agent', 'researcher', 'summarize', 'this'])).toEqual({
    agent: 'researcher',
    rest: ['summarize', 'this'],
  })
  expect(extractAgentOption(['summarize', '--agent', './agents/qa.yaml'])).toEqual({
    agent: './agents/qa.yaml',
    rest: ['summarize'],
  })
  expect(extractAgentOption(['--agent=./agents/qa.yaml', 'run'])).toEqual({
    agent: './agents/qa.yaml',
    rest: ['run'],
  })
  expect(extractAgentOption(['plain', 'prompt'])).toEqual({
    agent: undefined,
    rest: ['plain', 'prompt'],
  })
})

test('agent option rejects a missing or flag-like value', () => {
  expect(() => extractAgentOption(['--agent'])).toThrow(
    'The --agent option requires an agent name or file path',
  )
  expect(() => extractAgentOption(['--agent', '--resume'])).toThrow(
    'The --agent option requires an agent name or file path',
  )
  expect(() => extractAgentOption(['--agent='])).toThrow(
    'The --agent option requires an agent name or file path',
  )
})

test('the CLI rejects --agent on commands that own their runtime', async () => {
  const daemon = await runCli(['daemon', '--agent', 'researcher'])
  expect(daemon.exitCode).not.toBe(0)
  expect(daemon.stderr).toContain(
    'The --agent option is only supported for one-shot prompts',
  )

  const resume = await runCli(['--agent', 'researcher', '--resume', 'session-id'])
  expect(resume.exitCode).not.toBe(0)
  expect(resume.stderr).toContain(
    'The --agent option is only supported for one-shot prompts',
  )
})

test('help documents the --agent option', async () => {
  const result = await runCli(['--help'])
  expect(result.exitCode).toBe(0)
  expect(result.stdout).toContain('--agent <name|path>')
})

test('value-option parser consumes values and rejects positional tokens', () => {
  expect(parseValueOptions(
    ['--socket', '/tmp/x.sock', '--project-dir', '/workspace'],
    'daemon',
    ['--project-dir', '--socket'],
  )).toEqual(new Map([
    ['--socket', '/tmp/x.sock'],
    ['--project-dir', '/workspace'],
  ]))
  expect(() => parseValueOptions(
    ['--token', 'test-token', 'poll'],
    'telegram',
    ['--token'],
  )).toThrow('Unexpected telegram argument: poll')
})

test('daemon rejects unknown or misspelled flags instead of ignoring them', async () => {
  const misspelled = await runCli(['daemon', '--socke', '/tmp/x.sock'])
  expect(misspelled.exitCode).not.toBe(0)
  expect(misspelled.stderr).toContain('Unknown daemon option: --socke')

  const unknown = await runCli(['daemon', '--port', '8080'])
  expect(unknown.exitCode).not.toBe(0)
  expect(unknown.stderr).toContain('Unknown daemon option: --port')
})

test('daemon rejects a flag-like value where an option value is required', async () => {
  const result = await runCli(['daemon', '--socket', '--pid-file', '/tmp/x.pid'])
  expect(result.exitCode).not.toBe(0)
  expect(result.stderr).toContain('daemon option --socket requires a value')
})

test('daemon rejects unsupported positional arguments', async () => {
  const result = await runCli(['daemon', '--socket', '/tmp/x.sock', 'serve'])
  expect(result.exitCode).not.toBe(0)
  expect(result.stderr).toContain('Unexpected daemon argument: serve')
})

test('telegram rejects unknown or misspelled flags instead of ignoring them', async () => {
  const misspelled = await runCli(['telegram', '--token', 'test-token', '--prot', '1234'])
  expect(misspelled.exitCode).not.toBe(0)
  expect(misspelled.stderr).toContain('Unknown telegram option: --prot')

  const missingValue = await runCli(['telegram', '--token'])
  expect(missingValue.exitCode).not.toBe(0)
  expect(missingValue.stderr).toContain('telegram option --token requires a value')
})

test('telegram rejects unsupported positional arguments after consuming option values', async () => {
  const result = await runCli([
    'telegram',
    '--token',
    'test-token',
    '--host',
    '127.0.0.1',
    'poll',
  ])
  expect(result.exitCode).not.toBe(0)
  expect(result.stderr).toContain('Unexpected telegram argument: poll')
})
