// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { CLAUDE_CODE_INSTALLER, maintainClaudeCode, type CommandRunner } from '../src/llms/claudeCodeMaintenance.js'

function runner(code = 0, output = 'Claude Code is up to date (2.1.283)') {
  const calls: string[][] = []
  const run: CommandRunner = async argv => { calls.push([...argv]); return { code, output } }
  return { calls, run }
}

test('an installed CLI is updated at runtime start when a profile uses Claude Code', async () => {
  const r = runner()
  expect(await maintainClaudeCode({ inUse: true, environment: { PATH: '/bin' }, run: r.run, executable: () => '/h/.local/bin/claude' }))
    .toEqual({ action: 'updated', output: 'Claude Code is up to date (2.1.283)' })
  expect(r.calls).toEqual([['/h/.local/bin/claude', 'update']])
})

test('a missing CLI is installed with the official installer; Windows is pointed at the docs', async () => {
  const r = runner(0, 'Claude Code successfully installed!')
  expect((await maintainClaudeCode({ inUse: true, environment: {}, platform: 'linux', run: r.run, executable: () => undefined })).action).toBe('installed')
  expect(r.calls[0]![2]).toContain(CLAUDE_CODE_INSTALLER)
  const w = runner()
  expect((await maintainClaudeCode({ inUse: true, environment: {}, platform: 'win32', run: w.run, executable: () => undefined })).action).toBe('skipped')
  expect(w.calls).toEqual([])
})

test('nothing runs when no profile uses Claude Code or it is turned off, and failures are reported', async () => {
  const r = runner()
  expect((await maintainClaudeCode({ inUse: false, environment: {}, run: r.run, executable: () => '/bin/claude' })).action).toBe('skipped')
  expect((await maintainClaudeCode({ inUse: true, environment: { XERXES_CLAUDE_CODE_AUTOUPDATE: '0' }, run: r.run, executable: () => '/bin/claude' })).action).toBe('skipped')
  expect(r.calls).toEqual([])
  const failing = runner(1, 'network unreachable')
  expect(await maintainClaudeCode({ inUse: true, environment: {}, run: failing.run, executable: () => '/bin/claude' }))
    .toEqual({ action: 'failed', error: 'claude update exited 1: network unreachable' })
})
