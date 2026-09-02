// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, describe, expect, test } from 'bun:test'

import { HookRunner } from '../src/extensions/hooks.js'
import {
  loadShellHookConfigSync,
  parseShellHookConfig,
  registerShellHooks,
} from '../src/extensions/shellHooks.js'

const scratchDirs: string[] = []

afterEach(() => {
  for (const dir of scratchDirs.splice(0)) {
    rmSync(dir, { force: true, recursive: true })
  }
})

function scratch(): string {
  const dir = mkdtempSync(join(tmpdir(), 'xerxes-shell-hooks-'))
  scratchDirs.push(dir)
  return dir
}

describe('parseShellHookConfig', () => {
  test('accepts native point names and Claude Code event aliases', () => {
    const parsed = parseShellHookConfig({
      PreToolUse: [{ command: 'guard.sh', matcher: 'Bash|exec_command' }],
      on_turn_start: [{ command: 'echo hi', timeout_ms: 5000 }],
      SessionStart: [{ command: 'boot.sh' }],
    }, 'test')

    expect(parsed.tool_permission_check?.length).toBe(1)
    expect(parsed.tool_permission_check?.[0]?.matcher).toBe('Bash|exec_command')
    expect(parsed.on_turn_start?.[0]?.timeout_ms).toBe(5000)
    expect(parsed.on_session_start?.[0]?.command).toBe('boot.sh')
  })

  test('rejects unknown events and malformed entries loudly', () => {
    expect(() => parseShellHookConfig({ Nonsense: [{ command: 'x' }] }, 'test')).toThrow(/unknown hook event/)
    expect(() => parseShellHookConfig({ on_turn_start: 'oops' }, 'test')).toThrow(/must be a list/)
    expect(() => parseShellHookConfig({ on_turn_start: [{}] }, 'test')).toThrow(/command/)
    expect(() => parseShellHookConfig({ on_turn_start: [{ command: 'x', matcher: '(' }] }, 'test')).toThrow()
    expect(() => parseShellHookConfig({ on_turn_start: [{ command: 'x', timeout_ms: -1 }] }, 'test')).toThrow(/timeout_ms/)
  })
})

describe('loadShellHookConfigSync', () => {
  test('loads user hooks; workspace hooks only when trusted', () => {
    const home = scratch()
    const workspace = scratch()
    writeFileSync(join(home, 'config.yaml'), 'hooks:\n  on_turn_end:\n    - command: user.sh\n')
    writeFileSync(join(workspace, 'xerxes.yaml'), 'hooks:\n  on_turn_end:\n    - command: workspace.sh\n')

    const untrusted = loadShellHookConfigSync({ allowWorkspace: false, home, workspaceRoot: workspace })
    expect(untrusted.errors).toEqual([])
    expect(untrusted.hooks.on_turn_end?.map(spec => spec.command)).toEqual(['user.sh'])

    const trusted = loadShellHookConfigSync({ allowWorkspace: true, home, workspaceRoot: workspace })
    expect(trusted.hooks.on_turn_end?.map(spec => spec.command)).toEqual(['user.sh', 'workspace.sh'])
  })

  test('malformed hooks sections report errors and are skipped', () => {
    const home = scratch()
    writeFileSync(join(home, 'config.yaml'), 'hooks:\n  on_turn_end: nope\n')
    const loaded = loadShellHookConfigSync({ allowWorkspace: false, home })
    expect(loaded.errors.length).toBe(1)
    expect(loaded.hooks.on_turn_end).toBeUndefined()
  })
})

describe('shell hook execution', () => {
  test('observer hooks receive the payload as JSON on stdin', async () => {
    const dir = scratch()
    const marker = join(dir, 'payload.json')
    const runner = new HookRunner()
    registerShellHooks(runner, parseShellHookConfig({
      on_turn_start: [{ command: `cat > ${JSON.stringify(marker)}` }],
    }, 'test'), { cwd: dir })

    await runner.run('on_turn_start', { session_id: 'abc', model: 'm' })
    const payload = JSON.parse(readFileSync(marker, 'utf8')) as Record<string, unknown>
    expect(payload.hook_point).toBe('on_turn_start')
    expect(payload.session_id).toBe('abc')
  })

  test('permission hook: exit 2 denies with stderr as the reason; exit 0 allows', async () => {
    const runner = new HookRunner()
    registerShellHooks(runner, parseShellHookConfig({
      PreToolUse: [
        { command: 'grep -q danger && { echo "blocked dangerous call" >&2; exit 2; } || exit 0' },
      ],
    }, 'test'), { cwd: scratch() })

    const allowed = await runner.run('tool_permission_check', { arguments: { command: 'ls' }, toolName: 'exec_command' })
    expect(allowed).toEqual([{ allow: true, source: 'shell_hook' }])

    const denied = await runner.run('tool_permission_check', { arguments: { command: 'danger zone' }, toolName: 'exec_command' })
    expect(denied).toEqual([{ allow: false, reason: 'blocked dangerous call', source: 'shell_hook' }])
  })

  test('permission hook fails closed on crash', async () => {
    const runner = new HookRunner()
    registerShellHooks(runner, parseShellHookConfig({
      PreToolUse: [{ command: 'exit 42' }],
    }, 'test'), { cwd: scratch() })

    const results = await runner.run('tool_permission_check', { arguments: {}, toolName: 'exec_command' })
    expect(results).toEqual([
      expect.objectContaining({ allow: false, reason: expect.stringContaining('exited 42') }),
    ])
  })

  test('mutation hooks replace the threaded value from stdout JSON', async () => {
    const runner = new HookRunner()
    registerShellHooks(runner, parseShellHookConfig({
      before_tool_call: [{ command: 'printf \'{"arguments":{"path":"/safe"}}\'' }],
    }, 'test'), { cwd: scratch() })

    const mutated = await runner.run('before_tool_call', { arguments: { path: '/etc' }, toolName: 'ReadFile' })
    expect(mutated).toEqual({ path: '/safe' })
  })

  test('matchers gate tool hooks by name', async () => {
    const dir = scratch()
    const marker = join(dir, 'ran')
    const runner = new HookRunner()
    registerShellHooks(runner, parseShellHookConfig({
      after_tool_call: [{ command: `touch ${marker}`, matcher: '^exec_command$' }],
    }, 'test'), { cwd: dir })

    await runner.run('after_tool_call', { result: 'x', toolName: 'ReadFile' })
    await runner.run('after_tool_call', { result: 'x', toolName: 'exec_command' })
    // Bun's touch timing: assert via existsSync after both runs settle.
    const { existsSync } = await import('node:fs')
    expect(existsSync(marker)).toBe(true)
  })
})
