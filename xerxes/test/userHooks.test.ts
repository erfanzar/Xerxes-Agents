// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { HookRunner, resolveToolPermission, TOOL_PERMISSION_HOOK } from '../src/extensions/hooks.js'
import {
  executeUserHook,
  hasUserHooks,
  parseUserHooksConfig,
  registerUserHooks,
  USER_HOOK_SOURCE,
  type UserHookDefinition,
} from '../src/extensions/userHooks.js'

/** Cross-platform hook command: run one Bun eval script with no shell involved. */
function bunHook(script: string, extra: Partial<UserHookDefinition> = {}): UserHookDefinition {
  return { command: process.execPath, args: ['-e', script], ...extra }
}

const DRAIN_STDIN = 'for await (const _ of Bun.stdin.stream()) {}'

test('parseUserHooksConfig accepts a valid config and rejects malformed shapes', () => {
  expect(parseUserHooksConfig(undefined)).toEqual({})
  expect(parseUserHooksConfig(null)).toEqual({})
  expect(hasUserHooks(parseUserHooksConfig(undefined))).toBe(false)

  const config = parseUserHooksConfig({
    PreToolUse: [
      { command: 'guard.sh', matcher: '^(Bash|exec_command)$', timeoutMs: 1500 },
      { command: 'node', args: ['audit.js'] },
    ],
    TurnEnd: [{ command: 'log-turn.sh' }],
  })
  expect(hasUserHooks(config)).toBe(true)
  expect(config.PreToolUse).toHaveLength(2)
  expect(config.PreToolUse?.[0]).toMatchObject({ command: 'guard.sh', timeoutMs: 1500 })

  expect(() => parseUserHooksConfig('guard.sh')).toThrow('must be an object')
  expect(() => parseUserHooksConfig({ NoSuchEvent: [] })).toThrow('unknown hook event')
  expect(() => parseUserHooksConfig({ PreToolUse: 'guard.sh' })).toThrow('must be an array')
  expect(() => parseUserHooksConfig({ PreToolUse: [{}] })).toThrow('command')
  expect(() => parseUserHooksConfig({ PreToolUse: [{ command: 'x', matcher: '(' }] })).toThrow('regular expression')
  expect(() => parseUserHooksConfig({ PreToolUse: [{ command: 'x', args: 'y' }] })).toThrow('array of strings')
  expect(() => parseUserHooksConfig({ PreToolUse: [{ command: 'x', timeoutMs: -1 }] })).toThrow('positive number')
})

test('executeUserHook allows on exit 0 and denies on exit 2 with the stderr reason', async () => {
  const allow = await executeUserHook(bunHook(`${DRAIN_STDIN}`), 'PreToolUse', {
    event: 'PreToolUse',
    timestamp: new Date().toISOString(),
    tool_name: 'Bash',
  })
  expect(allow).toEqual({ kind: 'allow' })

  const deny = await executeUserHook(
    bunHook(`${DRAIN_STDIN}; console.error('destructive command blocked'); process.exit(2)`),
    'PreToolUse',
    { event: 'PreToolUse', timestamp: new Date().toISOString(), tool_name: 'Bash' },
  )
  expect(deny).toEqual({ kind: 'deny', reason: 'destructive command blocked' })
})

test('executeUserHook honors stdout verdicts: deny with reason and allow with updated arguments', async () => {
  const deny = await executeUserHook(
    bunHook(`${DRAIN_STDIN}; console.log(JSON.stringify({ decision: 'deny', reason: 'policy says no' }))`),
    'PreToolUse',
    { event: 'PreToolUse', timestamp: new Date().toISOString() },
  )
  expect(deny).toEqual({ kind: 'deny', reason: 'policy says no' })

  const rewrite = await executeUserHook(
    bunHook(`${DRAIN_STDIN}; console.log(JSON.stringify({ updated_arguments: { cmd: 'ls', args: [] } }))`),
    'PreToolUse',
    { event: 'PreToolUse', timestamp: new Date().toISOString() },
  )
  expect(rewrite).toEqual({ kind: 'allow', updatedArguments: { cmd: 'ls', args: [] } })

  // Stdout that is not a verdict document is ignored.
  const noisy = await executeUserHook(
    bunHook(`${DRAIN_STDIN}; console.log('just a log line')`),
    'PreToolUse',
    { event: 'PreToolUse', timestamp: new Date().toISOString() },
  )
  expect(noisy).toEqual({ kind: 'allow' })
})

test('executeUserHook fails closed on hook errors and timeouts', async () => {
  const failed = await executeUserHook(
    bunHook(`${DRAIN_STDIN}; process.exit(1)`),
    'PreToolUse',
    { event: 'PreToolUse', timestamp: new Date().toISOString() },
  )
  expect(failed.kind).toBe('deny')
  expect(failed.kind === 'deny' && failed.reason).toContain('exited with code 1')

  const missing = await executeUserHook(
    { command: 'xerxes-no-such-hook-binary', args: [] },
    'PreToolUse',
    { event: 'PreToolUse', timestamp: new Date().toISOString() },
  )
  expect(missing.kind).toBe('deny')

  const hung = await executeUserHook(
    bunHook('setTimeout(() => {}, 60_000)', { timeoutMs: 100 }),
    'PreToolUse',
    { event: 'PreToolUse', timestamp: new Date().toISOString() },
  )
  expect(hung.kind).toBe('deny')
  expect(hung.kind === 'deny' && hung.reason).toContain('timed out')
})

test('registered PreToolUse hooks gate tool permission with matcher filtering and argument rewrites', async () => {
  const hookRunner = new HookRunner()
  const counts = registerUserHooks(hookRunner, parseUserHooksConfig({
    PreToolUse: [
      bunHook(
        `${DRAIN_STDIN}; console.error('rm is not allowed'); process.exit(2)`,
        { matcher: '^Bash$' },
      ),
      bunHook(`${DRAIN_STDIN}; console.log(JSON.stringify({ updated_arguments: { path: 'safe.ts' } }))`),
    ],
  }))

  expect(counts[TOOL_PERMISSION_HOOK]).toBe(2)
  expect(hookRunner.hasHooks(TOOL_PERMISSION_HOOK)).toBe(true)

  const denied = await resolveToolPermission(hookRunner, { toolName: 'Bash', arguments: { command: 'rm -rf /' } })
  expect(denied.allowed).toBe(false)
  expect(denied.reason).toBe('rm is not allowed')
  expect(denied.denials[0]?.source).toBe(USER_HOOK_SOURCE)

  // The exit-2 hook is matcher-scoped to Bash, so ReadFile passes it and the
  // second hook's replacement arguments survive into the collapsed decision.
  const rewritten = await resolveToolPermission(hookRunner, { toolName: 'ReadFile', arguments: { path: 'a.ts' } })
  expect(rewritten.allowed).toBe(true)
  expect(rewritten.updatedArguments).toEqual({ path: 'safe.ts' })
})

test('observer hooks run at their lifecycle point without blocking the turn on failure', async () => {
  const errors: string[] = []
  const hookRunner = new HookRunner()
  registerUserHooks(hookRunner, parseUserHooksConfig({
    TurnEnd: [bunHook(`${DRAIN_STDIN}; process.exit(1)`)],
  }), { onError: message => errors.push(message) })

  // HookRunner isolates callback failures; the observer callback itself never
  // throws, so a broken TurnEnd hook cannot break or veto a turn.
  const results = await hookRunner.run('on_turn_end', { turnCount: 1 })
  expect(results).toEqual([])
})
