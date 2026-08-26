// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Seam tests for the daemon composition root.
//
// Every other test in this suite injects the port it is about to assert on,
// which proves the mechanism and says nothing about whether production ever
// connects it. That gap is not hypothetical: `onSessionModeChange` was
// declared, fired, and covered by a passing test while `cli.ts` subscribed
// nothing, so a mode change the model made updated the session and reached no
// client — the TUI kept rendering, and gating on, the mode the session had
// already left.
//
// These tests read the composition root itself, so a port that is added and
// never wired fails here rather than in someone's terminal.

import { expect, test } from 'bun:test'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

const SOURCE_ROOT = join(import.meta.dir, '..', 'src')

const readSource = (relative: string) => readFile(join(SOURCE_ROOT, relative), 'utf8')

/**
 * Hooks the composition root is expected to supply, with the reason each one
 * exists. A hook deliberately left unsubscribed belongs here with `wired:
 * false` and a written justification — never silently absent.
 */
const RUNTIME_HOOKS: ReadonlyArray<{ readonly name: string; readonly wired: boolean; readonly why: string }> = [
  { name: 'onSessionEvict', wired: true, why: 'reclaims a session\'s delegated children and memory' },
  { name: 'onTurnCancel', wired: true, why: 'Esc stops the whole delegation tree, not just the parent stream' },
  { name: 'onSessionModeChange', wired: true, why: 'a model-driven mode change must reach attached clients' },
  { name: 'subagentRetry', wired: true, why: 'retry of a dead subagent under its stable identity' },
]

test('every declared runtime hook is either wired at the composition root or justified', async () => {
  const [runtime, cli] = await Promise.all([
    readSource('daemon/runtime.ts'),
    readSource('cli.ts'),
  ])

  // The list above must stay in step with the interface it describes; a new
  // optional hook has to make a wiring decision, not inherit one by omission.
  const declared = [...runtime.matchAll(/readonly (on[A-Z]\w+|subagentRetry)\?:/g)].map(match => match[1])
  expect(new Set(declared)).toEqual(new Set(RUNTIME_HOOKS.map(hook => hook.name)))

  // Scoped to the runtime constructor call, not the whole file. A hook that
  // merely appears somewhere in cli.ts proves nothing — the forwarding into
  // InMemoryDaemonRuntime's options is the wire that was missing.
  const options = runtimeOptionsBlock(cli)
  for (const hook of RUNTIME_HOOKS) {
    const subscribed = new RegExp(`\\b${hook.name}\\b`).test(options)
    expect(
      { hook: hook.name, subscribed },
      `${hook.name} (${hook.why})`,
    ).toEqual({ hook: hook.name, subscribed: hook.wired })
  }
})

/** The options object literal handed to InMemoryDaemonRuntime in cli.ts. */
function runtimeOptionsBlock(cli: string): string {
  const marker = 'new InMemoryDaemonRuntime('
  const start = cli.indexOf(marker)
  expect(start, 'cli.ts must construct InMemoryDaemonRuntime').toBeGreaterThan(-1)
  let depth = 0
  for (let index = start + marker.length - 1; index < cli.length; index += 1) {
    const character = cli[index]
    if (character === '(') depth += 1
    else if (character === ')') {
      depth -= 1
      if (depth === 0) return cli.slice(start, index + 1)
    }
  }
  throw new Error('unbalanced InMemoryDaemonRuntime( call in cli.ts')
}

test('the daemon announces model-driven mode changes through the server', async () => {
  const [cli, server] = await Promise.all([
    readSource('cli.ts'),
    readSource('daemon/server.ts'),
  ])

  // Firing the hook is not enough — it has to land on something that emits to
  // clients. Assert the whole path, not just its first link.
  expect(cli).toContain('onSessionModeChange:')
  expect(cli).toContain('notifySessionModeChanged')
  expect(server).toContain('notifySessionModeChanged(sessionId: string)')
  expect(server).toMatch(/notifySessionModeChanged[\s\S]{0,600}emitStatus/)
})

test('deferred tool loading is reachable from production, even though it is off by default', async () => {
  const [cli, runner] = await Promise.all([
    readSource('cli.ts'),
    readSource('daemon/turnRunner.ts'),
  ])

  // The flag existed for a long time while every production call site used
  // definitions(), so turning it on changed nothing. The selector that honours
  // it has to be the one the turn actually calls — that stays true even though
  // the default is now off, because an escape hatch nobody can reach is the
  // same dead wiring in a different costume.
  expect(cli).toContain('deferredToolLoading:')
  expect(runner).toContain('definitionsForTranscript(state.messages)')
})

/**
 * Subsystems that must be CONSTRUCTED by production, not merely importable.
 *
 * The hook list above only guards DaemonRuntimeOptions. It did not catch the
 * durable-task subsystem, whose runtime, bridge and every consumer branch were
 * written and tested while nothing outside a test ever built one — so
 * `durableTaskBridge` was permanently undefined and every recording branch in
 * the subagent manager and the Cortex orchestrator was dead code. A factory
 * called only from tests is the signature of that bug.
 */
const CONSTRUCTED_SUBSYSTEMS: ReadonlyArray<{ readonly factory: string; readonly why: string }> = [
  { factory: 'bridgeDurableTaskLifecycle', why: 'durable record of subagent attempts across a crash' },
  { factory: 'createNativeSubagentHost', why: 'the delegated-turn host every subagent runs on' },
  { factory: 'createLocalWorkspaceProvider', why: 'backs the workspace CLI surface' },
]

test('subsystems with a factory are constructed outside their own tests', async () => {
  const [cli, workspaceCommand] = await Promise.all([
    readSource('cli.ts'),
    readSource('runtime/workspaceCommand.ts'),
  ])
  const production = `${cli}\n${workspaceCommand}`

  for (const subsystem of CONSTRUCTED_SUBSYSTEMS) {
    // A bare mention is not enough — an import with no call site is exactly the
    // shape this is looking for, so require the invocation.
    const invoked = new RegExp(`\\b${subsystem.factory}\\s*\\(`).test(production)
    expect({ factory: subsystem.factory, invoked }, `${subsystem.factory} (${subsystem.why})`)
      .toEqual({ factory: subsystem.factory, invoked: true })
  }
})
