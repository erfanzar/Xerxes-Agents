// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, describe, expect, spyOn, test } from 'bun:test'

import { ValidationError } from '../src/core/errors.js'
import { CronJob, CronScheduler, JobStore } from '../src/cron/index.js'
import { DeclarativeToolForge } from '../src/extensions/declarativeForge.js'
import {
  HookRunner,
  TOOL_PERMISSION_HOOK,
  resolveToolPermission,
  type HookFailureEvent,
} from '../src/extensions/hooks.js'
import { parseShellHookConfig, registerShellHooks } from '../src/extensions/shellHooks.js'
import { inspectWorkspaceHooks, workspaceShellHooks } from '../src/extensions/workspaceHooks.js'
import { LspManager } from '../src/lsp/manager.js'
import { ruffFormat } from '../src/lsp/ruffFormat.js'
import { MCPClient } from '../src/mcp/client.js'
import { MCPManager } from '../src/mcp/manager.js'
import { MCPConnectionError, type MCPCapabilityKind } from '../src/mcp/types.js'
import {
  CdpBrowserAdapter,
  type CdpConnection,
  type CdpConnectionFactory,
} from '../src/operators/cdpBrowser.js'
import { PtySessionManager } from '../src/operators/pty.js'
import { SpawnedAgentManager } from '../src/operators/subagents.js'

const tempDirs: string[] = []

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { force: true, recursive: true })
  }
})

function makeTempDir(prefix = 'xerxes-swarm-ext-mcp-'): string {
  const dir = mkdtempSync(join(tmpdir(), prefix))
  tempDirs.push(dir)
  return dir
}

describe('Extensions: shellHooks, HookRunner, and DeclarativeToolForge', () => {
  test('PreToolUse shell hooks with a matcher allow non-matching tools via resolveToolPermission while denying matches', async () => {
    const root = makeTempDir()
    const home = join(root, 'home')
    const workspace = join(root, 'workspace')
    writeFileSync(join(root, '.keep'), '')
    const { mkdirSync } = await import('node:fs')
    mkdirSync(home, { recursive: true })
    mkdirSync(workspace, { recursive: true })

    let executedCommands = 0
    const runner = new HookRunner()
    registerShellHooks(
      runner,
      parseShellHookConfig(
        {
          PreToolUse: [
            { command: 'guard-bash', matcher: '^Bash$' },
            { command: 'guard-write', matcher: '^WriteFile$' },
          ],
        },
        'test',
      ),
      {
        cwd: workspace,
        run: async command => {
          executedCommands += 1
          if (command === 'guard-bash') {
            return { code: 2, stdout: '', stderr: 'Bash blocked by policy' }
          }
          return { code: 0, stdout: '', stderr: '' }
        },
      },
    )

    // Non-matching tool must be allowed without invoking any shell process.
    const readDecision = await resolveToolPermission(runner, {
      toolName: 'ReadFile',
      arguments: { path: 'README.md' },
    })
    expect(readDecision).toEqual({ allowed: true, denials: [], reason: '' })
    expect(executedCommands).toBe(0)

    // Matching allowed tool invokes only its matching hook.
    const writeDecision = await resolveToolPermission(runner, {
      toolName: 'WriteFile',
      arguments: { path: 'out.txt' },
    })
    expect(writeDecision).toEqual({ allowed: true, denials: [], reason: '' })
    expect(executedCommands).toBe(1)

    // Matching denied tool fails closed with stderr reason while the non-matching sibling hook abstains cleanly.
    const bashDecision = await resolveToolPermission(runner, {
      toolName: 'Bash',
      arguments: { command: 'rm -rf /' },
    })
    expect(bashDecision.allowed).toBe(false)
    expect(bashDecision.reason).toBe('Bash blocked by policy')
    expect(executedCommands).toBe(2)

    // Also verify workspaceShellHooks keeps unmatched PreToolUse calls out of `recent` while allowing the tool.
    writeFileSync(
      join(home, 'config.json'),
      JSON.stringify({
        hooks: {
          PreToolUse: [{ command: 'exit 2', matcher: '^Bash$' }],
        },
      }),
    )
    const wsRunner = workspaceShellHooks({ home, allowWorkspace: false, reportError: () => {} })(workspace)
    const wsRead = await resolveToolPermission(wsRunner, { toolName: 'ReadFile', arguments: {} })
    expect(wsRead.allowed).toBe(true)
    expect(inspectWorkspaceHooks(wsRunner)?.recent).toHaveLength(0)
  })

  test('HookRunner snapshots callbacks across self-unregistration, supports once: true, and enforces hook timeouts', async () => {
    const quiet = spyOn(console, 'error').mockImplementation(() => {})
    try {
      const failures: HookFailureEvent[] = []
      const runner = new HookRunner({
        defaultTimeoutMs: 250,
        onFailure: event => {
          failures.push(event)
        },
      })

      // 1. Self-unregistering callback must NOT skip the next permission guard in registration order.
      const selfRemoving = () => {
        runner.unregister(TOOL_PERMISSION_HOOK, selfRemoving)
        return { allow: true }
      }
      runner.register(TOOL_PERMISSION_HOOK, selfRemoving)
      runner.register(TOOL_PERMISSION_HOOK, () => ({
        allow: false,
        reason: 'second guard still ran',
        source: 'guard-2',
      }))

      const firstDecision = await resolveToolPermission(runner, { toolName: 'Bash' })
      expect(firstDecision.allowed).toBe(false)
      expect(firstDecision.reason).toBe('second guard still ran')

      // 2. Per-hook timeout on permission hook fails closed; mutation hook timeout preserves prior value.
      const timeoutRunner = new HookRunner({
        onFailure: event => {
          failures.push(event)
        },
      })
      timeoutRunner.register(
        TOOL_PERMISSION_HOOK,
        () => new Promise(() => {}),
        { timeoutMs: 25 },
      )
      const timedOutPerm = await resolveToolPermission(timeoutRunner, { toolName: 'Bash' })
      expect(timedOutPerm.allowed).toBe(false)
      expect(timedOutPerm.reason).toContain('timed out after 25ms')

      let onceRuns = 0
      timeoutRunner.register(
        'before_tool_call',
        payload => {
          onceRuns += 1
          return { ...(payload.arguments as Record<string, unknown>), step1: true }
        },
        { once: true },
      )
      timeoutRunner.register(
        'before_tool_call',
        () => new Promise(() => {}),
        { timeoutMs: 25 },
      )
      timeoutRunner.register('before_tool_call', payload => ({
        ...(payload.arguments as Record<string, unknown>),
        step3: true,
      }))

      const firstMutated = await timeoutRunner.run('before_tool_call', { arguments: { init: 1 } })
      expect(firstMutated).toEqual({ init: 1, step1: true, step3: true })

      const secondMutated = await timeoutRunner.run('before_tool_call', { arguments: { init: 2 } })
      expect(secondMutated).toEqual({ init: 2, step3: true })
      expect(onceRuns).toBe(1)
      expect(failures.some(item => item.point === 'before_tool_call' && item.timedOut)).toBe(true)
    } finally {
      quiet.mockRestore()
    }
  })

  test('DeclarativeToolForge orders SemVer pre-releases below final releases and compares numeric pre-release identifiers', () => {
    const dir = makeTempDir()
    const forge = new DeclarativeToolForge(join(dir, 'forged.json'))

    for (const version of ['1.0.0-alpha.2', '1.0.0', '1.0.0-alpha.10', '1.0.0-alpha.1', '1.0.0-beta.1']) {
      forge.define({
        name: 'semver-pkg',
        version,
        description: `Release ${version}`,
        template: `v=${version}`,
      })
    }

    expect(forge.list().map(pkg => pkg.version)).toEqual([
      '1.0.0',
      '1.0.0-beta.1',
      '1.0.0-alpha.10',
      '1.0.0-alpha.2',
      '1.0.0-alpha.1',
    ])
    expect(forge.inspect('semver-pkg')?.version).toBe('1.0.0')
    expect(forge.run('semver-pkg', undefined, {})).toEqual({
      name: 'semver-pkg',
      version: '1.0.0',
      output: 'v=1.0.0',
    })
  })
})

describe('MCP: concurrent connect deduplication, AbortSignal cancellation, and dynamic list_changed refresh', () => {
  test('MCPClient deduplicates concurrent connect() calls without spawning duplicate stdio processes and sends notifications/cancelled on abort', async () => {
    const dir = makeTempDir()
    const spawnsFile = join(dir, 'spawns.log')
    const cancelledFile = join(dir, 'cancelled.json')

    const serverScript = String.raw`
      const fs = require('node:fs');
      fs.appendFileSync(${JSON.stringify(spawnsFile)}, process.pid + '\n');
      let buffer = '';
      process.stdin.setEncoding('utf8');
      process.stdin.on('data', chunk => {
        buffer += chunk;
        let idx;
        while ((idx = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, idx).trim();
          buffer = buffer.slice(idx + 1);
          if (!line) continue;
          const msg = JSON.parse(line);
          if (msg.method === 'notifications/cancelled') {
            fs.writeFileSync(${JSON.stringify(cancelledFile)}, JSON.stringify(msg.params));
            continue;
          }
          if (msg.id === undefined) continue;
          if (msg.method === 'initialize') {
            setTimeout(() => {
              process.stdout.write(JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: {
                  protocolVersion: '2024-11-05',
                  capabilities: { tools: { listChanged: true } },
                  serverInfo: { name: 'dedup-mcp', version: '1.0.0' },
                },
              }) + '\n');
            }, 30);
          } else if (msg.method === 'tools/list') {
            process.stdout.write(JSON.stringify({
              jsonrpc: '2.0',
              id: msg.id,
              result: { tools: [{ name: 'slow_op', inputSchema: { type: 'object' } }] },
            }) + '\n');
          } else if (msg.method === 'tools/call') {
            // Hold until cancelled.
          } else {
            process.stdout.write(JSON.stringify({
              jsonrpc: '2.0',
              id: msg.id,
              error: { code: -32601, message: 'not implemented' },
            }) + '\n');
          }
        }
      });
    `

    const client = new MCPClient({
      name: 'dedup',
      command: process.execPath,
      args: ['-e', serverScript],
      timeoutMs: 5_000,
    })

    try {
      await Promise.all([client.connect(), client.connect(), client.connect()])
      const spawnedPids = readFileSync(spawnsFile, 'utf8').trim().split('\n').filter(Boolean)
      expect(spawnedPids).toHaveLength(1)

      const controller = new AbortController()
      const callPromise = client.callTool('slow_op', {}, { signal: controller.signal })
      await Bun.sleep(20)
      controller.abort()
      await expect(callPromise).rejects.toBeInstanceOf(MCPConnectionError)

      for (let i = 0; i < 40; i += 1) {
        try {
          const cancelled = JSON.parse(readFileSync(cancelledFile, 'utf8')) as {
            requestId: number
            reason: string
          }
          expect(typeof cancelled.requestId).toBe('number')
          expect(cancelled.reason).toContain('cancelled')
          break
        } catch {
          await Bun.sleep(15)
        }
      }
      expect(JSON.parse(readFileSync(cancelledFile, 'utf8'))).toMatchObject({
        reason: 'Client cancelled request',
      })
    } finally {
      await client.disconnect()
    }
  })

  test('MCPClient propagates caller AbortSignal into Streamable HTTP fetch immediately instead of waiting for timeoutMs', async () => {
    let fetchSignalAborted = false
    const enteredCall = Promise.withResolvers<void>()

    const client = new MCPClient(
      {
        name: 'http-cancel',
        transport: 'streamable_http',
        url: 'https://mcp.example.test/mcp',
        timeoutMs: 30_000,
      },
      {
        fetch: async (_input, init) => {
          const body = JSON.parse(String(init?.body ?? '{}')) as { id?: number; method?: string }
          if (body.method === 'initialize') {
            return new Response(
              JSON.stringify({
                jsonrpc: '2.0',
                id: body.id,
                result: {
                  protocolVersion: '2025-06-18',
                  capabilities: { tools: {} },
                  serverInfo: { name: 'mock-http', version: '1.0.0' },
                },
              }),
              { headers: { 'Content-Type': 'application/json', 'Mcp-Session-Id': 'sess-1' } },
            )
          }
          if (body.method === 'notifications/initialized') {
            return new Response(null, { status: 202 })
          }
          if (body.method === 'tools/list') {
            return new Response(
              JSON.stringify({
                jsonrpc: '2.0',
                id: body.id,
                result: { tools: [{ name: 'hang_http', inputSchema: { type: 'object' } }] },
              }),
              { headers: { 'Content-Type': 'application/json' } },
            )
          }
          if (body.method === 'tools/call') {
            enteredCall.resolve()
            return await new Promise<Response>((_resolve, reject) => {
              init?.signal?.addEventListener(
                'abort',
                () => {
                  fetchSignalAborted = true
                  reject(new DOMException('The operation was aborted.', 'AbortError'))
                },
                { once: true },
              )
            })
          }
          return new Response(null, { status: 204 })
        },
      },
    )

    try {
      await client.connect()
      const controller = new AbortController()
      const startedAt = performance.now()
      const pending = client.callTool('hang_http', {}, { signal: controller.signal })
      await enteredCall.promise
      controller.abort('secret-abort-token')
      const err = await pending.catch((error: unknown) => error)
      expect(err).toBeInstanceOf(MCPConnectionError)
      expect(String(err)).not.toContain('secret-abort-token')
      expect(performance.now() - startedAt).toBeLessThan(1_000)
      await Bun.sleep(10)
      expect(fetchSignalAborted).toBe(true)
    } finally {
      await client.disconnect()
    }
  })

  test('MCPClient and MCPManager automatically refresh tools, resources, and prompts on list_changed notifications', async () => {
    let dynamicUnlocked = false
    let toolsListCalls = 0

    const dynamicServerScript = String.raw`
      let unlocked = false;
      let buffer = '';
      process.stdin.setEncoding('utf8');
      process.stdin.on('data', chunk => {
        buffer += chunk;
        let idx;
        while ((idx = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, idx).trim();
          buffer = buffer.slice(idx + 1);
          if (!line) continue;
          const msg = JSON.parse(line);
          if (msg.id === undefined) continue;
          if (msg.method === 'initialize') {
            process.stdout.write(JSON.stringify({
              jsonrpc: '2.0',
              id: msg.id,
              result: {
                protocolVersion: '2024-11-05',
                capabilities: {
                  tools: { listChanged: true },
                  resources: { listChanged: true },
                  prompts: { listChanged: true },
                },
                serverInfo: { name: 'dynamic-mcp', version: '1.0.0' },
              },
            }) + '\n');
          } else if (msg.method === 'tools/list') {
            const tools = unlocked
              ? [
                  { name: 'unlock', inputSchema: { type: 'object' } },
                  { name: 'newly_published', inputSchema: { type: 'object' } },
                ]
              : [{ name: 'unlock', inputSchema: { type: 'object' } }];
            process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: { tools } }) + '\n');
          } else if (msg.method === 'resources/list') {
            const resources = unlocked ? [{ uri: 'memo://live', name: 'Live' }] : [];
            process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: { resources } }) + '\n');
          } else if (msg.method === 'prompts/list') {
            const prompts = unlocked ? [{ name: 'live_prompt' }] : [];
            process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: msg.id, result: { prompts } }) + '\n');
          } else if (msg.method === 'tools/call') {
            if (msg.params.name === 'unlock') {
              unlocked = true;
              process.stdout.write(JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: { content: [{ type: 'text', text: 'unlocked' }] },
              }) + '\n');
              // Emit burst of list_changed notifications across all three capability kinds.
              process.stdout.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/tools/list_changed' }) + '\n');
              process.stdout.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/tools/list_changed' }) + '\n');
              process.stdout.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/resources/list_changed' }) + '\n');
              process.stdout.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/prompts/list_changed' }) + '\n');
            } else if (msg.params.name === 'newly_published') {
              process.stdout.write(JSON.stringify({
                jsonrpc: '2.0',
                id: msg.id,
                result: { content: [{ type: 'text', text: 'dynamic-ok' }] },
              }) + '\n');
            }
          }
        }
      });
    `
    void dynamicUnlocked
    void toolsListCalls

    const events: Array<{ name: string; kind: MCPCapabilityKind }> = []
    const allThreeUpdated = Promise.withResolvers<void>()
    const seenKinds = new Set<MCPCapabilityKind>()

    const manager = new MCPManager({
      onCapabilitiesChanged: event => {
        events.push(event)
        seenKinds.add(event.kind)
        if (seenKinds.size === 3) {
          allThreeUpdated.resolve()
        }
      },
    })

    try {
      expect(
        await manager.addServer({
          name: 'dynamic',
          command: process.execPath,
          args: ['-e', dynamicServerScript],
          timeoutMs: 5_000,
        }),
      ).toBe(true)

      expect(manager.getAllTools().map(tool => tool.name)).toEqual(['unlock'])
      await manager.callTool('unlock')
      await allThreeUpdated.promise

      expect(manager.getAllTools().map(tool => tool.name)).toEqual(['unlock', 'newly_published'])
      expect(manager.getAllResources().map(resource => resource.uri)).toEqual(['memo://live'])
      expect(manager.getAllPrompts().map(prompt => prompt.name)).toEqual(['live_prompt'])
      await expect(manager.callTool('newly_published')).resolves.toEqual({
        content: [{ type: 'text', text: 'dynamic-ok' }],
      })
      expect(events.map(item => item.name)).toEqual(expect.arrayContaining(['dynamic']))
    } finally {
      await manager.disconnectAll()
    }
  })
})

describe('LSP, Cron, and Operators fixes', () => {
  test('LspManager matches uppercase file extensions case-insensitively and ruffFormat accepts .PY/.PYI', async () => {
    const root = makeTempDir()
    const launched: string[] = []
    const manager = new LspManager(
      {
        servers: [
          { name: 'ts', command: 'ts-srv', languageId: 'typescript', extensions: ['.ts'] },
          { name: 'dts', command: 'dts-srv', languageId: 'typescript', extensions: ['.d.ts'] },
        ],
      },
      async options => {
        launched.push(options.command)
        return {
          execute: async req => `${options.command}:${req.action}`,
          close: async () => {},
        }
      },
    )

    try {
      expect(
        await manager.forWorkspace(root).execute({
          action: 'hover',
          filePath: 'COMPONENT.TS',
          line: 0,
          character: 0,
        }),
      ).toBe('ts-srv:hover')
      expect(
        await manager.forWorkspace(root).execute({
          action: 'hover',
          filePath: 'GLOBAL.D.TS',
          line: 0,
          character: 0,
        }),
      ).toBe('dts-srv:hover')

      // Verify ruffFormat accepts uppercase .PY / .PYI extensions with an injected formatter binary.
      const fakeRuff = join(root, 'fake-ruff.sh')
      writeFileSync(fakeRuff, '#!/bin/sh\ncat\n')
      chmodSync(fakeRuff, 0o755)
      const formatted = await ruffFormat(root, { filePath: 'MODULE.PYI', text: 'x: int = 1\n' }, undefined, fakeRuff)
      expect(formatted.text).toBe('x: int = 1\n')
    } finally {
      await manager.close()
    }
  })

  test('CronScheduler JobClock deducts cumulative active execution time across pause/resume instead of resetting full timeout', async () => {
    const dir = makeTempDir()
    const quiet = spyOn(console, 'error').mockImplementation(() => {})
    try {
      const store = new JobStore(join(dir, 'jobs.json'))
      store.add(
        new CronJob({
          id: 'cumulative-timeout',
          prompt: 'work',
          oneshot: true,
          maxRetries: 0,
          nextRunAt: '2026-05-15T12:00:00Z',
        }),
      )

      const scheduler = new CronScheduler(
        store,
        async (_job, signal, clock) => {
          // 45ms active slice #1
          await Bun.sleep(45)
          signal.throwIfAborted()
          // Pause for 60ms (must NOT count, and resume must NOT reset remaining budget back to 70ms)
          clock.pause()
          await Bun.sleep(60)
          clock.resume()
          // Calling resume() again while running must NOT reset the timer either.
          clock.resume()
          // 50ms active slice #2 -> cumulative active = 95ms > 70ms timeout!
          await new Promise<void>((resolve, reject) => {
            const timer = setTimeout(resolve, 50)
            signal.addEventListener(
              'abort',
              () => {
                clearTimeout(timer)
                reject(signal.reason)
              },
              { once: true },
            )
          })
          return 'unexpected-success'
        },
        { jobTimeout: 70 },
      )

      expect(await scheduler.tick(new Date('2026-05-15T12:00:00Z'))).toEqual([])
      await scheduler.waitForIdle()
      const persisted = store.get('cumulative-timeout')
      expect(persisted?.paused).toBe(true)
      expect(persisted?.metadata.last_error).toContain('timed out after 70ms')
    } finally {
      quiet.mockRestore()
    }
  })

  test('PtySessionManager preserves UTF-16 surrogate pairs across tight buffer eviction and chunk reads', async () => {
    const root = makeTempDir()
    // Limit pending output to force front-eviction in the middle of astral-plane emoji output.
    const manager = new PtySessionManager({ workspaceRoot: root, maxPendingOutputChars: 15 })
    const loneSurrogate = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?![\uD800-\uDBFF])[\uDC00-\uDFFF]/u
    void loneSurrogate
    const hasLoneSurrogate = (text: string): boolean => {
      for (let i = 0; i < text.length; i += 1) {
        const code = text.charCodeAt(i)
        if (code >= 0xd800 && code <= 0xdbff) {
          const next = text.charCodeAt(i + 1)
          if (!(next >= 0xdc00 && next <= 0xdfff)) return true
          i += 1
        } else if (code >= 0xdc00 && code <= 0xdfff) {
          return true
        }
      }
      return false
    }

    try {
      const opened = await manager.createSession(
        `${JSON.stringify(process.execPath)} -e 'process.stdout.write("🚀🌍🔥🎉✨💎🚀🌍🔥🎉✨💎\\n")'`,
        { yieldTimeMs: 500, maxOutputChars: 83 },
      )
      expect(hasLoneSurrogate(opened.stdout)).toBe(false)
      const rest = await manager.write(opened.sessionId, { yieldTimeMs: 100, maxOutputChars: 5 })
      expect(hasLoneSurrogate(rest.stdout)).toBe(false)
      const tail = await manager.write(opened.sessionId, { yieldTimeMs: 100, maxOutputChars: 1000 })
      expect(hasLoneSurrogate(tail.stdout)).toBe(false)
    } finally {
      await manager.closeAll()
    }
  })

  test('SpawnedAgentManager honors AbortSignal in spawn() and trims whitespace-padded handle IDs', async () => {
    let runnerInvocations = 0
    const manager = new SpawnedAgentManager({
      runner: async () => {
        runnerInvocations += 1
        return 'done'
      },
    })

    const aborted = new AbortController()
    aborted.abort()
    await expect(
      manager.spawn({ nickname: 'never-created', message: 'do work', signal: aborted.signal }),
    ).rejects.toThrow()
    expect(manager.hasHandle('never-created')).toBe(false)
    expect(runnerInvocations).toBe(0)

    const created = await manager.spawn({ nickname: 'worker-alpha', message: 'hello' })
    expect(created.id).toBe('worker-alpha')
    expect(manager.hasHandle('  worker-alpha  ')).toBe(true)

    const waited = await manager.wait(['  worker-alpha  '], 2_000)
    expect(waited.completed[0]?.id).toBe('worker-alpha')

    await manager.sendInput('  worker-alpha  ', { message: 'follow-up' })
    await manager.wait(['worker-alpha'], 2_000)
    expect(runnerInvocations).toBe(2)

    const closed = manager.close('  worker-alpha  ')
    expect(closed.closed).toBe(true)
    const resumed = manager.resume('  worker-alpha  ')
    expect(resumed.closed).toBe(false)
  })

  test('CdpBrowserAdapter find() rejects nested catastrophic-backtracking groups before reaching the page', async () => {
    const dir = makeTempDir()
    const commands: string[] = []
    const connection: CdpConnection = {
      close: () => {},
      command: async method => {
        commands.push(method)
        switch (method) {
          case 'Browser.getVersion':
            return { product: 'Fake' }
          case 'Target.createTarget':
            return { targetId: 't1' }
          case 'Target.attachToTarget':
            return { sessionId: 's1' }
          case 'Page.navigate':
            return { frameId: 'f1' }
          case 'Runtime.evaluate':
            return {
              result: {
                value: {
                  url: 'https://example.test',
                  title: 'Test',
                  contentPreview: 'body',
                  links: [],
                },
              },
            }
          case 'Target.detachFromTarget':
          case 'Target.closeTarget':
            return {}
          default:
            return {}
        }
      },
    }
    const connectionFactory: CdpConnectionFactory = { connect: async () => connection }
    const adapter = await CdpBrowserAdapter.connect('ws://127.0.0.1:9222/devtools/browser/1', {
      connectionFactory,
      screenshotDirectory: dir,
    })

    try {
      const page = await adapter.open({ url: 'https://example.test', waitMs: 0 })
      const evalCountBefore = commands.filter(method => method === 'Runtime.evaluate').length

      for (const unsafe of ['((a+))+', '(?:foo(bar+))+', '(((x*)))*']) {
        await expect(adapter.find(page.refId!, unsafe)).rejects.toBeInstanceOf(ValidationError)
      }
      expect(commands.filter(method => method === 'Runtime.evaluate')).toHaveLength(evalCountBefore)
    } finally {
      await adapter.close()
    }
  })

  test('CronScheduler handleFailure reads fresh persisted retry_count when metadata changes during execution', async () => {
    const dir = makeTempDir()
    const quiet = spyOn(console, 'error').mockImplementation(() => {})
    try {
      const store = new JobStore(join(dir, 'jobs.json'))
      store.add(
        new CronJob({
          id: 'fresh-retry',
          prompt: 'work',
          oneshot: true,
          maxRetries: 2,
          nextRunAt: '2026-05-15T12:00:00Z',
          metadata: { retry_count: 0 },
        }),
      )
      const scheduler = new CronScheduler(
        store,
        job => {
          // Simulate operator/recovery updating persisted metadata mid-run.
          const current = store.get(job.id)!
          store.update(job.id, { metadata: { ...current.metadata, retry_count: 2 } })
          throw new Error('run-failed')
        },
        { oneShotRetryBaseMs: 1_000 },
      )
      expect(await scheduler.tick(new Date('2026-05-15T12:00:00Z'))).toEqual([])
      const updated = store.get('fresh-retry')!
      // retry_count should advance from 2 -> 3 (> maxRetries 2), pausing the job immediately.
      expect(updated.metadata.retry_count).toBe(3)
      expect(updated.paused).toBe(true)
    } finally {
      quiet.mockRestore()
    }
  })
})
