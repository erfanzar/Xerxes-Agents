// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  bunDaemonLaunch,
  bunDaemonEnvironment,
  DAEMON_CONNECT_RETRY_MS,
  daemonBuildDecision,
  daemonCommandMatches,
  daemonPaths,
  formatBunDaemonStartupFailure,
  GatewayClient,
  NativeDaemonUnsupportedError,
  resolveBunDaemonEntry,
  resolveProjectDir,
  shellResultFromSlashResponse
} from '../gatewayClient.js'
import { isResponseFrame, normalizeEventType } from '../gatewayTypes.js'

describe('daemonPaths', () => {
  const prevHome = process.env.XERXES_HOME
  const prevSock = process.env.XERXES_DAEMON_SOCKET

  afterEach(() => {
    if (prevHome === undefined) delete process.env.XERXES_HOME
    else process.env.XERXES_HOME = prevHome
    if (prevSock === undefined) delete process.env.XERXES_DAEMON_SOCKET
    else process.env.XERXES_DAEMON_SOCKET = prevSock
  })

  it('derives the per-project digest socket under $XERXES_HOME/daemon/projects', () => {
    delete process.env.XERXES_DAEMON_SOCKET
    process.env.XERXES_HOME = '/tmp/xh'
    const dir = '/Users/me/proj'
    const digest = createHash('sha256').update(dir, 'utf8').digest('hex').slice(0, 16)
    const { socketPath, pidPath } = daemonPaths(dir)
    expect(socketPath).toBe(`/tmp/xh/daemon/projects/${digest}.sock`)
    expect(pidPath).toBe(`/tmp/xh/daemon/projects/${digest}.pid`)
  })

  it('defaults $XERXES_HOME to ~/.xerxes', () => {
    delete process.env.XERXES_HOME
    delete process.env.XERXES_DAEMON_SOCKET
    const { socketPath } = daemonPaths('/x')
    expect(socketPath.startsWith(join(homedir(), '.xerxes', 'daemon', 'projects'))).toBe(true)
    expect(socketPath.endsWith('.sock')).toBe(true)
  })

  it('honors XERXES_DAEMON_SOCKET override for the socket, keeps digest pid', () => {
    process.env.XERXES_HOME = '/tmp/xh'
    process.env.XERXES_DAEMON_SOCKET = '/run/custom.sock'
    const { socketPath, pidPath } = daemonPaths('/x')
    expect(socketPath).toBe('/run/custom.sock')
    expect(pidPath).toContain('/tmp/xh/daemon/projects/')
  })

  it('produces a 16-hex-char digest', () => {
    delete process.env.XERXES_DAEMON_SOCKET
    process.env.XERXES_HOME = '/tmp/xh'
    const { socketPath } = daemonPaths('/any/where')
    const name = socketPath.split('/').pop() ?? ''
    expect(name).toMatch(/^[0-9a-f]{16}\.sock$/)
  })

  it('canonicalizes subdirectories to the git repository root', () => {
    const root = mkdtempSync(join(tmpdir(), 'xerxes-gateway-'))
    try {
      execFileSync('git', ['init'], { cwd: root, stdio: 'ignore' })
      const child = join(root, 'xerxes', 'src', 'ui')
      mkdirSync(child, { recursive: true })
      expect(resolveProjectDir(child)).toBe(realpathSync(root))
    } finally {
      rmSync(root, { force: true, recursive: true })
    }
  })
})

describe('Bun daemon launcher', () => {
  it('passes the expected build identity to a spawned daemon without dropping host variables', () => {
    expect(bunDaemonEnvironment('build-123', { HOME: '/home/test', XERXES_DAEMON_BUILD_ID: 'old' })).toEqual({
      HOME: '/home/test',
      XERXES_DAEMON_BUILD_ID: 'build-123'
    })
  })

  it('honors configured Bun binary and entry paths while preserving v35 daemon arguments', () => {
    const root = mkdtempSync(join(tmpdir(), 'xerxes-bun-launch-'))
    try {
      const entryPath = join(root, 'runtime', 'cli.ts')
      mkdirSync(join(root, 'runtime'), { recursive: true })
      writeFileSync(entryPath, 'console.log("daemon")\n')

      const launch = bunDaemonLaunch('/work/project', '/run/xerxes.sock', '/run/xerxes.pid', {
        XERXES_TUI_BUN: '/opt/bun/bin/bun',
        XERXES_TUI_BUN_DAEMON: entryPath
      })

      expect(launch).toEqual({
        binary: '/opt/bun/bin/bun',
        entryPath,
        args: [
          entryPath,
          'daemon',
          '--project-dir',
          '/work/project',
          '--socket',
          '/run/xerxes.sock',
          '--pid-file',
          '/run/xerxes.pid'
        ]
      })
    } finally {
      rmSync(root, { force: true, recursive: true })
    }
  })

  it('resolves a relative configured entry from the selected project root', () => {
    const root = mkdtempSync(join(tmpdir(), 'xerxes-bun-entry-'))
    try {
      const entryPath = join(root, 'runtime', 'cli.ts')
      mkdirSync(join(root, 'runtime'), { recursive: true })
      writeFileSync(entryPath, 'console.log("daemon")\n')

      expect(resolveBunDaemonEntry(root, { XERXES_BUN_DAEMON: 'runtime/cli.ts' })).toBe(entryPath)
    } finally {
      rmSync(root, { force: true, recursive: true })
    }
  })

  it('finds the colocated TypeScript runtime without configuration', () => {
    const entryPath = resolveBunDaemonEntry('/tmp/independent-xerxes-project', {})

    expect(entryPath.endsWith(join('xerxes', 'src', 'cli.ts'))).toBe(true)
  })

  it('fails early for an explicitly configured missing entry instead of timing out a launch', () => {
    expect(() => resolveBunDaemonEntry('/work/project', { XERXES_TUI_BUN_DAEMON: 'missing/cli.ts' })).toThrow(
      'Configured Bun daemon entry does not exist'
    )
  })

  it('renders a Bun daemon startup failure for the TUI without retired Python wording', () => {
    const message = formatBunDaemonStartupFailure(
      new Error('Configured Bun daemon entry does not exist: /tmp/missing.ts')
    )

    expect(message).toBe('Bun daemon startup failed: Configured Bun daemon entry does not exist: /tmp/missing.ts')
    expect(message).not.toContain('Python')
  })

  it('retries a cold daemon at a short cadence without busy-spinning', async () => {
    vi.useFakeTimers()
    try {
      const client = new GatewayClient({ projectDir: process.cwd() })
      const privateClient = client as unknown as {
        spawnBunDaemon: (socketPath: string, pidPath: string) => void
        tryConnect: (socketPath: string) => Promise<boolean>
      }
      let attempts = 0
      let spawns = 0
      privateClient.tryConnect = async () => {
        attempts += 1
        return attempts === 3
      }
      privateClient.spawnBunDaemon = () => {
        spawns += 1
      }

      const starting = client.start()
      await vi.advanceTimersByTimeAsync(0)

      expect(DAEMON_CONNECT_RETRY_MS).toBe(25)
      expect(attempts).toBe(2)
      expect(spawns).toBe(1)

      await vi.advanceTimersByTimeAsync(DAEMON_CONNECT_RETRY_MS - 1)
      expect(attempts).toBe(2)

      await vi.advanceTimersByTimeAsync(1)
      await starting
      expect(attempts).toBe(3)
    } finally {
      vi.useRealTimers()
    }
  })

  it('replaces a verified stale local daemon before reporting gateway.ready', async () => {
    const client = new GatewayClient({
      expectedDaemonBuildId: 'expected-build',
      projectDir: process.cwd()
    })
    const privateClient = client as unknown as {
      ensureConnectedDaemonCurrent: (socketPath: string, pidPath: string) => Promise<boolean>
      spawnBunDaemon: (socketPath: string, pidPath: string) => void
      tryConnect: (socketPath: string) => Promise<boolean>
    }
    let checks = 0
    let spawns = 0
    privateClient.tryConnect = async () => true
    privateClient.ensureConnectedDaemonCurrent = async () => ++checks > 1
    privateClient.spawnBunDaemon = () => {
      spawns += 1
    }

    const ready = vi.fn()
    client.on('gateway.ready', ready)
    await client.start()

    expect(checks).toBe(2)
    expect(spawns).toBe(1)
    expect(ready).toHaveBeenCalledWith(
      expect.objectContaining({ payload: expect.objectContaining({ spawned: true }) })
    )
  })
})

describe('source daemon build compatibility', () => {
  const launch = {
    binary: 'bun',
    entryPath: '/checkout/src/cli.ts',
    args: [
      '/checkout/src/cli.ts',
      'daemon',
      '--project-dir',
      '/work/project',
      '--socket',
      '/run/xerxes.sock',
      '--pid-file',
      '/run/xerxes.pid'
    ]
  } as const

  it('matches only the exact source checkout daemon command', () => {
    expect(
      daemonCommandMatches(
        'bun /checkout/src/cli.ts daemon --project-dir /work/project --socket /run/xerxes.sock --pid-file /run/xerxes.pid',
        launch
      )
    ).toBe(true)
    expect(
      daemonCommandMatches(
        'bun /other/src/cli.ts daemon --project-dir /work/project --socket /run/xerxes.sock --pid-file /run/xerxes.pid',
        launch
      )
    ).toBe(false)
  })

  it('restarts only a mismatched default daemon with matching RPC, pid file, and process identity', () => {
    const local = {
      activeTurns: false,
      actualBuildId: 'old',
      commandMatches: true,
      daemonPid: 123,
      daemonProtocol: 35,
      daemonRuntime: 'bun-typescript',
      expectedBuildId: 'new',
      explicitSocket: false,
      pidFilePid: 123
    } as const

    expect(daemonBuildDecision(local)).toBe('restart')
    expect(daemonBuildDecision({ ...local, actualBuildId: 'new' })).toBe('current')
    expect(daemonBuildDecision({ ...local, actualBuildId: 'new', daemonProtocol: 34 })).toBe('reject')
    expect(daemonBuildDecision({ ...local, explicitSocket: true })).toBe('reject')
    expect(daemonBuildDecision({ ...local, activeTurns: true })).toBe('reject')
    expect(daemonBuildDecision({ ...local, pidFilePid: 999 })).toBe('reject')
    expect(daemonBuildDecision({ ...local, commandMatches: false })).toBe('reject')
  })
})

describe('native RPC compatibility boundary', () => {
  it.each([
    ['skills.manage', 'bun run xerxes skill'],
    ['reload.mcp', 'Restart the Bun daemon'],
    ['process.stop', 'Use /stop'],
    ['session.close', 'Native sessions are persistent']
  ])('rejects %s instead of fabricating a success response', async (method, guidance) => {
    const client = new GatewayClient({ projectDir: process.cwd() })

    await expect(client.request(method)).rejects.toEqual(
      expect.objectContaining({
        method,
        name: NativeDaemonUnsupportedError.name,
        message: expect.stringContaining(guidance)
      })
    )
  })
})

describe('normalizeEventType', () => {
  it('folds PascalCase (bridge) names to snake_case (socket)', () => {
    expect(normalizeEventType('TextPart')).toBe('text_part')
    expect(normalizeEventType('QuestionRequest')).toBe('question_request')
    expect(normalizeEventType('MCPLoadingBegin')).toBe('mcp_loading_begin')
    expect(normalizeEventType('SubagentEvent')).toBe('subagent_event')
  })

  it('passes snake_case through unchanged', () => {
    expect(normalizeEventType('text_part')).toBe('text_part')
    expect(normalizeEventType('init_done')).toBe('init_done')
  })

  it('leaves unknown names untouched', () => {
    expect(normalizeEventType('something_new')).toBe('something_new')
  })
})

describe('isResponseFrame', () => {
  it('treats id-bearing frames as responses', () => {
    expect(isResponseFrame({ id: 1, result: {} })).toBe(true)
    expect(isResponseFrame({ id: 0, result: {} })).toBe(true)
  })

  it('treats id-less event frames as not-responses', () => {
    expect(isResponseFrame({ method: 'event', params: {} })).toBe(false)
    expect(isResponseFrame({})).toBe(false)
  })
})

describe('shellResultFromSlashResponse', () => {
  it('does not turn an unsupported Bun bang command into a false code-0 success', () => {
    expect(shellResultFromSlashResponse({ ok: false, error: 'Bang commands are not supported.' })).toEqual({
      code: 127,
      stderr: 'Bang commands are not supported.',
      stdout: ''
    })
  })

  it('keeps successful slash output in the shell-response shape', () => {
    expect(shellResultFromSlashResponse({ ok: true, output: 'done', stderr: '' })).toEqual({
      code: 0,
      stderr: '',
      stdout: 'done'
    })
  })
})
