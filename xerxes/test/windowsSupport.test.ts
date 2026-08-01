// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Windows host support. Every branch here is reachable from a POSIX test run
// because the platform is an injected parameter rather than a read of
// `process.platform` — a Windows-only branch that only executes on Windows CI is
// a branch that regresses silently between Windows CI runs.

import { expect, test } from 'bun:test'
import { mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  controlChannelPath,
  defaultControlChannelPath,
  defaultInteractiveShell,
  environmentNamesMatch,
  fallbackExecutablePath,
  isNamedPipePath,
  isWindows,
  safeEnvironmentNames,
  shellCommandArgv,
} from '../src/core/hostPlatform.js'
import { defaultAcpRegistryDirectory } from '../src/acp/registry.js'
import { processCommandProbe } from '../src/core/processLiveness.js'
import { isBatchScript, planSpawn, WindowsSpawnError } from '../src/core/windowsSpawn.js'
import { daemonPaths } from '../src/daemon/paths.js'
import { PtySessionManager } from '../src/operators/pty.js'
import { TerminalRegistry } from '../src/runtime/terminalRegistry.js'
import { DaemonTranscriptStore, normalizeDaemonTranscript } from '../src/session/daemonTranscript.js'
import { controlChannelPath as uiControlChannelPath } from '../src/ui/lib/hostPlatform.js'

const DIGEST = 'a1b2c3d4e5f60718'

test('the control channel is a named pipe on Windows and a Unix socket elsewhere', () => {
  expect(controlChannelPath('/home/u/.xerxes/daemon/projects', DIGEST, 'win32'))
    .toBe(`\\\\.\\pipe\\xerxes-${DIGEST}`)
  expect(controlChannelPath('/home/u/.xerxes/daemon/projects', DIGEST, 'linux'))
    .toBe(`/home/u/.xerxes/daemon/projects/${DIGEST}.sock`)
  expect(defaultControlChannelPath('/home/u/.xerxes/daemon', 'win32')).toBe('\\\\.\\pipe\\xerxes-daemon')
  expect(defaultControlChannelPath('/home/u/.xerxes/daemon', 'darwin')).toBe('/home/u/.xerxes/daemon/xerxes.sock')
})

test('the runtime and TUI derive the identical control-channel address', () => {
  // The TUI cannot import runtime modules (it compiles under rootDir: src/ui),
  // so the two implementations are separate code. If they ever disagree the TUI
  // concludes no daemon is listening and silently starts a second one, so this
  // is the test that makes the duplication safe rather than merely tolerated.
  for (const platform of ['win32', 'linux', 'darwin'] as const) {
    expect(uiControlChannelPath('/base/projects', DIGEST, platform))
      .toBe(controlChannelPath('/base/projects', DIGEST, platform))
  }
})

test('daemonPaths keeps the pid file on disk while the channel becomes a pipe', () => {
  // A pipe name is not enumerable the way a directory is, so the pid file stays
  // a real file on Windows — it is what proves which process owns the channel.
  const windows = daemonPaths('/project', { XERXES_HOME: '/xh' }, 'win32')
  expect(isNamedPipePath(windows.socketPath)).toBe(true)
  expect(windows.pidPath.endsWith('.pid')).toBe(true)
  expect(isNamedPipePath(windows.pidPath)).toBe(false)

  const posix = daemonPaths('/project', { XERXES_HOME: '/xh' }, 'linux')
  expect(posix.socketPath.endsWith('.sock')).toBe(true)
  expect(isNamedPipePath(posix.socketPath)).toBe(false)
})

test('an explicit XERXES_DAEMON_SOCKET overrides the derived address on every platform', () => {
  const override = '\\\\.\\pipe\\custom-channel'
  expect(daemonPaths('/project', { XERXES_HOME: '/xh', XERXES_DAEMON_SOCKET: override }, 'win32').socketPath)
    .toBe(override)
  expect(daemonPaths('/project', { XERXES_HOME: '/xh', XERXES_DAEMON_SOCKET: '/tmp/x.sock' }, 'linux').socketPath)
    .toBe('/tmp/x.sock')
})

test('isNamedPipePath recognizes both pipe prefixes and rejects ordinary paths', () => {
  expect(isNamedPipePath('\\\\.\\pipe\\xerxes-abc')).toBe(true)
  expect(isNamedPipePath('\\\\?\\pipe\\xerxes-abc')).toBe(true)
  expect(isNamedPipePath('\\\\.\\PIPE\\xerxes-abc')).toBe(true)
  expect(isNamedPipePath('/home/u/.xerxes/daemon/projects/abc.sock')).toBe(false)
  expect(isNamedPipePath('C:\\Users\\u\\pipe\\abc')).toBe(false)
  expect(isNamedPipePath('')).toBe(false)
})

test('the default interactive shell follows the host convention', () => {
  expect(defaultInteractiveShell({ COMSPEC: 'C:\\Windows\\system32\\cmd.exe' }, 'win32'))
    .toBe('C:\\Windows\\system32\\cmd.exe')
  expect(defaultInteractiveShell({}, 'win32')).toBe('cmd.exe')
  expect(defaultInteractiveShell({ SHELL: '/bin/zsh' }, 'linux')).toBe('/bin/zsh')
  expect(defaultInteractiveShell({}, 'linux')).toBe('/bin/sh')
})

test('shellCommandArgv uses cmd.exe and PowerShell argument forms on Windows', () => {
  // `-c` is not a cmd.exe flag. Passing it starts an interactive shell that
  // never exits, so a PTY session would hang instead of running the command.
  expect(shellCommandArgv('C:\\Windows\\system32\\cmd.exe', 'git status', true, 'win32'))
    .toEqual(['C:\\Windows\\system32\\cmd.exe', '/d', '/s', '/c', 'git status'])
  expect(shellCommandArgv('powershell.exe', 'git status', true, 'win32'))
    .toEqual(['powershell.exe', '-NoLogo', '-NoProfile', '-Command', 'git status'])
  expect(shellCommandArgv('pwsh.exe', '', true, 'win32')).toEqual(['pwsh.exe', '-NoLogo', '-NoProfile'])
  // A login shell has no Windows equivalent, so `login` must not add `-l`.
  expect(shellCommandArgv('cmd.exe', 'dir', true, 'win32')).not.toContain('-l')
})

test('shellCommandArgv preserves the existing POSIX login behaviour', () => {
  expect(shellCommandArgv('/bin/bash', 'ls', true, 'linux')).toEqual(['/bin/bash', '-l', '-c', 'ls'])
  expect(shellCommandArgv('/bin/bash', 'ls', false, 'linux')).toEqual(['/bin/bash', '-c', 'ls'])
  expect(shellCommandArgv('/bin/sh', 'ls', true, 'linux')).toEqual(['/bin/sh', '-c', 'ls'])
  expect(shellCommandArgv('/bin/zsh', '', true, 'linux')).toEqual(['/bin/zsh', '-l'])
})

test('the sandbox environment allow-list covers what Windows children actually need', () => {
  const windows = safeEnvironmentNames('win32')
  // Without SystemRoot a child cannot load system DLLs; without PATHEXT it
  // cannot resolve `git` to `git.exe`; without TEMP anything writing a scratch
  // file fails. None of these are optional conveniences.
  for (const required of ['PATH', 'PATHEXT', 'COMSPEC', 'SystemRoot', 'TEMP']) {
    expect(windows).toContain(required)
  }
  expect(safeEnvironmentNames('linux')).toEqual(['PATH', 'HOME', 'LANG', 'LC_ALL', 'TERM'])
})

test('the sandbox PATH fallback resolves real directories on Windows', () => {
  // The previous Windows fallback was the empty string: every allow-listed
  // command then failed with ENOENT instead of running, so a strict sandbox
  // looked healthy while executing nothing at all.
  const path = fallbackExecutablePath({ SystemRoot: 'C:\\Windows' }, 'win32')
  expect(path).toContain('C:\\Windows\\system32')
  expect(path.length).toBeGreaterThan(0)
  expect(fallbackExecutablePath({}, 'win32')).toContain('C:\\Windows\\system32')
  expect(fallbackExecutablePath({}, 'linux')).toBe('/usr/bin:/bin')
})

test('environment names compare case-insensitively only on Windows', () => {
  // Windows environment blocks are case-insensitive, so a block-list that only
  // rejects NODE_OPTIONS would let node_options through to the same variable.
  expect(environmentNamesMatch('node_options', 'NODE_OPTIONS', 'win32')).toBe(true)
  expect(environmentNamesMatch('node_options', 'NODE_OPTIONS', 'linux')).toBe(false)
  expect(environmentNamesMatch('PATH', 'PATH', 'linux')).toBe(true)
})

test('process identity is read with ps on POSIX and a CIM query on Windows', () => {
  const [posixCommand, posixArgs] = processCommandProbe(4321, 'linux')
  expect(posixCommand).toBe('ps')
  expect(posixArgs).toEqual(['-p', '4321', '-o', 'command='])

  const [windowsCommand, windowsArgs] = processCommandProbe(4321, 'win32')
  expect(windowsCommand).toBe('powershell.exe')
  expect(windowsArgs.join(' ')).toContain('ProcessId=4321')
  expect(windowsArgs).toContain('-NoProfile')
})

test('a non-finite pid cannot reach the Windows identity query as syntax', () => {
  const [, args] = processCommandProbe(Number.NaN, 'win32')
  expect(args.join(' ')).toContain('ProcessId=-1')
  expect(args.join(' ')).not.toContain('NaN')
  expect(processCommandProbe(12.9, 'win32')[1].join(' ')).toContain('ProcessId=12')
})

test('planSpawn leaves POSIX argv untouched', () => {
  expect(planSpawn('git', ['status'], { platform: 'linux' })).toEqual({ argv: ['git', 'status'] })
})

test('planSpawn resolves through PATHEXT and runs a plain executable directly', () => {
  const plan = planSpawn('git', ['status'], {
    platform: 'win32',
    which: () => 'C:\\Program Files\\Git\\cmd\\git.exe',
  })
  expect(plan.argv).toEqual(['C:\\Program Files\\Git\\cmd\\git.exe', 'status'])
  expect(plan.windowsVerbatimArguments).toBeUndefined()
})

test('planSpawn wraps a .cmd shim in cmd.exe, which is the only way to run one', () => {
  // Every npm-installed CLI on Windows is a .cmd shim, so an MCP server launched
  // as `npx …` lands here; CreateProcess cannot execute a batch file at all.
  const plan = planSpawn('npx', ['-y', 'some-mcp-server'], {
    platform: 'win32',
    env: { COMSPEC: 'C:\\Windows\\system32\\cmd.exe' },
    which: () => 'C:\\Users\\u\\AppData\\Roaming\\npm\\npx.cmd',
  })
  expect(plan.windowsVerbatimArguments).toBe(true)
  expect(plan.argv.slice(0, 4)).toEqual(['C:\\Windows\\system32\\cmd.exe', '/d', '/s', '/c'])
  // One outer quote pair for `/s` to strip, with each token quoted inside it.
  expect(plan.argv[4]).toBe('""C:\\Users\\u\\AppData\\Roaming\\npm\\npx.cmd" "-y" "some-mcp-server""')
})

test('planSpawn keeps an argument containing spaces as a single token', () => {
  const plan = planSpawn('npx', ['--goal', 'audit the auth layer'], {
    platform: 'win32',
    env: { COMSPEC: 'cmd.exe' },
    which: () => 'C:\\npm\\npx.cmd',
  })
  expect(plan.argv[4]).toContain('"audit the auth layer"')
})

test('planSpawn refuses batch arguments that cmd.exe quoting cannot neutralize', () => {
  // cmd expands %VAR% inside double quotes, so no quoting makes a `%` inert.
  // Rejecting beats running a command the caller did not write (CVE-2024-27980).
  const attempt = (argument: string) =>
    planSpawn('npx', [argument], { platform: 'win32', which: () => 'C:\\npm\\npx.cmd' })
  expect(() => attempt('%PATH%')).toThrow(WindowsSpawnError)
  expect(() => attempt('a\nb')).toThrow(WindowsSpawnError)
  expect(() => attempt('a\rb')).toThrow(WindowsSpawnError)
  // A plain executable is not reinterpreted by a shell, so the same argument is
  // fine when no batch wrapping is involved.
  expect(() =>
    planSpawn('git', ['%PATH%'], { platform: 'win32', which: () => 'C:\\Git\\git.exe' })
  ).not.toThrow()
})

test('planSpawn passes an unresolvable name through so the runtime reports it', () => {
  expect(planSpawn('nope', ['x'], { platform: 'win32', which: () => null }).argv).toEqual(['nope', 'x'])
})

test('isBatchScript matches the extensions CreateProcess cannot execute', () => {
  expect(isBatchScript('C:\\npm\\npx.cmd')).toBe(true)
  expect(isBatchScript('C:\\npm\\NPX.CMD')).toBe(true)
  expect(isBatchScript('C:\\tools\\run.bat')).toBe(true)
  expect(isBatchScript('C:\\Git\\git.exe')).toBe(false)
  expect(isBatchScript('/usr/bin/git')).toBe(false)
})

test('the ACP registry root follows %APPDATA% on Windows and XDG elsewhere', () => {
  // XDG_CONFIG_HOME does not exist on Windows; writing agent.json there would
  // strand the registry where no ACP client looks for it.
  // The injected platform picks the joiner too, so every expectation below is
  // an exact literal that holds on a POSIX host and a Windows host alike.
  expect(defaultAcpRegistryDirectory({ APPDATA: 'C:\\Users\\u\\AppData\\Roaming' }, 'C:\\Users\\u', 'win32'))
    .toBe('C:\\Users\\u\\AppData\\Roaming\\agent-registry')
  expect(defaultAcpRegistryDirectory({}, 'C:\\Users\\u', 'win32'))
    .toBe('C:\\Users\\u\\AppData\\Roaming\\agent-registry')
  expect(defaultAcpRegistryDirectory({ XDG_CONFIG_HOME: '/xdg' }, '/home/u', 'linux')).toBe('/xdg/agent-registry')
  expect(defaultAcpRegistryDirectory({}, '/home/u', 'linux')).toBe('/home/u/.config/agent-registry')
})

test('isWindows only reports win32', () => {
  expect(isWindows('win32')).toBe(true)
  expect(isWindows('linux')).toBe(false)
  expect(isWindows('darwin')).toBe(false)
})

// Real-machine end-to-end cases. The injected-platform tests above prove the
// branching logic; these prove the branch that only a Windows host can reach.

test.skipIf(process.platform !== 'win32')(
  'a PTY session on Windows runs the command instead of dying instantly with exit 1',
  async () => {
    // Bun.spawn `detached` maps to DETACHED_PROCESS ("no console") on Windows,
    // which contradicts the ConPTY pseudoconsole the terminal option attaches:
    // the child exited with code 1 before printing a byte, so the F8 panel only
    // ever showed dead pty rows. `detached` is POSIX session leadership only.
    const terminals = new TerminalRegistry()
    const manager = new PtySessionManager({ terminals })
    const result = await manager.createSession('echo pty-windows-alive', { yieldTimeMs: 2_000 })
    try {
      expect(result.stdout).toContain('pty-windows-alive')
      expect(terminals.inspect(result.sessionId)?.output).toContain('pty-windows-alive')
    } finally {
      await manager.closeAll()
    }
  },
)

test.skipIf(process.platform !== 'win32')(
  'saving a transcript on Windows does not fsync a directory handle (EPERM)',
  async () => {
    // Windows cannot fsync a directory: the atomic transcript write crashed
    // every save with EPERM after the rename had already succeeded.
    const directory = await mkdtemp(join(tmpdir(), 'xerxes-windows-transcript-'))
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: directory })
    const transcript = normalizeDaemonTranscript({
      session_id: 'feed1234',
      messages: [{ role: 'user', content: 'persist me' }],
    }, { requestedSessionKey: 'feed1234', currentProjectDirectory: directory })
    if (!transcript) throw new Error('expected transcript to normalize')
    await store.save(transcript)
    expect((await store.load('feed1234'))?.messages).toHaveLength(1)
  },
)
