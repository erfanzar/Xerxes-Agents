// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  DAEMON_SERVICE_LABEL,
  DaemonServiceCommandError,
  daemonServicePaths,
  daemonServiceStatus,
  installDaemonService,
  renderLaunchdPlist,
  renderSystemdUnit,
  uninstallDaemonService,
  type BunDaemonServiceCommand,
  type DaemonServiceFilesystemPort,
  type DaemonServiceHost,
  type DaemonServiceProcessPort,
  type DaemonServiceProcessResult,
} from '../src/daemon/service.js'

const COMMAND: BunDaemonServiceCommand = {
  bunExecutable: 'bun',
  cliPath: '/opt/xerxes/dist/cli.js',
  daemonArgs: ['--socket', '/tmp/xerxes.sock'],
}

class MemoryFilesystem implements DaemonServiceFilesystemPort {
  readonly directories: string[] = []
  readonly files = new Map<string, string>()

  exists(path: string): boolean {
    return this.files.has(path) || this.directories.includes(path)
  }

  mkdir(path: string): void {
    this.directories.push(path)
  }

  readFile(path: string): string {
    const content = this.files.get(path)
    if (content === undefined) {
      throw new Error('missing file: ' + path)
    }
    return content
  }

  remove(path: string): void {
    this.files.delete(path)
  }

  writeFile(path: string, contents: string): void {
    this.files.set(path, contents)
  }
}

class RecordingProcess implements DaemonServiceProcessPort {
  readonly calls: string[][] = []
  alive = new Map<number, boolean>()
  handler: (argv: readonly string[]) => DaemonServiceProcessResult = () => processResult()

  isAlive(pid: number): boolean {
    return this.alive.get(pid) ?? false
  }

  run(argv: readonly string[]): DaemonServiceProcessResult {
    this.calls.push([...argv])
    return this.handler(argv)
  }
}

function processResult(stdout = '', exitCode = 0, stderr = ''): DaemonServiceProcessResult {
  return { exitCode, stderr, stdout }
}

function host(platform: string, filesystem = new MemoryFilesystem(), process = new RecordingProcess()): DaemonServiceHost {
  return {
    cwd: { cwd: () => '/workspace/xerxes' },
    filesystem,
    home: {
      home: () => '/home/agent',
      xerxesHome: () => '/home/agent/.xerxes',
    },
    platform: { current: () => platform },
    process,
  }
}

test('service renderers preserve launchd and systemd units with Bun CLI commands', () => {
  const paths = daemonServicePaths('/home/agent', '/home/agent/.xerxes')
  expect(paths).toEqual({
    defaultLogDirectory: '/home/agent/.xerxes/daemon/logs',
    defaultPidPath: '/home/agent/.xerxes/daemon/daemon.pid',
    launchdDirectory: '/home/agent/Library/LaunchAgents',
    launchdPlistPath: '/home/agent/Library/LaunchAgents/com.xerxes.daemon.plist',
    systemdDirectory: '/home/agent/.config/systemd/user',
    systemdUnitPath: '/home/agent/.config/systemd/user/xerxes-daemon.service',
  })

  const plist = renderLaunchdPlist(COMMAND, '/workspace/xerxes', '/var/log/xerxes')
  expect(plist).toContain(`<string>${DAEMON_SERVICE_LABEL}</string>`)
  expect(plist).toContain('<string>bun</string>')
  expect(plist).toContain('<string>/opt/xerxes/dist/cli.js</string>')
  expect(plist).toContain('<string>daemon</string>')
  expect(plist).toContain('<string>--socket</string>')
  expect(plist).toContain('/var/log/xerxes/daemon-stdout.log')
  expect(plist).not.toContain('python')

  const unit = renderSystemdUnit(COMMAND, '/workspace/xerxes')
  expect(unit).toContain('ExecStart="bun" "/opt/xerxes/dist/cli.js" "daemon" "--socket" "/tmp/xerxes.sock"')
  expect(unit).toContain('WorkingDirectory="/workspace/xerxes"')
  expect(unit).toContain('Restart=on-failure')
  expect(unit).not.toContain('python')
})

test('macOS service install, status, and uninstall use only injected filesystem and launchctl ports', async () => {
  const filesystem = new MemoryFilesystem()
  const process = new RecordingProcess()
  const serviceHost = host('darwin', filesystem, process)
  const paths = daemonServicePaths('/home/agent', '/home/agent/.xerxes')

  expect(await installDaemonService(serviceHost, { command: COMMAND })).toBe(
    `Installed: ${paths.launchdPlistPath}\nStarted via launchctl.`,
  )
  expect(filesystem.directories).toEqual([paths.defaultLogDirectory, paths.launchdDirectory])
  expect(filesystem.files.get(paths.launchdPlistPath)).toContain('<string>bun</string>')
  expect(process.calls).toEqual([['launchctl', 'load', paths.launchdPlistPath]])

  process.handler = argv => argv[0] === 'launchctl' && argv[1] === 'list'
    ? processResult('123\t0\tcom.xerxes.daemon\n')
    : processResult('', 1)
  expect(await daemonServiceStatus(serviceHost)).toBe('Running (launchd)\n123\t0\tcom.xerxes.daemon')

  process.handler = () => processResult('', 1, 'not loaded')
  expect(await daemonServiceStatus(serviceHost)).toBe('Not running (no launchd service)')
  expect(await uninstallDaemonService(serviceHost)).toBe(`Removed: ${paths.launchdPlistPath}`)
  expect(filesystem.files.has(paths.launchdPlistPath)).toBeFalse()
  expect(process.calls).toContainEqual(['launchctl', 'unload', paths.launchdPlistPath])
})

test('Linux service install requires systemctl success but uninstall intentionally tolerates manager failures', async () => {
  const filesystem = new MemoryFilesystem()
  const process = new RecordingProcess()
  const serviceHost = host('linux', filesystem, process)
  const paths = daemonServicePaths('/home/agent', '/home/agent/.xerxes')

  expect(await installDaemonService(serviceHost, { command: COMMAND, projectDirectory: '/project with space' })).toBe(
    `Installed: ${paths.systemdUnitPath}\nStarted via systemctl --user.`,
  )
  expect(filesystem.files.get(paths.systemdUnitPath)).toContain('WorkingDirectory="/project with space"')
  expect(process.calls).toEqual([
    ['systemctl', '--user', 'daemon-reload'],
    ['systemctl', '--user', 'enable', 'xerxes-daemon'],
    ['systemctl', '--user', 'start', 'xerxes-daemon'],
  ])

  process.handler = argv => argv.includes('is-active') ? processResult('active\n') : processResult('', 1)
  expect(await daemonServiceStatus(serviceHost)).toBe('Running (systemd)')
  process.handler = argv => argv.includes('is-active') ? processResult('inactive\n', 3) : processResult('', 1)
  expect(await daemonServiceStatus(serviceHost)).toBe('Not running (systemd: inactive)')

  expect(await uninstallDaemonService(serviceHost)).toBe(`Removed: ${paths.systemdUnitPath}`)
  expect(filesystem.files.has(paths.systemdUnitPath)).toBeFalse()
  expect(process.calls.slice(-3)).toEqual([
    ['systemctl', '--user', 'stop', 'xerxes-daemon'],
    ['systemctl', '--user', 'disable', 'xerxes-daemon'],
    ['systemctl', '--user', 'daemon-reload'],
  ])

  const failingFilesystem = new MemoryFilesystem()
  const failingProcess = new RecordingProcess()
  failingProcess.handler = argv => argv.includes('enable') ? processResult('', 1, 'permission denied') : processResult()
  await expect(installDaemonService(host('linux', failingFilesystem, failingProcess), { command: COMMAND }))
    .rejects.toBeInstanceOf(DaemonServiceCommandError)
  expect(failingFilesystem.files.has(paths.systemdUnitPath)).toBeTrue()
})

test('unsupported service hosts return a Bun command and status verifies only a real PID file through ports', async () => {
  const filesystem = new MemoryFilesystem()
  const process = new RecordingProcess()
  const serviceHost = host('win32', filesystem, process)
  const paths = daemonServicePaths('/home/agent', '/home/agent/.xerxes')

  expect(await installDaemonService(serviceHost, { command: COMMAND })).toBe(
    'Unsupported platform: win32. Run manually with `bun /opt/xerxes/dist/cli.js daemon --socket /tmp/xerxes.sock`.',
  )
  expect(filesystem.directories).toEqual([])
  expect(process.calls).toEqual([])
  expect(await daemonServiceStatus(serviceHost)).toBe('Not running')

  filesystem.writeFile(paths.defaultPidPath, '42\n')
  process.alive.set(42, true)
  expect(await daemonServiceStatus(serviceHost)).toBe('Running (PID: 42)')
  process.alive.set(42, false)
  expect(await daemonServiceStatus(serviceHost)).toBe('Stale PID file (PID 42 not running)')

  filesystem.writeFile(paths.defaultPidPath, 'not-a-pid')
  expect(await daemonServiceStatus(serviceHost)).toBe('Stale PID file (invalid PID)')
})
