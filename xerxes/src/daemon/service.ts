// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { join } from 'node:path'

export const DAEMON_SERVICE_LABEL = 'com.xerxes.daemon'
export const LAUNCHD_PLIST_NAME = `${DAEMON_SERVICE_LABEL}.plist`
export const SYSTEMD_UNIT_NAME = 'xerxes-daemon.service'
export const SYSTEMD_SERVICE_NAME = 'xerxes-daemon'

type MaybePromise<T> = T | Promise<T>

/** Explicit filesystem boundary. `mkdir` must create missing parent directories. */
export interface DaemonServiceFilesystemPort {
  exists(path: string): MaybePromise<boolean>
  mkdir(path: string): MaybePromise<void>
  readFile(path: string): MaybePromise<string>
  remove(path: string): MaybePromise<void>
  writeFile(path: string, contents: string): MaybePromise<void>
}

/** Completed invocation of a platform service manager. */
export interface DaemonServiceProcessResult {
  readonly exitCode: number
  readonly stderr: string
  readonly stdout: string
}

/** Explicit process boundary for launchctl/systemctl and PID liveness checks. */
export interface DaemonServiceProcessPort {
  isAlive(pid: number): MaybePromise<boolean>
  run(argv: readonly string[]): MaybePromise<DaemonServiceProcessResult>
}

/** Platform fact supplied by the embedding runtime instead of `process.platform`. */
export interface DaemonServicePlatformPort {
  current(): MaybePromise<string>
}

/** User-home paths supplied by the embedding runtime instead of environment discovery. */
export interface DaemonServiceHomePort {
  home(): MaybePromise<string>
  xerxesHome(): MaybePromise<string>
}

/** Working-directory fact supplied by the embedding runtime instead of `process.cwd()`. */
export interface DaemonServiceCwdPort {
  cwd(): MaybePromise<string>
}

/** All external state needed for service installation, removal, and inspection. */
export interface DaemonServiceHost {
  readonly cwd: DaemonServiceCwdPort
  readonly filesystem: DaemonServiceFilesystemPort
  readonly home: DaemonServiceHomePort
  readonly platform: DaemonServicePlatformPort
  readonly process: DaemonServiceProcessPort
}

/** The Bun command written into the platform unit file. */
export interface BunDaemonServiceCommand {
  /** Bun executable or absolute Bun path, for example `bun` or `/opt/homebrew/bin/bun`. */
  readonly bunExecutable: string
  /** Built Xerxes CLI file, for example `/opt/xerxes/dist/cli.js`. */
  readonly cliPath: string
  /** Explicit daemon arguments appended after the required `daemon` command. */
  readonly daemonArgs?: readonly string[]
}

export interface DaemonServiceInstallOptions {
  readonly command: BunDaemonServiceCommand
  /** Defaults to the caller's explicit cwd port. */
  readonly projectDirectory?: string
  /** Defaults to `<xerxesHome>/daemon/logs`. */
  readonly logDirectory?: string
}

export interface DaemonServiceStatusOptions {
  /** Defaults to `<xerxesHome>/daemon/daemon.pid` on unsupported service platforms. */
  readonly pidPath?: string
}

export interface DaemonServicePaths {
  readonly defaultLogDirectory: string
  readonly defaultPidPath: string
  readonly launchdDirectory: string
  readonly launchdPlistPath: string
  readonly systemdDirectory: string
  readonly systemdUnitPath: string
}

/** Raised when an install-time launchctl/systemctl command fails. */
export class DaemonServiceCommandError extends Error {
  readonly argv: readonly string[]
  readonly exitCode: number
  readonly stderr: string

  constructor(argv: readonly string[], result: DaemonServiceProcessResult) {
    const detail = result.stderr.trim()
    super(`Service command failed (${result.exitCode}): ${formatCommand(argv)}${detail ? `\n${detail}` : ''}`)
    this.name = 'DaemonServiceCommandError'
    this.argv = [...argv]
    this.exitCode = result.exitCode
    this.stderr = result.stderr
  }
}

/** Build all fixed service paths from an explicitly supplied user home. */
export function daemonServicePaths(home: string, xerxesHome: string): DaemonServicePaths {
  const normalizedHome = requiredText(home, 'home')
  const normalizedXerxesHome = requiredText(xerxesHome, 'xerxesHome')
  const launchdDirectory = join(normalizedHome, 'Library', 'LaunchAgents')
  const systemdDirectory = join(normalizedHome, '.config', 'systemd', 'user')
  return Object.freeze({
    defaultLogDirectory: join(normalizedXerxesHome, 'daemon', 'logs'),
    defaultPidPath: join(normalizedXerxesHome, 'daemon', 'daemon.pid'),
    launchdDirectory,
    launchdPlistPath: join(launchdDirectory, LAUNCHD_PLIST_NAME),
    systemdDirectory,
    systemdUnitPath: join(systemdDirectory, SYSTEMD_UNIT_NAME),
  })
}

/** Return the Bun command represented by a service unit. */
export function bunDaemonServiceArgv(command: BunDaemonServiceCommand): readonly string[] {
  const bunExecutable = requiredText(command.bunExecutable, 'bunExecutable')
  const cliPath = requiredText(command.cliPath, 'cliPath')
  const daemonArgs = (command.daemonArgs ?? []).map((argument, index) => requiredText(argument, `daemonArgs[${index}]`))
  return Object.freeze([bunExecutable, cliPath, 'daemon', ...daemonArgs])
}

/** Render the macOS launchd plist used by {@link installDaemonService}. */
export function renderLaunchdPlist(
  command: BunDaemonServiceCommand,
  projectDirectory: string,
  logDirectory: string,
): string {
  const argv = bunDaemonServiceArgv(command)
  const workingDirectory = requiredText(projectDirectory, 'projectDirectory')
  const logs = requiredText(logDirectory, 'logDirectory')
  const argumentsXml = argv.map(argument => `        <string>${xmlText(argument)}</string>`).join('\n')
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${DAEMON_SERVICE_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
${argumentsXml}
    </array>
    <key>WorkingDirectory</key>
    <string>${xmlText(workingDirectory)}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${xmlText(join(logs, 'daemon-stdout.log'))}</string>
    <key>StandardErrorPath</key>
    <string>${xmlText(join(logs, 'daemon-stderr.log'))}</string>
</dict>
</plist>
`
}

/** Render the Linux user-level systemd unit used by {@link installDaemonService}. */
export function renderSystemdUnit(command: BunDaemonServiceCommand, projectDirectory: string): string {
  const argv = bunDaemonServiceArgv(command)
  const workingDirectory = requiredText(projectDirectory, 'projectDirectory')
  return `[Unit]
Description=Xerxes Daemon — Background Agent
After=network.target

[Service]
Type=simple
ExecStart=${systemdWords(argv)}
WorkingDirectory=${systemdWord(workingDirectory)}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
`
}

/**
 * Install and start a launchd/systemd user service using only caller-owned ports.
 *
 * No PID files or synthetic status records are written: successful installation
 * writes the real platform unit and invokes its real service manager through
 * the supplied process port.
 */
export async function installDaemonService(
  host: DaemonServiceHost,
  options: DaemonServiceInstallOptions,
): Promise<string> {
  const platform = await normalizedPlatform(host)
  const paths = await hostPaths(host)
  const projectDirectory = options.projectDirectory === undefined
    ? await requiredHostText(host.cwd.cwd(), 'cwd')
    : requiredText(options.projectDirectory, 'projectDirectory')
  const logDirectory = options.logDirectory === undefined
    ? paths.defaultLogDirectory
    : requiredText(options.logDirectory, 'logDirectory')
  const argv = bunDaemonServiceArgv(options.command)

  if (platform === 'darwin') {
    await host.filesystem.mkdir(logDirectory)
    await host.filesystem.mkdir(paths.launchdDirectory)
    await host.filesystem.writeFile(paths.launchdPlistPath, renderLaunchdPlist(options.command, projectDirectory, logDirectory))
    await runRequired(host.process, ['launchctl', 'load', paths.launchdPlistPath])
    return `Installed: ${paths.launchdPlistPath}\nStarted via launchctl.`
  }
  if (platform === 'linux') {
    await host.filesystem.mkdir(logDirectory)
    await host.filesystem.mkdir(paths.systemdDirectory)
    await host.filesystem.writeFile(paths.systemdUnitPath, renderSystemdUnit(options.command, projectDirectory))
    await runRequired(host.process, ['systemctl', '--user', 'daemon-reload'])
    await runRequired(host.process, ['systemctl', '--user', 'enable', SYSTEMD_SERVICE_NAME])
    await runRequired(host.process, ['systemctl', '--user', 'start', SYSTEMD_SERVICE_NAME])
    return `Installed: ${paths.systemdUnitPath}\nStarted via systemctl --user.`
  }
  return `Unsupported platform: ${platform}. Run manually with \`${formatCommand(argv)}\`.`
}

/** Stop and remove the installed user service without manufacturing any replacement state. */
export async function uninstallDaemonService(host: DaemonServiceHost): Promise<string> {
  const platform = await normalizedPlatform(host)
  const paths = await hostPaths(host)
  if (platform === 'darwin') {
    if (!await host.filesystem.exists(paths.launchdPlistPath)) {
      return 'No launchd service found.'
    }
    await host.process.run(['launchctl', 'unload', paths.launchdPlistPath])
    await host.filesystem.remove(paths.launchdPlistPath)
    return `Removed: ${paths.launchdPlistPath}`
  }
  if (platform === 'linux') {
    if (!await host.filesystem.exists(paths.systemdUnitPath)) {
      return 'No systemd service found.'
    }
    await host.process.run(['systemctl', '--user', 'stop', SYSTEMD_SERVICE_NAME])
    await host.process.run(['systemctl', '--user', 'disable', SYSTEMD_SERVICE_NAME])
    await host.filesystem.remove(paths.systemdUnitPath)
    await host.process.run(['systemctl', '--user', 'daemon-reload'])
    return `Removed: ${paths.systemdUnitPath}`
  }
  return `Unsupported platform: ${platform}.`
}

/** Report actual platform service-manager state, or verify the real fallback PID through the process port. */
export async function daemonServiceStatus(
  host: DaemonServiceHost,
  options: DaemonServiceStatusOptions = {},
): Promise<string> {
  const platform = await normalizedPlatform(host)
  if (platform === 'darwin') {
    const result = await host.process.run(['launchctl', 'list', DAEMON_SERVICE_LABEL])
    return result.exitCode === 0
      ? `Running (launchd)\n${result.stdout.trim()}`
      : 'Not running (no launchd service)'
  }
  if (platform === 'linux') {
    const result = await host.process.run(['systemctl', '--user', 'is-active', SYSTEMD_SERVICE_NAME])
    const state = result.stdout.trim()
    return state === 'active' ? 'Running (systemd)' : `Not running (systemd: ${state})`
  }

  const paths = await hostPaths(host)
  const pidPath = options.pidPath === undefined ? paths.defaultPidPath : requiredText(options.pidPath, 'pidPath')
  if (!await host.filesystem.exists(pidPath)) {
    return 'Not running'
  }
  const pid = parsePid(await host.filesystem.readFile(pidPath))
  if (pid === undefined) {
    return 'Stale PID file (invalid PID)'
  }
  return await host.process.isAlive(pid)
    ? `Running (PID: ${pid})`
    : `Stale PID file (PID ${pid} not running)`
}

async function hostPaths(host: DaemonServiceHost): Promise<DaemonServicePaths> {
  const [home, xerxesHome] = await Promise.all([
    requiredHostText(host.home.home(), 'home'),
    requiredHostText(host.home.xerxesHome(), 'xerxesHome'),
  ])
  return daemonServicePaths(home, xerxesHome)
}

async function normalizedPlatform(host: DaemonServiceHost): Promise<string> {
  return (await requiredHostText(host.platform.current(), 'platform')).toLowerCase()
}

async function requiredHostText(value: MaybePromise<string>, field: string): Promise<string> {
  return requiredText(await value, field)
}

async function runRequired(process: DaemonServiceProcessPort, argv: readonly string[]): Promise<void> {
  const result = await process.run(argv)
  if (result.exitCode !== 0) {
    throw new DaemonServiceCommandError(argv, result)
  }
}

function formatCommand(argv: readonly string[]): string {
  return argv.map(shellWord).join(' ')
}

function parsePid(contents: string): number | undefined {
  const text = contents.trim()
  if (!/^[1-9]\d*$/.test(text)) {
    return undefined
  }
  const pid = Number(text)
  return Number.isSafeInteger(pid) ? pid : undefined
}

function requiredText(value: string, field: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new TypeError(`${field} must be a non-empty string`)
  }
  if (/[\u0000\r\n]/.test(value)) {
    throw new TypeError(`${field} must not contain a NUL or newline`)
  }
  return value.trim()
}

function shellWord(value: string): string {
  return /^[A-Za-z0-9_./:=+@%,-]+$/.test(value) ? value : `'${value.replaceAll("'", "'\\''")}'`
}

function systemdWord(value: string): string {
  const normalized = requiredText(value, 'systemd value')
  return `"${normalized.replaceAll('\\', '\\\\').replaceAll('"', '\\"')}"`
}

function systemdWords(argv: readonly string[]): string {
  return argv.map(systemdWord).join(' ')
}

function xmlText(value: string): string {
  return requiredText(value, 'plist value')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;')
}
