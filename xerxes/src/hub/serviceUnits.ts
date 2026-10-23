// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, isAbsolute, join } from 'node:path'

import { firstLine, type CommandRunner } from './commandRunner.js'

/**
 * Per-user background services: a systemd user unit on Linux, a launchd agent
 * on macOS. Nothing needs root. Other platforms get an explicit error, never a
 * silent no-op.
 */
export interface UserServiceSpec {
  /** systemd unit name without `.service`. */
  readonly name: string
  /** launchd label (reverse DNS). */
  readonly label: string
  readonly description: string
  /** argv[0] must be absolute: service managers do not search PATH. */
  readonly argv: readonly string[]
  readonly environment: Readonly<Record<string, string>>
  readonly logPath: string
}

export interface UserServiceContext {
  readonly platform: NodeJS.Platform
  readonly home: string
  readonly run: CommandRunner
  readonly sleep: (milliseconds: number) => Promise<void>
  readonly uid?: number
  readonly user?: string
}

export class UserServiceError extends Error {
  constructor(message: string) { super(message); this.name = 'UserServiceError' }
}

function plain(value: string, what: string): void {
  if (/[\0\r\n]/.test(value)) throw new UserServiceError(`${what} must not contain control characters`)
}

export function validateServiceSpec(spec: UserServiceSpec): void {
  if (!/^[a-z0-9][a-z0-9-]{0,63}$/.test(spec.name)) throw new UserServiceError('service name must be lowercase letters, digits and dashes')
  if (!/^[a-zA-Z0-9][a-zA-Z0-9.-]{0,127}$/.test(spec.label)) throw new UserServiceError('service label must be reverse-DNS letters, digits, dots and dashes')
  if (spec.argv.length === 0 || !isAbsolute(spec.argv[0]!)) throw new UserServiceError('the service executable must be an absolute path')
  if (!isAbsolute(spec.logPath) || /\s/.test(spec.logPath)) throw new UserServiceError('the service log path must be absolute and contain no spaces')
  plain(spec.description, 'description')
  for (const arg of spec.argv) plain(arg, 'service argument')
  for (const [key, value] of Object.entries(spec.environment)) {
    if (!/^[A-Z_][A-Z0-9_]*$/.test(key)) throw new UserServiceError(`invalid environment variable name ${JSON.stringify(key)}`)
    plain(value, `environment variable ${key}`)
  }
}

/** systemd word quoting: backslash and quote escaped, `%` and `$` doubled so nothing is expanded. */
function systemdQuote(value: string): string {
  return `"${value.replace(/\\/g, () => '\\\\').replace(/"/g, () => '\\"').replace(/%/g, () => '%%').replace(/\$/g, () => '$$')}"`
}

export function renderSystemdUnit(spec: UserServiceSpec): string {
  validateServiceSpec(spec)
  const log = spec.logPath.replace(/%/g, () => '%%')
  return [
    '[Unit]',
    `Description=${spec.description.replace(/%/g, () => '%%')}`,
    'After=network-online.target',
    'Wants=network-online.target',
    '',
    '[Service]',
    'Type=simple',
    `ExecStart=${spec.argv.map(systemdQuote).join(' ')}`,
    ...Object.entries(spec.environment).map(([key, value]) => `Environment=${systemdQuote(`${key}=${value}`)}`),
    'Restart=always',
    'RestartSec=5',
    `StandardOutput=append:${log}`,
    `StandardError=append:${log}`,
    '',
    '[Install]',
    'WantedBy=default.target',
    '',
  ].join('\n')
}

function xml(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&apos;')
}

export function renderLaunchAgent(spec: UserServiceSpec): string {
  validateServiceSpec(spec)
  const environment = Object.entries(spec.environment)
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
    '<plist version="1.0">',
    '<dict>',
    `  <key>Label</key><string>${xml(spec.label)}</string>`,
    '  <key>ProgramArguments</key>',
    '  <array>',
    ...spec.argv.map(arg => `    <string>${xml(arg)}</string>`),
    '  </array>',
    ...(environment.length === 0 ? [] : [
      '  <key>EnvironmentVariables</key>',
      '  <dict>',
      ...environment.map(([key, value]) => `    <key>${xml(key)}</key><string>${xml(value)}</string>`),
      '  </dict>',
    ]),
    '  <key>RunAtLoad</key><true/>',
    '  <key>KeepAlive</key><true/>',
    '  <key>ThrottleInterval</key><integer>5</integer>',
    `  <key>StandardOutPath</key><string>${xml(spec.logPath)}</string>`,
    `  <key>StandardErrorPath</key><string>${xml(spec.logPath)}</string>`,
    '</dict>',
    '</plist>',
    '',
  ].join('\n')
}

export function userServicePath(spec: Pick<UserServiceSpec, 'name' | 'label'>, platform: NodeJS.Platform, home: string): string | undefined {
  if (platform === 'linux') return join(home, '.config', 'systemd', 'user', `${spec.name}.service`)
  if (platform === 'darwin') return join(home, 'Library', 'LaunchAgents', `${spec.label}.plist`)
  return undefined
}

async function must(run: CommandRunner, argv: readonly string[], what: string): Promise<void> {
  const result = await run(argv, { timeoutMs: 30_000 })
  if (result.code !== 0 || result.timedOut) throw new UserServiceError(`${what} failed: ${firstLine(result.stderr || result.stdout)}`)
}

/** Write the unit, then (re)start it. Re-running replaces the previous definition and restarts it. */
export async function installUserService(spec: UserServiceSpec, context: UserServiceContext): Promise<{ readonly path: string; readonly notes: readonly string[] }> {
  const path = userServicePath(spec, context.platform, context.home)
  if (path === undefined) {
    throw new UserServiceError(`Background services are supported on Linux (systemd) and macOS (launchd), not ${context.platform}. Run the command shown with --no-service yourself.`)
  }
  await mkdir(dirname(spec.logPath), { recursive: true, mode: 0o700 })
  await mkdir(dirname(path), { recursive: true })
  const notes: string[] = []
  if (context.platform === 'linux') {
    await writeFile(path, renderSystemdUnit(spec), { mode: 0o644 })
    await must(context.run, ['systemctl', '--user', 'daemon-reload'], 'systemctl --user daemon-reload')
    await must(context.run, ['systemctl', '--user', 'enable', `${spec.name}.service`], `enabling ${spec.name}`)
    await must(context.run, ['systemctl', '--user', 'restart', `${spec.name}.service`], `starting ${spec.name}`)
    const user = context.user
    if (user !== undefined) {
      const linger = await context.run(['loginctl', 'show-user', user, '--property=Linger'], { timeoutMs: 10_000 })
      if (!/Linger=yes/.test(linger.stdout)) {
        const enabled = await context.run(['loginctl', 'enable-linger', user], { timeoutMs: 10_000 })
        if (enabled.code !== 0) notes.push(`The service stops when ${user} logs out. To keep it running, run once: sudo loginctl enable-linger ${user}`)
      }
    }
  } else {
    await writeFile(path, renderLaunchAgent(spec), { mode: 0o644 })
    const domain = `gui/${context.uid ?? process.getuid?.() ?? 0}`
    await context.run(['launchctl', 'bootout', `${domain}/${spec.label}`], { timeoutMs: 15_000 })
    let started = await context.run(['launchctl', 'bootstrap', domain, path], { timeoutMs: 15_000 })
    // A bootout still in progress makes the next bootstrap fail once; retry after it settles.
    if (started.code !== 0) {
      await context.sleep(1500)
      started = await context.run(['launchctl', 'bootstrap', domain, path], { timeoutMs: 15_000 })
    }
    if (started.code !== 0) throw new UserServiceError(`launchctl bootstrap failed: ${firstLine(started.stderr || started.stdout)}`)
  }
  return { path, notes }
}
