// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync } from 'node:fs'

import { xerxesHome } from '../core/paths.js'
import { PROVIDERS } from '../llms/providerRegistry.js'

export type DiagnosisSeverity = 'fail' | 'ok' | 'warn'

export interface Diagnosis {
  readonly fixHint: string
  readonly message: string
  readonly name: string
  readonly severity: DiagnosisSeverity
}

export interface DoctorOptions {
  readonly bunVersion?: string
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly fileExists?: (path: string) => boolean
  readonly findExecutable?: (name: string) => string | null
  readonly home?: string
  readonly platform?: NodeJS.Platform
}

export type DoctorCheck = (options?: DoctorOptions) => Diagnosis

/** Verify the Bun runtime that owns the TypeScript implementation is available. */
export function checkBunRuntime(options: DoctorOptions = {}): Diagnosis {
  const version = options.bunVersion ?? Bun.version
  if (version.trim()) return diagnosis('bun', 'ok', 'Bun ' + version)
  return diagnosis('bun', 'fail', 'Bun runtime version is unavailable', 'Install a supported Bun release.')
}

/** Warn when the installed xerxes command cannot be discovered through PATH. */
export function checkXerxesOnPath(options: DoctorOptions = {}): Diagnosis {
  const found = (options.findExecutable ?? Bun.which)('xerxes')
  if (found) return diagnosis('xerxes-on-path', 'ok', 'xerxes binary at ' + found)
  return diagnosis(
    'xerxes-on-path',
    'warn',
    'xerxes is not on PATH',
    'Run through bun or add the Bun package bin directory to PATH.',
  )
}

/** Report whether at least one configured provider credential environment variable is populated. */
export function checkProviderKeys(options: DoctorOptions = {}): Diagnosis {
  const environment = options.environment ?? process.env
  const keys = [...new Set(Object.values(PROVIDERS)
    .flatMap(provider => provider.apiKeyEnv === undefined ? [] : [provider.apiKeyEnv]))]
  const present = keys.filter(key => Boolean(environment[key]))
  if (present.length) {
    return diagnosis('provider-keys', 'ok', 'providers configured via environment: ' + present.join(', '))
  }
  return diagnosis(
    'provider-keys',
    'warn',
    'No provider API key is set in the environment',
    'Set a provider key such as OPENAI_API_KEY or ANTHROPIC_API_KEY, or configure a profile.',
  )
}

/** Verify the Bun runtime home directory exists, without creating it as a side effect. */
export function checkXerxesHome(options: DoctorOptions = {}): Diagnosis {
  const environment = options.environment ?? process.env
  const home = options.home ?? xerxesHome(environment)
  if ((options.fileExists ?? existsSync)(home)) {
    return diagnosis('xerxes-home', 'ok', 'XERXES_HOME present at ' + home)
  }
  return diagnosis(
    'xerxes-home',
    'warn',
    'XERXES_HOME has not been created at ' + home,
    'It is created on first run; this is usually safe to ignore.',
  )
}

/** Identify hosts where the Unix-socket daemon needs an alternate transport. */
export function checkPlatform(options: DoctorOptions = {}): Diagnosis {
  const platform = options.platform ?? process.platform
  if (platform === 'win32') {
    return diagnosis(
      'platform',
      'warn',
      'Native Windows lacks the Unix-socket daemon transport used by default',
      'Use WSL2 or configure the WebSocket control transport.',
    )
  }
  return diagnosis('platform', 'ok', platform + ' host')
}

const MACOS_COMPUTER_USE_BINARIES = ['/usr/sbin/screencapture', '/usr/bin/sips', '/usr/bin/osascript'] as const

/** Report whether the zero-install macOS computer_use backend can run on this host. */
export function checkComputerUse(options: DoctorOptions = {}): Diagnosis {
  const platform = options.platform ?? process.platform
  if (platform !== 'darwin') {
    return diagnosis('computer-use', 'ok', 'computer_use macOS backend is not applicable on ' + platform)
  }
  const exists = options.fileExists ?? existsSync
  const missing = MACOS_COMPUTER_USE_BINARIES.filter(path => !exists(path))
  if (missing.length) {
    return diagnosis(
      'computer-use',
      'warn',
      'computer_use macOS backend is missing system tools: ' + missing.join(', '),
      'screencapture, sips, and osascript ship with macOS; reinstall the OS command line tools if they are absent.',
    )
  }
  return diagnosis(
    'computer-use',
    'ok',
    'computer_use macOS backend available by default; grant Screen Recording and Accessibility to the terminal app',
  )
}

export const DEFAULT_DOCTOR_CHECKS: readonly DoctorCheck[] = Object.freeze([
  checkBunRuntime,
  checkPlatform,
  checkXerxesOnPath,
  checkProviderKeys,
  checkXerxesHome,
  checkComputerUse,
])

export const MINIMAL_DOCTOR_CHECKS: readonly DoctorCheck[] = Object.freeze([
  checkBunRuntime,
  checkXerxesOnPath,
])

/** Run the Bun-native diagnostic collection in deterministic declaration order. */
export function runAllDoctorChecks(
  options: DoctorOptions = {},
  checks: readonly DoctorCheck[] = DEFAULT_DOCTOR_CHECKS,
): readonly Diagnosis[] {
  return checks.map(check => check(options))
}

/** Run the low-cost subset suitable for routine CLI startup or update probes. */
export function runMinimalDoctorChecks(options: DoctorOptions = {}): readonly Diagnosis[] {
  return runAllDoctorChecks(options, MINIMAL_DOCTOR_CHECKS)
}

export function hasDoctorFailures(report: readonly Diagnosis[]): boolean {
  return report.some(diagnosis => diagnosis.severity === 'fail')
}

/** Render diagnostic results for a human terminal without exposing credentials. */
export function formatDoctorReport(report: readonly Diagnosis[]): string {
  const icons: Readonly<Record<DiagnosisSeverity, string>> = {
    ok: '✓',
    warn: '!',
    fail: '✗',
  }
  return report.map(item => {
    const hint = item.fixHint && item.severity !== 'ok' ? '\n    → ' + item.fixHint : ''
    return icons[item.severity] + ' ' + item.name + ': ' + item.message + hint
  }).join('\n')
}

function diagnosis(
  name: string,
  severity: DiagnosisSeverity,
  message: string,
  fixHint = '',
): Diagnosis {
  return Object.freeze({ name, severity, message, fixHint })
}
