// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync } from 'node:fs'

import { xerxesHome } from '../core/paths.js'
import { CliWriter, createCliStyle, type CliWriterOptions } from './cliStyle.js'
import { PROVIDERS } from '../llms/providerRegistry.js'
import { formatConfigProvenance, getConfigProvenance } from '../core/config.js'
import { formatAgentDefinitionLoadErrors } from '../agents/definitions.js'

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

/**
 * Report the host platform and the control transport it will use.
 *
 * Windows is a supported host: the per-project control channel is a named pipe
 * there rather than a Unix socket, which `node:net` reaches through the same
 * API. This check used to warn that native Windows was unusable; it now names
 * the transport so a user debugging a connection knows what to look for.
 */
export function checkPlatform(options: DoctorOptions = {}): Diagnosis {
  const platform = options.platform ?? process.platform
  if (platform === 'win32') {
    return diagnosis('platform', 'ok', 'win32 host; daemon control channel uses a named pipe')
  }
  return diagnosis('platform', 'ok', platform + ' host; daemon control channel uses a Unix socket')
}

/** Windows console programs Xerxes shells out to for clipboard and process identity. */
const WINDOWS_REQUIRED_BINARIES = ['powershell.exe', 'cmd.exe'] as const

/**
 * Verify the Windows console tools Xerxes depends on are reachable.
 *
 * `powershell.exe` is not optional decoration: it is how the runtime reads a
 * process command line (there is no `ps`) and how the TUI reaches the clipboard.
 * A stripped or PATH-broken host fails those quietly, so name it up front.
 */
export function checkWindowsTooling(options: DoctorOptions = {}): Diagnosis {
  const platform = options.platform ?? process.platform
  if (platform !== 'win32') {
    return diagnosis('windows-tooling', 'ok', 'Windows console tooling is not applicable on ' + platform)
  }
  const find = options.findExecutable ?? Bun.which
  const missing = WINDOWS_REQUIRED_BINARIES.filter(name => !find(name))
  if (missing.length) {
    return diagnosis(
      'windows-tooling',
      'warn',
      'Windows console tools are not on PATH: ' + missing.join(', '),
      'Clipboard access and daemon process-identity checks need them; add %SystemRoot%\\system32 '
        + 'and its WindowsPowerShell\\v1.0 subdirectory to PATH.',
    )
  }
  return diagnosis('windows-tooling', 'ok', 'Windows console tooling available')
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

/**
 * Report which layer supplied every non-default setting.
 *
 * Config resolves across defaults, the user file, an opt-in workspace file, the
 * environment and CLI overrides, and once merged a value used to carry no memory
 * of where it came from — so "why is my model set to that" had no answer short of
 * reading five sources by hand. Values are redacted by the formatter.
 */
export function checkConfigProvenance(_options: DoctorOptions = {}): Diagnosis {
  let report: string
  try {
    report = formatConfigProvenance(getConfigProvenance(), { changedOnly: true })
  } catch (error) {
    return diagnosis(
      'config-provenance',
      'warn',
      'Config provenance is unavailable',
      error instanceof Error ? error.message : String(error),
    )
  }
  return report.trim()
    ? diagnosis('config-provenance', 'ok', 'Config sources resolved', report)
    : diagnosis('config-provenance', 'ok', 'Every setting is at its built-in default')
}

/**
 * Report agent-spec files that failed to load or carried fixable diagnostics.
 *
 * Strict spec rejection used to be invisible: a broken YAML file silently
 * vanished from the catalog and `--agent <name>` only reported "Unknown agent".
 * The loader now captures formatted per-file errors; this check surfaces them
 * where users already look when something feels off. Like config provenance,
 * it reads the live catalog rather than injected options, because the errors
 * live in the loader's own load of the ambient user/project directories.
 */
export function checkAgentDefinitions(_options: DoctorOptions = {}): Diagnosis {
  const formatted = formatAgentDefinitionLoadErrors()
  if (formatted === undefined) {
    return diagnosis('agent-specs', 'ok', 'Agent definitions loaded')
  }
  // Load errors render as '- <path>: <ErrorName>: <message>'; lift the path
  // out so the fix hint names the spec file to open instead of restating it.
  const firstFile = /^- (.+?): [A-Za-z$_][\w$]*: /.exec(formatted)?.[1]
  return diagnosis(
    'agent-specs',
    'warn',
    formatted,
    firstFile
      ? `Fix the agent spec at ${firstFile}; affected agents are skipped until it loads cleanly.`
      : 'Fix the named agent spec files; affected agents are skipped until they load cleanly.',
  )
}

export function checkSubsystemStorage(options: DoctorOptions = {}): Diagnosis {
  const environment = options.environment ?? process.env
  const home = options.home ?? xerxesHome(environment)
  const exists = options.fileExists ?? existsSync
  const subsystems = [
    { name: 'scheduler', path: `${home}/scheduler` },
    { name: 'memory', path: `${home}/governed-memory` },
    { name: 'capabilities', path: `${home}/capabilities` },
    { name: 'telemetry', path: `${home}/telemetry` },
  ]
  const present = subsystems.filter(({ path }) => exists(path)).map(({ name }) => name)
  if (present.length === subsystems.length) {
    return diagnosis('subsystem-storage', 'ok', 'All subsystem storage directories are present: ' + present.join(', '))
  }
  const missing = subsystems.filter(({ path }) => !exists(path)).map(({ name }) => name)
  return diagnosis(
    'subsystem-storage',
    'warn',
    'Subsystem storage directories are missing: ' + missing.join(', '),
    'Run the relevant xerxes commands (schedule, memory, capability, telemetry) to create them.',
  )
}

export const DEFAULT_DOCTOR_CHECKS: readonly DoctorCheck[] = Object.freeze([
  checkBunRuntime,
  checkPlatform,
  checkXerxesOnPath,
  checkProviderKeys,
  checkXerxesHome,
  checkComputerUse,
  checkWindowsTooling,
  checkConfigProvenance,
  checkAgentDefinitions,
  checkSubsystemStorage,
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

/**
 * Render diagnostic results for a human terminal without exposing credentials.
 *
 * Kept as a pure string builder so callers can capture or compose the report.
 * Colour arrives through the injected writer; with styling off the output is
 * byte-identical to what this function produced before the CLI gained a
 * presentation layer, which is what keeps it safe to pipe.
 */
export function formatDoctorReport(
  report: readonly Diagnosis[],
  writerOptions: CliWriterOptions = { style: createCliStyle('none') },
): string {
  const lines: string[] = []
  const writer = new CliWriter({ ...writerOptions, write: line => lines.push(line) })
  for (const item of report) {
    writer.status(
      item.severity,
      item.name,
      item.message,
      item.fixHint && item.severity !== 'ok' ? item.fixHint : '',
    )
  }
  return lines.join('\n')
}

/**
 * Print a full `xerxes doctor` run: a heading, the checks, then a verdict.
 *
 * The verdict line exists because a wall of nine ticks makes the one warning in
 * the middle easy to miss; the summary states the count so the reader does not
 * have to audit the list themselves.
 */
export function printDoctorReport(
  report: readonly Diagnosis[],
  writerOptions: CliWriterOptions = {},
): void {
  const writer = new CliWriter(writerOptions)
  writer.heading('Xerxes doctor')
  writer.line()
  for (const item of report) {
    writer.status(
      item.severity,
      item.name,
      item.message,
      item.fixHint && item.severity !== 'ok' ? item.fixHint : '',
    )
  }
  const failures = report.filter(item => item.severity === 'fail').length
  const warnings = report.filter(item => item.severity === 'warn').length
  writer.line()
  if (failures > 0) {
    writer.status('fail', '', `${failures} check(s) failed, ${warnings} warning(s)`)
  } else if (warnings > 0) {
    writer.status('warn', '', `${warnings} warning(s); nothing is broken`)
  } else {
    writer.status('ok', '', `all ${report.length} checks passed`)
  }
}

function diagnosis(
  name: string,
  severity: DiagnosisSeverity,
  message: string,
  fixHint = '',
): Diagnosis {
  return Object.freeze({ name, severity, message, fixHint })
}
