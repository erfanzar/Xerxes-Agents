// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'

const BUILD_CONFIG_PATHS = new Set([
  '.github/workflows/ci.yml',
  '.github/workflows/ci.yaml',
  '.github/workflows/bun-ci.yml',
  '.github/workflows/bun-release.yml',
  'Dockerfile',
  'Makefile',
  'package.json',
  'bun.lock',
  'bun.lockb',
  'tsconfig.json',
  'xerxes/package.json',
  'xerxes/tsconfig.json',
  'xerxes/tsconfig.ui.json',
  'xerxes/vitest.ui.config.ts',
])

const CRITICAL_SOURCE_PREFIXES = [
  'xerxes/src/bridge/',
  'xerxes/src/daemon/',
  'xerxes/src/runtime/',
  'xerxes/src/security/',
  'xerxes/src/streaming/',
  'xerxes/src/tools/',
] as const

const VERIFICATION_MARKERS = [
  'git diff --check',
  'docker build',
  'bun test',
  'bun run check',
  'bun run build',
  'bunx tsc',
] as const

const RECENT_TOOL_EXECUTIONS = 50

export type ChangeSeverity = 'info' | 'warning' | 'error'

/** Parsed git status porcelain row. */
export class WorkspaceChange {
  readonly oldPath: string
  readonly path: string
  readonly status: string

  constructor(status: string, path: string, oldPath = '') {
    this.status = status
    this.path = path
    this.oldPath = oldPath
    Object.freeze(this)
  }

  get deleted(): boolean {
    return this.status.includes('D') && !this.untracked
  }

  get tracked(): boolean {
    return this.status !== '??'
  }

  get untracked(): boolean {
    return this.status === '??'
  }
}

export interface ChangeGuardFinding {
  readonly code: string
  readonly message: string
  readonly path: string
  readonly severity: ChangeSeverity
}

export interface ChangeGuardReportOptions {
  readonly findings?: readonly ChangeGuardFinding[]
  readonly statusAvailable?: boolean
  readonly verificationCommands?: readonly string[]
}

/** Immutable change-risk classification and recent verification evidence. */
export class ChangeGuardReport {
  readonly findings: readonly ChangeGuardFinding[]
  readonly statusAvailable: boolean
  readonly verificationCommands: readonly string[]

  constructor(options: ChangeGuardReportOptions = {}) {
    this.findings = Object.freeze((options.findings ?? []).map(finding => Object.freeze({ ...finding })))
    this.verificationCommands = Object.freeze([...(options.verificationCommands ?? [])])
    this.statusAvailable = options.statusAvailable ?? true
    Object.freeze(this)
  }

  get severity(): ChangeSeverity {
    if (this.findings.some(finding => finding.severity === 'error')) return 'error'
    if (this.findings.some(finding => finding.severity === 'warning')) return 'warning'
    return 'info'
  }

  get shouldNotify(): boolean {
    if (!this.findings.length) return false
    return this.severity === 'error' || this.verificationCommands.length === 0
  }

  /** Stable SHA-1 fingerprint suitable for duplicate-notification suppression. */
  get fingerprint(): string {
    const payload = JSON.stringify({
      findings: this.findings,
      verification_commands: this.verificationCommands,
      status_available: this.statusAvailable,
    })
    return createHash('sha1').update(payload, 'utf8').digest('hex')
  }
}

export interface CommandResult {
  readonly exitCode: number
  readonly stdout: string
}

export type CommandRunner = (
  args: readonly string[],
  options: { readonly cwd: string },
) => CommandResult | Promise<CommandResult>

export interface AnalyzeWorkspaceChangesOptions {
  readonly commandRunner?: CommandRunner
}

/**
 * Inspect git status using an injectable command runner.
 *
 * A missing Git executable or nonzero status produces an empty,
 * status-unavailable report rather than throwing into the turn loop.
 */
export async function analyzeWorkspaceChanges(
  cwd: string,
  toolExecutions: readonly Readonly<Record<string, unknown>>[] = [],
  options: AnalyzeWorkspaceChangesOptions = {},
): Promise<ChangeGuardReport> {
  const runner = options.commandRunner ?? defaultCommandRunner
  try {
    const result = await runner(['git', 'status', '--porcelain=v1', '--untracked-files=no'], { cwd })
    if (result.exitCode !== 0) return new ChangeGuardReport({ statusAvailable: false })
    return analyzeStatusLines(result.stdout.split(/\r?\n/), toolExecutions)
  } catch {
    return new ChangeGuardReport({ statusAvailable: false })
  }
}

/** Classify already-collected git porcelain lines without shelling out. */
export function analyzeStatusLines(
  lines: readonly string[],
  toolExecutions: readonly Readonly<Record<string, unknown>>[] = [],
): ChangeGuardReport {
  const changes = parsePorcelainStatus(lines)
  return new ChangeGuardReport({
    findings: findingsForChanges(changes),
    verificationCommands: recentVerificationCommands(toolExecutions),
  })
}

/** Parse the portable subset of git status porcelain consumed by the native guard. */
export function parsePorcelainStatus(lines: readonly string[]): WorkspaceChange[] {
  const changes: WorkspaceChange[] = []
  for (const raw of lines) {
    if (raw.length < 4) continue
    const status = raw.slice(0, 2)
    const pathText = raw.slice(3)
    const rename = pathText.indexOf(' -> ')
    const oldPath = rename < 0 ? '' : normalizeGitPath(pathText.slice(0, rename))
    const path = normalizeGitPath(rename < 0 ? pathText : pathText.slice(rename + 4))
    changes.push(new WorkspaceChange(status, path, oldPath))
  }
  return changes
}

/** Render a concise frontend notification for a non-empty risk report. */
export function formatChangeGuardNotification(report: ChangeGuardReport): string {
  if (!report.findings.length) return ''
  const lines = ['Risky workspace changes detected:']
  for (const finding of report.findings) {
    lines.push('- ' + finding.message + (finding.path ? ' [' + finding.path + ']' : ''))
  }
  if (report.verificationCommands.length) {
    lines.push('', 'Recent verification:')
    for (const command of report.verificationCommands.slice(0, 3)) lines.push('- ' + command)
  } else {
    lines.push('', 'No recent Bun test, Bun check/build, or git diff --check command was found in this session.')
  }
  return lines.join('\n')
}

function findingsForChanges(changes: readonly WorkspaceChange[]): ChangeGuardFinding[] {
  const findings: ChangeGuardFinding[] = []
  const deletedTests = sorted(changes.filter(change => change.deleted && isTestFile(change.path)).map(change => change.path))
  if (deletedTests.length) {
    findings.push(Object.freeze({
      severity: 'error',
      code: 'deleted-tests',
      path: samplePaths(deletedTests),
      message: deletedTests.length + ' tracked test file(s) were deleted',
    }))
  }

  const deletedSources = sorted(changes.filter(change => change.deleted && isSourceFile(change.path)).map(change => change.path))
  if (deletedSources.length) {
    findings.push(Object.freeze({
      severity: 'warning',
      code: 'deleted-source',
      path: samplePaths(deletedSources),
      message: deletedSources.length + ' source file(s) were deleted',
    }))
  }

  const buildConfigs = sorted(changes.filter(change => change.tracked && isBuildConfig(change.path)).map(change => change.path))
  if (buildConfigs.length) {
    findings.push(Object.freeze({
      severity: 'warning',
      code: 'build-config-changed',
      path: samplePaths(buildConfigs),
      message: 'build, install, lockfile, or CI configuration changed',
    }))
  }

  const critical = sorted(changes
    .filter(change => change.tracked && !change.deleted && isCriticalSource(change.path))
    .map(change => change.path))
  if (critical.length) {
    findings.push(Object.freeze({
      severity: 'warning',
      code: 'runtime-critical-changed',
      path: samplePaths(critical),
      message: 'runtime, daemon, bridge, security, streaming, or tool code changed',
    }))
  }
  return findings
}

function recentVerificationCommands(executions: readonly Readonly<Record<string, unknown>>[]): string[] {
  const commands: string[] = []
  for (const execution of executions.slice(-RECENT_TOOL_EXECUTIONS)) {
    const command = toolExecutionCommand(execution)
    if (!command) continue
    const normalized = command.split(/\s+/).filter(Boolean).join(' ')
    const lowered = normalized.toLowerCase()
    if (VERIFICATION_MARKERS.some(marker => lowered.includes(marker))) commands.push(truncate(normalized, 180))
  }
  return commands.slice(-5)
}

function toolExecutionCommand(execution: Readonly<Record<string, unknown>>): string {
  const name = typeof execution.name === 'string' ? execution.name.toLowerCase() : ''
  const inputs = objectValue(execution.inputs)
  if (!inputs) return ''
  for (const key of ['cmd', 'command', 'shell_command']) {
    const value = inputs[key]
    if (typeof value === 'string' && value.trim()) return value
  }
  if (name.includes('shell') || name.includes('exec') || name.includes('bash')) {
    for (const value of Object.values(inputs)) {
      if (typeof value === 'string' && value.trim()) return value
    }
  }
  return ''
}

async function defaultCommandRunner(
  args: readonly string[],
  options: { readonly cwd: string },
): Promise<CommandResult> {
  const process = Bun.spawn([...args], {
    cwd: options.cwd,
    stdin: 'ignore',
    stdout: 'pipe',
    stderr: 'pipe',
  })
  const stdout = await new Response(process.stdout).text()
  const exitCode = await process.exited
  return { exitCode, stdout }
}

function normalizeGitPath(value: string): string {
  return value.trim().replace(/^"|"$/g, '').replaceAll('\\', '/')
}

function isTestFile(path: string): boolean {
  return (path.startsWith('xerxes/test/') && /\.(test|spec)\.ts$/.test(path))
    || (path.startsWith('xerxes/src/ui/') && /\.(test|spec)\.tsx?$/.test(path))
}

function isSourceFile(path: string): boolean {
  return path.startsWith('xerxes/src/') && /\.tsx?$/.test(path)
}

function isBuildConfig(path: string): boolean {
  return BUILD_CONFIG_PATHS.has(path) || path.startsWith('.github/workflows/')
}

function isCriticalSource(path: string): boolean {
  return isSourceFile(path) && CRITICAL_SOURCE_PREFIXES.some(prefix => path.startsWith(prefix))
}

function samplePaths(paths: readonly string[], limit = 5): string {
  const sample = paths.slice(0, limit)
  return sample.join(', ') + (paths.length > limit ? ', +' + (paths.length - limit) + ' more' : '')
}

function truncate(value: string, limit: number): string {
  return value.length <= limit ? value : value.slice(0, Math.max(0, limit - 3)) + '...'
}

function sorted(values: readonly string[]): string[] {
  return [...values].sort((left, right) => left.localeCompare(right))
}

function objectValue(value: unknown): Readonly<Record<string, unknown>> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Readonly<Record<string, unknown>>
    : undefined
}
