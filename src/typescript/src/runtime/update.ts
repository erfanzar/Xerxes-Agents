// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { resolve } from 'node:path'

/** Environment variable containing a release-approved Bun package or source spec. */
export const BUN_PACKAGE_SPEC_ENV = 'XERXES_PACKAGE'
export const DEFAULT_GIT_TIMEOUT = 1_000
export const DEFAULT_BUN_UPDATE_TIMEOUT = 120_000
export const DEFAULT_PACKAGE_REGISTRY = 'https://registry.npmjs.org'

/** A completed child-process invocation supplied by the host or Bun runtime. */
export interface UpdateProcessResult {
  readonly exitCode: number
  readonly stderr: string
  readonly stdout: string
  readonly timedOut: boolean
}

/** Process boundary used for git inspection and an explicitly requested Bun update. */
export interface UpdateProcessOptions {
  readonly cwd: string
  readonly timeout: number
}

export type UpdateProcessRunner = (
  argv: readonly string[],
  options: UpdateProcessOptions,
) => Promise<UpdateProcessResult>

/** Fetch boundary for an explicitly requested registry-version lookup. */
export type UpdateFetch = (url: string, init?: RequestInit) => Promise<Response>

/** Local checkout state compared with the configured or discoverable upstream ref. */
export interface GitUpdateStatus {
  readonly aheadCount: number
  readonly behindCount: number
  readonly branch: string
  readonly error: string
  readonly headHash: string
  readonly isGit: boolean
  readonly upstream: string
  readonly upstreamHash: string
}

export interface GitUpdateStatusOptions {
  /** Explicitly refresh the remote-tracking ref before comparing it. This may use the network. */
  readonly fetch?: boolean
  readonly cwd?: string
  readonly runner?: UpdateProcessRunner
  readonly timeout?: number
}

/** A no-side-effect Bun global-package command plan. */
export interface BunUpdatePlan {
  readonly argv: readonly string[]
  readonly source: 'argument' | 'environment'
  readonly spec: string
}

export interface BunUpdatePlanOptions {
  readonly bunExecutable?: string
  readonly environment?: Readonly<Record<string, string | undefined>>
  /** An explicit npm package or Bun-supported source spec. */
  readonly packageSpec?: string
}

export interface BunUpdateExecutionOptions {
  readonly cwd?: string
  readonly runner?: UpdateProcessRunner
  readonly timeout?: number
}

/** Result of an update command that the caller explicitly chose to execute. */
export interface BunUpdateExecutionResult extends UpdateProcessResult {
  readonly command: readonly string[]
  readonly ok: boolean
}

export interface BunPackageUpdateCheck {
  readonly currentVersion?: string
  readonly error?: string
  readonly latestVersion?: string
  readonly packageName: string
  /** Omitted when no current version was supplied or versions cannot be compared safely. */
  readonly updateAvailable?: boolean
}

export interface BunPackageUpdateCheckOptions {
  /** Version known by the caller. This module never guesses a globally installed version. */
  readonly currentVersion?: string
  readonly fetch?: UpdateFetch
  readonly packageName: string
  readonly registryUrl?: string
}

export interface UpdateCommandOptions {
  readonly apply: boolean
  readonly check: boolean
  readonly currentVersion: string | undefined
  readonly cwd: string | undefined
  readonly dryRun: boolean
  readonly packageName: string | undefined
  readonly packageSpec: string | undefined
}

export interface UpdateCommandHost {
  readonly bunExecutable?: string
  readonly cwd?: string
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly fetch?: UpdateFetch
  readonly runner?: UpdateProcessRunner
  readonly timeout?: number
  readonly write?: (message: string) => void
}

export interface UpdateCommandResult {
  readonly applied: boolean
  readonly execution?: BunUpdateExecutionResult
  readonly git: GitUpdateStatus
  readonly packageCheck?: BunPackageUpdateCheck
  readonly plan?: BunUpdatePlan
}

/** Raised when an update request lacks an explicit safe target or uses unsupported flags. */
export class UpdateCommandError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'UpdateCommandError'
  }
}

/** Help for the intentionally opt-in Bun update command. */
export const UPDATE_HELP = `Inspect or explicitly run a Bun Xerxes update.

Usage:
  xerxes update
  xerxes update --check [--package <npm-package> --current-version <version>]
  xerxes update --dry-run [--spec <package-or-source-spec>]
  xerxes update --apply [--spec <package-or-source-spec>]

Without flags, this only inspects local git tracking state. --check may fetch
the git upstream and, only with --package, query that named npm registry entry.
--dry-run prints a Bun command but never runs it. --apply is required to run a
planned command. A package/source spec must be supplied with --spec or through
${BUN_PACKAGE_SPEC_ENV}; Xerxes does not assume a published package name.`

/** Run a command through Bun without a shell, applying a caller-provided timeout. */
export async function runUpdateProcess(
  argv: readonly string[],
  options: UpdateProcessOptions,
): Promise<UpdateProcessResult> {
  if (argv.length === 0) throw new UpdateCommandError('cannot run an empty update command')
  const timeout = validateTimeout(options.timeout)
  let child: ReturnType<typeof Bun.spawn>
  try {
    child = Bun.spawn([...argv], {
      cwd: options.cwd,
      stdin: 'ignore',
      stderr: 'pipe',
      stdout: 'pipe',
    })
  } catch (error) {
    return {
      exitCode: -1,
      stderr: errorMessage(error),
      stdout: '',
      timedOut: false,
    }
  }

  const stdout = child.stdout === undefined || typeof child.stdout === 'number'
    ? Promise.resolve('')
    : new Response(child.stdout).text()
  const stderr = child.stderr === undefined || typeof child.stderr === 'number'
    ? Promise.resolve('')
    : new Response(child.stderr).text()
  let timeoutHandle: ReturnType<typeof setTimeout> | undefined
  const completed = await Promise.race([
    child.exited.then(exitCode => ({ exitCode, timedOut: false })),
    new Promise<{ readonly exitCode: number; readonly timedOut: true }>(resolveTimeout => {
      timeoutHandle = setTimeout(() => {
        try {
          child.kill()
        } catch {
          // The process may have exited between the timeout and kill attempt.
        }
        resolveTimeout({ exitCode: -1, timedOut: true })
      }, timeout)
    }),
  ])
  if (timeoutHandle !== undefined) clearTimeout(timeoutHandle)
  if (completed.timedOut) await child.exited
  const [capturedStdout, capturedStderr] = await Promise.all([stdout, stderr])
  return {
    exitCode: completed.exitCode,
    stderr: completed.timedOut ? timeoutMessage(timeout, capturedStderr) : capturedStderr,
    stdout: capturedStdout,
    timedOut: completed.timedOut,
  }
}

/** Compare the current checkout with its upstream without fetching unless explicitly requested. */
export async function gitUpdateStatus(options: GitUpdateStatusOptions = {}): Promise<GitUpdateStatus> {
  const cwd = resolve(options.cwd ?? process.cwd())
  const timeout = validateTimeout(options.timeout ?? DEFAULT_GIT_TIMEOUT)
  const runner = options.runner ?? runUpdateProcess
  const runGit = (arguments_: readonly string[], commandTimeout = timeout): Promise<string> => {
    return gitOutput(arguments_, cwd, commandTimeout, runner)
  }

  try {
    if (await runGit(['rev-parse', '--is-inside-work-tree']) !== 'true') return emptyGitStatus(false)
  } catch {
    return emptyGitStatus(false)
  }

  let branch: string
  let headHash: string
  try {
    [branch, headHash] = await Promise.all([
      runGit(['rev-parse', '--abbrev-ref', 'HEAD']),
      runGit(['rev-parse', '--short=12', 'HEAD']),
    ])
  } catch (error) {
    return gitStatus({ isGit: true, error: errorMessage(error) })
  }

  let upstream: string
  try {
    upstream = await runGit(['rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}'])
  } catch {
    upstream = await fallbackUpstream(branch, runGit)
  }
  if (!upstream) return gitStatus({ isGit: true, branch, headHash, error: 'no upstream ref' })

  let fetchError = ''
  if (options.fetch ?? false) {
    const remote = upstream.split('/', 1)[0] || 'origin'
    try {
      await runGit(['fetch', '--quiet', '--no-tags', remote], Math.max(timeout, 10_000))
    } catch (error) {
      fetchError = 'fetch failed: ' + errorMessage(error)
    }
  }

  try {
    const [counts, upstreamHash] = await Promise.all([
      runGit(['rev-list', '--left-right', '--count', `HEAD...${upstream}`]),
      runGit(['rev-parse', '--short=12', upstream]),
    ])
    const [aheadCount, behindCount] = parseGitCounts(counts)
    return gitStatus({
      isGit: true,
      branch,
      headHash,
      upstream,
      upstreamHash,
      aheadCount,
      behindCount,
      error: fetchError,
    })
  } catch (error) {
    return gitStatus({
      isGit: true,
      branch,
      headHash,
      upstream,
      error: fetchError || errorMessage(error),
    })
  }
}

/** Render a compact, accurate description of a local git update comparison. */
export function formatGitUpdateStatus(status: GitUpdateStatus): string {
  if (!status.isGit) return 'not a git checkout'
  const head = status.headHash ? `HEAD ${status.headHash}` : 'HEAD unknown'
  const upstream = status.upstream || 'upstream'
  const upstreamHash = status.upstreamHash ? ` ${status.upstreamHash}` : ''
  const fetchSuffix = status.error.startsWith('fetch failed:') ? `; ${status.error}` : ''
  if (status.behindCount > 0 && status.aheadCount > 0) {
    const detail = `${upstream}${upstreamHash}; ${head}${fetchSuffix}`
    return `${status.behindCount} upstream commit(s) available; local ${status.aheadCount} ahead (${detail})`
  }
  if (status.behindCount > 0) {
    return `${status.behindCount} upstream commit(s) available (${upstream}${upstreamHash}; ${head}${fetchSuffix})`
  }
  if (status.aheadCount > 0) return `current upstream; local ${status.aheadCount} ahead (${head}${fetchSuffix})`
  if (status.error) return `unknown (${status.error}; ${head})`
  return `current (${head})`
}

/** Construct a Bun global-install command only from an explicit package or source spec. */
export function planBunUpdate(options: BunUpdatePlanOptions = {}): BunUpdatePlan {
  const environment = options.environment ?? process.env
  const explicit = options.packageSpec === undefined ? undefined : normalizePackageSpec(options.packageSpec)
  const configured = explicit === undefined
    ? environment[BUN_PACKAGE_SPEC_ENV] === undefined
      ? undefined
      : normalizePackageSpec(environment[BUN_PACKAGE_SPEC_ENV] ?? '')
    : undefined
  const spec = explicit ?? configured
  if (spec === undefined) {
    throw new UpdateCommandError(
      `No Bun package/source spec is configured. Pass --spec or set ${BUN_PACKAGE_SPEC_ENV}.`,
    )
  }
  const bunExecutable = options.bunExecutable === undefined
    ? process.execPath
    : normalizeExecutable(options.bunExecutable)
  return {
    argv: [bunExecutable, 'add', '--global', spec],
    source: explicit === undefined ? 'environment' : 'argument',
    spec,
  }
}

/** Execute a caller-supplied Bun update plan. Callers should require an explicit user action first. */
export async function executeBunUpdate(
  plan: BunUpdatePlan,
  options: BunUpdateExecutionOptions = {},
): Promise<BunUpdateExecutionResult> {
  const runner = options.runner ?? runUpdateProcess
  const result = await runner(plan.argv, {
    cwd: resolve(options.cwd ?? process.cwd()),
    timeout: validateTimeout(options.timeout ?? DEFAULT_BUN_UPDATE_TIMEOUT),
  })
  return { command: plan.argv, ok: result.exitCode === 0 && !result.timedOut, ...result }
}

/** Query a named npm registry entry only when the caller explicitly asks to check it. */
export async function checkBunPackageUpdate(
  options: BunPackageUpdateCheckOptions,
): Promise<BunPackageUpdateCheck> {
  const packageName = normalizeNpmPackageName(options.packageName)
  const currentVersion = options.currentVersion === undefined
    ? undefined
    : normalizeVersion(options.currentVersion, 'current version')
  const fetcher = options.fetch ?? ((url: string, init?: RequestInit) => globalThis.fetch(url, init))
  let url: string
  try {
    url = packageRegistryUrl(options.registryUrl ?? DEFAULT_PACKAGE_REGISTRY, packageName)
  } catch (error) {
    return packageCheck({
      packageName,
      ...(currentVersion === undefined ? {} : { currentVersion }),
      error: errorMessage(error),
    })
  }

  try {
    const response = await fetcher(url, { headers: { accept: 'application/json' } })
    if (!response.ok) {
      return packageCheck({
        packageName,
        ...(currentVersion === undefined ? {} : { currentVersion }),
        error: `registry returned HTTP ${response.status}`,
      })
    }
    const payload: unknown = await response.json()
    if (!isRecord(payload) || typeof payload.version !== 'string' || !payload.version.trim()) {
      return packageCheck({
        packageName,
        ...(currentVersion === undefined ? {} : { currentVersion }),
        error: 'registry response has no version string',
      })
    }
    const latestVersion = normalizeVersion(payload.version, 'registry version')
    if (currentVersion === undefined) return packageCheck({ packageName, latestVersion })
    const comparison = compareReleaseVersions(latestVersion, currentVersion)
    if (comparison === undefined) {
      return packageCheck({
        packageName,
        currentVersion,
        latestVersion,
        error: 'versions are not comparable semantic versions',
      })
    }
    return packageCheck({ packageName, currentVersion, latestVersion, updateAvailable: comparison > 0 })
  } catch (error) {
    return packageCheck({
      packageName,
      ...(currentVersion === undefined ? {} : { currentVersion }),
      error: errorMessage(error),
    })
  }
}

/** Compare two semantic-style versions. Returns undefined instead of guessing for unsupported formats. */
export function compareReleaseVersions(left: string, right: string): number | undefined {
  const leftVersion = parseReleaseVersion(left)
  const rightVersion = parseReleaseVersion(right)
  if (leftVersion === undefined || rightVersion === undefined) return undefined
  for (let index = 0; index < 3; index += 1) {
    const difference = (leftVersion.core[index] ?? 0) - (rightVersion.core[index] ?? 0)
    if (difference !== 0) return difference > 0 ? 1 : -1
  }
  return comparePrerelease(leftVersion.prerelease, rightVersion.prerelease)
}

/** Parse supported update flags. Python/uv-specific update flags are deliberately not supported. */
export function parseUpdateCommandOptions(args: readonly string[]): UpdateCommandOptions {
  let apply = false
  let check = false
  let currentVersion: string | undefined
  let cwd: string | undefined
  let dryRun = false
  let packageName: string | undefined
  let packageSpec: string | undefined

  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    if (argument === undefined) continue
    switch (argument) {
      case '--apply':
        apply = true
        break
      case '--check':
        check = true
        break
      case '--current-version':
        currentVersion = requiredOptionValue(args, ++index, argument)
        break
      case '--cwd':
        cwd = requiredOptionValue(args, ++index, argument)
        break
      case '--dry-run':
        dryRun = true
        break
      case '--package':
        packageName = requiredOptionValue(args, ++index, argument)
        break
      case '--spec':
        packageSpec = requiredOptionValue(args, ++index, argument)
        break
      case '--git':
      case '--force':
        throw new UpdateCommandError(`${argument} is not supported by the Bun update command.`)
      default:
        throw new UpdateCommandError('Unknown update option: ' + argument)
    }
  }
  if (apply && dryRun) throw new UpdateCommandError('--apply and --dry-run cannot be used together.')
  if (packageName !== undefined && !check) throw new UpdateCommandError('--package requires --check.')
  if (currentVersion !== undefined && packageName === undefined) {
    throw new UpdateCommandError('--current-version requires --package.')
  }
  return { apply, check, currentVersion, cwd, dryRun, packageName, packageSpec }
}

/** Run the CLI update surface: status by default, network checking and process execution only with explicit flags. */
export async function runUpdateCommand(
  args: readonly string[],
  host: UpdateCommandHost = {},
): Promise<UpdateCommandResult> {
  const options = parseUpdateCommandOptions(args)
  const write = host.write ?? console.log
  const runner = host.runner ?? runUpdateProcess
  const cwd = resolve(host.cwd ?? options.cwd ?? process.cwd())
  const gitTimeout = validateTimeout(host.timeout ?? DEFAULT_GIT_TIMEOUT)
  const updateTimeout = validateTimeout(host.timeout ?? DEFAULT_BUN_UPDATE_TIMEOUT)
  const git = await gitUpdateStatus({ cwd, fetch: options.check, runner, timeout: gitTimeout })
  write('Git: ' + formatGitUpdateStatus(git))

  let packageCheck: BunPackageUpdateCheck | undefined
  if (options.check && options.packageName !== undefined) {
    packageCheck = await checkBunPackageUpdate({
      packageName: options.packageName,
      ...(options.currentVersion === undefined ? {} : { currentVersion: options.currentVersion }),
      ...(host.fetch === undefined ? {} : { fetch: host.fetch }),
    })
    write(formatBunPackageUpdateCheck(packageCheck))
  } else if (options.check) {
    write('Package registry: skipped; pass --package <npm-package> to check a named registry entry.')
  } else {
    write('Package registry: not checked; use --check --package <npm-package> to opt in.')
  }

  let plan: BunUpdatePlan | undefined
  let execution: BunUpdateExecutionResult | undefined
  if (options.dryRun || options.apply) {
    plan = planBunUpdate({
      ...(host.bunExecutable === undefined ? {} : { bunExecutable: host.bunExecutable }),
      ...(host.environment === undefined ? {} : { environment: host.environment }),
      ...(options.packageSpec === undefined ? {} : { packageSpec: options.packageSpec }),
    })
  }
  if (options.dryRun && plan !== undefined) {
    write('Would run: ' + shellCommand(plan.argv))
  } else if (options.apply && plan !== undefined) {
    execution = await executeBunUpdate(plan, { cwd, runner, timeout: updateTimeout })
    if (!execution.ok) {
      throw new UpdateCommandError('Bun update command failed: ' + processFailure(execution))
    }
    if (execution.stdout.trim()) write(execution.stdout.trim())
    if (execution.stderr.trim()) write('Bun update stderr: ' + oneLine(execution.stderr))
    write('Bun update command exited successfully.')
  } else {
    write('No Bun update command was run. Use --dry-run to review a spec or --apply to execute one.')
  }

  return {
    applied: execution !== undefined,
    git,
    ...(packageCheck === undefined ? {} : { packageCheck }),
    ...(plan === undefined ? {} : { plan }),
    ...(execution === undefined ? {} : { execution }),
  }
}

/** Render an explicitly requested package-registry check without claiming an unconfigured package exists. */
export function formatBunPackageUpdateCheck(check: BunPackageUpdateCheck): string {
  const prefix = `Package registry check for ${check.packageName}:`
  if (check.latestVersion === undefined) {
    return `${prefix} unavailable (${oneLine(check.error ?? 'no version returned')})`
  }
  if (check.currentVersion === undefined) {
    return `${prefix} latest observed version is ${check.latestVersion}; no current version was supplied.`
  }
  if (check.updateAvailable === undefined) {
    const detail = oneLine(check.error ?? 'versions could not be compared safely')
    return `${prefix} observed ${check.latestVersion}; ${detail}.`
  }
  if (check.updateAvailable) return `${prefix} ${check.currentVersion} -> ${check.latestVersion} is available.`
  return `${prefix} current version ${check.currentVersion} matches or exceeds ${check.latestVersion}.`
}

async function gitOutput(
  arguments_: readonly string[],
  cwd: string,
  timeout: number,
  runner: UpdateProcessRunner,
): Promise<string> {
  const result = await runner(['git', ...arguments_], { cwd, timeout })
  if (result.exitCode !== 0 || result.timedOut) throw new UpdateCommandError(processFailure(result))
  return result.stdout.trim()
}

async function fallbackUpstream(
  branch: string,
  runGit: (arguments_: readonly string[]) => Promise<string>,
): Promise<string> {
  const candidates = [
    ...(branch && branch !== 'HEAD' ? [`origin/${branch}`] : []),
    'origin/main',
    'origin/master',
  ]
  for (const ref of [...new Set(candidates)]) {
    try {
      await runGit(['rev-parse', '--verify', ref])
      return ref
    } catch {
      // An absent remote-tracking ref is expected while probing local git state.
    }
  }
  return ''
}

function emptyGitStatus(isGit: boolean): GitUpdateStatus {
  return gitStatus({ isGit })
}

function gitStatus(values: Partial<GitUpdateStatus> & Pick<GitUpdateStatus, 'isGit'>): GitUpdateStatus {
  return {
    aheadCount: values.aheadCount ?? 0,
    behindCount: values.behindCount ?? 0,
    branch: values.branch ?? '',
    error: values.error ?? '',
    headHash: values.headHash ?? '',
    isGit: values.isGit,
    upstream: values.upstream ?? '',
    upstreamHash: values.upstreamHash ?? '',
  }
}

function parseGitCounts(raw: string): readonly [number, number] {
  const [ahead, behind, ...rest] = raw.trim().split(/\s+/)
  if (ahead === undefined || behind === undefined || rest.length > 0) {
    throw new UpdateCommandError(`unexpected git revision count: ${JSON.stringify(raw)}`)
  }
  return [parseNonnegativeInteger(ahead, 'ahead count'), parseNonnegativeInteger(behind, 'behind count')]
}

function packageRegistryUrl(registryUrl: string, packageName: string): string {
  let registry: URL
  try {
    registry = new URL(registryUrl)
  } catch {
    throw new UpdateCommandError('package registry URL is invalid')
  }
  if (registry.protocol !== 'https:' && registry.protocol !== 'http:') {
    throw new UpdateCommandError('package registry URL must use HTTP or HTTPS')
  }
  if (registry.search || registry.hash) {
    throw new UpdateCommandError('package registry URL cannot include a query or fragment')
  }
  const root = registry.toString().replace(/\/+$/, '')
  return `${root}/${encodeURIComponent(packageName)}/latest`
}

function packageCheck(values: BunPackageUpdateCheck): BunPackageUpdateCheck {
  return {
    packageName: values.packageName,
    ...(values.currentVersion === undefined ? {} : { currentVersion: values.currentVersion }),
    ...(values.error === undefined ? {} : { error: values.error }),
    ...(values.latestVersion === undefined ? {} : { latestVersion: values.latestVersion }),
    ...(values.updateAvailable === undefined ? {} : { updateAvailable: values.updateAvailable }),
  }
}

interface ParsedReleaseVersion {
  readonly core: readonly number[]
  readonly prerelease: readonly string[] | undefined
}

function parseReleaseVersion(value: string): ParsedReleaseVersion | undefined {
  const pattern = new RegExp([
    '^v?',
    '((?:0|[1-9]\\d*)(?:\\.(?:0|[1-9]\\d*)){0,2})',
    '(?:-([0-9A-Za-z-]+(?:\\.[0-9A-Za-z-]+)*))?',
    '(?:\\+[0-9A-Za-z-]+(?:\\.[0-9A-Za-z-]+)*)?$',
  ].join(''))
  const match = pattern.exec(value.trim())
  if (match === null) return undefined
  const coreText = match[1]
  if (coreText === undefined) return undefined
  const core = coreText.split('.').map(Number)
  if (!core.every(Number.isSafeInteger)) return undefined
  const prereleaseText = match[2]
  return {
    core,
    prerelease: prereleaseText === undefined ? undefined : prereleaseText.split('.'),
  }
}

function comparePrerelease(
  left: readonly string[] | undefined,
  right: readonly string[] | undefined,
): number {
  if (left === undefined && right === undefined) return 0
  if (left === undefined) return 1
  if (right === undefined) return -1
  const maximum = Math.max(left.length, right.length)
  for (let index = 0; index < maximum; index += 1) {
    const leftPart = left[index]
    const rightPart = right[index]
    if (leftPart === undefined) return -1
    if (rightPart === undefined) return 1
    if (leftPart === rightPart) continue
    const leftNumeric = /^\d+$/.test(leftPart)
    const rightNumeric = /^\d+$/.test(rightPart)
    if (leftNumeric && rightNumeric) {
      const difference = compareNumericIdentifiers(leftPart, rightPart)
      if (difference !== 0) return difference
      continue
    }
    if (leftNumeric) return -1
    if (rightNumeric) return 1
    return leftPart > rightPart ? 1 : -1
  }
  return 0
}

function compareNumericIdentifiers(left: string, right: string): number {
  const normalizedLeft = left.replace(/^0+/, '') || '0'
  const normalizedRight = right.replace(/^0+/, '') || '0'
  if (normalizedLeft.length !== normalizedRight.length) {
    return normalizedLeft.length > normalizedRight.length ? 1 : -1
  }
  if (normalizedLeft === normalizedRight) return 0
  return normalizedLeft > normalizedRight ? 1 : -1
}

function normalizePackageSpec(value: string): string {
  if (typeof value !== 'string') throw new UpdateCommandError('Bun package/source spec must be a string')
  const spec = value.trim()
  if (!spec || spec.startsWith('-') || /[\0\r\n]/.test(spec)) {
    throw new UpdateCommandError('Bun package/source spec must be non-empty and cannot start with a flag')
  }
  return spec
}

function normalizeNpmPackageName(value: string): string {
  if (typeof value !== 'string' || !/^(?:@[a-z0-9][a-z0-9._-]*\/)?[a-z0-9][a-z0-9._-]*$/.test(value)) {
    throw new UpdateCommandError('npm package name is invalid')
  }
  return value
}

function normalizeVersion(value: string, label: string): string {
  if (typeof value !== 'string') throw new UpdateCommandError(`${label} must be a string`)
  const version = value.trim()
  if (!version || /[\0\r\n]/.test(version)) throw new UpdateCommandError(`${label} is invalid`)
  return version
}

function normalizeExecutable(value: string): string {
  if (typeof value !== 'string' || !value.trim() || /[\0\r\n]/.test(value)) {
    throw new UpdateCommandError('Bun executable path is invalid')
  }
  return value.trim()
}

function validateTimeout(value: number): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new UpdateCommandError('update timeout must be a positive safe integer')
  }
  return value
}

function requiredOptionValue(args: readonly string[], index: number, flag: string): string {
  const value = args[index]?.trim()
  if (!value || value.startsWith('-')) throw new UpdateCommandError(`${flag} requires a value.`)
  return value
}

function parseNonnegativeInteger(value: string, label: string): number {
  if (!/^\d+$/.test(value)) throw new UpdateCommandError(`${label} is not a non-negative integer`)
  const parsed = Number(value)
  if (!Number.isSafeInteger(parsed)) throw new UpdateCommandError(`${label} is not a safe integer`)
  return parsed
}

function processFailure(result: Pick<UpdateProcessResult, 'exitCode' | 'stderr' | 'timedOut'>): string {
  if (result.timedOut) return oneLine(result.stderr) || 'process timed out'
  const detail = oneLine(result.stderr)
  return detail || `process exited with code ${result.exitCode}`
}

function timeoutMessage(timeout: number, stderr: string): string {
  const suffix = stderr.trim() ? `: ${stderr}` : ''
  return `process timed out after ${timeout}ms${suffix}`
}

function shellCommand(argv: readonly string[]): string {
  return argv.map(shellQuote).join(' ')
}

function shellQuote(value: string): string {
  return /^[A-Za-z0-9_@./:=+,-]+$/.test(value) ? value : JSON.stringify(value)
}

function oneLine(value: string): string {
  return value.trim().replace(/\s+/g, ' ').slice(0, 500)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
