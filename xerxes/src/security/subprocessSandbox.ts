// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { stat } from 'node:fs/promises'

import { XerxesError } from '../core/errors.js'
import { WorkspacePathError, WorkspacePathResolver } from '../tools/pathSafety.js'
import type { JsonObject } from '../types/toolCalls.js'
import type { SandboxBackend, SandboxExecutionRequest } from './sandbox.js'

const DEFAULT_MAX_OUTPUT_CHARS = 20_000
const DEFAULT_MAX_TIMEOUT_MS = 30_000
const MAX_ARGUMENT_BYTES = 64 * 1024
const MAX_OUTPUT_CHARS = 1_000_000
const SAFE_PARENT_ENVIRONMENT_NAMES = ['PATH', 'HOME', 'LANG', 'LC_ALL', 'TERM'] as const
const BLOCKED_ENVIRONMENT_NAMES = new Set(['BUN_INSTALL', 'BUN_OPTIONS', 'NODE_OPTIONS', 'NODE_PATH', 'PYTHONPATH'])
const ENVIRONMENT_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/
const UNSAFE_EXECUTABLE = /[;&|`$<>\0]/

export interface SubprocessSandboxBackendOptions {
  /** Exact executable names or paths that sandboxed requests may invoke. */
  readonly allowedCommands: Iterable<string>
  /** Tool names accepted by this backend. Defaults to the direct-argv `exec_command` tool. */
  readonly allowedTools?: Iterable<string>
  /** Explicit environment values added to the small safe parent-environment allow-list. */
  readonly environment?: Readonly<Record<string, string>>
  /** Metadata only: Bun cannot impose a portable memory rlimit on arbitrary child commands. */
  readonly memoryLimitMb?: number
  /** Maximum response characters retained across each stdout/stderr stream. */
  readonly maxOutputChars?: number
  /** Hard wall-clock cap for every child process, in milliseconds. */
  readonly maxTimeoutMs?: number
  /** Metadata only: a subprocess backend cannot enforce network isolation by itself. */
  readonly networkAccessRequested?: boolean
  /** Workspace root that bounds every request working directory. Defaults to the current directory. */
  readonly workingDirectory?: string
}

export interface SubprocessSandboxResult {
  readonly command: readonly string[]
  readonly cwd: string
  readonly exitCode: number
  readonly stderr: string
  readonly stdout: string
  readonly timedOut: false
  readonly truncated: boolean
}

/** Base class for clear, safe-to-surface failures from the local subprocess backend. */
export class SubprocessSandboxError extends XerxesError {}

export class SubprocessSandboxConfigurationError extends SubprocessSandboxError {}

export class SubprocessSandboxRequestError extends SubprocessSandboxError {
  readonly toolName: string

  constructor(toolName: string, message: string, details: Record<string, unknown> = {}) {
    super(`Subprocess sandbox rejected tool '${toolName}': ${message}`, details)
    this.toolName = toolName
  }
}

export class SubprocessSandboxTimeoutError extends SubprocessSandboxError {
  readonly timeoutMs: number
  readonly toolName: string

  constructor(toolName: string, timeoutMs: number) {
    super(`Subprocess sandbox execution of tool '${toolName}' timed out after ${timeoutMs}ms`, { timeoutMs })
    this.timeoutMs = timeoutMs
    this.toolName = toolName
  }
}

export class SubprocessSandboxAbortedError extends SubprocessSandboxError {
  readonly toolName: string

  constructor(toolName: string) {
    super(`Subprocess sandbox execution of tool '${toolName}' was cancelled`)
    this.toolName = toolName
  }
}

export class SubprocessSandboxExecutionError extends SubprocessSandboxError {
  readonly toolName: string

  constructor(toolName: string, message: string, details: Record<string, unknown> = {}) {
    super(`Subprocess sandbox execution of tool '${toolName}' failed: ${message}`, details)
    this.toolName = toolName
  }
}

interface Invocation {
  readonly arguments_: readonly string[]
  readonly command: string
  readonly maxOutputChars: number
  readonly timeoutMs: number
  readonly workdir: string
}

/**
 * Executes an allow-listed direct-argv tool request in a fresh Bun child process.
 *
 * This is process isolation, not a container: it bounds the child CWD, environment,
 * wall-clock runtime, argument size, and captured output, but does not claim filesystem,
 * network, or portable memory isolation for arbitrary executables.
 */
export class SubprocessSandboxBackend implements SandboxBackend {
  private readonly allowedCommands: ReadonlySet<string>
  private readonly allowedTools: ReadonlySet<string>
  private readonly environment: Readonly<Record<string, string>>
  private readonly maxOutputChars: number
  private readonly maxTimeoutMs: number
  private readonly memoryLimitMb: number | undefined
  private readonly networkAccessRequested: boolean
  private readonly paths: WorkspacePathResolver

  constructor(options: SubprocessSandboxBackendOptions) {
    this.allowedCommands = normalizeExecutableSet(options.allowedCommands)
    this.allowedTools = normalizeToolSet(options.allowedTools ?? ['exec_command'])
    this.environment = sanitizeEnvironment(options.environment)
    this.maxOutputChars = normalizePositiveInteger(
      options.maxOutputChars ?? DEFAULT_MAX_OUTPUT_CHARS,
      'maxOutputChars',
      MAX_OUTPUT_CHARS,
    )
    this.maxTimeoutMs = normalizePositiveInteger(
      options.maxTimeoutMs ?? DEFAULT_MAX_TIMEOUT_MS,
      'maxTimeoutMs',
      Number.MAX_SAFE_INTEGER,
    )
    this.memoryLimitMb = normalizeOptionalPositiveInteger(options.memoryLimitMb, 'memoryLimitMb')
    this.networkAccessRequested = options.networkAccessRequested ?? false
    this.paths = new WorkspacePathResolver(options.workingDirectory ?? process.cwd())
  }

  async execute(request: SandboxExecutionRequest): Promise<string> {
    if (request.signal?.aborted) {
      throw new SubprocessSandboxAbortedError(request.toolName)
    }
    const invocation = this.parseRequest(request.toolName, request.arguments)
    const cwd = await this.resolveCwd(request.toolName, invocation.workdir)
    return this.run(request, invocation, cwd)
  }

  isAvailable(): boolean {
    return true
  }

  getCapabilities(): Readonly<Record<string, unknown>> {
    return {
      backend: 'subprocess',
      available: true,
      isolationLevel: 'process',
      filesystemIsolation: false,
      networkIsolation: false,
      memoryLimitEnforced: false,
      environmentSanitized: true,
      workingDirectoryBounded: true,
      timeoutEnforced: true,
      outputBounded: true,
      allowedCommands: [...this.allowedCommands],
      allowedTools: [...this.allowedTools],
      maxOutputChars: this.maxOutputChars,
      maxTimeoutMs: this.maxTimeoutMs,
      networkAccessRequested: this.networkAccessRequested,
      ...(this.memoryLimitMb === undefined ? {} : { memoryLimitMb: this.memoryLimitMb }),
    }
  }

  private parseRequest(toolName: string, arguments_: JsonObject): Invocation {
    if (!this.allowedTools.has(toolName)) {
      throw new SubprocessSandboxRequestError(toolName, 'is not in the subprocess tool allow-list')
    }
    const command = requiredExecutable(arguments_, toolName)
    if (!this.allowedCommands.has(command)) {
      throw new SubprocessSandboxRequestError(toolName, `command ${JSON.stringify(command)} is not allow-listed`)
    }
    const args = optionalStringArray(arguments_, toolName, 'args')
    const argumentBytes = new TextEncoder().encode([command, ...args].join('\0')).byteLength
    if (argumentBytes > MAX_ARGUMENT_BYTES) {
      throw new SubprocessSandboxRequestError(toolName, `arguments exceed the ${MAX_ARGUMENT_BYTES}-byte limit`)
    }
    const workdir = optionalWorkdir(arguments_, toolName)
    const requestedTimeout = optionalPositiveInteger(arguments_, toolName, 'timeout_ms')
    if (requestedTimeout !== undefined && requestedTimeout > this.maxTimeoutMs) {
      throw new SubprocessSandboxRequestError(
        toolName,
        `timeout_ms exceeds the sandbox limit of ${this.maxTimeoutMs}ms`,
      )
    }
    const requestedOutput = optionalPositiveInteger(arguments_, toolName, 'max_output_chars')
    if (requestedOutput !== undefined && requestedOutput > this.maxOutputChars) {
      throw new SubprocessSandboxRequestError(
        toolName,
        `max_output_chars exceeds the sandbox limit of ${this.maxOutputChars}`,
      )
    }
    return {
      command,
      arguments_: args,
      workdir,
      timeoutMs: requestedTimeout ?? this.maxTimeoutMs,
      maxOutputChars: requestedOutput ?? this.maxOutputChars,
    }
  }

  private async resolveCwd(toolName: string, workdir: string): Promise<string> {
    let cwd: string
    try {
      cwd = await this.paths.resolve(workdir)
    } catch (error) {
      if (error instanceof WorkspacePathError) {
        throw new SubprocessSandboxRequestError(toolName, error.message, { field: 'workdir' })
      }
      throw error
    }
    try {
      if (!(await stat(cwd)).isDirectory()) {
        throw new SubprocessSandboxRequestError(toolName, 'workdir must be an existing workspace directory')
      }
    } catch (error) {
      if (error instanceof SubprocessSandboxRequestError) {
        throw error
      }
      throw new SubprocessSandboxRequestError(toolName, `workdir is unavailable: ${errorMessage(error)}`)
    }
    return cwd
  }

  private async run(request: SandboxExecutionRequest, invocation: Invocation, cwd: string): Promise<string> {
    const controller = new AbortController()
    let timedOut = false
    const abort = () => controller.abort(request.signal?.reason)
    request.signal?.addEventListener('abort', abort, { once: true })
    const timer = setTimeout(() => {
      timedOut = true
      controller.abort(new Error(`Sandbox timeout after ${invocation.timeoutMs}ms`))
    }, invocation.timeoutMs)

    try {
      const child = Bun.spawn([invocation.command, ...invocation.arguments_], {
        cwd,
        env: { ...this.environment },
        stdin: 'ignore',
        stdout: 'pipe',
        stderr: 'pipe',
        signal: controller.signal,
        killSignal: 'SIGKILL',
        maxBuffer: invocation.maxOutputChars * 8,
      })
      const [exitCode, stdout, stderr] = await Promise.all([
        child.exited,
        new Response(child.stdout).text(),
        new Response(child.stderr).text(),
      ])
      if (timedOut) {
        throw new SubprocessSandboxTimeoutError(request.toolName, invocation.timeoutMs)
      }
      if (request.signal?.aborted) {
        throw new SubprocessSandboxAbortedError(request.toolName)
      }
      const stdoutResult = capOutput(stdout, invocation.maxOutputChars)
      const stderrResult = capOutput(stderr, invocation.maxOutputChars)
      const result: SubprocessSandboxResult = {
        command: [invocation.command, ...invocation.arguments_],
        cwd: await this.paths.relative(cwd),
        exitCode,
        stdout: stdoutResult.text,
        stderr: stderrResult.text,
        timedOut: false,
        truncated: stdoutResult.truncated || stderrResult.truncated,
      }
      return JSON.stringify(result)
    } catch (error) {
      if (timedOut) {
        throw new SubprocessSandboxTimeoutError(request.toolName, invocation.timeoutMs)
      }
      if (request.signal?.aborted) {
        throw new SubprocessSandboxAbortedError(request.toolName)
      }
      if (error instanceof SubprocessSandboxError) {
        throw error
      }
      throw new SubprocessSandboxExecutionError(request.toolName, errorMessage(error))
    } finally {
      clearTimeout(timer)
      request.signal?.removeEventListener('abort', abort)
    }
  }
}

function normalizeExecutableSet(values: Iterable<string>): ReadonlySet<string> {
  const commands = new Set<string>()
  for (const value of values) {
    if (typeof value !== 'string' || !value || value.trim() !== value || UNSAFE_EXECUTABLE.test(value)) {
      throw new SubprocessSandboxConfigurationError('allowedCommands must contain direct executable names or paths')
    }
    commands.add(value)
  }
  if (!commands.size) {
    throw new SubprocessSandboxConfigurationError('allowedCommands must not be empty')
  }
  return commands
}

function normalizeToolSet(values: Iterable<string>): ReadonlySet<string> {
  const tools = new Set<string>()
  for (const value of values) {
    if (typeof value !== 'string' || !value.trim()) {
      throw new SubprocessSandboxConfigurationError('allowedTools must contain non-empty tool names')
    }
    tools.add(value)
  }
  if (!tools.size) {
    throw new SubprocessSandboxConfigurationError('allowedTools must not be empty')
  }
  return tools
}

function normalizePositiveInteger(value: number, name: string, maximum: number): number {
  if (!Number.isSafeInteger(value) || value <= 0 || value > maximum) {
    throw new SubprocessSandboxConfigurationError(`${name} must be an integer between 1 and ${maximum}`)
  }
  return value
}

function normalizeOptionalPositiveInteger(value: number | undefined, name: string): number | undefined {
  if (value === undefined) {
    return undefined
  }
  return normalizePositiveInteger(value, name, Number.MAX_SAFE_INTEGER)
}

function sanitizeEnvironment(values: Readonly<Record<string, string>> | undefined): Readonly<Record<string, string>> {
  const environment: Record<string, string> = {}
  for (const name of SAFE_PARENT_ENVIRONMENT_NAMES) {
    const value = process.env[name]
    if (value !== undefined) {
      environment[name] = value
    }
  }
  if (environment.PATH === undefined) {
    environment.PATH = process.platform === 'win32' ? '' : '/usr/bin:/bin'
  }
  for (const [name, value] of Object.entries(values ?? {})) {
    if (!ENVIRONMENT_NAME.test(name) || blockedEnvironmentName(name)) {
      throw new SubprocessSandboxConfigurationError(`environment variable ${JSON.stringify(name)} is not allowed`)
    }
    if (typeof value !== 'string') {
      throw new SubprocessSandboxConfigurationError(`environment variable ${JSON.stringify(name)} must be a string`)
    }
    environment[name] = value
  }
  return Object.freeze(environment)
}

function blockedEnvironmentName(name: string): boolean {
  return name === 'PATH'
    || name.startsWith('DYLD_')
    || name.startsWith('LD_')
    || BLOCKED_ENVIRONMENT_NAMES.has(name)
}

function requiredExecutable(arguments_: JsonObject, toolName: string): string {
  const value = arguments_.cmd
  if (typeof value !== 'string' || !value || value.trim() !== value || UNSAFE_EXECUTABLE.test(value)) {
    throw new SubprocessSandboxRequestError(
      toolName,
      'cmd must contain one direct executable name or path without shell syntax',
    )
  }
  return value
}

function optionalStringArray(arguments_: JsonObject, toolName: string, name: string): string[] {
  const value = arguments_[name]
  if (value === undefined) {
    return []
  }
  if (!Array.isArray(value)) {
    throw new SubprocessSandboxRequestError(toolName, `${name} must be an array of strings without null bytes`)
  }
  const strings: string[] = []
  for (const item of value) {
    if (typeof item !== 'string' || item.includes('\0')) {
      throw new SubprocessSandboxRequestError(toolName, `${name} must be an array of strings without null bytes`)
    }
    strings.push(item)
  }
  return strings
}

function optionalWorkdir(arguments_: JsonObject, toolName: string): string {
  const value = arguments_.workdir
  if (value === undefined) {
    return '.'
  }
  if (typeof value !== 'string' || !value.trim() || value.includes('\0')) {
    throw new SubprocessSandboxRequestError(toolName, 'workdir must be a non-empty workspace-relative path')
  }
  return value
}

function optionalPositiveInteger(arguments_: JsonObject, toolName: string, name: string): number | undefined {
  const value = arguments_[name]
  if (value === undefined) {
    return undefined
  }
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value <= 0) {
    throw new SubprocessSandboxRequestError(toolName, `${name} must be a positive integer`)
  }
  return value
}

function capOutput(output: string, maximum: number): { readonly text: string; readonly truncated: boolean } {
  if (output.length <= maximum) {
    return { text: output, truncated: false }
  }
  const suffix = '\n…[truncated]…'
  if (maximum <= suffix.length) {
    return { text: suffix.slice(0, maximum), truncated: true }
  }
  return { text: `${output.slice(0, maximum - suffix.length)}${suffix}`, truncated: true }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
