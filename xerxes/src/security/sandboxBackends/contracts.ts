// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfigurationError, ValidationError, XerxesError } from '../../core/errors.js'
import type { JsonObject, JsonValue } from '../../types/toolCalls.js'
import type { SandboxBackend, SandboxExecutionRequest } from '../sandbox.js'

const DEFAULT_MAX_OUTPUT_CHARS = 20_000
const DEFAULT_TIMEOUT_MS = 60_000
const ENVIRONMENT_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/
const UNSAFE_EXECUTABLE = /[\s;&|`$<>\0]/

/** Backends ported from the Python sandbox-backend family. */
export const SANDBOX_BACKEND_NAMES = Object.freeze([
  'docker',
  'daytona',
  'modal',
  'singularity',
  'ssh',
  'subprocess',
] as const)

export type SandboxBackendName = (typeof SANDBOX_BACKEND_NAMES)[number]

/** A direct-argv command delivered to a caller-owned sandbox host. */
export interface SandboxCommand {
  readonly argv: readonly string[]
  readonly cwd?: string
  /** Only caller-configured variables are present; process.env is never read here. */
  readonly environment: Readonly<Record<string, string>>
  readonly maxOutputChars: number
  readonly timeoutMs: number
}

/** Cancellation metadata accompanies, but is not serialized into, a sandbox command. */
export interface SandboxHostExecutionRequest {
  readonly command: SandboxCommand
  readonly signal?: AbortSignal
  readonly toolName: string
}

/** Confirmed result returned by an injected sandbox host. */
export interface SandboxCommandResult {
  readonly exitCode: number
  readonly metadata?: Readonly<Record<string, JsonValue>>
  readonly resourceId?: string
  readonly stderr: string
  readonly stdout: string
  readonly timedOut: boolean
  readonly truncated: boolean
}

/** Common command policy shared by all host-backed sandbox adapters. */
export interface SandboxCommandPolicyOptions {
  /** Tool names this adapter accepts. Defaults to the direct-argv `exec_command` tool. */
  readonly allowedTools?: Iterable<string>
  /** Explicit variables passed to the host. Ambient process variables are never inherited. */
  readonly environment?: Readonly<Record<string, string>>
  /** Maximum output each host may return. */
  readonly maxOutputChars?: number
  /** Upper bound for a tool-request timeout. */
  readonly maxTimeoutMs?: number
  /** Timeout used when a tool request does not declare `timeout_ms`. */
  readonly timeoutMs?: number
  /** CWD used when a request omits `workdir`; containment is owned by the injected host. */
  readonly workingDirectory?: string
}

/** Fully validated policy retained by an adapter. */
export interface SandboxCommandPolicy {
  readonly allowedTools: ReadonlySet<string>
  readonly environment: Readonly<Record<string, string>>
  readonly maxOutputChars: number
  readonly maxTimeoutMs: number
  readonly timeoutMs: number
  readonly workingDirectory: string | undefined
}

/** A named native adapter usable anywhere the existing sandbox router accepts a backend. */
export interface SandboxBackendAdapter extends SandboxBackend {
  readonly name: SandboxBackendName
}

/** Optional reachability probe a host can expose without forcing an SDK choice here. */
export interface SandboxHostProbe {
  isAvailable?(): boolean | Promise<boolean>
}

/** Configuration failures at the adapter/host boundary. */
export class SandboxBackendAdapterConfigurationError extends ConfigurationError {
  readonly backend: string

  constructor(backend: string, message: string, details: Record<string, unknown> = {}) {
    super(`sandboxBackends.${backend}`, message, details)
    this.backend = backend
  }
}

/** Invalid serializable tool input before it reaches a host adapter. */
export class SandboxBackendAdapterRequestError extends ValidationError {
  readonly backend: string
  readonly toolName: string

  constructor(backend: string, toolName: string, message: string, value: unknown = undefined) {
    super('sandboxRequest', `${backend}: ${message}`, value, { backend, toolName })
    this.backend = backend
    this.toolName = toolName
  }
}

/** A configured adapter could not find its injected external host or runtime. */
export class SandboxBackendUnavailableError extends XerxesError {
  readonly backend: string

  constructor(backend: string, message: string) {
    super(`Sandbox backend ${backend}: ${message}`)
    this.backend = backend
  }
}

/** An injected host returned data that cannot safely become a tool result. */
export class SandboxBackendProtocolError extends XerxesError {
  readonly backend: string

  constructor(backend: string, message: string) {
    super(`Sandbox backend ${backend}: ${message}`)
    this.backend = backend
  }
}

/** Validate a reusable command policy without touching the host environment or filesystem. */
export function normalizeSandboxCommandPolicy(
  backend: string,
  options: SandboxCommandPolicyOptions | undefined,
): SandboxCommandPolicy {
  const candidate: unknown = options
  if (candidate !== undefined && (candidate === null || typeof candidate !== 'object' || Array.isArray(candidate))) {
    throw new SandboxBackendAdapterConfigurationError(backend, 'command policy must be an object')
  }
  const source = options ?? {}
  const allowedTools = normalizeToolNames(backend, source.allowedTools ?? ['exec_command'])
  const maxTimeoutMs = positiveSafeInteger(backend, source.maxTimeoutMs ?? DEFAULT_TIMEOUT_MS, 'maxTimeoutMs')
  const timeoutMs = positiveSafeInteger(backend, source.timeoutMs ?? maxTimeoutMs, 'timeoutMs')
  if (timeoutMs > maxTimeoutMs) {
    throw new SandboxBackendAdapterConfigurationError(
      backend,
      'timeoutMs must not exceed maxTimeoutMs',
      { maxTimeoutMs, timeoutMs },
    )
  }
  return Object.freeze({
    allowedTools,
    environment: normalizeEnvironment(backend, source.environment),
    maxOutputChars: positiveSafeInteger(backend, source.maxOutputChars ?? DEFAULT_MAX_OUTPUT_CHARS, 'maxOutputChars'),
    maxTimeoutMs,
    timeoutMs,
    workingDirectory: optionalPath(backend, source.workingDirectory, 'workingDirectory'),
  })
}

/** Convert the existing router request into a bounded direct-argv host request. */
export function prepareSandboxCommand(
  backend: string,
  request: SandboxExecutionRequest,
  policy: SandboxCommandPolicy,
): SandboxHostExecutionRequest {
  if (request.signal?.aborted) {
    throw new SandboxBackendAdapterRequestError(backend, request.toolName, 'request was cancelled before host execution')
  }
  if (!policy.allowedTools.has(request.toolName)) {
    throw new SandboxBackendAdapterRequestError(backend, request.toolName, 'tool is not in the adapter allow-list')
  }
  const argv = commandArgv(backend, request.toolName, request.arguments)
  const requestedTimeout = optionalPositiveInteger(backend, request.toolName, request.arguments, 'timeout_ms')
  if (requestedTimeout !== undefined && requestedTimeout > policy.maxTimeoutMs) {
    throw new SandboxBackendAdapterRequestError(
      backend,
      request.toolName,
      `timeout_ms exceeds the adapter limit of ${policy.maxTimeoutMs}ms`,
      requestedTimeout,
    )
  }
  const requestedOutput = optionalPositiveInteger(backend, request.toolName, request.arguments, 'max_output_chars')
  if (requestedOutput !== undefined && requestedOutput > policy.maxOutputChars) {
    throw new SandboxBackendAdapterRequestError(
      backend,
      request.toolName,
      `max_output_chars exceeds the adapter limit of ${policy.maxOutputChars}`,
      requestedOutput,
    )
  }
  const requestedCwd = optionalPath(backend, request.arguments.workdir, 'workdir', request.toolName)
  const cwd = requestedCwd ?? policy.workingDirectory
  const command: SandboxCommand = Object.freeze({
    argv: Object.freeze(argv),
    ...(cwd === undefined ? {} : { cwd }),
    environment: policy.environment,
    maxOutputChars: requestedOutput ?? policy.maxOutputChars,
    timeoutMs: requestedTimeout ?? policy.timeoutMs,
  })
  return Object.freeze({
    toolName: request.toolName,
    command,
    ...(request.signal === undefined ? {} : { signal: request.signal }),
  })
}

/** Serialize a host-confirmed result into the existing sandbox router's string response surface. */
export function serializeSandboxCommandResult(
  backend: SandboxBackendName,
  command: SandboxCommand,
  result: SandboxCommandResult,
): string {
  validateResult(backend, result)
  return JSON.stringify({
    backend,
    command: command.argv,
    ...(command.cwd === undefined ? {} : { cwd: command.cwd }),
    exitCode: result.exitCode,
    stdout: result.stdout,
    stderr: result.stderr,
    timedOut: result.timedOut,
    truncated: result.truncated,
    ...(result.resourceId === undefined ? {} : { resourceId: result.resourceId }),
    ...(result.metadata === undefined ? {} : { metadata: result.metadata }),
  })
}

/** Return false when no external host was supplied; otherwise delegate an optional host probe. */
export async function hostAdapterIsAvailable(host: SandboxHostProbe | undefined): Promise<boolean> {
  const candidate: unknown = host
  if (candidate === undefined || candidate === null || typeof candidate !== 'object') return false
  const probe = (candidate as { readonly isAvailable?: unknown }).isAvailable
  if (probe === undefined) return true
  if (typeof probe !== 'function') {
    throw new SandboxBackendAdapterConfigurationError('host', 'isAvailable must be a function when present')
  }
  return await (probe as () => boolean | Promise<boolean>)()
}

/** Fail with a clear configuration error instead of attempting an implicit SDK or shell fallback. */
export function requiredHost<T>(backend: string, host: T | undefined, operation: string): T {
  const candidate: unknown = host
  if (candidate === undefined || candidate === null || typeof candidate !== 'object') {
    throw new SandboxBackendAdapterConfigurationError(
      backend,
      `a host adapter must be injected before ${operation}`,
    )
  }
  return candidate as T
}

/** Require a concrete host operation before making a caller-owned external call. */
export function requiredHostMethod<T>(backend: string, host: T | undefined, operation: string, method: string): T {
  const resolved = requiredHost(backend, host, operation)
  const candidate = resolved as unknown as Record<string, unknown>
  if (typeof candidate[method] !== 'function') {
    throw new SandboxBackendAdapterConfigurationError(
      backend,
      `injected host must expose ${method}() before ${operation}`,
    )
  }
  return resolved
}

/** Validate an explicit non-empty image, endpoint, host, or other free-text configuration value. */
export function requiredText(backend: string, value: string, field: string): string {
  if (typeof value !== 'string' || !value.trim() || value.includes('\0')) {
    throw new SandboxBackendAdapterConfigurationError(backend, `${field} must be a non-empty string without NUL bytes`)
  }
  return value.trim()
}

/** Validate a positive integer configuration value. */
export function positiveSafeInteger(backend: string, value: number, field: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new SandboxBackendAdapterConfigurationError(backend, `${field} must be a positive safe integer`, { value })
  }
  return value
}

function commandArgv(backend: string, toolName: string, arguments_: JsonObject): string[] {
  const command = arguments_.cmd
  if (typeof command !== 'string' || !command || command.trim() !== command || UNSAFE_EXECUTABLE.test(command)) {
    throw new SandboxBackendAdapterRequestError(
      backend,
      toolName,
      'cmd must contain one direct executable name or path without shell syntax',
      command,
    )
  }
  const args = arguments_.args
  if (args === undefined) return [command]
  if (!Array.isArray(args)) {
    throw new SandboxBackendAdapterRequestError(backend, toolName, 'args must be an array of strings without NUL bytes', args)
  }
  const values = [command]
  for (const argument of args) {
    if (typeof argument !== 'string' || argument.includes('\0')) {
      throw new SandboxBackendAdapterRequestError(
        backend,
        toolName,
        'args must be an array of strings without NUL bytes',
        args,
      )
    }
    values.push(argument)
  }
  return values
}

function normalizeToolNames(backend: string, values: Iterable<string>): ReadonlySet<string> {
  if (!isIterable(values)) {
    throw new SandboxBackendAdapterConfigurationError(backend, 'allowedTools must be iterable')
  }
  const tools = new Set<string>()
  for (const value of values) {
    if (typeof value !== 'string' || !value.trim() || value.includes('\0')) {
      throw new SandboxBackendAdapterConfigurationError(backend, 'allowedTools must contain non-empty tool names')
    }
    tools.add(value)
  }
  if (!tools.size) {
    throw new SandboxBackendAdapterConfigurationError(backend, 'allowedTools must not be empty')
  }
  return tools
}

function normalizeEnvironment(
  backend: string,
  environment: Readonly<Record<string, string>> | undefined,
): Readonly<Record<string, string>> {
  const result: Record<string, string> = {}
  for (const [name, value] of Object.entries(environment ?? {})) {
    if (!ENVIRONMENT_NAME.test(name)) {
      throw new SandboxBackendAdapterConfigurationError(backend, `environment variable ${JSON.stringify(name)} is invalid`)
    }
    if (typeof value !== 'string' || value.includes('\0')) {
      throw new SandboxBackendAdapterConfigurationError(
        backend,
        `environment variable ${JSON.stringify(name)} must be a string without NUL bytes`,
      )
    }
    result[name] = value
  }
  return Object.freeze(result)
}

function optionalPath(
  backend: string,
  value: unknown,
  field: string,
  toolName?: string,
): string | undefined {
  if (value === undefined) return undefined
  if (typeof value !== 'string' || !value.trim() || value.includes('\0')) {
    if (toolName === undefined) {
      throw new SandboxBackendAdapterConfigurationError(backend, `${field} must be a non-empty string without NUL bytes`)
    }
    throw new SandboxBackendAdapterRequestError(backend, toolName, `${field} must be a non-empty path without NUL bytes`, value)
  }
  return value
}

function optionalPositiveInteger(
  backend: string,
  toolName: string,
  arguments_: JsonObject,
  field: string,
): number | undefined {
  const value = arguments_[field]
  if (value === undefined) return undefined
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value <= 0) {
    throw new SandboxBackendAdapterRequestError(backend, toolName, `${field} must be a positive safe integer`, value)
  }
  return value
}

function validateResult(backend: SandboxBackendName, result: SandboxCommandResult): void {
  if (result === null || typeof result !== 'object' || Array.isArray(result)) {
    throw new SandboxBackendProtocolError(backend, 'host returned a non-object command result')
  }
  if (!Number.isSafeInteger(result.exitCode)) {
    throw new SandboxBackendProtocolError(backend, 'host result exitCode must be a safe integer')
  }
  if (typeof result.stdout !== 'string' || typeof result.stderr !== 'string') {
    throw new SandboxBackendProtocolError(backend, 'host result stdout and stderr must be strings')
  }
  if (typeof result.timedOut !== 'boolean' || typeof result.truncated !== 'boolean') {
    throw new SandboxBackendProtocolError(backend, 'host result must explicitly declare timedOut and truncated')
  }
  if (result.resourceId !== undefined && (typeof result.resourceId !== 'string' || !result.resourceId.trim())) {
    throw new SandboxBackendProtocolError(backend, 'host result resourceId must be a non-empty string when present')
  }
  if (result.metadata !== undefined && !isJsonValue(result.metadata)) {
    throw new SandboxBackendProtocolError(backend, 'host result metadata must be JSON-serializable')
  }
}

function isJsonValue(value: unknown): value is JsonValue {
  if (value === null || typeof value === 'boolean' || typeof value === 'number' || typeof value === 'string') {
    return Number.isFinite(value as number) || typeof value !== 'number'
  }
  if (Array.isArray(value)) return value.every(isJsonValue)
  if (typeof value !== 'object') return false
  const prototype = Object.getPrototypeOf(value)
  if (prototype !== Object.prototype && prototype !== null) return false
  return Object.values(value).every(isJsonValue)
}

function isIterable(value: unknown): value is Iterable<unknown> {
  return value !== null && typeof value === 'object' && Symbol.iterator in value
    && typeof (value as { readonly [Symbol.iterator]?: unknown })[Symbol.iterator] === 'function'
}
