// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Parameters for the MCP connection retry schedule. Delays are measured in seconds. */
export interface ReconnectPolicyOptions {
  readonly baseSeconds?: number
  readonly factor?: number
  readonly maxAttempts?: number
  readonly maxSeconds?: number
}

/** Exponential-backoff policy used when an MCP server disconnects. */
export class ReconnectPolicy {
  readonly baseSeconds: number
  readonly factor: number
  readonly maxAttempts: number
  readonly maxSeconds: number

  constructor(options: ReconnectPolicyOptions = {}) {
    this.maxAttempts = positiveInteger(options.maxAttempts ?? 5, 'maxAttempts')
    this.baseSeconds = positiveNumber(options.baseSeconds ?? 1, 'baseSeconds')
    this.factor = positiveNumber(options.factor ?? 2, 'factor')
    this.maxSeconds = positiveNumber(options.maxSeconds ?? 60, 'maxSeconds')
  }

  /** Return the delay after a one-based failed connection attempt. */
  delayForAttempt(attempt: number): number {
    const normalizedAttempt = Math.max(1, Math.floor(attempt))
    return Math.min(this.maxSeconds, this.baseSeconds * this.factor ** (normalizedAttempt - 1))
  }
}

/** Raised after the final MCP connection retry fails. Its message is always credential-scrubbed. */
export class MCPReconnectError extends Error {
  readonly attempts: number

  constructor(message: string, attempts: number) {
    super(scrubCredentials(message))
    this.name = new.target.name
    this.attempts = attempts
  }
}

export interface ReconnectWithBackoffOptions {
  /** Called after each failed attempt, before the next delay. */
  readonly onError?: (attempt: number, error: unknown) => void | Promise<void>
  readonly policy?: ReconnectPolicy | ReconnectPolicyOptions
  /** Injectable sleep implementation. It receives seconds, not milliseconds. */
  readonly sleep?: (seconds: number) => void | Promise<void>
}

/**
 * Retry a connection operation with exponential backoff.
 *
 * The retry hook receives the original error so hosts can apply their own
 * observability policy. The thrown terminal error only exposes a redacted
 * message so credentials from subprocess or HTTP diagnostics cannot escape
 * through the normal lifecycle API.
 */
export async function reconnectWithBackoff<T>(
  connect: () => T | Promise<T>,
  options: ReconnectWithBackoffOptions = {},
): Promise<T> {
  const policy = options.policy instanceof ReconnectPolicy
    ? options.policy
    : new ReconnectPolicy(options.policy)
  const sleep = options.sleep ?? sleepSeconds
  let lastError: unknown

  for (let attempt = 1; attempt <= policy.maxAttempts; attempt += 1) {
    try {
      return await connect()
    } catch (error) {
      lastError = error
      await options.onError?.(attempt, error)
      if (attempt >= policy.maxAttempts) {
        break
      }
      await sleep(policy.delayForAttempt(attempt))
    }
  }

  throw new MCPReconnectError(errorMessage(lastError), policy.maxAttempts)
}

/** Replace common API key, bearer token, and password fragments with a safe marker. */
export function scrubCredentials(text: string): string {
  return text
    .replace(/\b(password)\b\s*(?::|=|\s)\s*['"]?([^\s'";,]+)/gi, '$1=[redacted]')
    .replace(/\b(api[_-]?key)\b\s*(?::|=|\s)\s*['"]?([A-Za-z0-9._-]{8,})/gi, '$1=[redacted]')
    .replace(/\b(token)\b\s*(?::|=|\s)\s*['"]?([A-Za-z0-9._-]{16,})/gi, '$1=[redacted]')
    .replace(/\b(authorization\s*:\s*bearer)\s+([A-Za-z0-9._-]+)/gi, '$1=[redacted]')
    .replace(/\bsk-[A-Za-z0-9_-]{16,}\b/g, '[redacted]')
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }
  return String(error)
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) {
    throw new RangeError(name + ' must be an integer of at least 1')
  }
  return value
}

function positiveNumber(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(name + ' must be positive')
  }
  return value
}

function sleepSeconds(seconds: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, seconds * 1_000))
}
