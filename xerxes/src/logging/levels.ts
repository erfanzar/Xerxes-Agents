// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Logging levels shared by human and structured Bun loggers. */
export const LoggingLevel = Object.freeze({
  DEBUG: 'DEBUG',
  INFO: 'INFO',
  WARNING: 'WARNING',
  ERROR: 'ERROR',
  CRITICAL: 'CRITICAL',
} as const)

export type LoggingLevel = (typeof LoggingLevel)[keyof typeof LoggingLevel]
export type LogContext = Readonly<Record<string, unknown>>

/** Minimal output port so hosts and tests control logging destinations. */
export interface LogOutput {
  readonly isTTY?: boolean
  write(chunk: string): void
}

const LEVEL_PRIORITY: Readonly<Record<LoggingLevel, number>> = Object.freeze({
  DEBUG: 10,
  INFO: 20,
  WARNING: 30,
  ERROR: 40,
  CRITICAL: 50,
})

/** Normalize a level name, retaining Python console behavior for unknown names. */
export function normalizeLoggingLevel(value: string | undefined): LoggingLevel {
  const normalized = value?.trim().toUpperCase()
  return isLoggingLevel(normalized) ? normalized : LoggingLevel.INFO
}

/** Return whether an event at `candidate` should pass a logger's threshold. */
export function isLoggingEnabled(threshold: LoggingLevel, candidate: LoggingLevel): boolean {
  return LEVEL_PRIORITY[candidate] >= LEVEL_PRIORITY[threshold]
}

/** Create the process stdout adapter only when no host-provided output exists. */
export function defaultLogOutput(): LogOutput {
  return {
    isTTY: process.stdout.isTTY === true,
    write: chunk => { process.stdout.write(chunk) },
  }
}

/** Return a validated ISO timestamp from an injected wall clock. */
export function loggingTimestamp(clock: () => Date): string {
  const now = clock()
  if (!(now instanceof Date) || !Number.isFinite(now.getTime())) {
    throw new TypeError('logging clock must return a valid Date')
  }
  return now.toISOString()
}

/** Merge immutable context bindings without retaining the caller's object. */
export function mergeLogContext(...contexts: Array<LogContext | undefined>): LogContext {
  return Object.freeze(Object.assign({}, ...contexts.filter((context): context is LogContext => context !== undefined)))
}

function isLoggingLevel(value: string | undefined): value is LoggingLevel {
  return value === LoggingLevel.DEBUG
    || value === LoggingLevel.INFO
    || value === LoggingLevel.WARNING
    || value === LoggingLevel.ERROR
    || value === LoggingLevel.CRITICAL
}
