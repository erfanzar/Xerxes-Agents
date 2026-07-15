// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { StreamEvent } from '../streaming/events.js'
import {
  LoggingLevel,
  defaultLogOutput,
  isLoggingEnabled,
  loggingTimestamp,
  mergeLogContext,
  normalizeLoggingLevel,
  type LogContext,
  type LogOutput,
  type LoggingLevel as LoggingLevelName,
} from './levels.js'

/** ANSI palette retained from the Python console logger. */
export const COLORS = Object.freeze({
  BLACK: '\u001B[30m',
  RED: '\u001B[31m',
  GREEN: '\u001B[32m',
  YELLOW: '\u001B[33m',
  BLUE: '\u001B[34m',
  MAGENTA: '\u001B[35m',
  CYAN: '\u001B[36m',
  WHITE: '\u001B[37m',
  LIGHT_BLACK: '\u001B[90m',
  LIGHT_RED: '\u001B[91m',
  LIGHT_GREEN: '\u001B[92m',
  LIGHT_YELLOW: '\u001B[93m',
  LIGHT_BLUE: '\u001B[94m',
  LIGHT_MAGENTA: '\u001B[95m',
  LIGHT_CYAN: '\u001B[96m',
  LIGHT_WHITE: '\u001B[97m',
  RESET: '\u001B[0m',
  BOLD: '\u001B[1m',
  DIM: '\u001B[2m',
  ITALIC: '\u001B[3m',
  UNDERLINE: '\u001B[4m',
  BLUE_PURPLE: '\u001B[38;5;99m',
} as const)

export type ConsoleColor = keyof typeof COLORS

export const LEVEL_COLORS: Readonly<Record<LoggingLevelName, string>> = Object.freeze({
  DEBUG: COLORS.LIGHT_BLUE,
  INFO: COLORS.BLUE_PURPLE,
  WARNING: COLORS.YELLOW,
  ERROR: COLORS.LIGHT_RED,
  CRITICAL: COLORS.RED + COLORS.BOLD,
})

export interface ConsoleLogRecord {
  readonly context?: LogContext
  readonly level: LoggingLevelName
  readonly message: string
  readonly name: string
  readonly timestamp: string
}

export interface ColorFormatterOptions {
  readonly color?: boolean
}

/** Format human logs like the Python formatter, including every multiline segment. */
export class ColorFormatter {
  private readonly color: boolean

  constructor(options: ColorFormatterOptions = {}) {
    this.color = options.color ?? true
  }

  format(record: ConsoleLogRecord): string {
    const color = LEVEL_COLORS[record.level]
    const tag = this.color
      ? `${color}(${clockTime(record.timestamp)} ${record.name})${COLORS.RESET}`
      : `(${clockTime(record.timestamp)} ${record.name})`
    const suffix = contextSuffix(record.context)
    return record.message.split('\n').map(line => {
      const rendered = line + suffix
      return rendered ? `${tag} ${rendered}` : tag
    }).join('\n')
  }
}

export interface ConsoleLoggerOptions {
  readonly clock?: () => Date
  readonly color?: boolean
  readonly context?: LogContext
  readonly level?: string
  readonly name?: string
  readonly stream?: LogOutput
}

/**
 * Human-oriented logger with TTY-aware ANSI decoration and immutable context bindings.
 *
 * A logger writes only to its injected output port. The lazy module singleton is kept
 * solely for Python-style helper functions and never changes Node's global handlers.
 */
export class XerxesLogger {
  readonly name: string
  private readonly clock: () => Date
  private readonly color: boolean
  private readonly context: LogContext
  private readonly formatter: ColorFormatter
  private currentLevel: LoggingLevelName
  private readonly stream: LogOutput

  constructor(options: ConsoleLoggerOptions = {}) {
    this.name = options.name?.trim() || 'Xerxes'
    this.clock = options.clock ?? (() => new Date())
    this.stream = options.stream ?? defaultLogOutput()
    this.color = options.color ?? this.stream.isTTY === true
    this.context = mergeLogContext(options.context)
    this.currentLevel = normalizeLoggingLevel(options.level ?? process.env.XERXES_LOG_LEVEL)
    this.formatter = new ColorFormatter({ color: this.color })
  }

  get level(): LoggingLevelName {
    return this.currentLevel
  }

  get usesColor(): boolean {
    return this.color
  }

  /** Return a sibling logger that shares output and clock but extends this context. */
  bind(context: LogContext): XerxesLogger {
    return new XerxesLogger({
      name: this.name,
      level: this.currentLevel,
      stream: this.stream,
      clock: this.clock,
      color: this.color,
      context: mergeLogContext(this.context, context),
    })
  }

  /** Set the current threshold; unknown labels safely select INFO. */
  setLevel(level: string): void {
    this.currentLevel = normalizeLoggingLevel(level)
  }

  debug(message: string, context: LogContext = {}): ConsoleLogRecord | undefined {
    return this.log(LoggingLevel.DEBUG, message, context)
  }

  info(message: string, context: LogContext = {}): ConsoleLogRecord | undefined {
    return this.log(LoggingLevel.INFO, message, context)
  }

  warning(message: string, context: LogContext = {}): ConsoleLogRecord | undefined {
    return this.log(LoggingLevel.WARNING, message, context)
  }

  error(message: string, context: LogContext = {}): ConsoleLogRecord | undefined {
    return this.log(LoggingLevel.ERROR, message, context)
  }

  critical(message: string, context: LogContext = {}): ConsoleLogRecord | undefined {
    return this.log(LoggingLevel.CRITICAL, message, context)
  }

  log(level: string, message: string, context: LogContext = {}): ConsoleLogRecord | undefined {
    const normalized = normalizeLoggingLevel(level)
    if (!isLoggingEnabled(this.currentLevel, normalized)) return undefined
    const record = Object.freeze({
      timestamp: loggingTimestamp(this.clock),
      name: this.name,
      level: normalized,
      message: String(message),
      context: mergeLogContext(this.context, context),
    })
    this.stream.write(this.formatter.format(record) + '\n')
    return record
  }
}

let defaultLogger: XerxesLogger | undefined

/** Return the lazily-created convenience logger without mutating global handlers. */
export function getLogger(): XerxesLogger {
  defaultLogger ??= new XerxesLogger()
  return defaultLogger
}

/** Replace the convenience logger explicitly, primarily for hosts and tests. */
export function configureConsoleLogger(options: ConsoleLoggerOptions = {}): XerxesLogger {
  defaultLogger = new XerxesLogger(options)
  return defaultLogger
}

/** Python-compatible singleton accessor with a less ambiguous TypeScript name. */
export const getConsoleLogger = getLogger

/** Update the default logger threshold. */
export function setVerbosity(level: string): void {
  getLogger().setLevel(level)
}

export function logStep(
  stepName: string,
  description = '',
  color: string = 'CYAN',
  logger: XerxesLogger = getLogger(),
): ConsoleLogRecord | undefined {
  const tag = `[${stepName}]`
  const styled = paint(logger, tag, COLORS[color.toUpperCase() as ConsoleColor] ?? COLORS.CYAN)
  return logger.info(description ? `${styled} ${description}` : styled)
}

export function logThinking(agentName: string, logger: XerxesLogger = getLogger()): ConsoleLogRecord | undefined {
  const agent = paint(logger, `  (🧠 ${agentName})`, COLORS.BLUE)
  const suffix = paint(logger, ' is thinking...', COLORS.BLUE_PURPLE)
  return logger.info(agent + suffix)
}

export function logSuccess(message: string, logger: XerxesLogger = getLogger()): ConsoleLogRecord | undefined {
  return logger.info(paint(logger, `🚀 ${message}`, COLORS.BLUE))
}

export function logError(message: string, logger: XerxesLogger = getLogger()): ConsoleLogRecord | undefined {
  return logger.error(paint(logger, `❌ ${message}`, COLORS.LIGHT_RED))
}

export function logWarning(message: string, logger: XerxesLogger = getLogger()): ConsoleLogRecord | undefined {
  return logger.warning(paint(logger, `⚠️ ${message}`, COLORS.YELLOW))
}

export function logRetry(
  attempt: number,
  maxAttempts: number,
  error: string,
  logger: XerxesLogger = getLogger(),
): ConsoleLogRecord | undefined {
  return logger.warning(
    paint(logger, `⏳ Retry ${attempt}/${maxAttempts}: `, COLORS.YELLOW) + paint(logger, error, COLORS.LIGHT_RED),
  )
}

export function logDelegation(
  fromAgent: string,
  toAgent: string,
  logger: XerxesLogger = getLogger(),
): ConsoleLogRecord | undefined {
  const prefix = paint(logger, '📌 Delegation: ', COLORS.MAGENTA)
  const source = paint(logger, fromAgent, COLORS.CYAN)
  const target = paint(logger, toAgent, COLORS.CYAN)
  return logger.info(`${prefix}${source} → ${target}`)
}

export function logAgentStart(agent?: string, logger: XerxesLogger = getLogger()): ConsoleLogRecord | undefined {
  const label = agent ? `${agent} Agent is started.` : 'Agent is started.'
  return logger.info(' ' + paint(logger, label, COLORS.BLUE_PURPLE))
}

export function logTaskStart(
  taskName: string,
  agent?: string,
  logger: XerxesLogger = getLogger(),
): ConsoleLogRecord | undefined {
  const task = paint(logger, ` Task Started: ${taskName}`, COLORS.BLUE, COLORS.BOLD)
  const owner = agent ? ' ' + paint(logger, `(Agent: ${agent})`, COLORS.DIM) : ''
  return logger.info(task + owner)
}

export function logTaskComplete(
  taskName: string,
  duration?: number,
  logger: XerxesLogger = getLogger(),
): ConsoleLogRecord | undefined {
  const task = paint(logger, `🚀 Task Completed: ${taskName}`, COLORS.GREEN)
  const elapsed = duration ? ' ' + paint(logger, `(${duration.toFixed(2)}s)`, COLORS.DIM) : ''
  return logger.info(task + elapsed)
}

export interface StreamConsoleRendererOptions {
  readonly color?: boolean
  readonly stream?: LogOutput
}

/** Compact Bun-native renderer for the portable stream event vocabulary. */
export class StreamConsoleRenderer {
  private readonly color: boolean
  private openLine = false
  private readonly stream: LogOutput

  constructor(options: StreamConsoleRendererOptions = {}) {
    this.stream = options.stream ?? defaultLogOutput()
    this.color = options.color ?? this.stream.isTTY === true
  }

  render(event: StreamEvent): void {
    switch (event.type) {
      case 'text':
        this.writeText(event.text)
        return
      case 'thinking':
        this.writeText(this.paint(event.text, COLORS.MAGENTA))
        return
      case 'tool_start':
        this.writeLine(`${this.paint('🛠️', COLORS.BLUE_PURPLE)} Calling ${event.call.function.name}`)
        return
      case 'tool_end': {
        const status = event.result.permitted ? '✅' : '❌'
        const elapsed = event.result.durationMs ? ` in ${(event.result.durationMs / 1_000).toFixed(2)}s` : ''
        this.writeLine(`${this.paint(status, event.result.permitted ? COLORS.LIGHT_GREEN : COLORS.LIGHT_RED)} ${event.result.name} completed${elapsed}`)
        return
      }
      case 'permission_request':
        this.writeLine(`${this.paint('⚠️', COLORS.YELLOW)} Permission requested: ${event.request.description}`)
        return
      case 'provider_retry':
        this.writeLine(`${this.paint('⏳', COLORS.YELLOW)} Retry ${event.attempt}/${event.maxAttempts}: ${event.error}`)
        return
      case 'skill_suggestion':
        this.writeLine(`${this.paint('💡', COLORS.CYAN)} Suggested skill: ${event.skillName}`)
        return
      case 'turn_done':
        this.writeLine(`${this.paint('✓', COLORS.LIGHT_GREEN)} Turn completed (${event.toolCallsCount} tool call(s))`)
        return
    }
  }

  flush(): void {
    if (!this.openLine) return
    this.stream.write('\n')
    this.openLine = false
  }

  private writeText(value: string): void {
    this.stream.write(value)
    this.openLine = !value.endsWith('\n')
  }

  private writeLine(value: string): void {
    this.flush()
    this.stream.write(value + '\n')
  }

  private paint(value: string, ...styles: string[]): string {
    return this.color ? styles.join('') + value + COLORS.RESET : value
  }
}

let defaultStreamRenderer: StreamConsoleRenderer | undefined

/** Python-style default stream callback for lightweight non-TUI rendering. */
export function streamCallback(event: StreamEvent): void {
  defaultStreamRenderer ??= new StreamConsoleRenderer()
  defaultStreamRenderer.render(event)
}

/** Build an injected stream callback without sharing state with other renderers. */
export function createStreamCallback(options: StreamConsoleRendererOptions = {}): (event: StreamEvent) => void {
  const renderer = new StreamConsoleRenderer(options)
  return event => renderer.render(event)
}

function paint(logger: XerxesLogger, value: string, ...styles: string[]): string {
  return logger.usesColor ? styles.join('') + value + COLORS.RESET : value
}

function clockTime(timestamp: string): string {
  const match = timestamp.match(/T(\d{2}:\d{2}:\d{2})/)
  return match?.[1] ?? timestamp
}

function contextSuffix(context: LogContext | undefined): string {
  if (context === undefined || !Object.keys(context).length) return ''
  try {
    return ' ' + JSON.stringify(context)
  } catch {
    return ' [unserializable context]'
  }
}
