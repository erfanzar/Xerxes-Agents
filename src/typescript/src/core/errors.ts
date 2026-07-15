// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type ErrorDetails = Readonly<Record<string, unknown>>

/** Base error for failures with structured information safe to expose to callers. */
export class XerxesError extends Error {
  readonly details: ErrorDetails

  constructor(message: string, details: Record<string, unknown> = {}) {
    super(message)
    this.name = new.target.name
    this.details = Object.freeze({ ...details })
  }
}

export class AgentError extends XerxesError {
  readonly agentId: string

  constructor(agentId: string, message: string, details: Record<string, unknown> = {}) {
    super(`Agent ${agentId}: ${message}`, details)
    this.agentId = agentId
  }
}

export class FunctionExecutionError extends XerxesError {
  readonly functionName: string
  readonly cause: unknown

  constructor(
    functionName: string,
    message: string,
    cause: unknown = undefined,
    details: Record<string, unknown> = {},
  ) {
    super(`Function ${functionName}: ${message}`, details)
    this.functionName = functionName
    this.cause = cause
  }
}

export class XerxesTimeoutError extends XerxesError {
  readonly operation: string
  readonly timeout: number

  constructor(operation: string, timeout: number, details: Record<string, unknown> = {}) {
    super(`Operation ${operation} timed out after ${timeout} seconds`, details)
    this.operation = operation
    this.timeout = timeout
  }
}

export class ValidationError extends XerxesError {
  readonly field: string
  readonly value: unknown

  constructor(field: string, message: string, value: unknown = undefined, details: Record<string, unknown> = {}) {
    super(`Validation error for ${field}: ${message}`, details)
    this.field = field
    this.value = value
  }
}

export class RateLimitError extends XerxesError {
  readonly resource: string
  readonly limit: number
  readonly window: string
  readonly retryAfter: number | undefined

  constructor(
    resource: string,
    limit: number,
    window: string,
    retryAfter: number | undefined = undefined,
    details: Record<string, unknown> = {},
  ) {
    super(
      `Rate limit exceeded for ${resource}: ${limit} per ${window}${retryAfter ? `. Retry after ${retryAfter} seconds` : ''}`,
      details,
    )
    this.resource = resource
    this.limit = limit
    this.window = window
    this.retryAfter = retryAfter
  }
}

export class XerxesMemoryError extends XerxesError {
  readonly operation: string

  constructor(operation: string, message: string, details: Record<string, unknown> = {}) {
    super(`Memory operation ${operation}: ${message}`, details)
    this.operation = operation
  }
}

export class ClientError extends XerxesError {
  readonly clientType: string
  readonly cause: unknown

  constructor(
    clientType: string,
    message: string,
    cause: unknown = undefined,
    details: Record<string, unknown> = {},
  ) {
    super(`Client ${clientType}: ${message}`, details)
    this.clientType = clientType
    this.cause = cause
  }
}

export class ConfigurationError extends XerxesError {
  readonly configKey: string

  constructor(configKey: string, message: string, details: Record<string, unknown> = {}) {
    super(`Configuration ${configKey}: ${message}`, details)
    this.configKey = configKey
  }
}

export class AgentSpecError extends XerxesError {}

export class ToolPolicyViolation extends XerxesError {
  readonly toolName: string

  constructor(toolName: string, message: string, details: Record<string, unknown> = {}) {
    super(`Tool policy violation for ${toolName}: ${message}`, details)
    this.toolName = toolName
  }
}

export class ProviderError extends ClientError {}
