// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ToolDefinition } from '../types/toolCalls.js'
import type { AcpPermissionBoard } from './permissions.js'
import type { AcpSession } from './session.js'

/** JSON-RPC error codes used by the newline-delimited ACP transport. */
export const ACP_JSON_RPC_ERRORS = {
  internalError: -32603,
  invalidParams: -32602,
  invalidRequest: -32600,
  methodNotFound: -32601,
  parseError: -32700,
} as const

export type AcpJsonRpcId = number | string | null

export interface AcpJsonRpcRequest {
  readonly id?: AcpJsonRpcId
  readonly jsonrpc: '2.0'
  readonly method: string
  readonly params?: Record<string, unknown>
}

export interface AcpJsonRpcSuccess {
  readonly id: AcpJsonRpcId
  readonly jsonrpc: '2.0'
  readonly result: unknown
}

export interface AcpJsonRpcFailure {
  readonly error: {
    readonly code: number
    readonly message: string
  }
  readonly id: AcpJsonRpcId
  readonly jsonrpc: '2.0'
}

export interface AcpModelInfo {
  readonly id: string
  readonly name: string
}

/** A flat, tagged event sent inside a `session/update` notification. */
export interface AcpWireEvent {
  readonly kind: string
  readonly [key: string]: unknown
}

export type AcpEventEmitter = (event: AcpWireEvent) => void | Promise<void>

export interface AcpPromptRequest {
  readonly emit?: AcpEventEmitter
  readonly session: AcpSession
  readonly text: string
}

export type AcpPromptHandler = (request: AcpPromptRequest) => Promise<unknown> | unknown

/**
 * Minimal seam between the transport-agnostic ACP server and a concrete runtime.
 *
 * `AcpAgentRunner` implements this interface, but tests and embedders can supply a
 * smaller runner without bootstrapping a provider.
 */
export interface AcpPromptRunner {
  cancel(sessionId: string): boolean
  listModels?(): readonly AcpModelInfo[]
  listTools?(): readonly ToolDefinition[]
  pendingQuestions?(): readonly Record<string, unknown>[]
  resetSession?(sessionId: string): void
  respondQuestion?(inputId: string, answer: string): Record<string, unknown>
  runPrompt(request: AcpPromptRequest): Promise<unknown> | unknown
  setPermissionBoard?(board: AcpPermissionBoard): void
}

export function isAcpJsonRpcId(value: unknown): value is AcpJsonRpcId {
  return value === null || typeof value === 'number' || typeof value === 'string'
}

export function isAcpRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function acpJsonRpcSuccess(id: AcpJsonRpcId, result: unknown): AcpJsonRpcSuccess {
  return { jsonrpc: '2.0', id, result }
}

export function acpJsonRpcFailure(id: AcpJsonRpcId, code: number, message: string): AcpJsonRpcFailure {
  return { jsonrpc: '2.0', id, error: { code, message } }
}
