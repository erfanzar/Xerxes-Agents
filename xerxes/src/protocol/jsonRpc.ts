// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type JsonRpcId = number | string
export type JsonRpcPayload = Record<string, unknown>

export interface JsonRpcRequest {
  readonly id: JsonRpcId
  readonly jsonrpc: '2.0'
  readonly method: string
  readonly params: JsonRpcPayload
}

export interface JsonRpcSuccess {
  readonly id: JsonRpcId
  readonly jsonrpc: '2.0'
  readonly result: JsonRpcPayload
}

export interface JsonRpcFailure {
  readonly error: {
    readonly code: number
    readonly message: string
  }
  readonly id: JsonRpcId | null
  readonly jsonrpc: '2.0'
}

export interface DaemonEventFrame {
  readonly jsonrpc: '2.0'
  readonly method: 'event'
  readonly params: {
    readonly payload: JsonRpcPayload
    readonly type: string
  }
}

export function parseJsonRpcRequest(line: string): JsonRpcRequest {
  let value: unknown
  try {
    value = JSON.parse(line) as unknown
  } catch {
    throw new JsonRpcParseError('Invalid JSON')
  }
  if (!isRecord(value) || value.jsonrpc !== '2.0' || typeof value.method !== 'string') {
    throw new JsonRpcParseError('Invalid JSON-RPC request')
  }
  if (typeof value.id !== 'string' && typeof value.id !== 'number') {
    throw new JsonRpcParseError('JSON-RPC request id must be a string or number')
  }
  return {
    jsonrpc: '2.0',
    id: value.id,
    method: value.method,
    params: isRecord(value.params) ? value.params : {},
  }
}

export function jsonRpcSuccess(id: JsonRpcId, result: JsonRpcPayload): JsonRpcSuccess {
  return { jsonrpc: '2.0', id, result }
}

export function jsonRpcFailure(id: JsonRpcId | null, code: number, message: string): JsonRpcFailure {
  return { jsonrpc: '2.0', id, error: { code, message } }
}

export function daemonEvent(type: string, payload: JsonRpcPayload): DaemonEventFrame {
  return { jsonrpc: '2.0', method: 'event', params: { type, payload } }
}

export class JsonRpcParseError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'JsonRpcParseError'
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
