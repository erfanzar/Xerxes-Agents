// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { CommandDispatchResponse } from '../gatewayTypes.js'

export type RpcResult = Record<string, any>

export const asRpcResult = <T extends RpcResult = RpcResult>(value: unknown): T | null =>
  !value || typeof value !== 'object' || Array.isArray(value) ? null : (value as T)

export const asCommandDispatch = (value: unknown): CommandDispatchResponse | null => {
  const o = asRpcResult(value)

  if (!o || typeof o.type !== 'string') {
    return null
  }

  const t = o.type

  if (t === 'exec' || t === 'plugin') {
    return { type: t, ...(typeof o.output === 'string' ? { output: o.output } : {}) }
  }

  if (t === 'alias' && typeof o.target === 'string') {
    return { type: 'alias', target: o.target }
  }

  if (t === 'skill' && typeof o.name === 'string') {
    return { type: 'skill', name: o.name, ...(typeof o.message === 'string' ? { message: o.message } : {}) }
  }

  if (t === 'send' && typeof o.message === 'string') {
    return {
      type: 'send',
      message: o.message,
      ...(typeof o.notice === 'string' ? { notice: o.notice } : {})
    }
  }

  if (t === 'prefill' && typeof o.message === 'string') {
    return {
      type: 'prefill',
      message: o.message,
      ...(typeof o.notice === 'string' ? { notice: o.notice } : {})
    }
  }

  return null
}

export const rpcErrorMessage = (err: unknown) =>
  err instanceof Error && err.message ? err.message : typeof err === 'string' && err.trim() ? err : 'request failed'
