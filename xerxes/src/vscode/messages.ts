// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Frames between the Xerxes webview and the extension host. They carry the
 * same channels the desktop's Electron IPC does (see desktop/bridgeApi.ts),
 * so the panel runs the desktop renderer unchanged.
 */

/** Webview → host: one bridge request. */
export interface InvokeFrame { readonly kind: 'invoke'; readonly id: number; readonly channel: string; readonly args: readonly unknown[] }
/** Host → webview: the answer to one request. */
export type ResultFrame =
  | { readonly kind: 'result'; readonly id: number; readonly ok: true; readonly value: unknown }
  | { readonly kind: 'result'; readonly id: number; readonly ok: false; readonly error: string }
/** Host → webview: a value pushed on a channel (runtime events, update state, …). */
export interface PushFrame { readonly kind: 'push'; readonly channel: string; readonly value: unknown }

const CHANNEL = /^[a-z][a-z0-9:-]{0,63}$/i

/** Validate an untrusted frame from the webview. */
export function invokeFrame(value: unknown): InvokeFrame | null {
  if (!value || typeof value !== 'object') return null
  const frame = value as Record<string, unknown>
  if (frame.kind !== 'invoke' || !Number.isSafeInteger(frame.id) || typeof frame.channel !== 'string' || !CHANNEL.test(frame.channel) || !Array.isArray(frame.args) || frame.args.length > 8) return null
  return { kind: 'invoke', id: frame.id as number, channel: frame.channel, args: frame.args }
}

/** Validate an untrusted frame from the host (inside the webview). */
export function hostFrame(value: unknown): ResultFrame | PushFrame | null {
  if (!value || typeof value !== 'object') return null
  const frame = value as Record<string, unknown>
  if (frame.kind === 'result' && Number.isSafeInteger(frame.id)) {
    return frame.ok === true
      ? { kind: 'result', id: frame.id as number, ok: true, value: frame.value }
      : { kind: 'result', id: frame.id as number, ok: false, error: typeof frame.error === 'string' ? frame.error : 'Request failed' }
  }
  if (frame.kind === 'push' && typeof frame.channel === 'string' && CHANNEL.test(frame.channel)) return { kind: 'push', channel: frame.channel, value: frame.value }
  return null
}
