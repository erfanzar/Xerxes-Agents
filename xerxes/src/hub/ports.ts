// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Normalized daemon journal event. Private provider traffic is never journal traffic. */
export interface UpstreamEvent {
  readonly event_id: string
  readonly sequence: number
  readonly cursor: string
  /** Added by the pool from the verified subscription, for durable sink dedupe. */
  readonly stream_id?: string
  readonly runtime_id: string
  readonly instance_id: string
  readonly project_dir: string
  readonly session_id?: string
  readonly type: string
  readonly payload: Readonly<Record<string, unknown>>
  readonly occurred_at: string
}

/** Implementations persist events and their upstream cursor atomically, and dedupe replays. */
export interface HubEventSink {
  ingest(runtimeId: string, workspaceId: string, events: readonly UpstreamEvent[]): Promise<void> | void
  upstreamCursor(runtimeId: string): string | undefined
}

/** SSH is a host capability; tests inject processes, never contact a real host. */
export interface SshProcess {
  readonly stderr: ReadableStream<Uint8Array>
  readonly exited: Promise<number>
  kill(signal?: 'SIGTERM' | 'SIGKILL'): void
}

export interface SshPort {
  spawn(command: string, args: readonly string[]): SshProcess
}

export interface HubTunnel {
  readonly socketPath: string
  close(): Promise<void> | void
}

export class HubOperatorError extends Error {
  readonly retryable = false
  constructor(message: string) {
    super(message)
    this.name = 'HubOperatorError'
  }
}

export function isPrivateProviderType(type: string): boolean {
  return type.startsWith('provider.') || type.startsWith('provider_') || type.startsWith('remote_provider')
}

/** Validate at the daemon boundary rather than casting journal input. */
export function parseUpstreamEvent(value: unknown): UpstreamEvent | undefined {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined
  const row = value as Record<string, unknown>
  const text = (key: string): string | undefined => typeof row[key] === 'string' && row[key] ? row[key] : undefined
  const event_id = text('event_id'), cursor = text('cursor'), runtime_id = text('runtime_id')
  const instance_id = text('instance_id'), project_dir = text('project_dir'), type = text('type'), occurred_at = text('occurred_at')
  if (!event_id || !cursor || !runtime_id || !instance_id || !project_dir || !type || !occurred_at || !Number.isFinite(Date.parse(occurred_at)) || isPrivateProviderType(type)
    || !project_dir.startsWith('/') || /[\0\r\n]/.test(project_dir)
    || !Number.isSafeInteger(row.sequence) || Number(row.sequence) < 0
    || typeof row.payload !== 'object' || row.payload === null || Array.isArray(row.payload)) return undefined
  if (row.session_id !== undefined && typeof row.session_id !== 'string') return undefined
  return { event_id, sequence: Number(row.sequence), cursor, runtime_id, instance_id, project_dir, type,
    payload: row.payload as Record<string, unknown>, occurred_at,
    ...(typeof row.session_id === 'string' ? { session_id: row.session_id } : {}) }
}
