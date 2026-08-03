// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SpawnedAgentSnapshot } from '../operators/subagents.js'

export const SUBAGENT_SNAPSHOT_METADATA_KEY = 'xerxes_subagent_snapshots_v1'
export const SUBAGENT_DELIVERY_METADATA_KEY = 'xerxes_subagent_deliveries_v1'

const MAX_ARCHIVED_TEXT_CHARS = 16_000
const MAX_ARCHIVED_FILES = 1_000
const MAX_ARCHIVED_DELIVERIES = 2_000

export interface PersistedSubagentDelivery {
  readonly generation: string
  readonly id: string
}

/** Persist the bounded set of exact task generations already delivered to the parent. */
export function replacePersistedSubagentDeliveries(
  metadata: Record<string, unknown>,
  deliveries: readonly PersistedSubagentDelivery[],
): void {
  metadata[SUBAGENT_DELIVERY_METADATA_KEY] = deliveries
    .slice(-MAX_ARCHIVED_DELIVERIES)
    .map(delivery => ({ generation: delivery.generation, id: delivery.id }))
}

/** Read validated delivery markers from session metadata. */
export function persistedSubagentDeliveryValues(
  metadata: Readonly<Record<string, unknown>>,
): readonly PersistedSubagentDelivery[] {
  const value = metadata[SUBAGENT_DELIVERY_METADATA_KEY]
  if (!Array.isArray(value)) return []
  return Object.freeze(value.slice(-MAX_ARCHIVED_DELIVERIES).flatMap(item => {
    if (item === null || typeof item !== 'object' || Array.isArray(item)) return []
    const record = item as Readonly<Record<string, unknown>>
    return typeof record.id === 'string' && record.id.trim()
      && typeof record.generation === 'string' && record.generation.trim()
      ? [{ id: record.id.trim(), generation: record.generation.trim() }]
      : []
  }))
}

/** Replace the session-owned manifest with the manager's current complete view. */
export function replacePersistedSubagentSnapshots(
  metadata: Record<string, unknown>,
  snapshots: readonly SpawnedAgentSnapshot[],
): void {
  metadata[SUBAGENT_SNAPSHOT_METADATA_KEY] = snapshots.map(archivedSnapshotWire)
}

/** Merge terminal progress observed outside a tool call into the durable manifest. */
export function mergePersistedSubagentSnapshots(
  metadata: Record<string, unknown>,
  snapshots: readonly SpawnedAgentSnapshot[],
): void {
  const existing = persistedSubagentSnapshotValues(metadata)
  const byId = new Map<string, Record<string, unknown>>()
  for (const value of existing) {
    const id = typeof value.id === 'string' ? value.id : ''
    if (id) byId.set(id, value)
  }
  for (const snapshot of snapshots) byId.set(snapshot.id, archivedSnapshotWire(snapshot))
  metadata[SUBAGENT_SNAPSHOT_METADATA_KEY] = [...byId.values()]
}

export function persistedSubagentSnapshotValues(
  metadata: Readonly<Record<string, unknown>>,
): readonly Readonly<Record<string, unknown>>[] {
  const value = metadata[SUBAGENT_SNAPSHOT_METADATA_KEY]
  if (!Array.isArray(value)) return []
  return value.filter((item): item is Readonly<Record<string, unknown>> => (
    item !== null && typeof item === 'object' && !Array.isArray(item)
  ))
}

function archivedSnapshotWire(snapshot: SpawnedAgentSnapshot): Record<string, unknown> {
  const retryAware = snapshot as SpawnedAgentSnapshot & {
    readonly attempt?: unknown
    readonly generation?: unknown
  }
  return {
    id: snapshot.id,
    ...(typeof retryAware.attempt === 'number' || typeof retryAware.attempt === 'string'
      ? { attempt: retryAware.attempt }
      : {}),
    ...(typeof retryAware.generation === 'number' || typeof retryAware.generation === 'string'
      ? { generation: retryAware.generation }
      : {}),
    name: snapshot.name,
    title: snapshot.title,
    agent_id: snapshot.agentId,
    creator_id: snapshot.creatorAgentId ?? null,
    parent_id: snapshot.parentAgentId ?? null,
    model: snapshot.model ?? null,
    rules: snapshot.rules ?? [],
    toolsets: snapshot.toolsets ?? [],
    ...(snapshot.apiCalls === undefined ? {} : { api_calls: snapshot.apiCalls }),
    ...(snapshot.toolCalls === undefined ? {} : { tool_count: snapshot.toolCalls }),
    ...(snapshot.cacheReadTokens === undefined ? {} : { cache_read_tokens: snapshot.cacheReadTokens }),
    ...(snapshot.cacheCreationTokens === undefined ? {} : { cache_creation_tokens: snapshot.cacheCreationTokens }),
    ...(snapshot.inputTokens === undefined ? {} : { input_tokens: snapshot.inputTokens }),
    ...(snapshot.outputTokens === undefined ? {} : { output_tokens: snapshot.outputTokens }),
    ...(snapshot.reasoningTokens === undefined ? {} : { reasoning_tokens: snapshot.reasoningTokens }),
    files_read: snapshot.filesRead?.slice(0, MAX_ARCHIVED_FILES) ?? [],
    files_written: snapshot.filesWritten?.slice(0, MAX_ARCHIVED_FILES) ?? [],
    summary: boundedText(snapshot.completionSummary, 500) ?? null,
    status: snapshot.status,
    history_session_id: snapshot.historySessionId ?? null,
    created_at: snapshot.createdAt,
    updated_at: snapshot.updatedAt,
    prompt_profile: snapshot.promptProfile,
    source_agent_id: snapshot.sourceAgentId ?? null,
    last_input: boundedText(snapshot.lastInput, MAX_ARCHIVED_TEXT_CHARS) ?? null,
    last_output: boundedText(snapshot.lastOutput, MAX_ARCHIVED_TEXT_CHARS) ?? null,
    error: boundedText(snapshot.error, 2_000) ?? null,
    queue_size: snapshot.queueSize,
    closed: snapshot.closed,
  }
}

function boundedText(value: string | undefined, limit: number): string | undefined {
  if (!value) return undefined
  return value.length <= limit ? value : `${value.slice(0, limit - 1)}…`
}
