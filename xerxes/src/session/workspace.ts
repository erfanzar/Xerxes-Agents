// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'

import type { JsonValue } from '../types/toolCalls.js'

/** JSON-safe metadata retained with a workspace identity. */
export type WorkspaceMetadata = Readonly<Record<string, JsonValue>>

/** Python-readable, serializable wire shape for one workspace identity. */
export interface WorkspaceIdentityRecord {
  readonly workspace_id: string
  readonly name: string
  readonly root_path: string | null
  readonly created_at: string
  readonly metadata: WorkspaceMetadata
}

export interface WorkspaceIdentityOptions {
  readonly createdAt?: string
  readonly metadata?: Readonly<Record<string, JsonValue>>
  readonly name: string
  readonly rootPath?: string | null
  readonly workspaceId: string
}

/**
 * Stable, JSON-serializable identity for the long-lived container of sessions.
 *
 * Instances are immutable value objects. Their metadata is cloned and frozen,
 * so a caller cannot mutate a registry entry through a retained input object.
 */
export class WorkspaceIdentity {
  readonly createdAt: string
  readonly metadata: WorkspaceMetadata
  readonly name: string
  readonly rootPath: string | null
  readonly workspaceId: string

  constructor(options: WorkspaceIdentityOptions) {
    this.workspaceId = options.workspaceId
    this.name = options.name
    this.rootPath = options.rootPath ?? null
    this.createdAt = options.createdAt ?? ''
    this.metadata = copyMetadata(options.metadata ?? {})
    Object.freeze(this)
  }

  /** Return a fresh immutable Python-compatible record for JSON serialization. */
  toRecord(): WorkspaceIdentityRecord {
    return freezeRecord({
      workspace_id: this.workspaceId,
      name: this.name,
      root_path: this.rootPath,
      created_at: this.createdAt,
      metadata: copyMetadata(this.metadata),
    })
  }

  toJSON(): WorkspaceIdentityRecord {
    return this.toRecord()
  }

  /** Reconstruct an immutable identity from its Python-compatible wire record. */
  static fromRecord(value: unknown): WorkspaceIdentity {
    if (!isPlainRecord(value)) {
      throw new TypeError('workspace identity must be an object')
    }
    if (typeof value.workspace_id !== 'string' || typeof value.name !== 'string') {
      throw new TypeError('workspace identity must contain workspace_id and name strings')
    }
    const rootPath = hasOwn(value, 'root_path') ? nullableString(value.root_path, 'root_path') : null
    const createdAt = hasOwn(value, 'created_at') ? stringValue(value.created_at, 'created_at') : ''
    const metadata = hasOwn(value, 'metadata') ? metadataValue(value.metadata) : {}
    return new WorkspaceIdentity({
      workspaceId: value.workspace_id,
      name: value.name,
      rootPath,
      createdAt,
      metadata,
    })
  }
}

export interface WorkspaceManagerOptions {
  /** Injectable opaque-ID source for deterministic hosts and tests. */
  readonly idFactory?: () => string
  /** Injectable wall clock for deterministic creation timestamps. */
  readonly clock?: () => Date
}

export interface CreateWorkspaceOptions {
  readonly metadata?: Readonly<Record<string, JsonValue>>
  readonly name: string
  readonly rootPath?: string | null
  readonly workspaceId?: string
}

/**
 * Process-local registry of workspace identities.
 *
 * JavaScript map changes are atomic between awaits, so no lock is needed for
 * this synchronous registry. Durable session storage remains responsible for
 * retaining a workspace ID across process restarts.
 */
export class WorkspaceManager {
  private readonly clock: () => Date
  private readonly idFactory: () => string
  private readonly workspaces = new Map<string, WorkspaceIdentity>()

  constructor(options: WorkspaceManagerOptions = {}) {
    this.idFactory = options.idFactory ?? defaultWorkspaceId
    this.clock = options.clock ?? (() => new Date())
  }

  /** Register a workspace and return a defensive immutable identity snapshot. */
  createWorkspace(options: CreateWorkspaceOptions): WorkspaceIdentity {
    const workspace = new WorkspaceIdentity({
      workspaceId: options.workspaceId || this.nextWorkspaceId(),
      name: options.name,
      ...(options.rootPath === undefined ? {} : { rootPath: options.rootPath }),
      createdAt: clockTimestamp(this.clock),
      ...(options.metadata === undefined ? {} : { metadata: options.metadata }),
    })
    this.workspaces.set(workspace.workspaceId, workspace)
    return snapshot(workspace)
  }

  /** Return a defensive immutable snapshot, or undefined for an unknown ID. */
  getWorkspace(workspaceId: string): WorkspaceIdentity | undefined {
    const workspace = this.workspaces.get(workspaceId)
    return workspace === undefined ? undefined : snapshot(workspace)
  }

  /** Return immutable identity snapshots in workspace creation order. */
  listWorkspaces(): readonly WorkspaceIdentity[] {
    return Object.freeze([...this.workspaces.values()].map(snapshot))
  }

  private nextWorkspaceId(): string {
    const workspaceId = this.idFactory()
    if (!workspaceId.trim()) {
      throw new TypeError('workspace id factory returned an empty identifier')
    }
    return workspaceId
  }
}

function defaultWorkspaceId(): string {
  return randomUUID().replaceAll('-', '')
}

function clockTimestamp(clock: () => Date): string {
  const now = clock()
  if (!(now instanceof Date) || !Number.isFinite(now.getTime())) {
    throw new TypeError('workspace manager clock must return a valid Date')
  }
  return now.toISOString()
}

function snapshot(identity: WorkspaceIdentity): WorkspaceIdentity {
  return WorkspaceIdentity.fromRecord(identity.toRecord())
}

function freezeRecord(record: WorkspaceIdentityRecord): WorkspaceIdentityRecord {
  return Object.freeze({ ...record, metadata: copyMetadata(record.metadata) })
}

function metadataValue(value: unknown): WorkspaceMetadata {
  if (!isPlainRecord(value)) {
    throw new TypeError('workspace identity metadata must be a JSON object')
  }
  return copyMetadata(value)
}

function copyMetadata(metadata: Readonly<Record<string, unknown>>): WorkspaceMetadata {
  const copied: Record<string, JsonValue> = {}
  for (const [key, value] of Object.entries(metadata)) {
    copied[key] = copyJsonValue(value)
  }
  return Object.freeze(copied)
}

function copyJsonValue(value: unknown): JsonValue {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return value
  if (typeof value === 'number') {
    if (Number.isFinite(value)) return value
    throw new TypeError('workspace metadata must contain JSON-serializable values')
  }
  if (Array.isArray(value)) {
    return Object.freeze(value.map(copyJsonValue)) as unknown as JsonValue
  }
  if (isPlainRecord(value)) {
    const copied: Record<string, JsonValue> = {}
    for (const [key, nested] of Object.entries(value)) {
      copied[key] = copyJsonValue(nested)
    }
    return Object.freeze(copied) as JsonValue
  }
  throw new TypeError('workspace metadata must contain JSON-serializable values')
}

function nullableString(value: unknown, field: string): string | null {
  if (value === null || typeof value === 'string') return value
  throw new TypeError(`workspace identity ${field} must be a string or null`)
}

function stringValue(value: unknown, field: string): string {
  if (typeof value === 'string') return value
  throw new TypeError(`workspace identity ${field} must be a string`)
}

function hasOwn(value: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key)
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}
