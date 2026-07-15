// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { CURRENT_SESSION_SCHEMA_VERSION, type SessionRecordData } from './models.js'

/** A forward-only transformation from the preceding session schema version. */
export type SessionMigration = (record: SessionRecordData) => SessionRecordData

/** Registered migrations keyed by the version they produce. */
export const SESSION_MIGRATIONS = new Map<number, SessionMigration>()
/** Compatibility alias for consumers that used the Python session API name. */
export const MIGRATIONS = SESSION_MIGRATIONS

/** Register a migration that upgrades a record to `targetVersion`. */
export function registerMigration(targetVersion: number, migration: SessionMigration): SessionMigration {
  if (!Number.isInteger(targetVersion) || targetVersion < 2) {
    throw new RangeError(`Migration target version must be an integer >= 2; got ${targetVersion}`)
  }
  if (SESSION_MIGRATIONS.has(targetVersion)) {
    throw new Error(`Duplicate migration registered for version ${targetVersion}`)
  }
  SESSION_MIGRATIONS.set(targetVersion, migration)
  return migration
}

/** Alias that makes session migration call sites explicit. */
export const registerSessionMigration = registerMigration

/** Remove a migration, primarily for isolated test setup. */
export function unregisterMigration(targetVersion: number): boolean {
  return SESSION_MIGRATIONS.delete(targetVersion)
}

/**
 * Apply registered forward-only migrations until a record reaches `targetVersion`.
 *
 * Records already at or beyond the target are returned as-is because the runtime
 * deliberately never downgrades persisted data.
 */
export function migrateSessionRecord(
  record: SessionRecordData,
  targetVersion = CURRENT_SESSION_SCHEMA_VERSION,
): SessionRecordData {
  let current = schemaVersion(record.schema_version)
  if (current >= targetVersion) return record

  let migrated = record
  while (current < targetVersion) {
    const nextVersion = current + 1
    const migration = SESSION_MIGRATIONS.get(nextVersion)
    if (!migration) {
      throw new Error(
        `No migration registered to produce schema_version=${nextVersion}; current=${current}, target=${targetVersion}`,
      )
    }
    migrated = migration(migrated)
    migrated.schema_version = nextVersion
    current = nextVersion
  }
  return migrated
}

/** Compatibility alias for the original session subsystem. */
export const migrateRecord = migrateSessionRecord

function schemaVersion(value: unknown): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) return 1
  return value
}
