// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Database, type SQLQueryBindings } from 'bun:sqlite'
import { chmod, mkdir, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'

/**
 * SQLite store for the identity subsystem, in its own file
 * (`<hub state dir>/identity.sqlite`, 0600, WAL). Tokens, pairing codes and
 * invite codes are stored only as keyed hashes. All timestamps are epoch
 * milliseconds except grant `expires_at` (ISO, matching `HubGrant`) and audit
 * `occurred_at` (ISO).
 */

export interface UserRow {
  id: string
  kind: 'user' | 'agent'
  display_name: string
  is_owner: number
  created_at: number
  revoked_at: number | null
  revoked_reason: string | null
}

export interface DeviceRow {
  id: string
  user_id: string
  label: string
  status: 'pending' | 'active' | 'revoked'
  /** JSON array of scopes, or null for "no restriction". */
  scope_ceiling: string | null
  created_at: number
  activated_at: number | null
  last_seen_at: number | null
  absolute_expires_at: number
  revoked_at: number | null
  revoked_reason: string | null
}

export interface FamilyRow {
  id: string
  device_id: string
  generation: number
  rotated_at: number | null
  last_used_at: number
  created_at: number
  revoked_at: number | null
}

export interface SessionRow {
  id: string
  family_id: string
  device_id: string
  user_id: string
  hash: string
  created_at: number
  expires_at: number
  revoked_at: number | null
  stepup_at: number | null
}

export interface PairingRow {
  id: string
  kind: 'pair' | 'invite'
  user_id: string | null
  label: string
  role: string | null
  code_hash: string
  comparison_hash: string
  confirm_required: number
  status: 'open' | 'consumed' | 'cancelled' | 'locked'
  failed_attempts: number
  confirm_attempts: number
  created_at: number
  expires_at: number
  consumed_at: number | null
  device_id: string | null
}

export interface InviteRow {
  id: string
  created_by: string
  code_hash: string
  resources: string
  role: string
  confirm_required: number
  label: string
  created_at: number
  expires_at: number
  status: 'open' | 'consumed' | 'cancelled' | 'locked'
  failed_attempts: number
  consumed_at: number | null
  accepted_user_id: string | null
  comparison_hash: string
  confirm_attempts: number
  device_id: string | null
}

export interface PasskeyRow {
  id: string
  user_id: string
  public_key: string
  alg: number
  sign_count: number
  label: string
  created_at: number
  last_used_at: number | null
}

export interface PushSubscriptionRow {
  id: string
  user_id: string
  device_id: string
  endpoint: string
  p256dh: string
  auth: string
  created_at: number
  last_success_at: number | null
}

export interface GrantRow {
  id: number
  principal_id: string
  resource_type: 'hub' | 'workspace' | 'session'
  workspace_id: string | null
  session_id: string | null
  effect: 'allow' | 'deny'
  role: string | null
  scopes: string | null
  expires_at: string | null
  created_at: number
  created_by: string
}

export interface AuditRow {
  sequence: number
  schema_version: number
  event_id: string
  occurred_at: string
  actor: string
  action: string
  workspace_id: string | null
  session_id: string | null
  interaction_id: string | null
  submission_id: string | null
  request_hash: string
  decision: string
  outcome: string
  previous_hash: string
  event_hash: string
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS hub_users (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('user', 'agent')),
  display_name TEXT NOT NULL,
  is_owner INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  revoked_at INTEGER,
  revoked_reason TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS hub_users_single_owner ON hub_users(is_owner) WHERE is_owner = 1;

CREATE TABLE IF NOT EXISTS hub_devices (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES hub_users(id),
  label TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending', 'active', 'revoked')),
  scope_ceiling TEXT,
  created_at INTEGER NOT NULL,
  activated_at INTEGER,
  last_seen_at INTEGER,
  absolute_expires_at INTEGER NOT NULL,
  revoked_at INTEGER,
  revoked_reason TEXT
);
CREATE INDEX IF NOT EXISTS hub_devices_user ON hub_devices(user_id);

CREATE TABLE IF NOT EXISTS hub_token_families (
  id TEXT PRIMARY KEY,
  device_id TEXT NOT NULL REFERENCES hub_devices(id),
  generation INTEGER NOT NULL,
  rotated_at INTEGER,
  last_used_at INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  revoked_at INTEGER
);
CREATE INDEX IF NOT EXISTS hub_token_families_device ON hub_token_families(device_id);

CREATE TABLE IF NOT EXISTS hub_refresh_tokens (
  hash TEXT PRIMARY KEY,
  family_id TEXT NOT NULL REFERENCES hub_token_families(id),
  generation INTEGER NOT NULL,
  issued_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS hub_refresh_tokens_family ON hub_refresh_tokens(family_id);

CREATE TABLE IF NOT EXISTS hub_sessions (
  id TEXT PRIMARY KEY,
  family_id TEXT NOT NULL REFERENCES hub_token_families(id),
  device_id TEXT NOT NULL REFERENCES hub_devices(id),
  user_id TEXT NOT NULL REFERENCES hub_users(id),
  hash TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  revoked_at INTEGER,
  stepup_at INTEGER
);
CREATE INDEX IF NOT EXISTS hub_sessions_device ON hub_sessions(device_id);
CREATE INDEX IF NOT EXISTS hub_sessions_hash ON hub_sessions(hash);

CREATE TABLE IF NOT EXISTS hub_pairings (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('pair', 'invite')),
  user_id TEXT,
  label TEXT NOT NULL,
  role TEXT,
  code_hash TEXT NOT NULL,
  comparison_hash TEXT NOT NULL,
  confirm_required INTEGER NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('open', 'consumed', 'cancelled', 'locked')),
  failed_attempts INTEGER NOT NULL DEFAULT 0,
  confirm_attempts INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  consumed_at INTEGER,
  device_id TEXT
);

CREATE TABLE IF NOT EXISTS hub_invites (
  id TEXT PRIMARY KEY,
  created_by TEXT NOT NULL REFERENCES hub_users(id),
  code_hash TEXT NOT NULL,
  resources TEXT NOT NULL,
  role TEXT NOT NULL,
  confirm_required INTEGER NOT NULL,
  label TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('open', 'consumed', 'cancelled', 'locked')),
  failed_attempts INTEGER NOT NULL DEFAULT 0,
  consumed_at INTEGER,
  accepted_user_id TEXT,
  comparison_hash TEXT NOT NULL DEFAULT '',
  confirm_attempts INTEGER NOT NULL DEFAULT 0,
  device_id TEXT
);

CREATE TABLE IF NOT EXISTS hub_passkeys (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES hub_users(id),
  public_key TEXT NOT NULL,
  alg INTEGER NOT NULL,
  sign_count INTEGER NOT NULL,
  label TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  last_used_at INTEGER
);
CREATE INDEX IF NOT EXISTS hub_passkeys_user ON hub_passkeys(user_id);

CREATE TABLE IF NOT EXISTS hub_grants (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  principal_id TEXT NOT NULL REFERENCES hub_users(id),
  resource_type TEXT NOT NULL CHECK (resource_type IN ('hub', 'workspace', 'session')),
  workspace_id TEXT,
  session_id TEXT,
  effect TEXT NOT NULL CHECK (effect IN ('allow', 'deny')),
  role TEXT,
  scopes TEXT,
  expires_at TEXT,
  created_at INTEGER NOT NULL,
  created_by TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS hub_grants_principal ON hub_grants(principal_id);

CREATE TABLE IF NOT EXISTS hub_audit_events (
  sequence INTEGER PRIMARY KEY,
  schema_version INTEGER NOT NULL,
  event_id TEXT NOT NULL UNIQUE,
  occurred_at TEXT NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  workspace_id TEXT,
  session_id TEXT,
  interaction_id TEXT,
  submission_id TEXT,
  request_hash TEXT NOT NULL,
  decision TEXT NOT NULL,
  outcome TEXT NOT NULL,
  previous_hash TEXT NOT NULL,
  event_hash TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS hub_audit_no_update BEFORE UPDATE ON hub_audit_events
BEGIN SELECT RAISE(ABORT, 'hub_audit_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS hub_audit_no_delete BEFORE DELETE ON hub_audit_events
BEGIN SELECT RAISE(ABORT, 'hub_audit_events is append-only'); END;

CREATE TABLE IF NOT EXISTS hub_push_subscriptions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES hub_users(id),
  device_id TEXT NOT NULL REFERENCES hub_devices(id),
  endpoint TEXT NOT NULL UNIQUE,
  p256dh TEXT NOT NULL,
  auth TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  last_success_at INTEGER
);
CREATE INDEX IF NOT EXISTS hub_push_user ON hub_push_subscriptions(user_id);
`

export class HubIdentityStore {
  readonly db: Database

  private constructor(readonly path: string) {
    this.db = new Database(path, { strict: true })
    this.db.exec('PRAGMA journal_mode = WAL; PRAGMA synchronous = FULL; PRAGMA busy_timeout = 5000; PRAGMA foreign_keys = ON;')
    this.db.exec(SCHEMA)
  }

  /** Create the file with mode 0600 before SQLite touches it, then open it. */
  static async open(path: string): Promise<HubIdentityStore> {
    await mkdir(dirname(path), { recursive: true, mode: 0o700 })
    await writeFile(path, '', { flag: 'a', mode: 0o600 })
    await chmod(path, 0o600)
    return new HubIdentityStore(path)
  }

  one<T>(sql: string, ...params: SQLQueryBindings[]): T | undefined {
    return this.db.query<T, SQLQueryBindings[]>(sql).get(...params) ?? undefined
  }

  all<T>(sql: string, ...params: SQLQueryBindings[]): T[] {
    return this.db.query<T, SQLQueryBindings[]>(sql).all(...params)
  }

  /** Returns the number of changed rows. */
  run(sql: string, ...params: SQLQueryBindings[]): number {
    return this.db.query<unknown, SQLQueryBindings[]>(sql).run(...params).changes
  }

  /**
   * Atomic write section. `BEGIN IMMEDIATE` takes the write lock up front so a
   * CLI process and the Hub server never interleave read-then-write steps;
   * nested calls become savepoints.
   */
  transaction<T>(fn: () => T): T {
    if (this.depth > 0) {
      const name = `sp_${this.depth}`
      this.db.exec(`SAVEPOINT ${name}`)
      this.depth += 1
      try {
        const result = fn()
        this.db.exec(`RELEASE ${name}`)
        return result
      } catch (error) {
        this.db.exec(`ROLLBACK TO ${name}; RELEASE ${name}`)
        throw error
      } finally {
        this.depth -= 1
      }
    }
    this.db.exec('BEGIN IMMEDIATE')
    this.depth = 1
    try {
      const result = fn()
      this.db.exec('COMMIT')
      return result
    } catch (error) {
      this.db.exec('ROLLBACK')
      throw error
    } finally {
      this.depth = 0
    }
  }

  private depth = 0

  close(): void {
    this.db.close()
  }
}
