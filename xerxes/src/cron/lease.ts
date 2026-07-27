// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdirSync, readFileSync, statSync, unlinkSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

import { processCommand, processIsAlive } from '../core/processLiveness.js'

/**
 * A lease record grows a `command` so a recycled pid cannot impersonate the
 * holder: the daemon that took the lease recorded its own command line, and a
 * later claimant compares that against what the pid is running now.
 */
export interface CronLeaseRecord {
  readonly acquiredAt: string
  readonly command: string
  readonly ownerKey: string
  readonly pid: number
}

/** What the existing lease file means for the process trying to claim it. */
export type CronLeaseHolderState = 'owned' | 'refused' | 'stale'

/** `acquired` and `owned` both permit cron work; `refused` never does. */
export type CronLeaseState = 'acquired' | 'owned' | 'refused'

export interface CronLeaseOutcome {
  /** True when the caller may run cron work. */
  readonly held: boolean
  /** The record on disk after the attempt, when one could be read. */
  readonly holder: CronLeaseRecord | undefined
  readonly state: CronLeaseState
}

export interface CronLeaseClaim {
  /** Command line of a pid; '' when it cannot be read. */
  readonly commandOf: (pid: number) => string
  readonly isAlive: (pid: number) => boolean
  /** Resolved project directory of the claiming daemon. */
  readonly ownerKey: string
  readonly pid: number
}

export interface CronLeaseOptions {
  readonly commandOf?: (pid: number) => string
  readonly isAlive?: (pid: number) => boolean
  readonly ownerKey: string
  readonly pid?: number
}

const LEASE_FILE_MODE = 0o600
/**
 * Exclusive create and the subsequent write are two syscalls, so a claimant can
 * read a lease file the winner has created but not yet filled in. Treating that
 * empty read as abandoned would hand the lease to a second daemon during the
 * exact moment both are starting, so a young unreadable record is refused
 * rather than recovered.
 */
const UNREADABLE_LEASE_GRACE_MS = 2_000

/**
 * Take an exclusive cron lease for `ownerKey`, recovering one abandoned by a
 * dead holder.
 *
 * The lock is an exclusive-create (`wx`) file: the kernel picks the winner, so
 * no amount of concurrent daemons can both observe success. Recovery of a stale
 * lease unlinks and retries the create exactly once — never in a loop, because
 * two recoverers looping would take turns deleting each other's fresh lease and
 * both would eventually believe they hold it.
 */
export function acquireCronLease(path: string, options: CronLeaseOptions): CronLeaseOutcome {
  const claim = leaseClaim(options)
  const created = createLeaseFile(path, claim)
  if (created) return { held: true, holder: created, state: 'acquired' }

  const holder = readCronLease(path)
  if (holder) {
    const state = classifyCronLeaseHolder(holder, claim)
    if (state === 'owned') return { held: true, holder, state: 'owned' }
    if (state === 'refused') return { held: false, holder, state: 'refused' }
  } else if (leaseIsYoungerThanGrace(path)) {
    return { held: false, holder: undefined, state: 'refused' }
  }

  // Classification forks `ps`, so a competing recoverer has ample time to win
  // in between. Unlink only while the abandoned record is still the one on
  // disk; a blind unlink would delete the fresh lease that recoverer just took
  // and both processes would go on believing they hold it. A record that
  // changed under us is left for the next attempt to classify.
  if (!leaseStillOnDisk(path, holder)) {
    return { held: false, holder: readCronLease(path), state: 'refused' }
  }
  try {
    unlinkSync(path)
  } catch (error) {
    if (!isMissingFile(error)) throw error
  }
  const recovered = createLeaseFile(path, claim)
  if (recovered) return { held: true, holder: recovered, state: 'acquired' }
  // Someone else recovered first. Their lease is authoritative from here on.
  return { held: false, holder: readCronLease(path), state: 'refused' }
}

/**
 * Classify a lease we did not create.
 *
 * A holder that is alive but whose command line cannot be read is refused, not
 * recovered: an unreadable `ps` is not evidence that the daemon died, and
 * stealing the lease from a live daemon reintroduces the double-fire this lock
 * exists to prevent.
 */
export function classifyCronLeaseHolder(
  holder: CronLeaseRecord,
  claim: CronLeaseClaim,
): CronLeaseHolderState {
  if (holder.pid === claim.pid && holder.ownerKey === claim.ownerKey) return 'owned'
  if (!claim.isAlive(holder.pid)) return 'stale'
  const observed = claim.commandOf(holder.pid)
  if (!observed || !holder.command) return 'refused'
  // Same pid running a different program: the holder exited and the kernel
  // handed its pid to something unrelated, so the lease really is abandoned.
  return observed === holder.command ? 'refused' : 'stale'
}

/** Release a lease this process holds. Another owner's lease is left alone. */
export function releaseCronLease(path: string, options: CronLeaseOptions): boolean {
  const claim = leaseClaim(options)
  const holder = readCronLease(path)
  if (!holder || holder.pid !== claim.pid || holder.ownerKey !== claim.ownerKey) return false
  try {
    unlinkSync(path)
    return true
  } catch (error) {
    if (isMissingFile(error)) return false
    throw error
  }
}

/** Read the current lease record, or undefined when it is absent or unreadable. */
export function readCronLease(path: string): CronLeaseRecord | undefined {
  let raw: string
  try {
    raw = readFileSync(path, 'utf8')
  } catch {
    return undefined
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return undefined
  }
  if (!isRecord(parsed)) return undefined
  const pid = parsed.pid
  const ownerKey = parsed.owner_key
  if (typeof pid !== 'number' || !Number.isSafeInteger(pid) || pid <= 0) return undefined
  if (typeof ownerKey !== 'string' || !ownerKey) return undefined
  return {
    acquiredAt: typeof parsed.acquired_at === 'string' ? parsed.acquired_at : '',
    command: typeof parsed.command === 'string' ? parsed.command : '',
    ownerKey,
    pid,
  }
}

function leaseClaim(options: CronLeaseOptions): CronLeaseClaim {
  return {
    commandOf: options.commandOf ?? processCommand,
    isAlive: options.isAlive ?? processIsAlive,
    ownerKey: options.ownerKey,
    pid: options.pid ?? process.pid,
  }
}

function createLeaseFile(path: string, claim: CronLeaseClaim): CronLeaseRecord | undefined {
  const record: CronLeaseRecord = {
    acquiredAt: new Date().toISOString(),
    command: claim.commandOf(claim.pid),
    ownerKey: claim.ownerKey,
    pid: claim.pid,
  }
  const encoded = `${JSON.stringify({
    acquired_at: record.acquiredAt,
    command: record.command,
    owner_key: record.ownerKey,
    pid: record.pid,
  })}\n`
  mkdirSync(dirname(path), { recursive: true })
  try {
    writeFileSync(path, encoded, { encoding: 'utf8', flag: 'wx', mode: LEASE_FILE_MODE })
    return record
  } catch (error) {
    if (isAlreadyExists(error)) return undefined
    throw error
  }
}

function leaseStillOnDisk(path: string, classified: CronLeaseRecord | undefined): boolean {
  const current = readCronLease(path)
  if (!classified) return current === undefined
  return (
    current !== undefined &&
    current.pid === classified.pid &&
    current.ownerKey === classified.ownerKey &&
    current.acquiredAt === classified.acquiredAt
  )
}

function leaseIsYoungerThanGrace(path: string): boolean {
  try {
    return Date.now() - statSync(path).mtimeMs < UNREADABLE_LEASE_GRACE_MS
  } catch {
    return false
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isAlreadyExists(error: unknown): boolean {
  return isNodeError(error, 'EEXIST')
}

function isMissingFile(error: unknown): boolean {
  return isNodeError(error, 'ENOENT')
}

function isNodeError(error: unknown, code: string): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === code
}
