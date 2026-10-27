// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Wire and file shape of a lesson shared between machines, either through a
 * repository's `.xerxes/lessons/` folder or through the Hub. Pure and
 * browser-safe: no filesystem, process or network access.
 *
 * Merge rule (identical everywhere, so every transport converges):
 * - the set of ids only grows;
 * - the content of an id never changes (a correction is a new id that lists
 *   the ids it supersedes);
 * - `retired` absorbs `active` and never reverts.
 */

export const SHARED_LESSON_LIMITS = Object.freeze({
  cue: 240,
  lesson: 8000,
  evidence: 1000,
  host: 255,
  agentId: 200,
  reason: 500,
  supersedes: 50,
  /** Most lessons one `lessons.shared.list` reply carries. */
  listLessons: 1000,
  /** Most lessons one `lessons.shared.apply` request carries. */
  applyLessons: 200,
  /** Serialized request or reply ceiling. */
  payloadBytes: 1024 * 1024,
})

/** `lesson-` plus a lowercase RFC 4122 UUID, the id every LessonStore record already uses. Safe as a file name. */
export const SHARED_LESSON_ID_PATTERN = /^lesson-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/u
/** Path-independent repository identity: `git:` plus 32 lowercase hex characters. */
export const REPO_KEY_PATTERN = /^git:[0-9a-f]{32}$/u

export type SharedLessonStatus = 'active' | 'retired'

export interface SharedLessonOrigin {
  /** Host name of the machine where the lesson was first recorded. */
  readonly host: string
  readonly agent_id: string
}

export interface SharedLessonWire {
  readonly id: string
  readonly repo: string
  readonly status: SharedLessonStatus
  readonly cue: string
  readonly lesson: string
  readonly confidence: number
  /** Human-readable provenance of the original evidence (never re-verifiable on another machine). */
  readonly evidence: string
  readonly supersedes: readonly string[]
  readonly origin: SharedLessonOrigin
  /** ISO 8601 (`Date.prototype.toISOString` form). */
  readonly recorded_at: string
  readonly updated_at: string
  /** Present only when `status` is `retired`. */
  readonly retired_reason?: string
}

export type SharedLessonParse =
  | { readonly ok: true; readonly lesson: SharedLessonWire }
  | { readonly ok: false; readonly error: string }

const KEYS = new Set(['id', 'repo', 'status', 'cue', 'lesson', 'confidence', 'evidence', 'supersedes', 'origin', 'recorded_at', 'updated_at', 'retired_reason'])
const ORIGIN_KEYS = new Set(['host', 'agent_id'])

/** Strict validation: unknown keys, oversize text, control characters and non-canonical dates are refused. */
export function parseSharedLessonWire(value: unknown): SharedLessonParse {
  const fail = (error: string): SharedLessonParse => ({ ok: false, error })
  if (!isPlainObject(value)) return fail('a shared lesson must be an object')
  for (const key of Object.keys(value)) if (!KEYS.has(key)) return fail(`unknown shared lesson field: ${key.slice(0, 40)}`)
  const { id, repo, status, cue, lesson, confidence, evidence, supersedes, origin, recorded_at, updated_at, retired_reason } = value
  if (typeof id !== 'string' || !SHARED_LESSON_ID_PATTERN.test(id)) return fail('id must be lesson-<uuid>')
  if (typeof repo !== 'string' || !REPO_KEY_PATTERN.test(repo)) return fail('repo must be git:<32 hex>')
  if (status !== 'active' && status !== 'retired') return fail('status must be active or retired')
  if (!boundedText(cue, SHARED_LESSON_LIMITS.cue, false)) return fail(`cue must be one line of 1-${SHARED_LESSON_LIMITS.cue} characters`)
  if (!boundedText(lesson, SHARED_LESSON_LIMITS.lesson, true)) return fail(`lesson must be 1-${SHARED_LESSON_LIMITS.lesson} characters`)
  if (typeof confidence !== 'number' || !Number.isFinite(confidence) || confidence < 0 || confidence > 1) return fail('confidence must be a number from 0 to 1')
  if (!boundedText(evidence, SHARED_LESSON_LIMITS.evidence, true)) return fail(`evidence must be 1-${SHARED_LESSON_LIMITS.evidence} characters`)
  if (!Array.isArray(supersedes) || supersedes.length > SHARED_LESSON_LIMITS.supersedes
    || !supersedes.every(item => typeof item === 'string' && SHARED_LESSON_ID_PATTERN.test(item) && item !== id)
    || new Set(supersedes).size !== supersedes.length) {
    return fail(`supersedes must list at most ${SHARED_LESSON_LIMITS.supersedes} distinct other lesson ids`)
  }
  if (!isPlainObject(origin)) return fail('origin must be an object')
  for (const key of Object.keys(origin)) if (!ORIGIN_KEYS.has(key)) return fail(`unknown origin field: ${key.slice(0, 40)}`)
  if (!boundedText(origin.host, SHARED_LESSON_LIMITS.host, false)) return fail('origin.host must be one line of text')
  if (!boundedText(origin.agent_id, SHARED_LESSON_LIMITS.agentId, false)) return fail('origin.agent_id must be one line of text')
  if (!isCanonicalIso(recorded_at)) return fail('recorded_at must be an ISO 8601 timestamp')
  if (!isCanonicalIso(updated_at)) return fail('updated_at must be an ISO 8601 timestamp')
  if (status === 'active' && retired_reason !== undefined) return fail('retired_reason is only valid on a retired lesson')
  if (retired_reason !== undefined && !boundedText(retired_reason, SHARED_LESSON_LIMITS.reason, true)) return fail(`retired_reason must be 1-${SHARED_LESSON_LIMITS.reason} characters`)
  return {
    ok: true,
    lesson: {
      id, repo, status, cue, lesson, confidence, evidence, supersedes: [...supersedes] as string[],
      origin: { host: origin.host as string, agent_id: origin.agent_id as string },
      recorded_at, updated_at,
      ...(retired_reason === undefined ? {} : { retired_reason: retired_reason as string }),
    },
  }
}

/**
 * The state that wins when two copies of one id meet. Retired absorbs active;
 * between two retired copies the earlier retirement is kept so the result does
 * not depend on arrival order. Two active copies are the same lesson: keep current.
 */
export function mergeSharedLesson(current: SharedLessonWire | undefined, incoming: SharedLessonWire): SharedLessonWire {
  if (current === undefined) return incoming
  if (current.id !== incoming.id) throw new Error('mergeSharedLesson: ids differ')
  if (current.status === 'retired' && incoming.status === 'retired') {
    return incoming.updated_at < current.updated_at ? incoming : current
  }
  if (current.status === 'retired') return current
  if (incoming.status === 'retired') return incoming
  return current
}

/** Whether `merge(current, incoming)` would change what the holder of `current` stores. */
export function sharedLessonChanges(current: SharedLessonWire | undefined, incoming: SharedLessonWire): boolean {
  return mergeSharedLesson(current, incoming) !== current
}

function boundedText(value: unknown, max: number, multiline: boolean): value is string {
  if (typeof value !== 'string' || !value.trim() || value.length > max) return false
  // eslint-disable-next-line no-control-regex
  return multiline ? !/[\0-\x08\x0b\x0c\x0e-\x1f\x7f]/u.test(value) : !/[\0-\x1f\x7f]/u.test(value)
}

function isCanonicalIso(value: unknown): value is string {
  if (typeof value !== 'string' || value.length > 40) return false
  const time = Date.parse(value)
  return Number.isFinite(time) && new Date(time).toISOString() === value
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
