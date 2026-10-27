// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isAbsolute } from 'node:path'
import { LessonSharing, LessonSyncError } from '../memory/lessonSharing.js'
import { parseSharedLessonWire, REPO_KEY_PATTERN, SHARED_LESSON_ID_PATTERN, SHARED_LESSON_LIMITS } from '../memory/sharedLessonWire.js'
import type { JsonRpcPayload } from '../protocol/jsonRpc.js'

/** Authorization happens before dispatch; validation here applies equally to local and signed callers. */
export async function sharedLessonsRpc(method: string, params: Readonly<Record<string, unknown>>, sharing?: LessonSharing): Promise<JsonRpcPayload> {
  const fail = (code: string, error: string): JsonRpcPayload => ({ ok: false, code, error })
  if (process.env.XERXES_LESSON_SYNC === 'off') return fail('disabled', 'Lesson sharing is disabled by XERXES_LESSON_SYNC=off.')
  if (!sharing) return fail('disabled', 'Lesson sharing is not configured by this daemon host.')
  const apply = method === 'lessons.shared.apply'
  const keys = apply ? ['project_dir', 'repo', 'lessons'] : ['project_dir', 'after_id']
  if (Object.keys(params).some(key => !keys.includes(key))) return fail('invalid_request', 'Unknown lesson sharing parameter.')
  if (typeof params.project_dir !== 'string' || !isAbsolute(params.project_dir)) return fail('invalid_request', 'project_dir must be an absolute existing project directory.')
  if (Buffer.byteLength(JSON.stringify(params)) > SHARED_LESSON_LIMITS.payloadBytes) return fail('invalid_request', 'Lesson sharing requests are limited to 1 MiB.')
  try {
    if (!apply) {
      if (params.after_id !== undefined && (typeof params.after_id !== 'string' || !SHARED_LESSON_ID_PATTERN.test(params.after_id))) return fail('invalid_request', 'after_id must be a lesson id when supplied.')
      const snapshot = await sharing.snapshot(params.project_dir, { markShared: false })
      const candidates = snapshot.lessons.filter(row => params.after_id === undefined || row.id > String(params.after_id))
        .sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0)
      const lessons = []
      // Leave room for the non-null cursor and the JSON-RPC envelope.
      let bytes = Buffer.byteLength(JSON.stringify({ ok: true, repo: snapshot.repo, lessons: [], next_after_id: null })) + 1024
      for (const lesson of candidates.slice(0, SHARED_LESSON_LIMITS.listLessons)) {
        const size = Buffer.byteLength(JSON.stringify(lesson)) + (lessons.length ? 1 : 0)
        if (bytes + size > SHARED_LESSON_LIMITS.payloadBytes) break
        lessons.push(lesson); bytes += size
      }
      const next_after_id = lessons.length < candidates.length ? lessons.at(-1)!.id : null
      await sharing.markShared(lessons)
      return { ok: true, repo: snapshot.repo, lessons, next_after_id }
    }
    if (typeof params.repo !== 'string' || !REPO_KEY_PATTERN.test(params.repo) || !Array.isArray(params.lessons) || params.lessons.length > SHARED_LESSON_LIMITS.applyLessons) {
      return fail('invalid_request', 'repo must be git:<32 hex>; lessons must be an array of at most 200 shared lessons.')
    }
    for (const value of params.lessons) {
      const parsed = parseSharedLessonWire(value)
      if (!parsed.ok) return fail('invalid_request', parsed.error)
      if (parsed.lesson.repo !== params.repo) return fail('repo_mismatch', 'Every lesson must belong to the requested repository.')
    }
    const repo = await sharing.repoForWorkspace(params.project_dir)
    if (repo !== params.repo) return fail('repo_mismatch', 'The requested repository does not match this workspace.')
    return { ok: true, ...await sharing.apply(params.project_dir, params.lessons, 'hub') }
  } catch (error) {
    return fail(error instanceof LessonSyncError ? error.code : 'invalid_request', error instanceof Error ? error.message : 'Lesson sharing failed.')
  }
}
