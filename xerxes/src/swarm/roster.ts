// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import {
  COORDINATOR_NAME,
  type SwarmMemberRecord,
  type SwarmMemberRole,
  type SwarmMemberSnapshot,
} from './types.js'

export const MEMBER_NAME_PATTERN = /^[a-z0-9][a-z0-9_-]{0,23}$/
const NARRATOR_NAME_PATTERN = /^narrator(?:-\d+)?$/
const MONITOR_NAME_PATTERN = /^monitor(?:-\d+)?$/
const NUMBERED_SUFFIX_PATTERN = /^([a-z0-9][a-z0-9_-]*)-(\d+)$/

export function isNarratorName(name: string): boolean {
  return NARRATOR_NAME_PATTERN.test(name)
}

export function isMonitorName(name: string): boolean {
  return MONITOR_NAME_PATTERN.test(name)
}

export function roleForMemberName(name: string): SwarmMemberRole {
  if (name === COORDINATOR_NAME) return 'coordinator'
  if (isNarratorName(name)) return 'narrator'
  if (isMonitorName(name)) return 'monitor'
  return 'worker'
}

export function isReservedMemberName(name: string): boolean {
  return name === 'all'
    || name === 'user'
    || name === COORDINATOR_NAME
    || name.startsWith(`${COORDINATOR_NAME}-`)
}

export function assertValidWorkerName(name: string, field = 'name'): string {
  const trimmed = name.trim()
  if (!MEMBER_NAME_PATTERN.test(trimmed)) {
    throw new ValidationError(
      field,
      'worker name must be 1-24 chars of lowercase [a-z0-9_-] starting with a letter or digit',
      name,
    )
  }
  if (isReservedMemberName(trimmed)) {
    throw new ValidationError(
      field,
      `"${trimmed}" is reserved; choose a role-descriptive worker name`,
      name,
    )
  }
  return trimmed
}

export function suggestReplacementName(existingName: string, taken: ReadonlySet<string>): string {
  const match = NUMBERED_SUFFIX_PATTERN.exec(existingName)
  const stem = match?.[1] ?? existingName
  for (let index = 2; index <= 999; index += 1) {
    const candidate = `${stem}-${index}`
    if (candidate.length <= 24 && !taken.has(candidate) && !isReservedMemberName(candidate)) {
      return candidate
    }
  }
  return `${stem.slice(0, 20)}-2`
}

export interface PlannedWorkerSlice {
  readonly name: string
  readonly group?: string
  readonly effectiveBrief: string
}

export function expandPlannedWorkers(params: {
  readonly baseName: string
  readonly brief: string
  readonly n?: number
  readonly tasks?: readonly string[]
  readonly existingNames: ReadonlySet<string>
  readonly existingGroups: ReadonlySet<string>
  readonly killedNames: ReadonlySet<string>
}): PlannedWorkerSlice[] {
  const base = assertValidWorkerName(params.baseName, 'name')
  const brief = params.brief.trim()
  if (!brief) {
    throw new ValidationError('brief', 'must be a non-empty assignment brief')
  }

  const hasMultiCopies = params.n !== undefined && params.n > 1
  const hasTasks = params.tasks !== undefined
  if (params.n !== undefined && (!Number.isInteger(params.n) || params.n < 1)) {
    throw new ValidationError('n', 'must be a positive integer', params.n)
  }
  if (hasMultiCopies && hasTasks) {
    throw new ValidationError('tasks', 'cannot set both n > 1 and tasks on the same add call')
  }
  if (hasTasks && params.tasks.length === 0) {
    throw new ValidationError('tasks', 'must contain at least one slice target when provided')
  }

  const assertUnused = (candidate: string): void => {
    if (params.killedNames.has(candidate)) {
      const replacement = suggestReplacementName(candidate, params.existingNames)
      throw new ValidationError(
        'name',
        `worker "${candidate}" was previously killed and killed names cannot be reused; use "${replacement}" instead`,
        candidate,
      )
    }
    if (params.existingNames.has(candidate) || params.existingGroups.has(candidate)) {
      throw new ValidationError('name', `member or group "${candidate}" already exists on the roster`, candidate)
    }
  }

  if (hasTasks) {
    assertUnused(base)
    const slices: PlannedWorkerSlice[] = []
    for (const [index, rawTask] of params.tasks.entries()) {
      const taskText = rawTask.trim()
      if (!taskText) {
        throw new ValidationError('tasks', `task slice at index ${index} must not be empty`)
      }
      const memberName = assertValidWorkerName(`${base}-${index + 1}`, 'name')
      assertUnused(memberName)
      slices.push({
        name: memberName,
        group: base,
        effectiveBrief: `${brief}\n\nAssigned slice (${index + 1}/${params.tasks.length}): ${taskText}`,
      })
    }
    return slices
  }

  if (hasMultiCopies && params.n !== undefined) {
    assertUnused(base)
    const slices: PlannedWorkerSlice[] = []
    for (let index = 1; index <= params.n; index += 1) {
      const memberName = assertValidWorkerName(`${base}-${index}`, 'name')
      assertUnused(memberName)
      slices.push({
        name: memberName,
        group: base,
        effectiveBrief: `${brief}\n\nCopy ${index} of ${params.n} in group "${base}".`,
      })
    }
    return slices
  }

  assertUnused(base)
  return [{ name: base, effectiveBrief: brief }]
}

/**
 * Resolve a target name to either a single member or all living members of a group.
 */
export function resolveTargetMembers(
  members: ReadonlyMap<string, SwarmMemberRecord>,
  targetName: string,
  options: { readonly includeKilled?: boolean } = {},
): SwarmMemberRecord[] {
  const exact = members.get(targetName)
  if (exact) {
    if (!options.includeKilled && exact.killed) {
      return [exact]
    }
    return [exact]
  }
  const groupMatches: SwarmMemberRecord[] = []
  for (const member of members.values()) {
    const numberedMatch = NUMBERED_SUFFIX_PATTERN.exec(member.name)
    const matchesGroup = member.group === targetName || numberedMatch?.[1] === targetName
    if (!matchesGroup) continue
    if (!options.includeKilled && member.killed) continue
    groupMatches.push(member)
  }
  return groupMatches
}

export function memberMatchesGroupPrefix(
  member: SwarmMemberRecord,
  mentionToken: string,
  allMemberNames: ReadonlySet<string>,
): boolean {
  if (member.group === mentionToken) return true
  if (allMemberNames.has(mentionToken)) return false
  const numbered = NUMBERED_SUFFIX_PATTERN.exec(member.name)
  return numbered?.[1] === mentionToken
}

export function canManageWorker(caller: string, target: SwarmMemberRecord): boolean {
  if (caller === COORDINATOR_NAME) return true
  return target.added_by === caller
}

export function toMemberSnapshot(
  record: SwarmMemberRecord,
  extra?: {
    readonly conversationId?: string
    readonly currentActivity?: string
  },
): SwarmMemberSnapshot {
  const activeSummary = record.state === 'working' && !record.killed && extra?.currentActivity?.trim()
    ? extra.currentActivity.trim()
    : undefined
  return {
    name: record.name,
    ...(record.group === undefined ? {} : { group: record.group }),
    role: record.role,
    ...(record.added_by === undefined ? {} : { added_by: record.added_by }),
    model: record.model,
    model_display_name: record.model_display_name,
    workspace: record.workspace,
    ...(record.workspace_path ? { workspace_path: record.workspace_path } : {}),
    channels: [...record.channels],
    state: record.state,
    killed: record.killed,
    ...(extra?.conversationId ? { conversation_id: extra.conversationId } : {}),
    ...(activeSummary === undefined ? {} : { current_activity: activeSummary }),
    ...(record.parked_since === undefined ? {} : { parked_since: record.parked_since }),
    ...(record.mentions_only === undefined ? {} : { mentions_only: record.mentions_only }),
    last_read_seq: record.last_read_seq,
    ...(record.last_post_seq === undefined ? {} : { last_post_seq: record.last_post_seq }),
    ...(record.last_post_time === undefined ? {} : { last_post_time: record.last_post_time }),
    ...(record.error === undefined ? {} : { error: record.error }),
  }
}
