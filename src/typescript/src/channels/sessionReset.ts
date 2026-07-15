// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const ResetTrigger = {
  TIMEOUT: 'timeout',
  MSG_COUNT: 'msg_count',
  MESSAGE_COUNT: 'msg_count',
  MANUAL: 'manual',
} as const

export type ResetTrigger = (typeof ResetTrigger)[keyof typeof ResetTrigger]

export interface SessionResetPolicy {
  readonly messageCount: number
  readonly timeoutMinutes: number
  readonly trigger: ResetTrigger
}

export interface SessionResetPolicyInput {
  /** Python/YAML-compatible alias for messageCount. */
  readonly msg_count?: number
  readonly messageCount?: number
  /** Python/YAML-compatible alias for timeoutMinutes. */
  readonly timeout_minutes?: number
  readonly timeoutMinutes?: number
  readonly trigger?: ResetTrigger
}

export interface ShouldResetOptions {
  readonly lastMessageAt?: Date
  readonly manualRequest?: boolean
  readonly messageCount: number
  readonly now?: Date
}

/** Normalize a policy once at its configuration boundary. */
export function createSessionResetPolicy(input: SessionResetPolicyInput = {}): SessionResetPolicy {
  const messageCount = input.messageCount ?? input.msg_count ?? 50
  const timeoutMinutes = input.timeoutMinutes ?? input.timeout_minutes ?? 60
  if (!Number.isSafeInteger(messageCount) || messageCount < 1) {
    throw new RangeError('messageCount must be a positive safe integer')
  }
  if (!Number.isFinite(timeoutMinutes) || timeoutMinutes < 0) {
    throw new RangeError('timeoutMinutes must be a non-negative finite number')
  }
  const trigger = input.trigger ?? ResetTrigger.MANUAL
  if (!Object.values(ResetTrigger).includes(trigger)) {
    throw new TypeError('unknown reset trigger: ' + String(trigger))
  }
  return { trigger, timeoutMinutes, messageCount }
}

/**
 * Decide whether an external-chat session needs a fresh conversation.
 *
 * An explicit /new or /reset always wins. Message-count thresholds are
 * inclusive, matching the original public policy behavior.
 */
export function shouldReset(
  policy: SessionResetPolicy | SessionResetPolicyInput,
  options: ShouldResetOptions,
): boolean {
  if (options.manualRequest) return true
  const normalized = createSessionResetPolicy(policy)
  if (!Number.isSafeInteger(options.messageCount) || options.messageCount < 0) {
    throw new RangeError('messageCount must be a non-negative safe integer')
  }
  if (normalized.trigger === ResetTrigger.MANUAL) return false
  if (normalized.trigger === ResetTrigger.MESSAGE_COUNT) {
    return options.messageCount >= normalized.messageCount
  }
  if (options.lastMessageAt === undefined) return false
  const last = dateMilliseconds(options.lastMessageAt, 'lastMessageAt')
  const now = dateMilliseconds(options.now ?? new Date(), 'now')
  return last < now - normalized.timeoutMinutes * 60_000
}

function dateMilliseconds(value: Date, name: string): number {
  const timestamp = value.getTime()
  if (!Number.isFinite(timestamp)) throw new TypeError(name + ' must be a valid Date')
  return timestamp
}
