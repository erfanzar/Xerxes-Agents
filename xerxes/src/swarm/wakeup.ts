// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  memberCanSeeChannel,
  memberExplicitlySubscribedToChannel,
} from './channels.js'
import { memberMatchesGroupPrefix } from './roster.js'
import {
  COORDINATOR_NAME,
  WILDCARD_CHANNEL,
  type SwarmBoardMessage,
  type SwarmMemberRecord,
  type SwarmWorkerErrorInfo,
} from './types.js'

const QUOTA_OR_CAPACITY_PATTERN = /\b429\b|quota|rate[\s_-]?limit|too many requests|exhausted|capacity|overloaded|insufficient_quota/iu

/**
 * Detect provider quota, rate-limit, and capacity exhaustion errors so the
 * coordinator can switch the affected worker's model in place via `swarm{action:"model"}`.
 */
export function isQuotaOrCapacityError(message: string): boolean {
  return QUOTA_OR_CAPACITY_PATTERN.test(message)
}

export function doesMessageWakeCoordinator(
  message: SwarmBoardMessage,
  since: number,
  mentionsOnly: boolean,
): boolean {
  if (message.seq <= since) return false
  if (message.agent === COORDINATOR_NAME) return false
  if (!mentionsOnly) return true
  if (message.broadcast) return true
  if (message.mentions.includes(COORDINATOR_NAME)) return true
  return message.channel_mentions.length > 0
}

export function doesMessageWakeWorker(
  member: SwarmMemberRecord,
  message: SwarmBoardMessage,
  since: number,
  mentionsOnly: boolean,
  allMemberNames: ReadonlySet<string>,
): boolean {
  if (member.killed) return false
  if (message.seq <= since) return false
  if (message.agent === member.name) return false
  if (!memberCanSeeChannel(member.role, member.channels, message.channel)) {
    return false
  }
  if (!mentionsOnly) {
    return true
  }
  if (message.broadcast) {
    return true
  }
  if (message.mentions.includes(member.name)) {
    return true
  }
  for (const mention of message.mentions) {
    if (memberMatchesGroupPrefix(member, mention, allMemberNames)) {
      return true
    }
  }
  for (const channelMention of message.channel_mentions) {
    if (memberExplicitlySubscribedToChannel(member.role, member.channels, channelMention)) {
      return true
    }
  }
  return false
}

export function doesMessageReviveStoppedWorker(
  member: SwarmMemberRecord,
  message: SwarmBoardMessage,
  allMemberNames: ReadonlySet<string>,
): boolean {
  if (member.killed || member.state !== 'stopped') return false
  if (message.agent === member.name) return false
  if (!memberCanSeeChannel(member.role, member.channels, message.channel)) return false
  if (message.broadcast) return true
  if (message.mentions.includes(member.name)) return true
  for (const mention of message.mentions) {
    if (memberMatchesGroupPrefix(member, mention, allMemberNames)) {
      return true
    }
  }
  for (const channelMention of message.channel_mentions) {
    if (memberExplicitlySubscribedToChannel(member.role, member.channels, channelMention)) {
      return true
    }
  }
  return false
}

export function formatBoardWakeNotice(head: number, lastReadSeq: number): string {
  return `[board] the latest message is #${head} (you have read up to #${lastReadSeq}).`
}

export function formatWorkerErrorWakeNotice(err: SwarmWorkerErrorInfo): string {
  if (err.quota_exhausted) {
    return `[swarm] Worker "${err.member_name}" failed with a quota/capacity error: ${err.message}. Call swarm{action:"models"} to inspect available models and call swarm{action:"model", name:"${err.member_name}", model:"<model>", reason:"quota exhausted"} to switch it in place while preserving its conversation (never kill and add a replacement).`
  }
  return `[swarm] Worker "${err.member_name}" failed and is now waiting: ${err.message}. Inspect the worker with read_board or swarm{action:"status"} and decide whether to retry, switch models via swarm{action:"model"}, or intervene.`
}

export type CoordinatorWaitOutcome =
  | { readonly kind: 'board'; readonly woke: true; readonly message: SwarmBoardMessage }
  | { readonly kind: 'worker_error'; readonly woke: true; readonly worker_error: SwarmWorkerErrorInfo }
  | { readonly kind: 'steered'; readonly woke: false; readonly steered: true }

interface PendingCoordinatorWaiter {
  readonly since: number
  readonly mentionsOnly: boolean
  readonly resolve: (result: CoordinatorWaitOutcome) => void
  readonly cleanup: () => void
}

/**
 * Event-driven zero-polling wait hub for `coordinator` calls to `swarm_wait`.
 */
export class CoordinatorWaitHub {
  private readonly waiters = new Set<PendingCoordinatorWaiter>()

  get hasActiveWaiters(): boolean {
    return this.waiters.size > 0
  }

  waitForMessage(params: {
    readonly since: number
    readonly mentionsOnly: boolean
    readonly signal?: AbortSignal
    readonly wakeSignal?: AbortSignal
  }): Promise<CoordinatorWaitOutcome> {
    if (params.signal?.aborted) {
      return Promise.reject(new DOMException('swarm_wait cancelled', 'AbortError'))
    }
    if (params.wakeSignal?.aborted) {
      return Promise.resolve({ kind: 'steered', woke: false, steered: true })
    }

    return new Promise((resolve, reject) => {
      const onAbort = (): void => {
        waiter.cleanup()
        reject(new DOMException('swarm_wait cancelled', 'AbortError'))
      }
      const onSteer = (): void => {
        waiter.cleanup()
        resolve({ kind: 'steered', woke: false, steered: true })
      }
      const waiter: PendingCoordinatorWaiter = {
        since: params.since,
        mentionsOnly: params.mentionsOnly,
        resolve: (outcome) => {
          waiter.cleanup()
          resolve(outcome)
        },
        cleanup: () => {
          this.waiters.delete(waiter)
          params.signal?.removeEventListener('abort', onAbort)
          params.wakeSignal?.removeEventListener('abort', onSteer)
        },
      }

      params.signal?.addEventListener('abort', onAbort, { once: true })
      params.wakeSignal?.addEventListener('abort', onSteer, { once: true })
      this.waiters.add(waiter)
    })
  }

  notifyMessage(message: SwarmBoardMessage): boolean {
    let consumed = false
    for (const waiter of [...this.waiters]) {
      if (doesMessageWakeCoordinator(message, waiter.since, waiter.mentionsOnly)) {
        consumed = true
        waiter.resolve({ kind: 'board', woke: true, message })
      }
    }
    return consumed
  }

  notifyWorkerError(err: SwarmWorkerErrorInfo): boolean {
    if (this.waiters.size === 0) return false
    const snapshot: SwarmWorkerErrorInfo = {
      member_name: err.member_name,
      message: err.message,
      quota_exhausted: err.quota_exhausted,
    }
    let consumed = false
    for (const waiter of [...this.waiters]) {
      consumed = true
      waiter.resolve({ kind: 'worker_error', woke: true, worker_error: snapshot })
    }
    return consumed
  }

  dispose(): void {
    for (const waiter of [...this.waiters]) {
      waiter.cleanup()
    }
  }
}

export function coordinatorDefaultChannels(): string[] {
  return [WILDCARD_CHANNEL]
}
