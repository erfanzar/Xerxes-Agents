// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { HubActor } from '../protocol.js'

/**
 * What the voice subsystem needs from the XERXES supervisor. The Hub supplies
 * the implementation; voice never reaches the supervisor, a workspace or a
 * daemon any other way. Voice can submit, read status and cancel the
 * supervisor's own turn. It cannot approve, confirm or review anything:
 * those routes do not exist on this interface.
 */

export interface SupervisorSubmitOutcome {
  /** False when the supervisor refused the message (for example the actor may not write to the thread). */
  readonly accepted: boolean
  /** True when this idempotency key was already submitted and nothing new started. */
  readonly duplicate: boolean
  readonly messageId?: string
}

export interface SupervisorThreadStatus {
  readonly busy: boolean
  readonly agentsWorking?: number
  /** Facts for the realtime model to relay. Bounded by the caller. */
  readonly summary?: string
}

export interface SupervisorReply {
  readonly id: string
  /** Message id (from `submit`) this reply answers. Absent when the supervisor cannot say. */
  readonly inReplyTo?: string
  /** Supervisor-authored, user-facing text. Never thinking, tool output or another user's private content. */
  readonly text: string
}

export interface SupervisorVoiceBridge {
  /** Whether this actor may read and write the thread. Checked before a voice conversation opens. */
  authorize(threadId: string, actor: HubActor): Promise<boolean>
  /** Submit exactly once per `idempotencyKey`; a repeated key returns `duplicate: true`. */
  submit(threadId: string, text: string, actor: HubActor, idempotencyKey: string): Promise<SupervisorSubmitOutcome>
  status(threadId: string): Promise<SupervisorThreadStatus>
  /** Cancel the supervisor's current turn on this thread. Never stops background agents. */
  cancel(threadId: string, actor: HubActor): Promise<{ readonly cancelled: boolean }>
  onReply(threadId: string, listener: (reply: SupervisorReply) => void): () => void
}

/**
 * Guard in front of any bridge: a submission key is sent to the bridge once.
 * A repeat while the first is in flight shares its result; a repeat afterwards
 * returns the remembered outcome marked `duplicate`. A failed submission is
 * forgotten, so an explicit retry with the same key reaches a bridge that
 * dedupes by key itself; nothing here retries on its own.
 */
export class ExactlyOnceSubmitter {
  private readonly entries = new Map<string, Promise<SupervisorSubmitOutcome>>()

  constructor(private readonly bridge: SupervisorVoiceBridge, private readonly maxKeys = 2048) {}

  async submit(threadId: string, text: string, actor: HubActor, idempotencyKey: string): Promise<SupervisorSubmitOutcome> {
    const key = `${threadId}\0${actor.principal_id}\0${idempotencyKey}`
    const existing = this.entries.get(key)
    if (existing !== undefined) {
      const outcome = await existing
      return { ...outcome, duplicate: true }
    }
    const pending = this.bridge.submit(threadId, text, actor, idempotencyKey)
    this.entries.set(key, pending)
    while (this.entries.size > this.maxKeys) {
      const oldest = this.entries.keys().next().value
      if (oldest === undefined) break
      this.entries.delete(oldest)
    }
    try {
      return await pending
    } catch (error) {
      if (this.entries.get(key) === pending) this.entries.delete(key)
      throw error
    }
  }
}
