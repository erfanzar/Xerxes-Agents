// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useCallback, useRef, useState } from 'react'

import {
  appendQueuedMessage,
  type QueuedMessage,
  queuedMessage,
  queuedMessageDisplays,
  shiftQueuedMessage
} from '../domain/queuedMessage.js'

// Mutates `arr` in place; returned reference is the same input array, kept
// so callers can chain. Use `Array.prototype.toSpliced` if you need a copy.
export function removeAtInPlace<T>(arr: T[], i: number): T[] {
  if (i < 0 || i >= arr.length) {
    return arr
  }

  arr.splice(i, 1)

  return arr
}

export function useQueue() {
  const queueRef = useRef<QueuedMessage[]>([])
  const activeSessionRef = useRef<null | string>(null)
  const sessionQueuesRef = useRef(new Map<string, QueuedMessage[]>())
  const heldForSwitchRef = useRef(new Set<QueuedMessage>())
  const [queuedDisplay, setQueuedDisplay] = useState<string[]>([])
  const queueEditRef = useRef<number | null>(null)
  const [queueEditIdx, setQueueEditIdx] = useState<number | null>(null)

  const syncQueue = useCallback(() => setQueuedDisplay(queuedMessageDisplays(queueRef.current)), [])

  const setQueueEdit = useCallback((idx: number | null) => {
    queueEditRef.current = idx
    setQueueEditIdx(idx)
  }, [])

  /**
   * Bind the visible queue to one live session.
   *
   * Follow-ups are authored for the session that was active when they were
   * queued. Switching tabs must not drain them into the newly selected
   * session, so keep inactive queues in-memory and restore them on return.
   * Messages entered before the first session exists remain attached to that
   * first session, preserving startup pre-queue behavior.
   */
  const activateSessionQueue = useCallback(
    (sessionId: string) => {
      const previousSessionId = activeSessionRef.current
      const held = heldForSwitchRef.current
      heldForSwitchRef.current = new Set()

      if (previousSessionId === sessionId) {
        return
      }

      if (previousSessionId) {
        // A prompt typed while this switch was in flight was authored for
        // the session being opened, not the one being left behind.
        const carried = queueRef.current.filter(message => held.has(message))
        sessionQueuesRef.current.set(previousSessionId, queueRef.current.filter(message => !held.has(message)))
        queueRef.current = [...(sessionQueuesRef.current.get(sessionId) ?? []), ...carried]
      }

      activeSessionRef.current = sessionId
      sessionQueuesRef.current.delete(sessionId)
      setQueueEdit(null)
      syncQueue()
    },
    [setQueueEdit, syncQueue]
  )

  const holdForSwitch = useCallback(
    (message: QueuedMessage) => {
      heldForSwitchRef.current.add(message)
      queueRef.current = appendQueuedMessage(queueRef.current, message)
      syncQueue()
    },
    [syncQueue]
  )

  // A failed or refused switch leaves the user on the session they were in,
  // with the held prompt visible in its queue. Still marked held, it would be
  // carried off by whichever unrelated switch came next.
  const releaseSwitchHold = useCallback(() => {
    heldForSwitchRef.current = new Set()
  }, [])

  const enqueue = useCallback(
    (submitText: string, displayText = submitText) => {
      queueRef.current = appendQueuedMessage(queueRef.current, queuedMessage(displayText, submitText))
      syncQueue()
    },
    [syncQueue]
  )

  const dequeue = useCallback(() => {
    const { message, rest } = shiftQueuedMessage(queueRef.current)

    queueRef.current = rest
    syncQueue()

    return message
  }, [syncQueue])

  const replaceQ = useCallback(
    (i: number, submitText: string, displayText = submitText) => {
      queueRef.current[i] = { ...queuedMessage(displayText, submitText), images: queueRef.current[i]?.images }
      syncQueue()
    },
    [syncQueue]
  )

  const removeQ = useCallback(
    (i: number) => {
      const before = queueRef.current.length

      removeAtInPlace(queueRef.current, i)

      if (queueRef.current.length !== before) {
        syncQueue()
      }
    },
    [syncQueue]
  )

  return {
    activateSessionQueue,
    dequeue,
    enqueue,
    holdForSwitch,
    queueEditIdx,
    queueEditRef,
    queueRef,
    queuedDisplay,
    releaseSwitchHold,
    removeQ,
    replaceQ,
    setQueueEdit,
    syncQueue
  }
}
