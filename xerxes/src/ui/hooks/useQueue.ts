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
  const [queuedDisplay, setQueuedDisplay] = useState<string[]>([])
  const queueEditRef = useRef<number | null>(null)
  const [queueEditIdx, setQueueEditIdx] = useState<number | null>(null)

  const syncQueue = useCallback(() => setQueuedDisplay(queuedMessageDisplays(queueRef.current)), [])

  const setQueueEdit = useCallback((idx: number | null) => {
    queueEditRef.current = idx
    setQueueEditIdx(idx)
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
      queueRef.current[i] = queuedMessage(displayText, submitText)
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
    dequeue,
    enqueue,
    queueEditIdx,
    queueEditRef,
    queueRef,
    queuedDisplay,
    removeQ,
    replaceQ,
    setQueueEdit,
    syncQueue
  }
}
