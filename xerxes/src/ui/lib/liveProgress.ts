// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { Msg, TodoItem } from '../types.js'

export const countPendingTodos = (todos: readonly TodoItem[]) =>
  todos.filter(todo => todo.status === 'in_progress' || todo.status === 'pending').length

export const isTodoDone = (todos: readonly TodoItem[]) =>
  todos.length > 0 && todos.every(todo => todo.status === 'completed' || todo.status === 'cancelled')

export const isToolShelfMessage = (msg: Msg | undefined) =>
  Boolean(msg?.kind === 'trail' && !msg.text && !msg.thinking?.trim() && msg.tools?.length)

export const canHoldToolShelf = (msg: Msg | undefined) =>
  Boolean(msg?.kind === 'trail' && !msg.text && (msg.thinking?.trim() || msg.tools?.length))

export interface TrailDetailVisibility {
  subagents: boolean
  thinking: boolean
  tools: boolean
}

export const messageHasVisibleDetails = (msg: Msg | undefined, visibility: TrailDetailVisibility) =>
  Boolean(msg && ((visibility.thinking && msg.thinking?.trim()) || (visibility.tools && msg.tools?.length)))

export const trailHasVisibleContent = (msg: Msg | undefined, visibility: TrailDetailVisibility) =>
  Boolean(msg?.kind === 'trail' && messageHasVisibleDetails(msg, visibility))

/** Keep transcript virtualization aligned with what MessageLine can paint. */
export const trailHasRenderableContent = (msg: Msg | undefined) =>
  trailHasVisibleContent(msg, { subagents: false, thinking: true, tools: true })

export const mergeToolShelfInto = (target: Msg, source: Msg): Msg => ({
  ...target,
  tools: [...(target.tools ?? []), ...(source.tools ?? [])]
})

/** Remove only the archived occurrences from the cumulative live tool list. */
export const unarchivedToolLines = (segments: readonly Msg[], liveTools: readonly string[]) => {
  const archivedCounts = new Map<string, number>()

  for (const line of segments.flatMap(segment => segment.tools ?? [])) {
    archivedCounts.set(line, (archivedCounts.get(line) ?? 0) + 1)
  }

  return liveTools.filter(line => {
    const remaining = archivedCounts.get(line) ?? 0

    if (remaining <= 0) {
      return true
    }

    archivedCounts.set(line, remaining - 1)

    return false
  })
}

export const appendToolShelfMessage = (prev: readonly Msg[], msg: Msg): Msg[] => {
  if (!isToolShelfMessage(msg)) {
    return [...prev, msg]
  }

  // A tool result belongs to the phase immediately above it. Looking past a
  // newer thinking row for an older tool shelf reorders the transcript into
  // "all tools under the first thought", which makes each later reasoning
  // phase appear to replace the one before it. Only merge with the tail;
  // barriers and newer phases therefore keep their chronological boundary.
  const tailIndex = prev.length - 1
  const tail = prev[tailIndex]

  if (canHoldToolShelf(tail)) {
    const next = [...prev]

    next[tailIndex] = mergeToolShelfInto(tail!, msg)

    return next
  }

  return [...prev, msg]
}
