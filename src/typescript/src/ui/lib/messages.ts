// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { Msg, Role } from '../types.js'

import {
  appendToolShelfMessage,
  type TrailDetailVisibility,
  trailHasRenderableContent,
  trailHasVisibleContent
} from './liveProgress.js'

export const appendTranscriptMessage = (prev: Msg[], msg: Msg): Msg[] => appendToolShelfMessage(prev, msg)

/** Roll back one optimistic row without deleting an older equal-text turn. */
export const removeTranscriptMessage = (prev: readonly Msg[], target: Msg): Msg[] =>
  prev.filter(message => message !== target)

/**
 * Native skill slash commands submit their model turn inside the daemon. Turn
 * the optimistic slash bookkeeping row into the authored user row without
 * duplicating it; direct handler calls still receive one visible user row.
 */
export const promoteSlashToUserMessage = (prev: readonly Msg[], command: string): Msg[] => {
  for (let index = prev.length - 1; index >= 0; index -= 1) {
    const item = prev[index]

    if (item?.kind === 'slash' && item.text === command) {
      const next = [...prev]

      next[index] = { role: 'user', text: command }

      return next
    }
  }

  return [...prev, { role: 'user', text: command }]
}

/**
 * The intro is a startup-only screen. Keep it in local session state for
 * metadata refreshes, but omit it as soon as the transcript has real content.
 * A slash row only records a local command dispatch: it is not content on its
 * own, so opening an interactive command must leave the startup screen intact.
 */
export const hasTranscriptContent = (items: readonly Msg[]): boolean =>
  items.some(
    item => item.kind !== 'intro' && item.kind !== 'slash' && (item.kind !== 'trail' || trailHasRenderableContent(item))
  )

export const visibleTranscriptMessages = (items: readonly Msg[]): Msg[] => {
  const renderable = (item: Msg) => item.kind !== 'trail' || trailHasRenderableContent(item)

  if (!hasTranscriptContent(items)) {
    return items.filter(item => item.kind !== 'slash' && renderable(item))
  }

  return items.filter(item => item.kind !== 'intro' && renderable(item))
}

/** Remove detail trails that are invisible in the current /details mode. */
export const visibleTranscriptDetails = (items: readonly Msg[], visibility: TrailDetailVisibility): Msg[] =>
  items.filter(item => item.kind !== 'trail' || trailHasVisibleContent(item, visibility))

export const upsert = (prev: Msg[], role: Role, text: string): Msg[] =>
  prev.at(-1)?.role === role ? [...prev.slice(0, -1), { role, text }] : [...prev, { role, text }]
