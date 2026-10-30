// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useCallback, useLayoutEffect, useRef, useState, type ReactElement, type ReactNode, type RefObject } from 'react'

import type { SelectedSession } from '../../client/hubStore.js'
import { useServices } from '../context.js'
import { createTranscriptScroll } from '../transcriptScroll.js'
import { BlockList } from './BlockList.js'

type TranscriptScroll = ReturnType<typeof createTranscriptScroll>

export interface FollowScroll {
  /** The reader scrolled away from the bottom; new content no longer follows. */
  readonly away: boolean
  readonly toBottom: () => void
  readonly beforePrepend: () => void
}

/**
 * Document-scrolling chat behaviour for one list: open at the bottom, follow
 * new content while the reader is there, never yank a reader who scrolled up,
 * and report it so a "Jump to latest" pill can show. `onMove` receives true
 * when the reader scrolled up near the top (time to load older history).
 */
export function useFollowScroll(list: RefObject<HTMLElement | null>, ready: boolean, deps: readonly unknown[], onMove?: (nearTop: boolean) => void): FollowScroll {
  const follow = useRef<TranscriptScroll | undefined>(undefined)
  const [away, setAway] = useState(false)
  const moveHandler = useRef(onMove)
  moveHandler.current = onMove

  useLayoutEffect(() => {
    follow.current = createTranscriptScroll({
      y: () => window.scrollY,
      height: () => document.documentElement.scrollHeight,
      viewport: () => window.innerHeight,
      top: () => list.current?.getBoundingClientRect().top ?? Infinity,
      blocks: () => Array.from(list.current?.querySelectorAll<HTMLElement>('[data-block-id]') ?? [])
        .filter(element => element.getClientRects().length > 0)
        .map(element => ({ id: element.dataset.blockId!, top: element.getBoundingClientRect().top })),
      scrollTo: y => window.scrollTo(0, y),
    })
    const onScroll = (): void => {
      const current = follow.current
      if (current === undefined) return
      const nearTop = current.moved()
      setAway(!current.pinned())
      moveHandler.current?.(nearTop)
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', onScroll)
      follow.current = undefined
    }
  }, [list])

  useLayoutEffect(() => {
    if (!ready) return
    follow.current?.changed()
    setAway(!(follow.current?.pinned() ?? true))
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the caller lists what changes the content
  }, [ready, ...deps])

  const toBottom = useCallback(() => { follow.current?.toBottom(); setAway(false) }, [])
  const beforePrepend = useCallback(() => { follow.current?.beforePrepend() }, [])
  return { away, toBottom, beforePrepend }
}

/** Floating pill shown while the reader is scrolled up; it names what is waiting below. */
export function JumpToLatest({ visible, waiting = 0, onJump }: { visible: boolean; waiting?: number; onJump: () => void }): ReactElement | null {
  if (!visible) return null
  return (
    <button type="button" className="jump-latest" onClick={onJump}>
      {waiting > 0 ? `${waiting} waiting for you · Jump to latest` : 'Jump to latest'} <span aria-hidden="true">↓</span>
    </button>
  )
}

/**
 * The task conversation. `children` render after the last block (the todo
 * list and the questions/permissions waiting for an answer), so opening the
 * task at the bottom shows them right above the composer.
 */
export function TaskTranscript({ selected, children }: { selected: SelectedSession; children?: ReactNode }): ReactElement {
  const { controller, store } = useServices()
  const root = useRef<HTMLDivElement>(null)
  const loading = useRef(false)
  const prepend = useRef<() => void>(() => {})

  const loadOlder = useCallback(async (automatic = false): Promise<void> => {
    const current = store.getSnapshot().selected
    if (loading.current || !current || current.workspaceId !== selected.workspaceId || current.sessionId !== selected.sessionId
      || current.phase !== 'ready' || !current.hasOlder || current.loadingOlder || (automatic && current.olderError !== undefined)) return
    loading.current = true
    prepend.current()
    try { await controller.loadOlder() }
    finally { loading.current = false }
  }, [controller, store, selected.workspaceId, selected.sessionId])

  const follow = useFollowScroll(
    root,
    selected.phase === 'ready',
    [selected.older, selected.tail.blocks, selected.tail.pending, selected.tail.todos],
    nearTop => { if (nearTop) void loadOlder(true) },
  )
  prepend.current = follow.beforePrepend

  const blocks = [...selected.older, ...selected.tail.blocks]
  return (
    <div ref={root} className="task-transcript" aria-label="Task conversation" aria-busy={selected.phase === 'loading'}>
      {selected.hasOlder && selected.phase === 'ready' ? (
        <div className="older">
          {selected.olderError === undefined ? null : <p role="alert" className="field-error">{selected.olderError}</p>}
          <button type="button" className="btn btn-quiet" disabled={selected.loadingOlder} onClick={() => { void loadOlder() }}>
            {selected.loadingOlder ? 'Loading…' : 'Load earlier messages'}
          </button>
        </div>
      ) : null}
      {selected.phase === 'ready' && blocks.length === 0 ? <p className="empty">Nothing recorded for this task yet.</p> : null}
      <BlockList blocks={blocks} prefix="task" />
      {children}
      <JumpToLatest visible={follow.away} waiting={selected.tail.pending.length} onJump={follow.toBottom} />
    </div>
  )
}
