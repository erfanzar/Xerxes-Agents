// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useCallback, useEffect, useLayoutEffect, useRef, type ReactElement } from 'react'

import type { SelectedSession } from '../../client/hubStore.js'
import { useServices } from '../context.js'
import { createTranscriptScroll } from '../transcriptScroll.js'
import { BlockList } from './BlockList.js'

export function TaskTranscript({ selected }: { selected: SelectedSession }): ReactElement {
  const { controller, store } = useServices()
  const root = useRef<HTMLDivElement>(null)
  const scroll = useRef<ReturnType<typeof createTranscriptScroll> | undefined>(undefined)
  const loading = useRef(false)

  useLayoutEffect(() => {
    scroll.current = createTranscriptScroll({
      y: () => window.scrollY,
      height: () => document.documentElement.scrollHeight,
      viewport: () => window.innerHeight,
      top: () => root.current?.getBoundingClientRect().top ?? Infinity,
      blocks: () => Array.from(root.current?.querySelectorAll<HTMLElement>('[data-block-id]') ?? [])
        .filter(element => element.getClientRects().length > 0)
        .map(element => ({ id: element.dataset.blockId!, top: element.getBoundingClientRect().top })),
      scrollTo: y => window.scrollTo(0, y),
    })
    return () => { scroll.current = undefined }
  }, [])

  useLayoutEffect(() => {
    if (selected.phase === 'ready') scroll.current?.changed()
  }, [selected.phase, selected.older, selected.tail.blocks])

  const loadOlder = useCallback(async (automatic = false): Promise<void> => {
    const current = store.getSnapshot().selected
    if (loading.current || !current || current.workspaceId !== selected.workspaceId || current.sessionId !== selected.sessionId
      || current.phase !== 'ready' || !current.hasOlder || current.loadingOlder || (automatic && current.olderError !== undefined)) return
    loading.current = true
    scroll.current?.beforePrepend()
    try { await controller.loadOlder() }
    finally { loading.current = false }
  }, [controller, store, selected.workspaceId, selected.sessionId])

  useEffect(() => {
    const onScroll = (): void => { if (scroll.current?.moved()) void loadOlder(true) }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [loadOlder])

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
    </div>
  )
}
