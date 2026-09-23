// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useCallback, useLayoutEffect, useRef, useState, type RefObject } from 'react'

/** A newly attached session starts at its tail; reading older content pauses following. */
export function useTranscriptScroll(sessionId: string | null, options: { more?: boolean; loading?: boolean; automatic?: boolean; load: () => Promise<void> }): { ref: RefObject<HTMLDivElement | null>; loadOlder: () => Promise<void>; following: boolean; scrollToLatest: () => void } {
  const ref = useRef<HTMLDivElement>(null)
  // Mirrored into React state so the feed can offer a way back. Following
  // could only be re-armed by scrolling into a 64px band at the bottom by
  // hand, and nothing on screen said the tail was still moving — so
  // submitting while scrolled up rendered your own message off-screen.
  const [pinned, setPinned] = useState(true)
  const session = useRef(sessionId)
  const following = useRef(true)
  const lastFollowTop = useRef<number | null>(null)
  const currentOptions = useRef(options)
  currentOptions.current = options
  const pending = useRef(false)
  const scrollToLatest = useCallback((): void => {
    const element = ref.current
    if (!element) return
    following.current = true
    setPinned(true)
    element.scrollTop = element.scrollHeight
    lastFollowTop.current = element.scrollTop
  }, [])
  const loadOlder = useCallback(async (): Promise<void> => {
    const element = ref.current
    if (!element || pending.current || currentOptions.current.loading || !currentOptions.current.more) return
    pending.current = true
    following.current = false
    setPinned(false)
    const identity = session.current
    const top = element.getBoundingClientRect().top
    const anchor = [...element.querySelectorAll<HTMLElement>('[data-history-anchor]')].find(node => node.getBoundingClientRect().bottom > top)
    const anchorId = anchor?.dataset.historyAnchor
    const offset = anchor ? anchor.getBoundingClientRect().top - top : 0
    const height = element.scrollHeight
    const scroll = element.scrollTop
    try {
      await currentOptions.current.load()
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
      if (identity !== session.current) return
      const retained = anchorId ? [...element.querySelectorAll<HTMLElement>('[data-history-anchor]')].find(node => node.dataset.historyAnchor === anchorId) : undefined
      element.scrollTop = retained ? element.scrollTop + retained.getBoundingClientRect().top - element.getBoundingClientRect().top - offset : scroll + element.scrollHeight - height
    } finally { pending.current = false }
  }, [])

  useLayoutEffect(() => {
    const element = ref.current
    if (!element) return
    if (session.current !== sessionId) {
      session.current = sessionId
      following.current = true
      setPinned(true)
    }
    const follow = (): void => {
      if (following.current && element.clientHeight > 0) {
        element.scrollTop = element.scrollHeight
        lastFollowTop.current = element.scrollTop
      }
    }
    const onScroll = (): void => {
      if (element.clientHeight > 0) {
        // A scroll event from our own tail jump may arrive after more content
        // has grown. It must not be mistaken for the user scrolling upward.
        if (following.current && lastFollowTop.current !== null && element.scrollTop >= lastFollowTop.current) {
          // Browser scroll anchoring may also move the viewport down as the
          // composer or transcript settles; only upward movement unpins it.
          lastFollowTop.current = element.scrollTop
          return
        }
        lastFollowTop.current = null
        following.current = element.scrollHeight - element.scrollTop - element.clientHeight < 64
        setPinned(following.current)
        if (!following.current && element.scrollTop < 160 && currentOptions.current.automatic !== false) void loadOlder()
      }
    }
    // Measure after React commits the replay, not before its height is known.
    follow()
    element.addEventListener('scroll', onScroll, { passive: true })
    // Markdown, fonts, and window resizing can change height after that commit.
    const observer = new ResizeObserver(follow)
    observer.observe(element)
    for (const child of element.children) observer.observe(child)
    return () => {
      element.removeEventListener('scroll', onScroll)
      observer.disconnect()
    }
  })
  return { ref, loadOlder, following: pinned, scrollToLatest }
}
