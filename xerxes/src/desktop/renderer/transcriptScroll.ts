// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useLayoutEffect, useRef, type RefObject } from 'react'

/** A newly attached session starts at its tail; reading older content pauses following. */
export function useTranscriptScroll(sessionId: string | null): RefObject<HTMLDivElement | null> {
  const ref = useRef<HTMLDivElement>(null)
  const session = useRef(sessionId)
  const following = useRef(true)
  const lastFollowTop = useRef<number | null>(null)

  useLayoutEffect(() => {
    const element = ref.current
    if (!element) return
    if (session.current !== sessionId) {
      session.current = sessionId
      following.current = true
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
  return ref
}
