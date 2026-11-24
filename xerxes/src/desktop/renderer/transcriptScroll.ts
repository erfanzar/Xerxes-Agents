// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useCallback, useLayoutEffect, useRef, useState, type RefObject } from 'react'

/** Pages one click may fetch while older history stays folded out of sight. */
const MAX_INVISIBLE_PAGES = 10
/** Growth that counts as something new on screen, not a summary line's reflow. */
const VISIBLE_GROWTH_PX = 24
/** Distance from the bottom edge at which an intentional downward scroll re-pins auto-follow. */
export const REPIN_BOTTOM_THRESHOLD_PX = 24
/** Tolerance (px) within which the viewport is considered resting on the bottom edge. */
const BOTTOM_EDGE_TOLERANCE_PX = 2

export interface TranscriptScrollDecisionInput {
  readonly wasFollowing: boolean
  readonly prevScrollTop: number
  readonly scrollTop: number
  readonly clientHeight: number
  readonly scrollHeight: number
  readonly lastFollowTop: number | null
}

/**
 * Pure follow-state transition for a scroll event on the transcript viewport.
 *
 * - Clamps `scrollTop` and `prevScrollTop` to `[0, maxScrollTop]` so macOS elastic
 *   overscroll rebounding past the bottom edge (`1035 -> 1020 -> 1000`) or small
 *   height shrinks while at the tail are NEVER misread as upward scrolls.
 * - Any genuine upward scroll away from the bottom (`clampedTop < clampedPrev`)
 *   immediately unpins auto-follow (even a 1px upward scroll from the bottom).
 * - While unpinned, only scrolling downward back into the bottom band (or content no
 *   longer overflowing) re-pins auto-follow.
 */
export function nextTranscriptFollowState(input: TranscriptScrollDecisionInput): {
  readonly following: boolean
  readonly nextLastFollowTop: number | null
} {
  const { wasFollowing, prevScrollTop, scrollTop, clientHeight, scrollHeight, lastFollowTop } = input
  if (clientHeight <= 0) {
    return { following: wasFollowing, nextLastFollowTop: lastFollowTop }
  }
  const maxScrollTop = Math.max(0, scrollHeight - clientHeight)
  if (maxScrollTop <= BOTTOM_EDGE_TOLERANCE_PX) {
    return { following: true, nextLastFollowTop: 0 }
  }
  const clampedTop = Math.min(Math.max(0, scrollTop), maxScrollTop)
  const clampedPrev = Math.min(Math.max(0, prevScrollTop), maxScrollTop)
  const distanceFromBottom = maxScrollTop - clampedTop

  // Any genuine upward scroll (within valid [0, maxScrollTop] bounds) immediately unpins.
  // Elastic overscroll rebounds (`1035 -> 1020 -> 1000`) clamp to `maxScrollTop -> maxScrollTop`
  // so `clampedTop < clampedPrev` is false and they stay pinned at the bottom.
  if (clampedTop < clampedPrev) {
    return { following: false, nextLastFollowTop: null }
  }
  // Already at (or elastically overscrolled past) the very bottom edge: stay pinned.
  if (distanceFromBottom <= BOTTOM_EDGE_TOLERANCE_PX) {
    return { following: true, nextLastFollowTop: clampedTop }
  }
  if (wasFollowing) {
    if (lastFollowTop !== null && clampedTop >= Math.min(lastFollowTop, maxScrollTop) - 1) {
      return { following: true, nextLastFollowTop: clampedTop }
    }
    const stillAtTail = distanceFromBottom <= REPIN_BOTTOM_THRESHOLD_PX
    return { following: stillAtTail, nextLastFollowTop: stillAtTail ? clampedTop : null }
  }
  if (clampedTop > clampedPrev && distanceFromBottom <= REPIN_BOTTOM_THRESHOLD_PX) {
    return { following: true, nextLastFollowTop: clampedTop }
  }
  return { following: false, nextLastFollowTop: null }
}

function clampScrollPosition(el: HTMLElement): number {
  const maxTop = Math.max(0, el.scrollHeight - el.clientHeight)
  return Math.min(Math.max(0, el.scrollTop), maxTop)
}

export interface HistoryScrollAnchor {
  readonly anchorId: string | undefined
  readonly anchorOffset: number
  readonly offset: number
  readonly scrollHeight: number
  readonly scrollTop: number
}

export interface HistoryAnchorContainer {
  readonly scrollTop: number
  readonly scrollHeight: number
  getBoundingClientRect(): { readonly top: number; readonly bottom: number }
  querySelectorAll<E extends Element = HTMLElement>(selector: string): ArrayLike<E>
}

/** Capture the first visible `[data-history-anchor]` element and scroll metrics before prepending older history. */
export function captureHistoryScrollAnchor(element: HistoryAnchorContainer): HistoryScrollAnchor {
  const top = element.getBoundingClientRect().top
  const nodes = Array.from(element.querySelectorAll<HTMLElement>('[data-history-anchor]'))
  const anchor = nodes.find(node => node.getBoundingClientRect().bottom > top)
  const anchorOffset = anchor ? anchor.getBoundingClientRect().top - top : 0
  return {
    anchorId: anchor?.dataset.historyAnchor,
    anchorOffset,
    offset: anchorOffset,
    scrollHeight: element.scrollHeight,
    scrollTop: element.scrollTop,
  }
}

/** Compute the restored `scrollTop` that keeps the captured anchor at the same viewport offset after prepending. */
export function restoreHistoryScrollAnchor(element: HistoryAnchorContainer, anchor: HistoryScrollAnchor): number {
  const nodes = Array.from(element.querySelectorAll<HTMLElement>('[data-history-anchor]'))
  const retained = anchor.anchorId
    ? nodes.find(node => node.dataset.historyAnchor === anchor.anchorId)
    : undefined
  if (retained) {
    return Math.max(
      0,
      element.scrollTop + retained.getBoundingClientRect().top - element.getBoundingClientRect().top - anchor.anchorOffset,
    )
  }
  return Math.max(0, anchor.scrollTop + element.scrollHeight - anchor.scrollHeight)
}

/** A newly attached session starts at its tail; reading older content pauses following. */
export function useTranscriptScroll(
  sessionId: string | null,
  options: { more?: boolean; loading?: boolean; automatic?: boolean; load: () => Promise<void> },
): {
  ref: RefObject<HTMLDivElement | null>
  loadOlder: () => Promise<void>
  following: boolean
  scrollToLatest: () => void
  unpin: () => void
} {
  const ref = useRef<HTMLDivElement>(null)
  const [pinned, setPinned] = useState(true)
  const session = useRef(sessionId)
  const following = useRef(true)
  const lastFollowTop = useRef<number | null>(null)
  const lastScrollTop = useRef(0)
  const currentOptions = useRef(options)
  currentOptions.current = options
  const pending = useRef(false)
  const pendingPromise = useRef<Promise<void> | null>(null)
  /** The session whose too-short transcript was already topped up once on open. */
  const filledFor = useRef<string | null>(null)

  const syncPinned = useCallback((next: boolean): void => {
    following.current = next
    setPinned(current => (current === next ? current : next))
  }, [])

  const unpin = useCallback((): void => {
    lastFollowTop.current = null
    syncPinned(false)
  }, [syncPinned])

  const scrollToLatest = useCallback((): void => {
    const element = ref.current
    if (!element) return
    syncPinned(true)
    const maxTop = Math.max(0, element.scrollHeight - element.clientHeight)
    element.scrollTop = maxTop
    lastFollowTop.current = maxTop
    lastScrollTop.current = maxTop
  }, [syncPinned])

  const loadOlder = useCallback(async (): Promise<void> => {
    if (pendingPromise.current) return pendingPromise.current
    const element = ref.current
    if (!element || pending.current || currentOptions.current.loading || !currentOptions.current.more) return
    const run = (async (): Promise<void> => {
      pending.current = true
      syncPinned(false)
      lastFollowTop.current = null
      const identity = session.current
      const anchor = captureHistoryScrollAnchor(element)
      try {
        for (let page = 0; page < MAX_INVISIBLE_PAGES; page++) {
          await currentOptions.current.load()
          await new Promise<void>(resolve => setTimeout(resolve, 32))
          await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
          if (identity !== session.current) return
          const grew = element.scrollHeight - anchor.scrollHeight >= VISIBLE_GROWTH_PX
          if (grew || !currentOptions.current.more || currentOptions.current.automatic === false) break
        }
        element.scrollTop = restoreHistoryScrollAnchor(element, anchor)
        lastScrollTop.current = clampScrollPosition(element)
      } finally {
        pending.current = false
        pendingPromise.current = null
      }
    })()
    pendingPromise.current = run
    return run
  }, [syncPinned])

  // Persistent scroll/wheel/resize/mutation subscriptions scoped to the session.
  useLayoutEffect(() => {
    const element = ref.current
    if (!element) return
    if (session.current !== sessionId) {
      session.current = sessionId
      syncPinned(true)
      lastFollowTop.current = null
    }

    const follow = (): void => {
      if (element.clientHeight <= 0) return
      const maxTop = Math.max(0, element.scrollHeight - element.clientHeight)
      if (following.current) {
        // Never overwrite scrollTop when already at or elastically past maxTop
        // (prevents fighting macOS bottom rubber-band physics).
        if (element.scrollTop < maxTop - 1) {
          element.scrollTop = maxTop
        }
        lastFollowTop.current = maxTop
        lastScrollTop.current = maxTop
        return
      }
      // Only re-pin automatically if the transcript genuinely fits inside the viewport.
      if (!pending.current && maxTop <= BOTTOM_EDGE_TOLERANCE_PX) {
        syncPinned(true)
        lastFollowTop.current = 0
        lastScrollTop.current = 0
      }
    }

    const onWheel = (event: WheelEvent): void => {
      if (event.deltaY < 0 && element.clientHeight > 0 && element.scrollHeight > element.clientHeight + BOTTOM_EDGE_TOLERANCE_PX) {
        lastFollowTop.current = null
        syncPinned(false)
      }
    }

    const onScroll = (): void => {
      if (element.clientHeight <= 0) return
      const currentTop = element.scrollTop
      const prevTop = lastScrollTop.current
      const decision = nextTranscriptFollowState({
        wasFollowing: following.current,
        prevScrollTop: prevTop,
        scrollTop: currentTop,
        clientHeight: element.clientHeight,
        scrollHeight: element.scrollHeight,
        lastFollowTop: lastFollowTop.current,
      })
      const clampedTop = clampScrollPosition(element)
      lastScrollTop.current = clampedTop
      lastFollowTop.current = decision.nextLastFollowTop
      syncPinned(decision.following)
      if (!decision.following && clampedTop <= prevTop && clampedTop < 160 && currentOptions.current.automatic !== false) {
        void loadOlder()
      }
    }

    follow()
    element.addEventListener('wheel', onWheel, { passive: true })
    element.addEventListener('scroll', onScroll, { passive: true })

    // Observe content children for layout growth while ignoring floating controls like .jump-latest.
    const observed = new Set<Element>()
    const resizeObserver = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(follow) : null
    const observeChild = (child: Element): void => {
      if (
        !resizeObserver ||
        observed.has(child) ||
        child.classList.contains('jump-latest') ||
        child.classList.contains('jump-latest-dock') ||
        child.hasAttribute('data-scroll-ignore')
      ) {
        return
      }
      observed.add(child)
      resizeObserver.observe(child)
    }
    if (resizeObserver) {
      resizeObserver.observe(element)
      for (const child of element.children) observeChild(child)
    }
    const mutationObserver =
      typeof MutationObserver !== 'undefined' && resizeObserver
        ? new MutationObserver(() => {
            for (const child of element.children) observeChild(child)
          })
        : null
    mutationObserver?.observe(element, { childList: true })

    return () => {
      element.removeEventListener('wheel', onWheel)
      element.removeEventListener('scroll', onScroll)
      mutationObserver?.disconnect()
      resizeObserver?.disconnect()
    }
  }, [sessionId, loadOlder, syncPinned])

  // Lightweight post-commit tail follow & one-shot short-history top-up without churning listeners.
  useLayoutEffect(() => {
    const element = ref.current
    if (!element || element.clientHeight <= 0) return
    const maxTop = Math.max(0, element.scrollHeight - element.clientHeight)
    if (following.current) {
      if (element.scrollTop < maxTop - 1) {
        element.scrollTop = maxTop
      }
      lastFollowTop.current = maxTop
      lastScrollTop.current = maxTop
    } else if (!pending.current && maxTop <= BOTTOM_EDGE_TOLERANCE_PX) {
      syncPinned(true)
      lastFollowTop.current = 0
      lastScrollTop.current = 0
    }

    if (
      filledFor.current !== sessionId &&
      maxTop <= 1 &&
      currentOptions.current.more &&
      !currentOptions.current.loading &&
      currentOptions.current.automatic !== false
    ) {
      filledFor.current = sessionId
      void loadOlder()
    }
  })

  return { ref, loadOlder, following: pinned, scrollToLatest, unpin }
}
