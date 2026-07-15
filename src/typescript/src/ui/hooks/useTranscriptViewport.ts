// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Glue hook: turns the pure virtualHeights math into a live transcript window.
// Tracks terminal size + scroll position (sticky to bottom by default), and
// returns only the rows that should be mounted.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { TranscriptRow } from '../app/gatewayState.js'
import { computeVisibleWindow, estimateRowHeight, resolveScrollTop } from '../lib/virtualHeights.js'

// Approximate non-transcript chrome (banner + status rule + input + a little
// slack). The estimate only needs to be close; over-reserving just mounts a
// couple fewer rows.
const RESERVED_ROWS = 8

function terminalSize(): { rows: number; cols: number } {
  return {
    rows: process.stdout.rows ?? 24,
    cols: process.stdout.columns ?? 80
  }
}

export interface TranscriptViewport {
  rows: TranscriptRow[]
  cols: number
  hiddenAbove: number
  hiddenBelow: number
  sticky: boolean
  scrollBy: (deltaLines: number) => void
  scrollToBottom: () => void
}

export function useTranscriptViewport(transcript: TranscriptRow[], extraReserved = 0): TranscriptViewport {
  const [size, setSize] = useState(terminalSize)
  const [scrollTop, setScrollTop] = useState(0)
  const [sticky, setSticky] = useState(true)

  // Track terminal resizes.
  useEffect(() => {
    const onResize = () => setSize(terminalSize())
    process.stdout.on('resize', onResize)
    return () => {
      process.stdout.off('resize', onResize)
    }
  }, [])

  const heights = useMemo(() => transcript.map(row => estimateRowHeight(row, size.cols)), [transcript, size.cols])

  const viewportHeight = Math.max(3, size.rows - RESERVED_ROWS - extraReserved)

  const window = useMemo(
    () => computeVisibleWindow(heights, viewportHeight, sticky ? Number.MAX_SAFE_INTEGER : scrollTop),
    [heights, viewportHeight, sticky, scrollTop]
  )

  // When sticky, keep scrollTop pinned to the (new) bottom as content grows.
  const stickRef = useRef(sticky)
  stickRef.current = sticky
  useEffect(() => {
    if (stickRef.current) {
      setScrollTop(window.maxScrollTop)
    }
  }, [window.maxScrollTop])

  const scrollBy = useCallback(
    (deltaLines: number) => {
      setSticky(prevSticky => {
        const base = prevSticky ? window.maxScrollTop : scrollTop
        const next = resolveScrollTop(window.maxScrollTop, false, base + deltaLines)
        setScrollTop(next)
        // Re-stick if we scrolled to (or past) the bottom.
        return next >= window.maxScrollTop
      })
    },
    [window.maxScrollTop, scrollTop]
  )

  const scrollToBottom = useCallback(() => {
    setSticky(true)
    setScrollTop(window.maxScrollTop)
  }, [window.maxScrollTop])

  const rows = window.end >= window.start ? transcript.slice(window.start, window.end + 1) : []

  return {
    rows,
    cols: size.cols,
    hiddenAbove: window.start,
    hiddenBelow: Math.max(0, transcript.length - 1 - window.end),
    sticky,
    scrollBy,
    scrollToBottom
  }
}
