// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { fenceOpenAt, findStableBoundary, splitStreaming, splitStreamingRender, STREAMING_CHUNKS_EMPTY } from '../lib/streamingMarkdown.js'

describe('fenceOpenAt', () => {
  it('is false outside any fence', () => {
    expect(fenceOpenAt('a\n\nb', 4)).toBe(false)
  })
  it('is true between an opening fence and its close', () => {
    const s = '```ts\ncode here\n'
    expect(fenceOpenAt(s, s.length)).toBe(true)
  })
  it('is false once the fence closes', () => {
    const s = '```\ncode\n```\n'
    expect(fenceOpenAt(s, s.length)).toBe(false)
  })
})

describe('findStableBoundary', () => {
  it('returns -1 when there is no blank-line boundary yet', () => {
    expect(findStableBoundary('still typing the first paragraph')).toBe(-1)
  })
  it('finds the last blank-line boundary', () => {
    const text = 'para one\n\npara two in progress'
    expect(findStableBoundary(text)).toBe('para one\n\n'.length)
  })
  it('refuses a boundary that lands inside an open code fence', () => {
    // blank line is inside the fence; no safe boundary → -1
    const text = '```\nline one\n\nline two'
    expect(findStableBoundary(text)).toBe(-1)
  })
  it('allows a boundary after a closed fence', () => {
    const text = '```\ncode\n```\n\nnext para'
    expect(findStableBoundary(text)).toBe('```\ncode\n```\n\n'.length)
  })
})

describe('splitStreaming (monotonic)', () => {
  it('keeps everything unstable until the first boundary', () => {
    const s = splitStreaming('typing...', '')
    expect(s.stablePrefix).toBe('')
    expect(s.unstableSuffix).toBe('typing...')
  })

  it('freezes the prefix once a boundary appears', () => {
    const text = 'first block\n\nsecond'
    const s = splitStreaming(text, '')
    expect(s.stablePrefix).toBe('first block\n\n')
    expect(s.unstableSuffix).toBe('second')
  })

  it('only advances the prefix, never retreats', () => {
    const t1 = 'a\n\nb' // boundary after 'a\n\n'
    const s1 = splitStreaming(t1, '')
    expect(s1.stablePrefix).toBe('a\n\n')

    // next delta extends b, still no new boundary → prefix unchanged
    const t2 = 'a\n\nb more text'
    const s2 = splitStreaming(t2, s1.stablePrefix)
    expect(s2.stablePrefix).toBe('a\n\n')
    expect(s2.unstableSuffix).toBe('b more text')

    // a new boundary appears → prefix grows
    const t3 = 'a\n\nb more text\n\nc'
    const s3 = splitStreaming(t3, s2.stablePrefix)
    expect(s3.stablePrefix).toBe('a\n\nb more text\n\n')
    expect(s3.unstableSuffix).toBe('c')
  })

  it('resets when the text no longer starts with the previous prefix', () => {
    const s = splitStreaming('brand new turn', 'old prefix\n\n')
    expect(s.stablePrefix).toBe('')
    expect(s.unstableSuffix).toBe('brand new turn')
  })
})

describe('splitStreamingRender (render-ready chunks)', () => {
  it('keeps everything in the tail until the first block boundary', () => {
    const r = splitStreamingRender('still typing', STREAMING_CHUNKS_EMPTY)
    expect(r.chunks).toEqual([])
    expect(r.tail).toBe('still typing')
    expect(r.state.prefix).toBe('')
  })

  it('strips the trailing blank line from stabilized chunks', () => {
    const r = splitStreamingRender('# Heading\n\nnext', STREAMING_CHUNKS_EMPTY)
    expect(r.chunks).toEqual(['# Heading'])
    expect(r.tail).toBe('next')
  })

  it('appends chunks monotonically across deltas without mutating prior chunks', () => {
    let state = STREAMING_CHUNKS_EMPTY
    const deltas = [
      'para one',
      'para one\n\npara tw',
      'para one\n\npara two\n\n- a\n- b',
      'para one\n\npara two\n\n- a\n- b\n\ntail'
    ]

    const seen: string[][] = []
    for (const delta of deltas) {
      const r = splitStreamingRender(delta, state)
      seen.push([...r.chunks])
      state = r.state
    }

    expect(seen[0]).toEqual([])
    expect(seen[1]).toEqual(['para one'])
    expect(seen[2]).toEqual(['para one', 'para two'])
    expect(seen[3]).toEqual(['para one', 'para two', '- a\n- b'])
    // chunk identity: earlier chunk strings are never rewritten
    expect(seen[2]![0]).toBe(seen[1]![0])
  })

  it('never splits inside an open code fence', () => {
    let state = STREAMING_CHUNKS_EMPTY

    const mid = splitStreamingRender('intro\n\n```ts\nconst x = 1\n\nconst y = 2', state)
    state = mid.state
    expect(mid.chunks).toEqual(['intro'])
    expect(mid.tail).toBe('```ts\nconst x = 1\n\nconst y = 2')

    const done = splitStreamingRender('intro\n\n```ts\nconst x = 1\n\nconst y = 2\n```\n\nafter', state)
    expect(done.chunks).toEqual(['intro', '```ts\nconst x = 1\n\nconst y = 2\n```'])
    expect(done.tail).toBe('after')
  })

  it('keeps multi-block stabilized segments as one chunk with internal spacing intact', () => {
    const r = splitStreamingRender('a\n\nb\n\nc', STREAMING_CHUNKS_EMPTY)
    expect(r.chunks).toEqual(['a\n\nb'])
    expect(r.tail).toBe('c')
  })

  it('resets cleanly when a new turn reuses the component', () => {
    let state = STREAMING_CHUNKS_EMPTY
    state = splitStreamingRender('old reply\n\nmore', state).state
    expect(state.chunks).toEqual(['old reply'])

    const next = splitStreamingRender('new turn text', state)
    expect(next.chunks).toEqual([])
    expect(next.tail).toBe('new turn text')
  })

  it('chunks joined with the tail reproduce the original buffer minus stripped boundaries', () => {
    let state = STREAMING_CHUNKS_EMPTY
    const full = '# T\n\npara text\n\n- one\n- two\n\n```\ncode\n```\n\nlast'
    const r = splitStreamingRender(full, state)
    state = r.state
    // One big delta stabilizes all complete blocks into a single chunk; the
    // internal blank lines stay intact so inter-block spacing survives.
    expect(r.chunks).toEqual(['# T\n\npara text\n\n- one\n- two\n\n```\ncode\n```'])
    expect(r.tail).toBe('last')
    expect(`${r.chunks.join('\n\n')}\n\n${r.tail}`).toBe(full)
  })
})
