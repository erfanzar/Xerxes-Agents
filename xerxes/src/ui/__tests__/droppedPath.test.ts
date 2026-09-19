// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'
import { looksLikeDroppedPath } from '../app/useComposerState.js'

describe('pasted command and file hints', () => {
  it('does not suggest image attachment for slash-command arguments containing dots or paths', () => {
    for (const text of ['/model gpt-4.1 --provider pinned', '/file /tmp/index.ts', '/search error.path',
      '/mcp add --url https://example.invalid/mcp', '/custom-skill ./file.ts', ' /model gpt-4.1 ']) {
      expect(looksLikeDroppedPath(text), text).toBe(false)
    }
  })
  it('still recognizes dropped paths, including spaces, URI, root files and quoted paths', () => {
    for (const text of ['/tmp/image.png', '/Users/user/My Pictures/image.png', '/image.png',
      '"/My Pictures/image.png"', '~/Pictures/image.png', './image.png', 'file:///tmp/image.png', 'C:\\Pictures\\image.png']) {
      expect(looksLikeDroppedPath(text), text).toBe(true)
    }
    expect(looksLikeDroppedPath('/model gpt-4.1\n/mode plan')).toBe(false)
    expect(looksLikeDroppedPath('ordinary prose')).toBe(false)
  })
})
