// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { diffLineKind, parseInline, parseMarkdown } from '../lib/markdown.js'
import { highlightLine, isHighlightable } from '../lib/syntax.js'
import { DARK_THEME } from '../theme.js'

describe('parseMarkdown blocks', () => {
  it('parses headings with levels', () => {
    expect(parseMarkdown('# Title')).toEqual([{ type: 'heading', level: 1, text: 'Title' }])
    expect(parseMarkdown('### Sub')[0]).toMatchObject({ type: 'heading', level: 3 })
  })

  it('groups consecutive lines into a paragraph', () => {
    const b = parseMarkdown('line one\nline two\n\nnext para')
    expect(b).toHaveLength(2)
    expect(b[0]).toEqual({ type: 'paragraph', text: 'line one\nline two' })
    expect(b[1]).toEqual({ type: 'paragraph', text: 'next para' })
  })

  it('captures a fenced code block with language and raw lines', () => {
    const b = parseMarkdown('```ts\nconst x = 1\n// c\n```')
    expect(b).toEqual([{ type: 'code', lang: 'ts', lines: ['const x = 1', '// c'] }])
  })

  it('does not treat markdown inside a fence as structure', () => {
    const b = parseMarkdown('```\n# not a heading\n- not a list\n```')
    expect(b[0]).toMatchObject({ type: 'code', lines: ['# not a heading', '- not a list'] })
  })

  it('parses unordered and ordered lists', () => {
    expect(parseMarkdown('- a\n- b')).toEqual([{ type: 'list', ordered: false, items: ['a', 'b'] }])
    expect(parseMarkdown('1. one\n2. two')).toEqual([{ type: 'list', ordered: true, items: ['one', 'two'] }])
  })

  it('parses blockquotes', () => {
    expect(parseMarkdown('> quoted\n> more')).toEqual([{ type: 'quote', lines: ['quoted', 'more'] }])
  })

  it('parses a horizontal rule', () => {
    expect(parseMarkdown('---')).toEqual([{ type: 'hr' }])
  })

  it('parses a table with header + rows', () => {
    const b = parseMarkdown('| a | b |\n| - | - |\n| 1 | 2 |')
    expect(b[0]).toEqual({ type: 'table', header: ['a', 'b'], rows: [['1', '2']] })
  })

  it('handles CRLF input', () => {
    expect(parseMarkdown('# T\r\npara')).toEqual([
      { type: 'heading', level: 1, text: 'T' },
      { type: 'paragraph', text: 'para' }
    ])
  })
})

describe('parseInline', () => {
  it('splits code / bold / italic / link / text', () => {
    expect(parseInline('a `b` c')).toEqual([
      { type: 'text', text: 'a ' },
      { type: 'code', text: 'b' },
      { type: 'text', text: ' c' }
    ])
    expect(parseInline('**bold**')).toEqual([{ type: 'bold', text: 'bold' }])
    expect(parseInline('_em_')).toEqual([{ type: 'italic', text: 'em' }])
    expect(parseInline('[txt](http://x)')).toEqual([{ type: 'link', text: 'txt', url: 'http://x' }])
  })

  it('returns a single text span for plain input', () => {
    expect(parseInline('just text')).toEqual([{ type: 'text', text: 'just text' }])
  })
})

describe('diffLineKind', () => {
  it('classifies add/del/meta', () => {
    expect(diffLineKind('+added')).toBe('add')
    expect(diffLineKind('-removed')).toBe('del')
    expect(diffLineKind('@@ -1 +1 @@')).toBe('meta')
    expect(diffLineKind('+++ b/file')).toBe('meta')
    expect(diffLineKind('  context')).toBeNull()
  })
})

describe('syntax highlight', () => {
  it('knows common languages + aliases', () => {
    expect(isHighlightable('python')).toBe(true)
    expect(isHighlightable('tsx')).toBe(true)
    expect(isHighlightable('cobol')).toBe(false)
  })
  it('tokenizes keywords and strings distinctly', () => {
    const toks = highlightLine('const s = "hi"', 'ts', DARK_THEME)
    const kw = toks.find(t => t[1] === 'const')
    const strTok = toks.find(t => t[1] === '"hi"')
    expect(kw?.[0]).toBe(DARK_THEME.color.border)
    expect(strTok?.[0]).toBe(DARK_THEME.color.accent)
  })
})
