// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Markdown } from '../components/Markdown.js'

const html = (text: string): string => renderToStaticMarkup(<Markdown text={text} />).replaceAll('<!-- -->', '')

describe('phone markdown', () => {
  it('renders bold, inline code and italics instead of literal markers', () => {
    const out = html('**Where things stand** run `git status` and *then* stop')
    expect(out).toContain('<strong>Where things stand</strong>')
    expect(out).toContain('<code>git status</code>')
    expect(out).toContain('<em>then</em>')
    expect(out).not.toContain('**')
    expect(out).not.toContain('`')
  })

  it('nests indented bullets under their parent item', () => {
    const out = html('- **Commits:**\n  - `03ef4850`: computer control\n  - `565b1c15`: release\n- Last')
    expect(out).toBe('<div class="md"><ul><li><strong>Commits:</strong><ul><li><code>03ef4850</code>: computer control</li><li><code>565b1c15</code>: release</li></ul></li><li>Last</li></ul></div>')
  })

  it('keeps ordered list numbering and joins wrapped item lines', () => {
    const out = html('3. first\n   continues\n4. second')
    expect(out).toContain('<ol start="3"><li>first continues</li><li>second</li></ol>')
  })

  it('renders fenced code, headings, quotes, rules and tables', () => {
    const out = html('## Title\n\n```sh\nbun test\n```\n\n> quoted\n\n---\n\n| a | b |\n|---|---|\n| 1 | `2` |')
    expect(out).toContain('<h3>Title</h3>')
    expect(out).toContain('<pre class="md-code" data-lang="sh"><code>bun test</code></pre>')
    expect(out).toContain('<blockquote><p>quoted</p></blockquote>')
    expect(out).toContain('<hr/>')
    expect(out).toContain('<th>a</th>')
    expect(out).toContain('<td><code>2</code></td>')
  })

  it('separates paragraphs on blank lines and keeps single line breaks', () => {
    expect(html('one\ntwo\n\nthree')).toBe('<div class="md"><p>one<br/>two</p><p>three</p></div>')
  })

  it('never injects markup and only links http(s) or mailto targets', () => {
    const out = html('<img src=x onerror=alert(1)> [ok](https://example.com) [bad](javascript:alert(1))')
    expect(out).toContain('&lt;img')
    expect(out).not.toContain('<img')
    expect(out).toContain('<a href="https://example.com" target="_blank" rel="noopener noreferrer">ok</a>')
    expect(out).not.toContain('javascript:')
    expect(out).toContain('bad')
  })

  it('renders an unclosed fence (streaming) as code', () => {
    expect(html('```\npartial')).toContain('<pre class="md-code"><code>partial</code></pre>')
  })
})
