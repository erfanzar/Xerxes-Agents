// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { markdownBlocks, Markdown } from '../src/desktop/renderer/markdown.js'

const render = (text: string): string => renderToStaticMarkup(createElement(Markdown, { text }))
/** Visible text of rendered markup: highlighted code wraps tokens in spans. */
const visible = (html: string): string => html.replace(/<[^>]+>/g, '')

test('agent markdown renders headings, lists, and emphasis', () => {
  const html = render('# Plan\n\n- **first** thing\n- *second* thing\n\n1. ordered\n')
  expect(html).toContain('<h1>Plan</h1>')
  expect(html).toContain('<strong>first</strong>')
  expect(html).toContain('<em>second</em>')
  expect(html).toContain('<ul>')
  expect(html).toContain('ordered')
})

test('fenced code renders as a pre block, not styled like inline code', () => {
  const html = render('before\n\n```ts\nconst a = 1\nconst b = 2\n```\n\nafter')
  expect(html).toContain('md__code')
  expect(html).toContain('data-lang="ts"')
  expect(visible(html)).toContain('const a = 1\nconst b = 2')
  // The fence body must not inherit the inline-code chip border.
  expect(html).toContain('<pre class="md__code" data-lang="ts"><code>')
  expect(html).toContain('before')
  expect(html).toContain('after')
})

test('an unclosed fence from a streaming cut still renders as code', () => {
  const html = render('text\n```py\nprint(1)')
  expect(html).toContain('md__code')
  expect(visible(html)).toContain('print(1)')
})

test('inline code, links, and tables render without any raw HTML passing through', () => {
  const html = render(
    'use `bun test` and see [the docs](https://example.com)\n\n| tool | state |\n| --- | --- |\n| ls | done |\n',
  )
  expect(html).toContain('<code>bun test</code>')
  expect(html).toContain('md__link')
  expect(html).toContain('href="https://example.com"')
  expect(html).toContain('md__table')
  expect(html).toContain('<th>tool</th>')
  expect(html).toContain('<td>done</td>')
})

test('link markup can never inject attributes or elements', () => {
  const html = render('[x](javascript:alert(1)) <img src=x onerror=alert(1)> **bold**')
  // A javascript: link is never an anchor at all: it renders as inert text
  // with its target as the tooltip. The img tag is plain text because only
  // recognized markdown shapes become elements.
  expect(html).not.toContain('href=')
  expect(html).toContain('md__link--inert')
  expect(html).not.toContain('<img')
  expect(html).toContain('&lt;img')
})

test('blockquotes and horizontal rules fold into their own blocks', () => {
  const html = render('> quoted line\n\ntext\n\n---\n\nmore')
  expect(html).toContain('<blockquote>')
  expect(html).toContain('quoted line')
  expect(html).toContain('<hr')
})

test('markdownBlocks keeps streaming-friendly tolerance for partial tables', () => {
  // A pipe row without its divider yet is a paragraph, not a broken table.
  const html = render('| tool |\n')
  expect(html).not.toContain('md__table')
  expect(html).toContain('| tool |')
})

test('the tail glues to the end of trailing prose, not a line of its own', () => {
  const caret = createElement('span', { className: 'caret' })
  const html = renderToStaticMarkup(createElement(Markdown, { text: 'first line\nstill going', tail: caret }))
  // One paragraph, caret inside it — never <p>…</p><span class="caret">.
  expect(html).toContain('still going<span class="caret">')
  expect(html.match(/<p>/g)).toHaveLength(1)

  // Structural endings can't carry it inline — it follows the block instead.
  // The fence is wrapped so it can carry a language label and a copy button.
  const fenced = renderToStaticMarkup(createElement(Markdown, { text: 'prose\n```ts\nx\n```', tail: caret }))
  expect(fenced).toContain('</pre></figure><span class="caret">')
})

test('a fenced block names its language and offers a copy control', () => {
  const html = renderToStaticMarkup(createElement(Markdown, { text: '```ts\nconst a = 1\n```' }))
  expect(html).toContain('<span class="md__lang">ts</span>')
  expect(html).toContain('aria-label="Copy code"')
  expect(html).toContain('<code><span class="hljs-keyword">const</span>')
  expect(visible(html)).toContain('const a = 1')
  // An unlabelled fence still gets the bar, so the copy control never moves.
  expect(renderToStaticMarkup(createElement(Markdown, { text: '```\nplain\n```' }))).toContain('<span class="md__lang">text</span>')
})

test('a known fence language is syntax-highlighted as React spans, an unknown one stays plain', () => {
  const html = render('```ts\nconst a = 1\n```')
  expect(html).toContain('<span class="hljs-keyword">const</span>')
  expect(render('```nosuchlang\nconst a = 1\n```')).toContain('<code>const a = 1</code>')
  // Highlighting never lets markup in the code through as elements.
  const hostile = render('```html\n<img src=x onerror=alert(1)>\n```')
  expect(hostile).not.toContain('<img')
  expect(hostile).toContain('&lt;')
})

test('inline and display math render as MathML, and prices stay text', () => {
  const inline = render('Area is $x^2$ here.')
  expect(inline).toContain('<math')
  expect(inline).toContain('<msup>')
  expect(inline).not.toContain('annotation')
  const display = render('$$\n\\frac{a}{b}\n$$')
  expect(display).toContain('md__mathblock')
  expect(display).toContain('<mfrac>')
  expect(render('It costs $5 and $10 today.')).not.toContain('<math')
  // Shell variables in prose are not math.
  expect(render('Copy it to $HOME/$USER first.')).not.toContain('<math')
  expect(render('Then export $PATH:$HOME again.')).not.toContain('<math')
  // An opening $$ with no close is prose.
  expect(render('$$ not closed')).not.toContain('md__mathblock')
})

test('math cannot produce links, scripts or unknown elements', () => {
  const html = render('$\\href{javascript:alert(1)}{x}$ and $$\\url{https://evil.example}$$')
  expect(html).not.toContain('href=')
  expect(html).not.toContain('<a')
  expect(html).not.toContain('<script')
})

test('a mermaid fence becomes a diagram block that shows its source until drawn', () => {
  const html = render('```mermaid\ngraph TD; A-->B\n```')
  expect(html).toContain('md__mermaid')
  expect(html).toContain('Drawing the diagram')
  expect(html).toContain('graph TD; A--&gt;B')
  expect(html).toContain('aria-label="Copy diagram source"')
  // The source is never interpreted as markup.
  expect(render('```mermaid\ngraph TD; A["<img src=x onerror=alert(1)>"]\n```')).not.toContain('<img')
})

test('mixed inline list content shares one text flow beside its marker', () => {
  const html = render('- **Capture** the diff with `git diff` and read it.\n- [ ] Keep *all* text together.')
  expect(html).toContain('<span class="md__item"><strong>Capture</strong> the diff with <code>git diff</code> and read it.</span>')
  expect(html).toContain('<span class="md__item">Keep <em>all</em> text together.</span>')
})
