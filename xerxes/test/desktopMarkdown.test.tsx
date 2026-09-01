// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { markdownBlocks, Markdown } from '../src/desktop/renderer/markdown.js'

const render = (text: string): string => renderToStaticMarkup(createElement(Markdown, { text }))

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
  expect(html).toContain('const a = 1\nconst b = 2')
  // The fence body must not inherit the inline-code chip border.
  expect(html).toContain('<pre class="md__code" data-lang="ts"><code>')
  expect(html).toContain('before')
  expect(html).toContain('after')
})

test('an unclosed fence from a streaming cut still renders as code', () => {
  const html = render('text\n```py\nprint(1)')
  expect(html).toContain('md__code')
  expect(html).toContain('print(1)')
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
  // React neutralizes javascript: URLs by rewriting the href to a throwing
  // placeholder; the img tag is plain text because only recognized markdown
  // shapes become elements.
  expect(html).toContain('React has blocked a javascript: URL')
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
  const fenced = renderToStaticMarkup(createElement(Markdown, { text: 'prose\n```ts\nx\n```', tail: caret }))
  expect(fenced).toContain('</code></pre><span class="caret">')
})
