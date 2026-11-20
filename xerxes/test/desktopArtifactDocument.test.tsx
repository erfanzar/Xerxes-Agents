// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { documentOutline, documentTitle } from '../src/desktop/renderer/RailSections.js'

test('a Markdown artifact\'s leading "# Title" becomes the page title and is shown once', () => {
  const doc = documentTitle('# October report: GLM serving\n\n## Summary\nText.', 'Fallback')
  expect(doc.title).toBe('October report: GLM serving')
  expect(doc.body).not.toContain('# October report')
  expect(doc.body).toContain('## Summary')
  // No leading heading: the artifact's own title, the body untouched.
  const plain = documentTitle('Intro paragraph.\n# Later heading', 'Saved title')
  expect(plain).toEqual({ title: 'Saved title', body: 'Intro paragraph.\n# Later heading' })
  // A level-2 first line is not a page title.
  expect(documentTitle('## Section\nbody', 'T').title).toBe('T')
})

test('the outline lists h1-h3 in order, strips emphasis and skips fenced code', () => {
  const body = [
    '## Summary',
    'text',
    '```sh',
    '# not a heading',
    '```',
    '### **Throughput** up 20x',
    '#### too deep',
    '# Appendix #',
  ].join('\n')
  expect(documentOutline(body)).toEqual([
    { level: 2, text: 'Summary' },
    { level: 3, text: 'Throughput up 20x' },
    { level: 1, text: 'Appendix' },
  ])
  expect(documentOutline('no headings here')).toEqual([])
})
