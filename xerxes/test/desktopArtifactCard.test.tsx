// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { ArtifactCard, artifactOfTool } from '../src/desktop/renderer/ArtifactCard.js'
import { ToolCallRow } from '../src/desktop/renderer/Execution.js'
import { ArtifactsSection, type ArtifactRow } from '../src/desktop/renderer/RailSections.js'
import type { ToolItem } from '../src/desktop/renderer/types.js'

const item = (over: Partial<ToolItem>): ToolItem => ({
  id: 't1', verb: 'ArtifactTool', arg: '', dur: '0.1s', state: 'done', name: 'ArtifactTool',
  input: JSON.stringify({ action: 'create', title: 'Plan', kind: 'markdown', content: '# x' }),
  output: 'Created artifact "Plan: a (b) \\"c\\"" (id c2b490e7ea, version 1). It is in the Artifacts panel; mention it briefly.',
  ...over,
})

test('a finished ArtifactTool call names its artifact; anything else is not a card', () => {
  expect(artifactOfTool(item({}))).toEqual({ action: 'created', title: 'Plan: a (b) \\"c\\"', id: 'c2b490e7ea', version: 1, kind: 'markdown' })
  expect(artifactOfTool(item({ output: 'Updated artifact "Plan" (id abc123, version 4).', input: '{"kind":"code"}' })))
    .toEqual({ action: 'updated', title: 'Plan', id: 'abc123', version: 4, kind: 'code' })
  expect(artifactOfTool(item({ state: 'working' }))).toBeNull()
  expect(artifactOfTool(item({ state: 'failed', output: 'Tool execution failed: x' }))).toBeNull()
  expect(artifactOfTool(item({ name: 'exec_command' }))).toBeNull()
})

test('the transcript renders the card with an Open button instead of a tool row', () => {
  const html = renderToStaticMarkup(createElement(ToolCallRow, { item: item({ output: 'Created artifact "Plan" (id abc123, version 1).' }), label: 'Artifact' }))
  expect(html).toContain('artifact-card')
  expect(html).toContain('>Plan<')
  expect(html).toContain('>Open<')
  expect(html).not.toContain('execution-row')
})

test('ArtifactCard parses embed:true for html artifacts and renders sleek full-width markup without hex colors', () => {
  const parsed = artifactOfTool(item({
    input: JSON.stringify({ action: 'create', title: 'Simurgh 3D Torus', kind: 'html', embed: true, content: '<canvas></canvas>' }),
    output: 'Created artifact "Simurgh 3D Torus" (id art3d01, version 1).',
  }))
  expect(parsed).toEqual({
    action: 'created',
    title: 'Simurgh 3D Torus',
    id: 'art3d01',
    version: 1,
    kind: 'html',
    embed: true,
  })

  const docHtml = renderToStaticMarkup(createElement(ArtifactCard, {
    artifact: { id: 'doc1', title: 'Eyvan whole-project review', version: 3, action: 'updated', kind: 'markdown' },
  }))
  expect(docHtml).toContain('class="artifact-card"')
  expect(docHtml).toContain('class="artifact-card__row"')
  expect(docHtml).toContain('>Eyvan whole-project review<')
  expect(docHtml).toContain('>Document<')
  expect(docHtml).toContain('>v3<')
  expect(docHtml).toContain('>Open<')
  expect(docHtml).toContain('→')
  expect(docHtml).not.toContain('artifact-card__preview')
  expect(docHtml.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])

  const interactiveHtml = renderToStaticMarkup(createElement(ArtifactCard, {
    artifact: parsed!,
  }))
  expect(interactiveHtml).toContain('data-kind="html"')
  expect(interactiveHtml).toContain('data-interactive="true"')
  expect(interactiveHtml).toContain('data-expanded="true"')
  expect(interactiveHtml).toContain('>Interactive<')
  expect(interactiveHtml).toContain('artifact-card__preview')
  expect(interactiveHtml).toContain('aria-expanded="true"')
  expect(interactiveHtml).toContain('artifact-card__embed')
  expect(interactiveHtml.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])

  const css = readFileSync(join(import.meta.dir, '../src/desktop/renderer/layout.css'), 'utf8')
  expect(css).toContain('.artifact-card{display:flex;flex-direction:column;width:100%;max-width:none;')
  expect(css).not.toContain('.artifact-card{display:flex;align-items:center;gap:12px;max-width:520px;')
})

test('ArtifactsSection renders live WidgetFrame for html artifacts and toggles to highlighted HTML source', () => {
  const htmlRow: ArtifactRow = {
    id: 'scene-3d',
    title: 'Cosmic Simurgh Torus Knot',
    kind: 'html',
    size: 2048,
    createdAt: '2026-11-22T00:00:00.000Z',
    updatedAt: '2026-11-22T00:05:00.000Z',
    pinned: false,
    version: 2,
  }
  const content = '<div class="scene"><canvas id="c"></canvas><script>window.Xerxes3D.createScene(c)</script></div>'

  const previewHtml = renderToStaticMarkup(createElement(ArtifactsSection, {
    sessionKey: 'sess-1',
    online: true,
    target: { id: htmlRow.id, count: 1 },
    initialDocument: { row: htmlRow, content },
  }))
  expect(previewHtml).toContain('rsec__widget-stage')
  expect(previewHtml).toContain('sandbox="allow-scripts"')
  expect(previewHtml).toContain('aria-label="Reload preview"')
  expect(previewHtml).toContain('aria-pressed="true" title="Show live interactive preview">Preview</button>')
  expect(previewHtml).toContain('aria-pressed="false" title="Show HTML source">Source</button>')
  expect(previewHtml).toContain('>v2<')
  expect(previewHtml).not.toContain('class="rsec__source"')

  const sourceHtml = renderToStaticMarkup(createElement(ArtifactsSection, {
    sessionKey: 'sess-1',
    online: true,
    target: { id: htmlRow.id, count: 1 },
    initialDocument: { row: htmlRow, content },
    initialSource: true,
  }))
  expect(sourceHtml).toContain('class="rsec__source"')
  expect(sourceHtml).toContain('hljs-tag')
  expect(sourceHtml).toContain('aria-pressed="true" title="Show HTML source">Source</button>')
  expect(sourceHtml).not.toContain('rsec__widget-stage')
})

test('ArtifactsSection renders Markdown review with unified header metadata and polished list badges', () => {
  const mdRow: ArtifactRow = {
    id: 'review-1',
    title: 'Eyvan whole-project review',
    kind: 'markdown',
    size: 4096,
    createdAt: '2026-11-22T00:00:00.000Z',
    updatedAt: '2026-11-22T00:10:00.000Z',
    pinned: true,
    version: 3,
  }
  const mdHtml = renderToStaticMarkup(createElement(ArtifactsSection, {
    sessionKey: 'sess-1',
    online: true,
    target: { id: mdRow.id, count: 1 },
    initialDocument: { row: mdRow, content: '# Eyvan whole-project review\n\n## Findings\n\nAll fifteen checks documented cleanly.' },
  }))
  expect(mdHtml).toContain('class="artdoc"')
  expect(mdHtml).toContain('<h1 class="artdoc__title">Eyvan whole-project review</h1>')
  expect(mdHtml).toContain('class="artdoc__chips rsec__meta"')
  expect(mdHtml).not.toContain('<p class="rsec__meta">')
  expect(mdHtml).toContain('min read')
  expect(mdHtml).toContain('>v3<')

  const listHtml = renderToStaticMarkup(createElement(ArtifactsSection, {
    sessionKey: 'sess-1',
    online: true,
    initialRows: [
      mdRow,
      { id: 'scene-3d', title: 'Simurgh 3D', kind: 'html', size: 1024, createdAt: mdRow.createdAt, updatedAt: mdRow.updatedAt, pinned: false, version: 2 },
    ],
  }))
  expect(listHtml).toContain('rsec__row--artifact')
  expect(listHtml).toContain('>Doc<')
  expect(listHtml).toContain('>Interactive<')
  expect(listHtml).toContain('>v3<')
  expect(listHtml).toContain('>v2<')
})
