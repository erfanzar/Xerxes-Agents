// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'

import { ActivityGroup } from '../src/desktop/renderer/App.js'
import { dedupeArtifacts } from '../src/desktop/renderer/ToolTimeline.js'
import type { Block, ToolItem } from '../src/desktop/renderer/types.js'

test('a collapsed tool group still shows the artifact card it saved', () => {
  const blocks: Block[] = [{ kind: 'tools', id: 1, running: false, items: [
    { id: 'r', verb: 'ReadFile', name: 'ReadFile', arg: 'a.py', dur: '1s', state: 'done', input: '{}', output: 'text' },
    { id: 'a', verb: 'Artifact', name: 'ArtifactTool', arg: 'Design', dur: '1s', state: 'done', input: '{"action":"create","kind":"markdown"}', output: 'Created artifact "Architecture: RL Training" (id abcdef1234, version 1)' },
  ] }]
  const html = renderToStaticMarkup(<ActivityGroup blocks={blocks} active={false} turnActive={false} />)
  // The group body is not rendered while collapsed, yet the card is.
  expect(html).not.toContain('a.py')
  expect(html).toContain('class="artifact-card"')
  expect(html).toContain('Architecture: RL Training')
})

test('multiple updates to the same artifact ID collapse to one card with the latest version while distinct IDs stay separate', () => {
  const items: ToolItem[] = [
    { id: 'a1', verb: 'Artifact', name: 'ArtifactTool', arg: 'Review', dur: '0.2s', state: 'done', input: '{"action":"create","kind":"markdown"}', output: 'Created artifact "Eyvan whole-project review" (id eyvan-01, version 1)' },
    { id: 'a2', verb: 'Artifact', name: 'ArtifactTool', arg: 'Review', dur: '0.3s', state: 'done', input: '{"action":"update","kind":"markdown"}', output: 'Updated artifact "Eyvan whole-project review" (id eyvan-01, version 2)' },
    { id: 'a3', verb: 'Artifact', name: 'ArtifactTool', arg: 'Review', dur: '0.3s', state: 'done', input: '{"action":"update","kind":"markdown"}', output: 'Updated artifact "Eyvan whole-project review" (id eyvan-01, version 3)' },
    { id: 'b1', verb: 'Artifact', name: 'ArtifactTool', arg: 'Scene', dur: '0.4s', state: 'done', input: '{"action":"create","kind":"html","embed":true}', output: 'Created artifact "Simurgh 3D Topology" (id scene-99, version 1)' },
    { id: 'b2', verb: 'Artifact', name: 'ArtifactTool', arg: 'Scene', dur: '0.2s', state: 'done', input: '{"action":"update","kind":"html"}', output: 'Updated artifact "Simurgh 3D Topology" (id scene-99, version 2)' },
  ]

  const deduped = dedupeArtifacts(items)
  expect(deduped).toHaveLength(2)
  expect(deduped[0]).toEqual({
    id: 'eyvan-01',
    title: 'Eyvan whole-project review',
    version: 3,
    action: 'updated',
    kind: 'markdown',
  })
  expect(deduped[1]).toEqual({
    id: 'scene-99',
    title: 'Simurgh 3D Topology',
    version: 2,
    action: 'updated',
    kind: 'html',
    embed: true,
  })

  const blocks: Block[] = [{ kind: 'tools', id: 2, running: false, items }]
  const html = renderToStaticMarkup(<ActivityGroup blocks={blocks} active={false} turnActive={false} />)
  const cardMatches = html.match(/class="artifact-card"/g) ?? []
  expect(cardMatches).toHaveLength(2)
  expect(html).toContain('Eyvan whole-project review')
  expect(html).toContain('>v3<')
  expect(html).not.toContain('updated to v2')
  expect(html).not.toContain('>v1<')
  expect(html).toContain('Simurgh 3D Topology')
  expect(html).toContain('>v2<')
})
