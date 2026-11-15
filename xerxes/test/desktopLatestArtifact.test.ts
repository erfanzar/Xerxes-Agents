// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { latestArtifactId } from '../src/desktop/renderer/ArtifactCard.js'
import type { Block, ToolItem } from '../src/desktop/renderer/types.js'

const artifactTool = (id: string, title: string): ToolItem => ({
  id: `call-${id}`, verb: 'Artifact', arg: title, dur: '', state: 'done', name: 'ArtifactTool',
  input: '{"action":"create","kind":"markdown"}', output: `Created artifact "${title}" (id ${id}, version 1)`,
})

test('the Plan ready card opens the newest artifact this conversation saved, not an older pinned one', () => {
  const blocks: Block[] = [
    { kind: 'tools', id: 1, running: false, items: [artifactTool('aaaaaa1111', 'Old plan')] },
    { kind: 'agent', id: 2, text: 'ok', streaming: false },
    { kind: 'tools', id: 3, running: false, items: [artifactTool('bbbbbb2222', 'New plan'), { ...artifactTool('x', 'y'), name: 'ReadFile', output: 'text' }] },
  ]
  expect(latestArtifactId(blocks)).toBe('bbbbbb2222')
  expect(latestArtifactId([{ kind: 'agent', id: 1, text: 'no artifacts', streaming: false }])).toBeUndefined()
})
