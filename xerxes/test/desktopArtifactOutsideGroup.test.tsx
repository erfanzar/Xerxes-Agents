// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'

import { ActivityGroup } from '../src/desktop/renderer/App.js'
import type { Block } from '../src/desktop/renderer/types.js'

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
