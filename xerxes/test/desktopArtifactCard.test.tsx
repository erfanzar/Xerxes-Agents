// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { artifactOfTool } from '../src/desktop/renderer/ArtifactCard.js'
import { ToolCallRow } from '../src/desktop/renderer/Execution.js'
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
