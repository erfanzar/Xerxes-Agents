// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { loadAgentDefinitions } from '../src/agents/definitions.js'
import { toolListAdmits } from '../src/agents/toolPatterns.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { addLearningToolsToBuiltinAgents, registerSessionSearchTool } from '../src/tools/agentMetaTools.js'
import { MODEL_FACING_LEARNING_TOOLS, registerLessonTools } from '../src/tools/lessonTools.js'

// These tools shipped registered but absent from every built-in agent's tool
// list, so no model could call them (0 calls in 95 real sessions). This pins
// that the shipped default agent actually sees them, with usage guidance.
test('the shipped default agent admits session_search, lesson_record and lesson_recall', async () => {
  const cwd = await mkdtemp(join(tmpdir(), 'xerxes-learning-tools-'))
  try {
    const definitions = loadAgentDefinitions({ cwd })
    const before = definitions.get('default')!
    expect(before.source).toBe('built-in')
    addLearningToolsToBuiltinAgents(definitions, MODEL_FACING_LEARNING_TOOLS)
    const agent = definitions.get('default')!
    for (const name of MODEL_FACING_LEARNING_TOOLS) {
      expect(agent.tools.length === 0 || toolListAdmits(agent.tools, name)).toBe(true)
    }
    // Human-review tools stay out of the model's default surface.
    expect(toolListAdmits(agent.tools, 'lesson_review')).toBe(false)
  } finally { await rm(cwd, { recursive: true, force: true }) }
})

test('user-defined agents keep their own allow lists', () => {
  const definitions = new Map([['mine', { name: 'mine', source: 'user', tools: ['ReadFile'] } as never]])
  addLearningToolsToBuiltinAgents(definitions, MODEL_FACING_LEARNING_TOOLS)
  expect((definitions.get('mine') as unknown as { tools: string[] }).tools).toEqual(['ReadFile'])
})

test('the learning tools carry when-to-use guidance and are not deferred', () => {
  const registry = new ToolRegistry()
  registerLessonTools(registry, { store: () => { throw new Error('unused') }, activeWorkspace: () => '/tmp' })
  registerSessionSearchTool(registry, { async search() { return [] } } as never)
  const names = registry.guidanceForTools(['session_search', 'lesson_record', 'lesson_recall', 'lesson_review']).map(segment => segment.name)
  expect(names).toEqual(expect.arrayContaining(['session_search', 'lesson_record', 'lesson_recall']))
  expect(names).not.toContain('lesson_review')
})
