// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { defineCallableSchema } from '../src/core/utils.js'
import {
  ChainType,
  CortexTool,
  NativeTemplate,
  ProcessType,
  PromptTemplate,
  TemplateSyntaxError,
  extractTemplateVariables,
  interpolateInputs,
  validateInputsForTemplate,
} from '../src/cortex/core/index.js'

test('Cortex process and chain values retain the documented workflow vocabulary', () => {
  expect(ProcessType).toEqual({
    SEQUENTIAL: 'sequential',
    HIERARCHICAL: 'hierarchical',
    PARALLEL: 'parallel',
    CONSENSUS: 'consensus',
    PLANNED: 'planned',
  })
  expect(ChainType).toEqual({ LINEAR: 'linear', BRANCHING: 'branching', LOOP: 'loop' })
})

test('Cortex string interpolation supports scalar, JSON, validation, and explicit failures', () => {
  expect(interpolateInputs('Hello {name}; {count}; {enabled}; {empty}', {
    name: 'World',
    count: 42,
    enabled: true,
    empty: null,
  })).toBe('Hello World; 42; True; ')
  expect(interpolateInputs('Data: {data}; Items: {items}', {
    data: { key: 'val' },
    items: [1, 2, 3],
  })).toBe('Data: {"key": "val"}; Items: [1, 2, 3]')
  expect(extractTemplateVariables('{name} {name} {_private}')).toEqual(new Set(['name', '_private']))
  expect(validateInputsForTemplate('Hello {name}', { name: 'Ada' })).toEqual({ valid: true, errors: [] })
  expect(validateInputsForTemplate('{name}', { name: 'Ada', extra: true }, false)).toEqual({
    valid: false,
    errors: ['Unexpected variable: extra'],
  })
  expect(() => interpolateInputs('{missing}', {})).toThrow("Missing required template variable 'missing'")
  expect(() => interpolateInputs('{value}', { value: new Date() })).toThrow('Unsupported type object')
})

test('NativeTemplate evaluates documented branches, loops, tuple items, and loop metadata without code evaluation', () => {
  const template = new NativeTemplate(`
{% if enabled %}Enabled{% else %}Disabled{% endif %}
{% for name, score in scores.items() %}{{ loop.index }}. {{ name }}={{ score }}{% if not loop.last %}, {% endif %}{% endfor %}
{{ globalThis.process }}
`)

  expect(template.render({
    enabled: true,
    scores: { Ada: 9, Grace: 10 },
  })).toContain('Enabled1. Ada=9, 2. Grace=10')
  expect(template.render({ enabled: false, scores: {} })).toContain('Disabled')
  expect(template.render({ enabled: true, scores: {} })).not.toContain(process.execPath)
  expect(template.variables()).toEqual(new Set(['enabled', 'scores', 'globalThis']))
  expect(() => new NativeTemplate('{% include "other" %}')).toThrow(TemplateSyntaxError)
})

test('PromptTemplate renders every documented Cortex prompt shape through the native renderer', () => {
  const prompts = new PromptTemplate()
  const tool = new CortexTool({
    name: 'lookup',
    description: 'Look up a fact.',
    function: () => 'fact',
    autoGenerateSchema: false,
  })

  const agent = prompts.renderAgentPrompt({
    role: 'Researcher',
    goal: 'Gather evidence',
    backstory: 'A careful analyst',
    instructions: 'Cite primary sources.',
    rules: ['Be concise'],
    tools: [tool],
  })
  expect(agent).toContain('Instructions:\nCite primary sources.')
  expect(agent).toContain('- lookup: Look up a fact.')
  expect(agent).toContain('- Be concise')

  const task = prompts.renderTaskPrompt({
    description: 'Review the design',
    expectedOutput: 'A verified report',
    context: 'Previous evidence',
    constraints: ['No network calls'],
  })
  expect(task).toContain('Context from previous tasks:\nPrevious evidence')
  expect(task).toContain('- No network calls')

  const delegation = prompts.renderManagerDelegation(
    [{ role: 'Writer', goal: 'Write clearly' }],
    [{ description: 'Draft the report', expectedOutput: 'Report' }],
  )
  expect(delegation).toContain('1. Draft the report')
  expect(delegation).toContain('Writer: Write clearly')
  expect(delegation).toContain('Expected: Report')

  const review = prompts.renderManagerReview({
    agentRole: 'Writer',
    taskDescription: 'Draft the report',
    output: 'Draft complete',
  })
  expect(review).toContain('Review the following output from Writer:')

  const consensus = prompts.renderConsensus('Choose a design', { Designer: 'Clear', Engineer: 'Feasible' })
  expect(consensus).toContain('Designer:\nClear')
  expect(consensus).toContain('Engineer:\nFeasible')

  const planner = prompts.renderPlanner({
    objective: 'Ship the port',
    agents: [{ role: 'Implementer', goal: 'Write native code', tools: [tool] }],
  })
  expect(planner).toContain('OBJECTIVE: Ship the port')
  expect(planner).toContain('Tools: CortexTool')
  expect(planner).toContain('CONTEXT: No additional context provided')

  const step = prompts.renderStepExecution({
    action: 'verify',
    description: 'Run focused tests',
    arguments: { suite: 'cortex' },
    context: 'Code has changed',
  })
  expect(step).toContain('- suite: cortex')
  expect(step).toContain('CONTEXT FROM PREVIOUS STEPS:\nCode has changed')
})

test('PromptTemplate custom compilation exposes undeclared variables and explicit simple interpolation mode', () => {
  const prompts = new PromptTemplate()
  const custom = prompts.createCustomTemplate('{% for item in items %}{{ item }}{% endfor %} {{ owner }}')
  expect(custom?.render({ items: ['a', 'b'], owner: 'Ada' })).toBe('ab Ada')
  expect(prompts.getTemplateVariables('{% for item in items %}{{ item }} {{ owner }}{% endfor %}')).toEqual(
    new Set(['items', 'owner']),
  )

  const disabled = new PromptTemplate({ renderingEnabled: false })
  expect(disabled.render('Hello {name}', { name: 'World' })).toBe('Hello World')
  expect(disabled.createCustomTemplate('Hello {{ name }}')).toBeUndefined()
  expect(disabled.getTemplateVariables('{{ name }}')).toEqual(new Set())
})

test('CortexTool reuses canonical callable schemas and preserves explicit parameter descriptors', () => {
  const lookup = defineCallableSchema(
    (topic: string) => `Fact for ${topic}`,
    {
      name: 'lookupFact',
      description: 'Look up one fact.',
      parameters: {
        type: 'object',
        properties: { topic: { type: 'string' } },
        required: ['topic'],
      },
    },
  )
  const generated = CortexTool.fromFunction(lookup)
  expect(generated).toMatchObject({ name: 'lookupFact', description: 'Look up one fact.' })
  expect(generated.toFunctionJson()).toEqual({
    type: 'function',
    function: {
      name: 'lookupFact',
      description: 'Look up one fact.',
      parameters: {
        type: 'object',
        properties: { topic: { type: 'string' } },
        required: ['topic'],
      },
    },
  })

  const explicit = new CortexTool({
    name: 'calculate',
    description: 'Calculate a value.',
    function: () => undefined,
    autoGenerateSchema: false,
    parameters: { type: 'object', properties: { value: { type: 'number' } } },
  })
  expect(explicit.toFunctionJson().function.parameters).toEqual({
    type: 'object',
    properties: { value: { type: 'number' } },
  })

  const empty = new CortexTool({
    name: 'empty',
    description: 'No inputs.',
    function: () => undefined,
    autoGenerateSchema: false,
  })
  expect(empty.toFunctionJson().function.parameters).toEqual({ type: 'object', properties: {}, required: [] })
})
