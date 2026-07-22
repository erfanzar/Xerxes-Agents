// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { loadAgentSpec } from '../src/agents/agentSpec.js'
import { AgentSpecError } from '../src/core/errors.js'
import {
  BUILTIN_AGENTS,
  listAgentDefinitionLoadErrors,
  loadAgentDefinitions,
  loadBuiltinAgentDefinitions,
  type AgentDefinition,
} from '../src/agents/definitions.js'
import {
  AgentOrchestrator,
  AgentSwitchTrigger,
  registerDefaultSwitchTriggers,
  type OrchestratedAgent,
} from '../src/agents/orchestrator.js'

test('agent specs resolve inheritance, subagents, blocks, and prompt substitutions', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-spec-'))
  try {
    await writeFile(join(root, 'base.md'), 'Role: ${ROLE:-general}; unresolved: ${unknown}\n', 'utf8')
    await writeFile(join(root, 'base.yaml'), `version: 1
agent:
  name: base
  system_prompt_path: ./base.md
  system_prompt_args:
    ROLE: base
  tools: [ReadFile]
  allowed_tools:
    - ReadFile
  subagents:
    helper:
      path: ./helper.yaml
      description: Base helper
  max_depth: 2
`, 'utf8')
    await writeFile(join(root, 'child.yaml'), `version: 1
agent:
  extend: ./base.yaml
  name: child
  system_prompt_args:
    ROLE: coder
  tools:
    - WriteFile
  allowed_tools: null
  exclude_tools: [exec_command]
  subagents:
    reviewer:
      path: ./reviewer.yaml
      description: Reviews changes
`, 'utf8')

    const spec = loadAgentSpec(join(root, 'child.yaml'))
    expect(spec).toMatchObject({
      name: 'child',
      systemPrompt: 'Role: coder; unresolved: ${unknown}\n',
      tools: ['WriteFile'],
      allowedTools: null,
      excludeTools: ['exec_command'],
      maxDepth: 2,
    })
    expect(Object.keys(spec.subagents).sort()).toEqual(['helper', 'reviewer'])
    expect(spec.subagents.reviewer?.path).toBe(join(root, 'reviewer.yaml'))
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('built-in definitions are TypeScript-owned and retain resolved specialist prompts', () => {
  const definitions = loadBuiltinAgentDefinitions()
  expect([...definitions.keys()].sort()).toEqual([
    'coder',
    'default',
    'objective',
    'planner',
    'researcher',
    'reviewer',
    'tester',
  ])
  expect(BUILTIN_AGENTS.get('coder')?.systemPrompt).toContain('coding specialist focused on software engineering implementation')
  expect(definitions.get('default')?.tools).toEqual(expect.arrayContaining([
    'SpawnAgents',
    'agent_memory_status',
    'agent_memory_read',
    'agent_memory_write',
    'agent_memory_append',
    'agent_memory_list',
    'agent_memory_search',
    'agent_memory_journal',
  ]))
  expect(definitions.get('default')?.subagents).toMatchObject({
    coder: { description: 'Good at general software engineering tasks.' },
    objective: { description: 'Hard-goal execution loop with verification gates.' },
    planner: { description: 'Read-only implementation planning and architecture design.' },
    researcher: { description: 'Fast codebase exploration with prompt-enforced read-only behavior.' },
    reviewer: { description: 'Independent read-only code review with prioritized findings.' },
    tester: { description: 'Focused test authoring and verification without recursive delegation.' },
  })
  expect(Object.isFrozen(definitions.get('default')?.subagents)).toBeTrue()
  expect(Object.isFrozen(definitions.get('default')?.subagents?.coder)).toBeTrue()
  expect(definitions.get('researcher')?.tools).toEqual(expect.arrayContaining([
    'agent_memory_status',
    'agent_memory_read',
    'agent_memory_write',
    'agent_memory_append',
    'agent_memory_list',
    'agent_memory_search',
    'agent_memory_journal',
  ]))
  expect(definitions.get('objective')?.tools).toContain('agent_memory_journal')
  expect(definitions.get('objective')?.tools).toEqual(expect.arrayContaining([
    'AgentTool',
    'SpawnAgents',
    'AwaitAgents',
    'TaskOutputTool',
  ]))
  expect(definitions.get('objective')?.subagents).toMatchObject({
    coder: { description: 'Focused implementation for a disjoint part of the objective.' },
    researcher: { description: 'Read-only evidence gathering for a bounded objective question.' },
    reviewer: { description: 'Independent read-only review when changed paths or diff context are supplied.' },
    tester: { description: 'Focused test authoring and verification for the current objective.' },
  })
  expect(definitions.get('reviewer')?.allowedTools).toEqual([
    'ReadFile',
    'GlobTool',
    'GrepTool',
    'ListDir',
  ])
  expect(definitions.get('reviewer')?.excludeTools).toEqual(expect.arrayContaining([
    'AgentTool',
    'SpawnAgents',
    'WriteFile',
    'FileEditTool',
    'exec_command',
  ]))
  expect(definitions.get('reviewer')?.systemPrompt).toContain('read-only code review specialist')
  expect(definitions.get('tester')?.allowedTools).toEqual(expect.arrayContaining([
    'ReadFile',
    'WriteFile',
    'FileEditTool',
    'exec_command',
    'ListDir',
  ]))
  expect(definitions.get('tester')?.excludeTools).toEqual(expect.arrayContaining([
    'AgentTool',
    'SpawnAgents',
  ]))
  expect(definitions.get('tester')?.systemPrompt).toContain('testing specialist')
})

test('definition loader applies user/project precedence, multi-agent files, and isolated errors', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-definitions-'))
  const user = join(root, 'user')
  const projectAgents = join(root, '.xerxes', 'agents')
  try {
    await mkdir(user, { recursive: true })
    await mkdir(join(projectAgents, 'nested'), { recursive: true })
    await writeFile(join(user, 'shared.yaml'), `version: 1
agent:
  name: shared
  system_prompt: user prompt
`, 'utf8')
    await writeFile(join(user, 'broken.yaml'), '- this is not an agent mapping\n', 'utf8')
    await writeFile(join(projectAgents, 'shared.md'), `---
description: Project override
tools: [ReadFile, GrepTool]
max_depth: 3
---
project prompt
`, 'utf8')
    await writeFile(join(root, 'agents.yaml'), `version: 1
agents:
  embedded:
    system_prompt: embedded prompt
    allowed_tools:
      - ReadFile
  parent:
    system_prompt: parent prompt
    subagents:
      audit:
        path: ./.xerxes/agents/nested/reviewer.yaml
        description: Nested audit alias
`, 'utf8')
    await writeFile(join(projectAgents, 'nested', 'reviewer.yaml'), `version: 1
agent:
  name: internal-reviewer
  system_prompt: nested reviewer prompt
  allowed_tools: [ReadFile]
`, 'utf8')

    const definitions = loadAgentDefinitions({
      builtinDefinitions: new Map(),
      cwd: root,
      userDirectory: user,
      projectDirectory: projectAgents,
    })
    expect(definitions.get('shared')).toMatchObject({
      source: 'project',
      description: 'Project override',
      systemPrompt: 'project prompt',
      tools: ['ReadFile', 'GrepTool'],
      maxDepth: 3,
    })
    expect(definitions.get('embedded')).toMatchObject({
      systemPrompt: 'embedded prompt',
      allowedTools: ['ReadFile'],
      source: 'project',
    })
    // Referenced-only profiles never claim the plain alias: audit is reachable
    // exclusively through its creator-bound catalog key.
    expect(definitions.get('audit')).toBeUndefined()
    const auditKey = definitions.get('parent')?.subagents?.audit?.resolvedProfile ?? ''
    expect(auditKey).toStartWith('@catalog:audit:')
    expect(definitions.get(auditKey)).toMatchObject({
      name: 'audit',
      source: 'project',
      systemPrompt: 'nested reviewer prompt',
      allowedTools: ['ReadFile'],
    })
    expect(listAgentDefinitionLoadErrors()).toHaveLength(1)
    expect(listAgentDefinitionLoadErrors()[0]).toContain('broken.yaml')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('creator catalogs bind colliding aliases to their declared paths and omit broken references', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-catalog-'))
  const projectAgents = join(root, '.xerxes', 'agents')
  const globalCoder: AgentDefinition = {
    allowedTools: ['WriteFile'],
    description: 'global writable coder',
    excludeTools: [],
    isolation: '',
    maxDepth: 3,
    model: 'global-model',
    name: 'coder',
    source: 'built-in',
    systemPrompt: 'global coder prompt',
    tools: ['WriteFile'],
  }
  try {
    await mkdir(projectAgents, { recursive: true })
    await writeFile(join(projectAgents, 'readonly-coder.yaml'), `version: 1
agent:
  name: internal-readonly-coder
  system_prompt: creator-local readonly coder
  model: child-model
  allowed_tools: [ReadFile]
`, 'utf8')
    await writeFile(join(root, 'agents.yaml'), `version: 1
agents:
  parent:
    system_prompt: parent prompt
    subagents:
      coder:
        path: ./.xerxes/agents/readonly-coder.yaml
        description: Creator-local coder
      missing:
        path: ./.xerxes/agents/missing.yaml
        description: Broken child
`, 'utf8')

    const definitions = loadAgentDefinitions({
      builtinDefinitions: new Map([['coder', globalCoder]]),
      cwd: root,
      projectDirectory: projectAgents,
      userDirectory: join(root, 'user'),
    })
    const reference = definitions.get('parent')?.subagents?.coder
    expect(reference?.resolvedProfile).toStartWith('@catalog:coder:')
    expect(definitions.get(reference?.resolvedProfile ?? '')).toMatchObject({
      name: 'coder',
      model: 'child-model',
      systemPrompt: 'creator-local readonly coder',
      allowedTools: ['ReadFile'],
    })
    expect(definitions.get('coder')).toMatchObject({ model: 'global-model', allowedTools: ['WriteFile'] })
    expect(definitions.get('parent')?.subagents?.missing).toBeUndefined()
    expect(listAgentDefinitionLoadErrors().some(error => error.includes('missing.yaml'))).toBeTrue()
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('orchestrator routes capability and recovery triggers while recording switches', () => {
  const errors: AgentSwitchTrigger[] = []
  const orchestrator = new AgentOrchestrator({
    now: () => new Date('2026-07-13T12:00:00.000Z'),
    onTriggerError: trigger => errors.push(trigger),
  })
  const general: OrchestratedAgent = {
    id: 'general',
    fallbackAgentId: 'recovery',
    switchTriggers: [AgentSwitchTrigger.CAPABILITY_BASED, AgentSwitchTrigger.ERROR_RECOVERY],
    capabilities: [{ name: 'research', description: 'Researches', performanceScore: 1 }],
  }
  const specialist: OrchestratedAgent = {
    id: 'specialist',
    capabilities: [{ name: 'code', description: 'Writes code', performanceScore: 2 }],
  }
  const recovery: OrchestratedAgent = { id: 'recovery' }
  orchestrator.registerAgent(general)
  orchestrator.registerAgent(specialist)
  orchestrator.registerAgent(recovery)
  registerDefaultSwitchTriggers(orchestrator)

  expect(orchestrator.shouldSwitchAgent({ required_capability: 'code' })).toBe('specialist')
  expect(orchestrator.shouldSwitchAgent({ execution_error: true })).toBe('recovery')
  orchestrator.switchAgent('specialist', 'specialized work')
  expect(orchestrator.currentAgentId).toBe('specialist')
  expect(orchestrator.executionHistory).toEqual([{
    action: 'agent_switch',
    type: 'agent_switch',
    from: 'general',
    to: 'specialist',
    reason: 'specialized work',
    timestamp: '2026-07-13T12:00:00.000Z',
  }])
  orchestrator.registerSwitchTrigger(AgentSwitchTrigger.CUSTOM, () => {
    throw new Error('bad custom trigger')
  })
  expect(orchestrator.shouldSwitchAgent({})).toBeUndefined()
  expect(errors).toEqual([AgentSwitchTrigger.CUSTOM])
})

test('agent specs reject mapping values for scalar fields and missing prompt files', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-spec-invalid-'))
  try {
    await writeFile(join(root, 'mapping.yaml'), `version: 1
agent:
  name:
    nested: value
  system_prompt: hello
`, 'utf8')
    expect(() => loadAgentSpec(join(root, 'mapping.yaml'))).toThrow(AgentSpecError)
    expect(() => loadAgentSpec(join(root, 'mapping.yaml'))).toThrow('agent.name must be a scalar')

    await writeFile(join(root, 'missing-prompt.yaml'), `version: 1
agent:
  name: missing-prompt
  system_prompt_path: ./does-not-exist.md
`, 'utf8')
    expect(() => loadAgentSpec(join(root, 'missing-prompt.yaml'))).toThrow(AgentSpecError)
    expect(() => loadAgentSpec(join(root, 'missing-prompt.yaml'))).toThrow('System prompt file not found')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
