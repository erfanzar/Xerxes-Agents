// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { loadAgentSpec } from '../src/agents/agentSpec.js'
import {
  BUILTIN_AGENTS,
  listAgentDefinitionLoadErrors,
  loadAgentDefinitions,
  loadBuiltinAgentDefinitions,
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
  expect([...definitions.keys()].sort()).toEqual(['coder', 'default', 'objective', 'planner', 'researcher'])
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
})

test('definition loader applies user/project precedence, multi-agent files, and isolated errors', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-definitions-'))
  const user = join(root, 'user')
  const projectAgents = join(root, '.xerxes', 'agents')
  try {
    await mkdir(user, { recursive: true })
    await mkdir(projectAgents, { recursive: true })
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
    expect(listAgentDefinitionLoadErrors()).toHaveLength(1)
    expect(listAgentDefinitionLoadErrors()[0]).toContain('broken.yaml')
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
