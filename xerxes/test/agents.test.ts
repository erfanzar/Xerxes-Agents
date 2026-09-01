// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, spyOn, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  AGENT_SPEC_ISOLATION_MODES,
  drainAgentSpecDiagnostics,
  loadAgentSpec,
} from '../src/agents/agentSpec.js'
import { AgentSpecError } from '../src/core/errors.js'
import {
  BUILTIN_AGENTS,
  formatAgentDefinitionLoadErrors,
  listAgentDefinitionLoadErrors,
  loadAgentDefinitions,
  loadBuiltinAgentDefinitions,
  parseAgentMarkdown,
  resolveAgentDefinition,
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
    'creator',
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
  expect(definitions.get('creator')?.tools).toEqual(expect.arrayContaining([
    'AgentPresetInspectTool',
    'AgentPresetTool',
    'CreatorRuntimeTool',
  ]))
  expect(definitions.get('creator')?.systemPrompt).toContain('Creator mode')
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

test('agent specs accept explicit isolation modes and inherit them across extend chains', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-spec-isolation-'))
  try {
    const write = async (name: string, isolation?: string): Promise<string> => {
      const path = join(root, name)
      const field = isolation === undefined ? '' : `  isolation: ${isolation === '' ? "''" : isolation}\n`
      await writeFile(path, `version: 1\nagent:\n  name: ${name.replace('.yaml', '')}\n  system_prompt: prompt\n${field}`, 'utf8')
      return path
    }

    expect(loadAgentSpec(await write('worktree.yaml', 'worktree')).isolation).toBe('worktree')
    expect(loadAgentSpec(await write('shared.yaml', 'shared')).isolation).toBe('shared')
    // An explicit empty value opts out instead of inheriting.
    expect(loadAgentSpec(await write('unset.yaml', '')).isolation).toBe('')
    // Absent fields keep inheriting across extend chains.
    await writeFile(join(root, 'base.yaml'), `version: 1
agent:
  name: base
  system_prompt: base prompt
  isolation: worktree
`, 'utf8')
    await writeFile(join(root, 'child.yaml'), `version: 1
agent:
  extend: ./base.yaml
  name: child
`, 'utf8')
    expect(loadAgentSpec(join(root, 'child.yaml')).isolation).toBe('worktree')
    // A child may still override the inherited mode explicitly.
    await writeFile(join(root, 'override.yaml'), `version: 1
agent:
  extend: ./base.yaml
  name: override
  isolation: shared
`, 'utf8')
    expect(loadAgentSpec(join(root, 'override.yaml')).isolation).toBe('shared')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('agent specs reject unknown isolation modes instead of silently downgrading children', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-spec-isolation-invalid-'))
  try {
    const path = join(root, 'typo-mode.yaml')
    await writeFile(path, `version: 1
agent:
  name: typo-mode
  system_prompt: prompt
  isolation: git-worktree
`, 'utf8')
    expect(() => loadAgentSpec(path)).toThrow(AgentSpecError)
    expect(() => loadAgentSpec(path)).toThrow(/isolation must be one of.*got 'git-worktree'/u)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('unknown agent-spec fields surface through the loader error channel without dropping healthy siblings', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-spec-unknown-'))
  const projectAgents = join(root, '.xerxes', 'agents')
  try {
    await mkdir(projectAgents, { recursive: true })
    const misspelled = join(projectAgents, 'misspelled-field.yaml')
    await writeFile(misspelled, `version: 1
agent:
  name: misspelled-field
  system_prompt: prompt
  isolaton: worktree
`, 'utf8')
    await writeFile(join(projectAgents, 'healthy-sibling.yaml'), `version: 1
agent:
  name: healthy-sibling
  system_prompt: healthy prompt
`, 'utf8')

    // Direct loads fail with an AgentSpecError naming the file and the field...
    expect(() => loadAgentSpec(misspelled)).toThrow(AgentSpecError)
    expect(() => loadAgentSpec(misspelled)).toThrow(/contains unknown agent-spec field 'isolaton'/u)

    // ...and directory loads record the failure per file while unrelated specs still load.
    const options = {
      builtinDefinitions: new Map(),
      cwd: root,
      userDirectory: join(root, 'no-user-agents'),
      projectDirectory: projectAgents,
    }
    const definitions = loadAgentDefinitions(options)
    expect(definitions.get('healthy-sibling')?.systemPrompt).toBe('healthy prompt')
    expect(definitions.get('misspelled-field')).toBeUndefined()
    const errors = listAgentDefinitionLoadErrors()
    expect(errors.some(error => error.includes('misspelled-field.yaml') && error.includes("'isolaton'"))).toBeTrue()

    // Unknown top-level sections are rejected the same way.
    const topLevelTypo = join(projectAgents, 'top-level-typo.yaml')
    await writeFile(topLevelTypo, `verison: 1
agent:
  name: top-level-typo
  system_prompt: prompt
`, 'utf8')
    expect(() => loadAgentSpec(topLevelTypo)).toThrow(/contains unknown agent-spec section 'verison'/u)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('definition loading surfaces collected spec errors on stderr once per distinct set', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-spec-surface-'))
  const projectAgents = join(root, '.xerxes', 'agents')
  const emitted: string[] = []
  try {
    await mkdir(projectAgents, { recursive: true })
    await writeFile(join(projectAgents, 'healthy.yaml'), `version: 1
agent:
  name: healthy
  system_prompt: healthy prompt
`, 'utf8')
    await writeFile(join(projectAgents, 'invalid.yaml'), `version: 1
agent:
  name: invalid
  system_prompt: prompt
  isolation: git-worktree
`, 'utf8')
    const options = {
      builtinDefinitions: new Map(),
      cwd: root,
      userDirectory: join(root, 'no-user-agents'),
      projectDirectory: projectAgents,
    }

    const spy = spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
      emitted.push(args.map(part => String(part)).join(' '))
    })
    try {
      // The failing spec vanishes from the catalog, but the reason is announced.
      const definitions = loadAgentDefinitions(options)
      expect(definitions.get('healthy')?.systemPrompt).toBe('healthy prompt')
      expect(definitions.get('invalid')).toBeUndefined()
      expect(emitted).toHaveLength(1)
      expect(emitted[0]).toContain('[xerxes] 1 agent definition issue(s)')
      expect(emitted[0]).toContain('invalid.yaml')
      expect(emitted[0]).toContain("got 'git-worktree'")

      // Re-resolving the unchanged catalog does not flood stderr with repeats.
      loadAgentDefinitions(options)
      loadAgentDefinitions(options)
      expect(emitted).toHaveLength(1)
    } finally {
      spy.mockRestore()
    }

    // A newly broken file is a changed error set and is announced again.
    await writeFile(join(projectAgents, 'worse.yaml'), '- not an agent mapping\n', 'utf8')
    const secondSpy = spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
      emitted.push(args.map(part => String(part)).join(' '))
    })
    try {
      loadAgentDefinitions(options)
      expect(emitted).toHaveLength(2)
      expect(emitted[1]).toContain('[xerxes] 2 agent definition issue(s)')
      expect(emitted[1]).toContain('worse.yaml')
    } finally {
      secondSpy.mockRestore()
    }
    expect(formatAgentDefinitionLoadErrors()).toContain('invalid.yaml')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('YAML description is accepted as a deprecated alias of when_to_use', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-spec-alias-'))
  try {
    // Description alone resolves like when_to_use with no diagnostic noise.
    await writeFile(join(root, 'aliased.yaml'), `version: 1
agent:
  name: aliased
  system_prompt: prompt
  description: Runs QA passes.
`, 'utf8')
    const aliased = loadAgentSpec(join(root, 'aliased.yaml'))
    expect(aliased.whenToUse).toBe('Runs QA passes.')
    expect(drainAgentSpecDiagnostics()).toEqual([])

    // Both spellings: when_to_use wins and a note rides the load-error channel.
    await writeFile(join(root, 'both.yaml'), `version: 1
agent:
  name: both
  system_prompt: prompt
  description: markdown spelling
  when_to_use: canonical spelling
`, 'utf8')
    const both = loadAgentSpec(join(root, 'both.yaml'))
    expect(both.whenToUse).toBe('canonical spelling')
    const notes = drainAgentSpecDiagnostics()
    expect(notes).toHaveLength(1)
    expect(notes[0]).toContain("'description' is deprecated")
    expect(notes[0]).toContain('both.yaml')

    // Through the definition loader the same note reaches listAgentDefinitionLoadErrors.
    await mkdir(join(root, '.xerxes', 'agents'), { recursive: true })
    await writeFile(join(root, '.xerxes', 'agents', 'both-loader.yaml'), `version: 1
agent:
  name: both-loader
  system_prompt: prompt
  description: loader alias
  when_to_use: loader canonical
`, 'utf8')
    const definitions = loadAgentDefinitions({
      builtinDefinitions: new Map(),
      cwd: root,
      userDirectory: join(root, 'no-user-agents'),
      projectDirectory: join(root, '.xerxes', 'agents'),
    })
    expect(definitions.get('both-loader')?.description).toBe('loader canonical')
    expect(listAgentDefinitionLoadErrors().some(error =>
      error.includes('both-loader.yaml') && error.includes("'description' is deprecated"),
    )).toBeTrue()
  } finally {
    drainAgentSpecDiagnostics()
    await rm(root, { recursive: true, force: true })
  }
})

test('markdown frontmatter enforces recognized fields and isolation modes', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-md-frontmatter-'))
  const writeMarkdown = async (name: string, fields: string): Promise<string> => {
    const path = join(root, name)
    await writeFile(path, `---\n${fields}---\nbody prompt\n`, 'utf8')
    return path
  }
  try {
    // An isolation typo used to be accepted and silently downgraded children.
    const badIsolation = await writeMarkdown('bad-isolation.md', 'description: x\nisolation: worktrees\n')
    expect(() => parseAgentMarkdown(badIsolation)).toThrow(AgentSpecError)
    expect(() => parseAgentMarkdown(badIsolation)).toThrow(/frontmatter\.isolation must be one of.*got 'worktrees'/u)

    // A depth_limit typo used to be silently ignored, keeping the default depth.
    const badDepthKey = await writeMarkdown('bad-depth-key.md', 'description: x\ndepth_limit: 9\n')
    expect(() => parseAgentMarkdown(badDepthKey)).toThrow(AgentSpecError)
    expect(() => parseAgentMarkdown(badDepthKey)).toThrow(/contains unknown field 'depth_limit'/u)

    // Every documented field still parses, and directory loads record rejections per file.
    const valid = await writeMarkdown(
      'valid.md',
      'description: Notes keeper\nmodel: gpt-4o\nmax_depth: 3\ntools: [ReadFile]\nisolation: shared\n',
    )
    expect(parseAgentMarkdown(valid)).toMatchObject({
      description: 'Notes keeper',
      model: 'gpt-4o',
      maxDepth: 3,
      isolation: 'shared',
      systemPrompt: 'body prompt',
      tools: ['ReadFile'],
    })

    await mkdir(join(root, '.xerxes', 'agents'), { recursive: true })
    await writeFile(join(root, '.xerxes', 'agents', 'broken.md'), '---\ndepth_limit: 9\n---\nprompt\n', 'utf8')
    await writeFile(join(root, '.xerxes', 'agents', 'fine.md'), '---\ndescription: fine\n---\nprompt\n', 'utf8')
    const definitions = loadAgentDefinitions({
      builtinDefinitions: new Map(),
      cwd: root,
      userDirectory: join(root, 'no-user-agents'),
      projectDirectory: join(root, '.xerxes', 'agents'),
    })
    expect(definitions.get('fine')?.systemPrompt).toBe('prompt')
    expect(definitions.has('broken')).toBeFalse()
    expect(listAgentDefinitionLoadErrors().some(error =>
      error.includes('broken.md') && error.includes("'depth_limit'"),
    )).toBeTrue()
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('subagent entries reject missing, empty, or mistyped paths instead of dropping them', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-subagent-paths-'))
  try {
    const cases: ReadonlyArray<readonly [string, string, RegExp]> = [
      ['missing-path.yaml', '    helper:\n      description: no path declared\n', /subagents\.helper must declare a non-empty string 'path'/u],
      ['empty-path.yaml', '    helper:\n      path: ""\n', /subagents\.helper must declare a non-empty string 'path'/u],
      ['typo-path.yaml', '    helper:\n      pth: ./helper.yaml\n', /subagents\.helper contains unknown agent-spec field 'pth'/u],
      ['null-entry.yaml', '    helper:\n', /subagents\.helper must be a mapping with a 'path' field/u],
      ['list-entry.yaml', '    helper: [a, b]\n', /subagents\.helper must be a mapping with a 'path' field/u],
    ]
    for (const [name, subagentsBlock, pattern] of cases) {
      const path = join(root, name)
      await writeFile(path, `version: 1\nagent:\n  name: ${name.replace('.yaml', '')}\n  system_prompt: prompt\n  subagents:\n${subagentsBlock}`, 'utf8')
      expect(() => loadAgentSpec(path), name).toThrow(AgentSpecError)
      expect(() => loadAgentSpec(path), name).toThrow(pattern)
    }

    // Shorthand string entries and full mappings keep working side by side.
    const valid = join(root, 'valid.yaml')
    await writeFile(valid, `version: 1
agent:
  name: valid-parent
  system_prompt: prompt
  subagents:
    shorthand: ./shorthand-child.yaml
    full:
      path: ./full-child.yaml
      description: Full entry
`, 'utf8')
    const spec = loadAgentSpec(valid)
    expect(spec.subagents.shorthand?.path).toBe(join(root, 'shorthand-child.yaml'))
    expect(spec.subagents.full).toMatchObject({ path: join(root, 'full-child.yaml'), description: 'Full entry' })
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('bundled specs survive strict validation and every documented field still resolves', async () => {
  // Every bundled YAML spec must keep loading under the strict parser.
  const bundled = loadBuiltinAgentDefinitions()
  expect([...bundled.keys()].sort()).toEqual([
    'coder',
    'creator',
    'default',
    'objective',
    'planner',
    'researcher',
    'reviewer',
    'tester',
  ])
  for (const definition of bundled.values()) {
    expect(AGENT_SPEC_ISOLATION_MODES as readonly string[]).toContain(definition.isolation)
  }
  expect(BUILTIN_AGENTS.get('default')?.isolation).toBe('')

  // A spec using every documented field resolves end to end.
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-spec-full-'))
  try {
    await writeFile(join(root, 'prompt.md'), 'Role: ${ROLE:-general}\n', 'utf8')
    await writeFile(join(root, 'full.yaml'), `version: 1
agent:
  name: kitchen-sink
  system_prompt_path: ./prompt.md
  system_prompt_args:
    ROLE: sink
  model: gpt-4o
  when_to_use: Full documentation coverage.
  tools:
    - ReadFile
  allowed_tools:
    - ReadFile
  exclude_tools:
    - exec_command
  max_depth: 2
  isolation: worktree
  subagents:
    helper:
      path: ./helper.yaml
      description: Helper child
`, 'utf8')
    const spec = loadAgentSpec(join(root, 'full.yaml'))
    expect(spec).toMatchObject({
      name: 'kitchen-sink',
      systemPrompt: 'Role: sink\n',
      model: 'gpt-4o',
      whenToUse: 'Full documentation coverage.',
      tools: ['ReadFile'],
      allowedTools: ['ReadFile'],
      excludeTools: ['exec_command'],
      maxDepth: 2,
      isolation: 'worktree',
    })
    expect(spec.subagents.helper?.path).toBe(join(root, 'helper.yaml'))
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('resolveAgentDefinition resolves catalog names, YAML paths, and Markdown paths', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-resolve-'))
  try {
    // Isolated user/project directories keep the host's real agents out of the
    // catalog so the assertions only see built-ins plus these fixtures.
    const options = {
      cwd: root,
      userDirectory: join(root, 'no-user-agents'),
      projectDirectory: join(root, 'no-project-agents'),
    }
    await writeFile(join(root, 'qa.yaml'), `version: 1
agent:
  name: qa
  when_to_use: Runs QA passes.
  system_prompt: You are QA.
  tools: [ReadFile]
`, 'utf8')
    await writeFile(join(root, 'notes.md'), `---
description: Notes agent
tools: [ReadFile]
---
You keep notes.
`, 'utf8')

    const named = resolveAgentDefinition('researcher', options)
    expect(named.name).toBe('researcher')

    const fromYaml = resolveAgentDefinition('./qa.yaml', options)
    expect(fromYaml).toMatchObject({
      name: 'qa',
      description: 'Runs QA passes.',
      systemPrompt: 'You are QA.',
      tools: ['ReadFile'],
      source: 'cli',
    })

    const fromMarkdown = resolveAgentDefinition('./notes.md', options)
    expect(fromMarkdown).toMatchObject({
      name: 'notes',
      description: 'Notes agent',
      systemPrompt: 'You keep notes.',
      tools: ['ReadFile'],
      source: 'cli',
    })

    expect(() => resolveAgentDefinition('does-not-exist', options)).toThrow(AgentSpecError)
    expect(() => resolveAgentDefinition('does-not-exist', options)).toThrow(
      /Unknown agent 'does-not-exist'\. Available agents: .*researcher/,
    )
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
