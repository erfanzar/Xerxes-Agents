// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { parseSkillMarkdown, skillPromptSection } from '../src/extensions/skills.js'
import { PromptContextBuilder } from '../src/runtime/promptContext.js'
import {
  PromptProfile,
  getPromptProfileConfig,
  resolvePromptProfileConfig,
} from '../src/runtime/promptProfiles.js'

test('full prompt profile enables the complete native context surface', () => {
  const config = getPromptProfileConfig(PromptProfile.FULL)

  expect(config).toMatchObject({
    profile: 'full',
    includeRuntimeInfo: true,
    includeWorkspaceInfo: true,
    includeSandboxInfo: true,
    includeSkillsIndex: true,
    includeEnabledSkills: true,
    includeToolsList: true,
    includeGuardrails: true,
    includeBootstrap: true,
    includeRelevantMemories: true,
    includeUserProfile: true,
    includeRepoMap: true,
    includeGitInfo: true,
    maxMemoriesInjected: 5,
  })
  // FULL keeps generous-but-finite caps so one oversized skill body or an
  // unbounded tool list cannot grow every turn's prompt without limit.
  expect(config.maxSkillInstructionsLength).toBe(8_000)
  expect(config.maxToolsListed).toBe(100)
  expect(Object.isFrozen(config)).toBe(true)
})

test('compact, minimal, and none profiles enforce their documented gates and caps', () => {
  const compact = getPromptProfileConfig(' COMPACT ')
  const minimal = getPromptProfileConfig(PromptProfile.MINIMAL)
  const none = getPromptProfileConfig(PromptProfile.NONE)

  expect(compact).toMatchObject({
    profile: 'compact', includeWorkspaceInfo: false, includeBootstrap: false, includeRepoMap: false,
    maxSkillInstructionsLength: 500, maxToolsListed: 20,
  })
  expect(minimal).toMatchObject({
    profile: 'minimal', includeRuntimeInfo: false, includeSandboxInfo: true, includeToolsList: true,
    includeGuardrails: true, includeRelevantMemories: false, includeUserProfile: false, maxToolsListed: 10,
  })
  expect(none).toMatchObject({
    profile: 'none', includeRuntimeInfo: false, includeSandboxInfo: false, includeToolsList: false,
    includeGuardrails: false, includeBootstrap: false, includeRelevantMemories: false,
  })
})

test('resolution defaults to full and validates explicit profile configuration', () => {
  expect(resolvePromptProfileConfig(undefined).profile).toBe(PromptProfile.FULL)
  expect(resolvePromptProfileConfig({
    ...getPromptProfileConfig(PromptProfile.COMPACT),
    maxToolsListed: 3,
  }).maxToolsListed).toBe(3)
  expect(() => getPromptProfileConfig('verbose')).toThrow('Unknown prompt profile: verbose')
  expect(() => resolvePromptProfileConfig({
    ...getPromptProfileConfig(PromptProfile.FULL),
    maxMemoriesInjected: -1,
  })).toThrow('maxMemoriesInjected must be a non-negative safe integer')
  expect(() => resolvePromptProfileConfig({
    ...getPromptProfileConfig(PromptProfile.FULL),
    profile: 'unsupported' as PromptProfile,
  })).toThrow('Unknown prompt profile: unsupported')
})

test('prompt-profile rendering preserves full defaults and exact instruction and tool limits', async () => {
  const skill = parseSkillMarkdown([
    '---',
    'name: exact-limit',
    'description: Exact limit skill',
    '---',
    'abcdefghij',
  ].join('\n'), '/workspace/skills/exact-limit/SKILL.md')
  const host = {
    captureRuntimeInfo: () => ({
      platform: 'test',
      runtimeVersion: 'Bun test',
      timestamp: '2026-07-13T12:00:00.000Z',
      timezone: 'UTC',
      workingDirectory: '/workspace',
      workspaceName: 'workspace',
      xerxesVersion: '0.3.0',
    }),
  }
  const builder = new PromptContextBuilder({ host })
  const fullDefault = await builder.assembleSystemPromptPrefix({ enabledSkills: [skill], toolNames: ['one', 'two'] })
  const fullExplicit = await builder.assembleSystemPromptPrefix({
    profile: PromptProfile.FULL,
    enabledSkills: [skill],
    toolNames: ['one', 'two'],
  })
  const exactSkillLimit = await builder.assembleSystemPromptPrefix({
    profile: {
      ...getPromptProfileConfig(PromptProfile.FULL),
      maxSkillInstructionsLength: skillPromptSection(skill).length,
      maxToolsListed: 2,
    },
    enabledSkills: [skill],
    toolNames: ['one', 'two'],
  })
  const cappedTools = await builder.assembleSystemPromptPrefix({
    profile: { ...getPromptProfileConfig(PromptProfile.FULL), maxToolsListed: 2 },
    toolNames: ['one', 'two', 'three'],
  })

  expect(fullDefault).toBe(fullExplicit)
  expect(exactSkillLimit).toContain('abcdefghij')
  expect(exactSkillLimit).not.toContain('abcdefghij...')
  expect(cappedTools).toContain('  ... and 1 more')
  expect(cappedTools).not.toContain('  - three')
})

test('full profile renders orchestration engagement rules only with agent tools', async () => {
  const host = {
    captureRuntimeInfo: () => ({
      platform: 'test',
      runtimeVersion: 'Bun test',
      timestamp: '2026-07-13T12:00:00.000Z',
      timezone: 'UTC',
      workingDirectory: '/workspace',
      workspaceName: 'workspace',
      xerxesVersion: '0.3.0',
    }),
  }
  const builder = new PromptContextBuilder({ host })
  const agentTools = ['SpawnAgents', 'AwaitAgents', 'TaskListTool', 'ReadFile']

  const withAgents = await builder.assembleSystemPromptPrefix({ toolNames: agentTools })
  expect(withAgents).toContain('[Orchestration]')
  // Active supervision: cadence, interleaved work, and proactive management.
  expect(withAgents).toContain('every two minutes of wall time')
  expect(withAgents).toContain('interleave useful local work')
  expect(withAgents).toContain('inventory and peek tools rather than blind waiting')
  expect(withAgents).toContain('Nudge stuck agents, stop irrelevant or superseded ones')
  // Await-all semantics for required results stay honored.
  expect(withAgents).toContain('prefer await-all joins for required results')

  const withoutAgents = await builder.assembleSystemPromptPrefix({ toolNames: ['ReadFile'] })
  expect(withoutAgents).not.toContain('[Orchestration]')

  // Sub-agent profiles stay slim even when agent tools are listed.
  const compact = await builder.assembleSystemPromptPrefix({
    profile: PromptProfile.COMPACT,
    toolNames: agentTools,
  })
  expect(compact).not.toContain('[Orchestration]')
  const minimal = await builder.assembleSystemPromptPrefix({
    profile: PromptProfile.MINIMAL,
    toolNames: agentTools,
  })
  expect(minimal).not.toContain('[Orchestration]')
})

test('verification rules render for main and sub-agent profiles, never for NONE', async () => {
  const host = {
    captureRuntimeInfo: () => ({
      platform: 'test',
      runtimeVersion: 'Bun test',
      timestamp: '2026-07-13T12:00:00.000Z',
      timezone: 'UTC',
      workingDirectory: '/workspace',
      workspaceName: 'workspace',
      xerxesVersion: '0.3.0',
    }),
  }
  const builder = new PromptContextBuilder({ host })

  for (const profile of [PromptProfile.FULL, PromptProfile.COMPACT, PromptProfile.MINIMAL]) {
    const prefix = await builder.assembleSystemPromptPrefix({ profile, toolNames: ['ReadFile'] })
    expect(prefix).toContain('[Verification]')
    expect(prefix).toContain('fresh evidence from this session')
    expect(prefix).toContain('a claim without a tool result is a guess')
    expect(prefix).toContain('Verify the environment before blaming it')
    expect(prefix).toContain('correct it explicitly and immediately')
  }

  const none = await builder.assembleSystemPromptPrefix({ profile: PromptProfile.NONE })
  expect(none).not.toContain('[Verification]')
})

test('orchestration block does not weaken full-profile caps', async () => {
  const oversizedSkill = parseSkillMarkdown([
    '---',
    'name: oversized',
    'description: Oversized skill',
    '---',
    's'.repeat(20_000),
  ].join('\n'), '/workspace/skills/oversized/SKILL.md')
  const host = {
    captureRuntimeInfo: () => ({
      platform: 'test',
      runtimeVersion: 'Bun test',
      timestamp: '2026-07-13T12:00:00.000Z',
      timezone: 'UTC',
      workingDirectory: '/workspace',
      workspaceName: 'workspace',
      xerxesVersion: '0.3.0',
    }),
  }
  const builder = new PromptContextBuilder({ host })
  const toolNames = ['SpawnAgents', ...Array.from({ length: 150 }, (_, index) => 'tool-' + index)]
  const prefix = await builder.assembleSystemPromptPrefix({
    profile: PromptProfile.FULL,
    enabledSkills: [oversizedSkill],
    toolNames,
  })

  expect(prefix).toContain('[Orchestration]')
  // Skill-body and tool-list caps still hold with agent tools present.
  const skillSection = skillPromptSection(oversizedSkill)
  expect(prefix).toContain(skillSection.slice(0, 8_000) + '...')
  expect(prefix).not.toContain(skillSection.slice(0, 8_001))
  expect(prefix).toContain('  - tool-98')
  expect(prefix).not.toContain('  - tool-99')
  expect(prefix).toContain('  ... and 51 more')
})

test('full profile caps oversized skill bodies and long tool lists at finite bounds', async () => {
  const oversizedSkill = parseSkillMarkdown([
    '---',
    'name: oversized',
    'description: Oversized skill',
    '---',
    's'.repeat(20_000),
  ].join('\n'), '/workspace/skills/oversized/SKILL.md')
  const host = {
    captureRuntimeInfo: () => ({
      platform: 'test',
      runtimeVersion: 'Bun test',
      timestamp: '2026-07-13T12:00:00.000Z',
      timezone: 'UTC',
      workingDirectory: '/workspace',
      workspaceName: 'workspace',
      xerxesVersion: '0.3.0',
    }),
  }
  const builder = new PromptContextBuilder({ host })
  const toolNames = Array.from({ length: 150 }, (_, index) => 'tool-' + index)
  const context = await builder.build({
    profile: PromptProfile.FULL,
    enabledSkills: [oversizedSkill],
    toolNames,
  })

  const config = getPromptProfileConfig(PromptProfile.FULL)
  expect(config.maxSkillInstructionsLength).toBe(8_000)
  expect(config.maxToolsListed).toBe(100)
  // The 20k skill body is clipped at the FULL cap with an explicit marker.
  const skillSection = skillPromptSection(oversizedSkill)
  expect(skillSection.length).toBeGreaterThan(8_000)
  expect(context.enabledSkillsSection).toBe(
    '[Enabled Skill Instructions]\n' + skillSection.slice(0, 8_000) + '...\n',
  )
  // Only the first 100 tools render; the rest collapse into a summary line.
  expect(context.toolsSection).toContain('  - tool-99')
  expect(context.toolsSection).not.toContain('  - tool-100')
  expect(context.toolsSection).toContain('  ... and 50 more')
})
