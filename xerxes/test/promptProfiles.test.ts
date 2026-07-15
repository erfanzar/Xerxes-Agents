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
  expect(config.maxSkillInstructionsLength).toBeUndefined()
  expect(config.maxToolsListed).toBeUndefined()
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
