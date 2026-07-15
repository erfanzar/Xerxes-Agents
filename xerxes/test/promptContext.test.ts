// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { HookRunner } from '../src/extensions/hooks.js'
import { SkillRegistry, type Skill } from '../src/extensions/skills.js'
import { SandboxMode } from '../src/security/sandbox.js'
import {
  PromptContextBuilder,
  type PromptContextHost,
  type RuntimeInfo,
} from '../src/runtime/promptContext.js'
import { PromptProfile } from '../src/runtime/promptProfiles.js'

const runtimeInfo: RuntimeInfo = {
  platform: 'Darwin 25.0',
  runtimeVersion: 'Bun test',
  timestamp: '2026-07-13T12:00:00.000Z',
  timezone: 'Europe/Istanbul',
  workingDirectory: '/workspace/xerxes',
  workspaceName: 'xerxes',
  xerxesVersion: '0.2.0',
}

const CONTEXT_COMPOSITION_TEST = [
  'prompt context composes native skills, hooks, policy, sandbox, memories, git,',
  'and repo map through ports',
].join(' ')

test(CONTEXT_COMPOSITION_TEST, async () => {
  const hooks = new HookRunner()
  hooks.register('bootstrap_files', () => ['# Project notes', 'Use Bun.'])
  const skills = new SkillRegistry()
  const skill = sampleSkill()
  skills.register(skill)
  const host = fakeHost()
  const builder = new PromptContextBuilder({
    host,
    hookRunner: hooks,
    skillRegistry: skills,
    workspaceRoot: '/workspace/xerxes',
    guardrails: ['No destructive commands'],
    sandboxConfig: {
      mode: SandboxMode.STRICT,
      sandboxedTools: ['exec_command'],
      elevatedTools: ['apply_patch'],
    },
    memoryProvider: async (agentId, maximum) => [`${agentId}:${maximum}`, 'first\nsecond'],
    userProfileProvider: async agentId => 'prefers concise reports for ' + agentId,
  })

  const context = await builder.build({
    agentId: 'coder',
    enabledSkills: [skill],
    toolNames: ['ReadFile', 'exec_command', 'web.search_query'],
  })
  const prompt = await builder.assembleSystemPromptPrefix({
    agentId: 'coder',
    enabledSkills: [skill],
    toolNames: ['ReadFile', 'exec_command', 'web.search_query'],
  })

  expect(context.runtimeSection).toContain('Runtime: Bun test')
  expect(context.bootstrapSection).toContain('# Project notes')
  expect(context.skillsSection).toContain('native-test')
  expect(context.enabledSkillsSection).toContain('Use the native test skill.')
  expect(context.memorySection).toContain('first second')
  expect(context.userProfileSection).toContain('coder')
  expect(context.repoMapSection).toContain('src/app.ts')
  expect(context.gitSection).toContain('Branch: main')
  expect(context.sandboxSection).toContain('Sandboxed tools: exec_command')
  expect(prompt).toContain('[Identity]')
  expect(prompt).toContain('[Tooling]')
  expect(prompt).toContain('[Safety]')
  expect(prompt).toContain('[Skills & Instructions]')
  expect(prompt).toContain('[Workspace Context]')
  expect(prompt).toContain('[Sandbox Runtime]')
  expect(prompt).toContain('[Runtime]')
  expect(prompt).toContain('[Execution Policy]')
  expect(prompt).toContain('[Output Style]')
})

test('profile gates prevent optional ports and caps visible tool and skill prompt content', async () => {
  let gitCalls = 0
  let mapCalls = 0
  let memoryCalls = 0
  let profileCalls = 0
  const host: PromptContextHost = {
    captureRuntimeInfo: () => runtimeInfo,
    buildRepoMap: async () => {
      mapCalls += 1
      return 'map'
    },
    gitContext: async () => {
      gitCalls += 1
      return { branch: 'main', dirtyCount: 0, recentCommits: [] }
    },
  }
  const builder = new PromptContextBuilder({
    host,
    workspaceRoot: runtimeInfo.workingDirectory,
    memoryProvider: async () => {
      memoryCalls += 1
      return ['memory']
    },
    userProfileProvider: async () => {
      profileCalls += 1
      return 'profile'
    },
  })
  const enabled = [{ ...sampleSkill(), instructions: 'x'.repeat(550) }]
  const compact = await builder.build({
    profile: PromptProfile.COMPACT,
    enabledSkills: enabled,
    toolNames: Array.from({ length: 22 }, (_, index) => 'tool-' + index),
  })
  const minimal = await builder.assembleSystemPromptPrefix({ profile: PromptProfile.MINIMAL, toolNames: ['a', 'b'] })
  const none = await builder.buildNonePrefix()

  expect(compact.workspaceSection).toBe('')
  expect(compact.bootstrapSection).toBe('')
  expect(compact.repoMapSection).toBe('')
  expect(compact.enabledSkillsSection).toContain('...')
  expect(compact.toolsSection).toContain('... and 2 more')
  expect(gitCalls).toBe(2)
  expect(mapCalls).toBe(0)
  expect(memoryCalls).toBe(1)
  expect(profileCalls).toBe(1)
  expect(minimal).not.toContain('[Runtime]')
  expect(minimal).not.toContain('[Skills & Instructions]')
  expect(none).toBe('You are Xerxes, a runtime-managed AI agent operating inside a controlled tool environment.')
})

test('failed optional ports and providers are omitted instead of stopping a prompt build', async () => {
  const builder = new PromptContextBuilder({
    host: {
      captureRuntimeInfo: () => runtimeInfo,
      buildRepoMap: async () => { throw new Error('no map') },
      gitContext: async () => { throw new Error('no git') },
    },
    workspaceRoot: runtimeInfo.workingDirectory,
    memoryProvider: async () => { throw new Error('no memory') },
    userProfileProvider: async () => { throw new Error('no profile') },
  })

  const context = await builder.build()

  expect(context.gitSection).toBe('')
  expect(context.repoMapSection).toBe('')
  expect(context.memorySection).toBe('')
  expect(context.userProfileSection).toBe('')
  expect(context.runtimeSection).toContain('Bun test')
})

function fakeHost(): PromptContextHost {
  return {
    captureRuntimeInfo: () => runtimeInfo,
    buildRepoMap: async () => 'src/app.ts: export function run()',
    gitContext: async () => ({ branch: 'main', dirtyCount: 2, recentCommits: ['abc first'] }),
  }
}

function sampleSkill(): Skill {
  return {
    sourcePath: '/workspace/skills/native-test/SKILL.md',
    instructions: 'Use the native test skill.',
    metadata: {
      name: 'native-test',
      description: 'native skill',
      author: '',
      configVars: [],
      dependencies: [],
      platforms: [],
      requiredTools: [],
      resources: [],
      setupCommand: '',
      source: 'test',
      subcommands: [],
      tags: [],
      trustLevel: 'local',
      version: '1.0',
    },
  }
}
