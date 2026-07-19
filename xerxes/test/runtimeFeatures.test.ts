// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  HookRunner,
  PluginRegistry,
  PolicyAction,
  RuntimeExtensionDependencyError,
  SandboxMode,
  SkillRegistry,
  ToolPolicy,
  composeRuntimeFeatures,
  createOperatorRuntimeConfig,
  parseSkillMarkdown,
  resolveRuntimeExtensionDirectories,
  type RuntimeFeatureFilesystem,
} from '../src/index.js'

test('runtime feature directory discovery is workspace-explicit, stable, and conventional', async () => {
  const filesystem: RuntimeFeatureFilesystem = {
    join: (...segments) => segments.join('/').replaceAll('//', '/'),
    resolve: path => path.replace(/\/$/, ''),
    isDirectory: path => path === '/workspace/plugins' || path === '/workspace/skills',
  }
  const directories = await resolveRuntimeExtensionDirectories({
    workspaceRoot: '/workspace',
    pluginDirectories: ['plugins', '/shared/plugins', 'plugins'],
    skillDirectories: ['skills'],
  }, filesystem)

  expect(directories).toEqual({
    pluginDirectories: ['/workspace/plugins', '/shared/plugins'],
    skillDirectories: ['/workspace/skills'],
  })
  await expect(resolveRuntimeExtensionDirectories({ pluginDirectories: ['plugins'] }, filesystem))
    .rejects.toThrow('requires an explicit workspaceRoot')
})

test('runtime composition derives policy, agent overrides, hook wiring, and real operator state', async () => {
  const plugins = new PluginRegistry()
  plugins.registerPlugin({ name: 'local-plugin' })
  plugins.registerTool('plugin_tool', () => 'native', undefined, 'local-plugin')
  let hookCalls = 0
  plugins.registerHook('on_turn_start', () => {
    hookCalls += 1
    return 'observed'
  }, undefined, 'local-plugin')
  const skills = new SkillRegistry()
  skills.register(parseSkillMarkdown(`---
name: review
required_tools: [plugin_tool]
---
Review native changes.`, '/workspace/skills/review/SKILL.md'))
  const policy = new ToolPolicy({ optionalTools: ['exec_command', 'blocked_tool'] })
  const state = await composeRuntimeFeatures({
    enabled: true,
    discoverConventionalExtensions: false,
    enabledSkills: ['review', 'not-installed'],
    guardrails: ['global-guardrail'],
    policy,
    sandbox: { mode: SandboxMode.WARN, sandboxedTools: ['exec_command'] },
    operator: createOperatorRuntimeConfig({ enabled: true, powerToolsEnabled: true }),
    agentOverrides: {
      constrained: {
        enabledSkills: [],
        guardrails: ['agent-guardrail'],
        policy: new ToolPolicy({ deny: ['agent_only'] }),
        sandbox: { mode: SandboxMode.OFF },
      },
    },
  }, {
    pluginRegistry: plugins,
    skillRegistry: skills,
    toolLookup: { hasTool: () => false },
  })

  try {
    expect(state.enabled).toBeTrue()
    expect(state.getEnabledSkills().map(skill => skill.metadata.name)).toEqual(['review'])
    expect(state.getMissingEnabledSkillNames()).toEqual(['not-installed'])
    expect(state.getEnabledSkillNames('constrained')).toEqual([])
    expect(state.getGuardrails('constrained')).toEqual(['agent-guardrail'])
    expect(state.policyEngine.check('exec_command')).toBe(PolicyAction.ALLOW)
    expect(state.policyEngine.check('blocked_tool')).toBe(PolicyAction.DENY)
    expect(state.policyEngine.check('agent_only', 'constrained')).toBe(PolicyAction.DENY)
    expect(policy.evaluate('exec_command')).toBe(PolicyAction.DENY)
    expect(state.getSandboxRouter()).toBeDefined()
    expect(state.getSandboxRouter('constrained')).toBeDefined()
    expect(state.operatorState?.toolDefinitions().map(tool => tool.function.name)).toContain('exec_command')
    expect(state.registeredHooks.on_turn_start).toBe(1)
    expect(await state.hookRunner.run('on_turn_start')).toEqual(['observed'])
    expect(hookCalls).toBe(1)
  } finally {
    await state.close()
  }
})

test('runtime composition discovers conventional extensions and validates plugin-provided tools', async () => {
  const workspace = await mkdtemp(join(tmpdir(), 'xerxes-runtime-features-'))
  try {
    const pluginsDirectory = join(workspace, 'plugins')
    const skillDirectory = join(workspace, 'skills', 'inspect')
    await mkdir(pluginsDirectory, { recursive: true })
    await mkdir(skillDirectory, { recursive: true })
    await writeFile(join(pluginsDirectory, 'native.mjs'), `
export function register(registry) {
  registry.registerPlugin({ name: 'native-plugin', version: '1.0.0' })
  registry.registerTool('native_tool', () => 'native', undefined, 'native-plugin')
  registry.registerHook('on_turn_end', () => 'native-hook', undefined, 'native-plugin')
}
`, 'utf8')
    await writeFile(join(skillDirectory, 'SKILL.md'), `---
name: inspect
required_tools: [native_tool]
---
Inspect native runtime state.
`, 'utf8')

    const state = await composeRuntimeFeatures({ workspaceRoot: workspace, enabledSkills: ['inspect'] })
    try {
      expect(state.discovery.pluginDirectories).toEqual([pluginsDirectory])
      expect(state.discovery.skillDirectories).toEqual([join(workspace, 'skills')])
      expect(state.discovery.pluginNames).toEqual(['native-plugin'])
      expect(state.discovery.skillNames).toEqual(['inspect'])
      expect(state.pluginRegistry.getTool('native_tool')).toBeDefined()
      expect(state.getEnabledSkills().map(skill => skill.metadata.name)).toEqual(['inspect'])
      expect(await state.hookRunner.run('on_turn_end')).toEqual(['native-hook'])
    } finally {
      await state.close()
    }
  } finally {
    await rm(workspace, { force: true, recursive: true })
  }
})

test('runtime composition reports all declared extension dependency errors before installing hooks', async () => {
  const plugins = new PluginRegistry()
  plugins.registerPlugin({ name: 'dependent-plugin', dependencies: ['missing-plugin'] })
  plugins.registerHook('on_turn_start', () => 'must-not-register', undefined, 'dependent-plugin')
  const skills = new SkillRegistry()
  skills.register(parseSkillMarkdown(`---
name: dependent-skill
dependencies: [missing-skill]
required_tools: [missing-tool]
---
Missing dependencies.`, '/workspace/skills/dependent/SKILL.md'))
  const hooks = new HookRunner()

  await expect(composeRuntimeFeatures({ discoverConventionalExtensions: false }, {
    hookRunner: hooks,
    pluginRegistry: plugins,
    skillRegistry: skills,
  })).rejects.toMatchObject({
    name: 'RuntimeExtensionDependencyError',
    errors: [
      "Plugin dependency issue: Plugin 'dependent-plugin' requires missing dependency 'missing-plugin'",
      "Skill dependency issue: Skill 'dependent-skill' requires missing dependency 'missing-skill'",
      "Skill dependency issue: Skill 'dependent-skill' requires missing tool 'missing-tool'",
    ],
  } satisfies Partial<RuntimeExtensionDependencyError>)
  expect(hooks.hasHooks('on_turn_start')).toBeFalse()
})
