// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Demonstrate native plugins, skills, policy, hooks, context, and sandbox routing. */

import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  HookRunner,
  LoopDetector,
  PluginRegistry,
  PluginType,
  PolicyEngine,
  PromptContextBuilder,
  SandboxMode,
  SandboxRouter,
  SkillRegistry,
  ToolPolicy,
  createSystemPromptContextHost,
} from '../src/typescript/src/index.js'
import { divider, runMain } from './native_demo_support.js'

export function webSearch(query: string): string {
  return `Search results for: ${query}`
}

export async function runOpenClawCapabilitiesDemo(): Promise<readonly string[]> {
  const lines: string[] = []
  const plugins = new PluginRegistry()
  plugins.registerPlugin({
    name: 'search_plugin',
    version: '1.0.0',
    pluginType: PluginType.TOOL,
    description: 'Web search demonstration tool',
  })
  plugins.registerTool('web_search', (...args) => webSearch(String(args[0] ?? '')), undefined, 'search_plugin')
  lines.push(`Plugins: ${plugins.pluginNames.join(', ')}`)
  lines.push(`Tools: ${Object.keys(plugins.getAllTools()).join(', ')}`)

  const directory = await mkdtemp(join(tmpdir(), 'xerxes-native-skill-'))
  try {
    const skillDirectory = join(directory, 'web_research')
    await mkdir(skillDirectory, { recursive: true })
    await writeFile(join(skillDirectory, 'SKILL.md'), skillMarkdown(), 'utf8')
    const skills = new SkillRegistry()
    lines.push(`Skills discovered: ${(await skills.discover(directory)).join(', ')}`)
    lines.push(skills.markdownIndex())
  } finally {
    await rm(directory, { force: true, recursive: true })
  }

  const policy = new PolicyEngine({
    globalPolicy: new ToolPolicy({ deny: ['execute_shell', 'delete_file'] }),
    agentPolicies: { admin_agent: new ToolPolicy({ allow: ['execute_shell', 'web_search', 'read_file'] }) },
  })
  lines.push(`Policy global web_search: ${policy.check('web_search')}`)
  lines.push(`Policy global execute_shell: ${policy.check('execute_shell')}`)
  lines.push(`Policy admin execute_shell: ${policy.check('execute_shell', 'admin_agent')}`)
  try {
    policy.enforce('delete_file', 'reader_agent')
  } catch (error) {
    lines.push(`Policy enforcement: ${error instanceof Error ? error.message : String(error)}`)
  }

  const detector = new LoopDetector({ sameCallWarning: 3, sameCallCritical: 5 })
  for (let index = 0; index < 6; index += 1) {
    const event = detector.recordCall('web_search', { query: 'same query' })
    lines.push(`Loop call ${index + 1}: ${event.severity}`)
  }

  const hooks = new HookRunner()
  hooks.register('before_tool_call', payload => ({ ...(payload.arguments as Record<string, unknown>), auditTraced: true }))
  hooks.register('after_tool_call', payload => String(payload.result ?? '').toLowerCase().includes('secret') ? '[REDACTED]' : payload.result)
  hooks.register('bootstrap_files', payload => `[Bootstrap] ${String(payload.agentId ?? 'unknown')} initialized`)
  const args = { arguments: { query: 'hello' } }
  const result = { result: 'Contains Secret Data' }
  lines.push(`Hook arguments: ${JSON.stringify(hooks.run('before_tool_call', args))}`)
  lines.push(`Hook result: ${String(hooks.run('after_tool_call', result))}`)

  const prompt = new PromptContextBuilder({
    host: createSystemPromptContextHost({ xerxesVersion: 'native-demo' }),
    hookRunner: hooks,
    guardrails: ['Respect user privacy', 'Do not execute destructive operations'],
    sandboxConfig: { mode: SandboxMode.WARN, sandboxedTools: ['execute_shell'] },
  })
  const prefix = await prompt.assembleSystemPromptPrefix({ agentId: 'demo_agent', toolNames: ['web_search', 'read_file', 'execute_shell'] })
  lines.push(`Prompt context generated: ${prefix.includes('Guardrails') && prefix.includes('execute_shell')}`)

  const sandbox = new SandboxRouter({
    config: {
      mode: SandboxMode.STRICT,
      sandboxedTools: ['execute_shell', 'execute_typescript'],
      elevatedTools: ['read_file'],
    },
  })
  for (const toolName of ['execute_shell', 'read_file', 'web_search']) {
    const decision = sandbox.decide(toolName)
    lines.push(`Sandbox ${toolName}: ${decision.context} (${decision.reason})`)
  }
  return lines
}

async function main(): Promise<void> {
  divider('OPENCLAW-CLASS CAPABILITIES (native Bun)')
  for (const line of await runOpenClawCapabilitiesDemo()) console.log(line)
}

function skillMarkdown(): string {
  return `---
name: web_research
description: Search the web and synthesize findings
version: "1.0"
tags: [research, web]
---

# Web Research Skill

Break a question into sub-questions, use an injected search port, and clearly label synthesis.`
}

if (import.meta.main) runMain(main)
