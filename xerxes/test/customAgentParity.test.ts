// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, spyOn, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { loadAgentDefinitions, parseAgentMarkdown, subagentCatalogForAgent } from '../src/agents/definitions.js'
import { createNativeSubagentHost } from '../src/daemon/subagentHost.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { parseSkillMarkdown, SkillRegistry } from '../src/extensions/skills.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { registerClaudeAgentTools } from '../src/tools/claudeTools/agentOps.js'
import type { JsonObject } from '../src/types/toolCalls.js'

async function fixture(body: string, run: (root: string, path: string) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-agent-parity-'))
  const directory = join(root, '.xerxes/agents/review')
  await mkdir(directory, { recursive: true })
  const path = join(directory, 'specialist.md')
  await writeFile(path, body)
  try { await run(root, path) } finally { await rm(root, { recursive: true, force: true }) }
}

test('Markdown discovery is recursive and nearest repository definitions win', async () => {
  await fixture('---\nname: specialist\ndescription: Outer specialist\n---\nOuter instructions.', async root => {
    await mkdir(join(root, '.git'))
    const nested = join(root, 'packages/app')
    const localAgents = join(nested, '.xerxes/agents')
    await mkdir(localAgents, { recursive: true })
    const outer = loadAgentDefinitions({ cwd: nested, userDirectory: join(root, 'empty') })
    expect(outer.get('specialist')?.description).toBe('Outer specialist')
    await writeFile(join(localAgents, 'local.md'), '---\nname: specialist\ndescription: Nearest specialist\n---\nLocal instructions.')
    const definitions = loadAgentDefinitions({ cwd: nested, userDirectory: join(root, 'empty') })
    expect(definitions.get('specialist')?.systemPrompt).toBe('Local instructions.')
    expect(subagentCatalogForAgent(definitions, 'default').specialist?.resolvedProfile).toBe('specialist')
    // A Markdown agent used as the main agent can discover other profiles.
    expect(subagentCatalogForAgent(definitions, 'specialist').reviewer).toBeDefined()
  })
})

test('Claude-style Markdown fields map to native settings and reject invalid values', async () => {
  await fixture('---\nname: specialist\ndescription: Review code\ntools: Read, Grep, Bash\ndisallowedTools: Bash\nmodel: inherit\nmaxTurns: 3\neffort: high\nbackground: true\npermissionMode: plan\nskills: [audit]\n---\nReview code.', async (_root, path) => {
    expect(parseAgentMarkdown(path)).toMatchObject({
      name: 'specialist', model: '', maxTurns: 3, effort: 'high', background: true,
      permissionMode: 'plan', skills: ['audit'], promptMode: 'replace',
      tools: ['ReadFile', 'GrepTool', 'exec_command', 'write_stdin', 'list_terminal_sessions', 'close_terminal_session', 'check_command', 'list_commands', 'kill_command', 'pty_open', 'pty_write', 'pty_list', 'pty_close'],
      excludeTools: ['exec_command', 'write_stdin', 'list_terminal_sessions', 'close_terminal_session', 'check_command', 'list_commands', 'kill_command', 'pty_open', 'pty_write', 'pty_list', 'pty_close'],
    })
    for (const field of ['maxTurns: 0', 'effort: enormous', 'background: yes', 'permissionMode: unknown', 'tools: [42]']) {
      await writeFile(path, `---\nname: specialist\ndescription: Test\n${field}\n---\nTest.`)
      expect(() => parseAgentMarkdown(path)).toThrow()
    }
    await writeFile(path, '---\nname: specialist\ndescription: Test\ntools: []\n---\nTest.')
    expect(parseAgentMarkdown(path).allowedTools).toEqual([])
  })
})

test('native child uses its own persona, filtered tools, full preloaded skills and effective model/effort', async () => {
  await fixture('---\nname: specialist\ndescription: Review code\ntools: Read, Write, Skill\ndisallowedTools: Write\nmodel: file-model\neffort: high\nskills: [audit]\npermissionMode: plan\n---\nYou are the specialist persona.', async root => {
    const requests: CompletionRequest[] = []
    const llm: LlmClient = { async *stream(request) { requests.push(request); yield { content: 'Reviewed.' } } }
    const tools = new ToolRegistry()
    for (const name of ['ReadFile', 'WriteFile', 'SkillTool', 'exec_command']) tools.register({ type: 'function', function: { name, description: name, parameters: { type: 'object', properties: {} } } }, () => 'fixture')
    const skillRegistry = new SkillRegistry()
    const fullInstructions = 'Audit detail. '.repeat(2_000) + 'END OF PRELOADED SKILL'
    skillRegistry.register(parseSkillMarkdown(`---\nname: audit\n---\n${fullInstructions}`, join(root, 'SKILL.md')))
    const host = createNativeSubagentHost({
      agentDefinitions: loadAgentDefinitions({ cwd: root, userDirectory: join(root, 'empty') }),
      cwd: root, eventBus: new DaemonSubagentEventBus(), llm, model: 'parent-model', reasoningEffort: 'low',
      skillRegistry, permissionMode: 'accept-all', tools: tools.definitions(), toolExecutor: tools,
    })
    try {
      const child = await host.managerPort.spawn({ creatorAgentId: 'default', promptProfile: 'specialist', message: 'Review this change',
        permissionMode: 'accept-all', agent: { id: 'specialist', name: 'specialist', model: 'explicit-model' } })
      expect((await host.managerPort.wait([child.id], 2_000)).completed[0]?.status).toBe('completed')
      const request = requests[0]!
      expect(request.model).toBe('explicit-model')
      expect(request.thinking?.effort).toBe('high')
      expect(request.tools?.map(tool => tool.function.name)).toEqual(['ReadFile', 'SkillTool'])
      const prompt = JSON.stringify(request.messages)
      expect(prompt).toContain('You are the specialist persona.')
      expect(prompt).toContain('END OF PRELOADED SKILL')
      expect(prompt).not.toContain('You are Xerxes, an AI coding assistant.')
      expect(host.managerPort.listHandles()[0]?.rules).toContain('permission:plan')
    } finally { await host.manager.shutdown() }
  })
})

test('agent tool exposes types, accepts description and resumes the same native child conversation', async () => {
  await fixture('---\nname: specialist\ndescription: Audit code\n---\nAudit carefully.', async root => {
    const requests: CompletionRequest[] = []
    const llm: LlmClient = { async *stream(request) { requests.push(request); yield { content: 'Reviewed.' } } }
    const tools = new ToolRegistry()
    const definitions = loadAgentDefinitions({ cwd: root, userDirectory: join(root, 'empty') })
    const host = createNativeSubagentHost({ agentDefinitions: definitions, cwd: root, eventBus: new DaemonSubagentEventBus(), llm, model: 'parent-model', permissionMode: 'accept-all', tools: [], toolExecutor: tools })
    registerClaudeAgentTools(tools, { manager: host.managerPort, availableAgents: [{ name: 'specialist', description: 'Audit code' }] })
    const call = (args: JsonObject, sessionId = 'owner') => tools.execute({ id: 'agent', type: 'function', function: { name: 'AgentTool', arguments: args } }, { agentId: 'default', sessionId, metadata: {} })
    try {
      expect(tools.definitions().find(tool => tool.function.name === 'AgentTool')?.function.description).toContain('specialist: Audit code')
      const first = JSON.parse(await call({ subagent_type: 'specialist', prompt: 'First audit', description: 'Audit code' }))
      expect(first.status).toBe('completed')
      const resumed = JSON.parse(await call({ resume: first.id, prompt: 'Continue the audit' }))
      expect(resumed.id).toBe(first.id)
      expect(host.managerPort.listHandles()).toHaveLength(1)
      expect(JSON.stringify(requests.at(-1)?.messages)).toContain('First audit')
      expect(JSON.stringify(requests.at(-1)?.messages)).toContain('Continue the audit')
      await expect(call({ resume: first.id, prompt: 'Foreign audit' }, 'other-owner')).rejects.toThrow()
    } finally { await host.manager.shutdown() }
  })
})

test('frontmatter background returns before a child completes and inherits tools when omitted', async () => {
  await fixture('---\nname: specialist\ndescription: Audit code\nbackground: true\n---\nAudit carefully.', async root => {
    let release!: () => void
    const gate = new Promise<void>(resolve => { release = resolve })
    const requests: CompletionRequest[] = []
    const llm: LlmClient = { async *stream(request): AsyncGenerator<LlmDelta> { requests.push(request); await gate; yield { content: 'Done.' } } }
    const tools = new ToolRegistry()
    tools.register({ type: 'function', function: { name: 'ReadFile', description: 'Read', parameters: { type: 'object', properties: {} } } }, () => 'read')
    const host = createNativeSubagentHost({ agentDefinitions: loadAgentDefinitions({ cwd: root, userDirectory: join(root, 'empty') }), cwd: root, eventBus: new DaemonSubagentEventBus(), llm, model: 'parent-model', permissionMode: 'accept-all', tools: tools.definitions(), toolExecutor: tools })
    registerClaudeAgentTools(tools, { manager: host.managerPort })
    const wait = spyOn(host.managerPort, 'wait')
    try {
      const result = JSON.parse(await tools.execute({ id: 'agent', type: 'function', function: { name: 'AgentTool', arguments: { subagent_type: 'specialist', prompt: 'Audit this', description: 'Audit', run_in_background: false, timeout: 0.01 } } }, { agentId: 'default', sessionId: 'owner', metadata: {} }))
      expect(result.status).not.toBe('completed')
      expect(result.rules).toContain('background')
      const batch = JSON.parse(await tools.execute({ id: 'batch', type: 'function', function: { name: 'SpawnAgents', arguments: { agents: [{ subagent_type: 'specialist', prompt: 'Second audit', title: 'Second audit' }], wait: true, timeout: 0.01 } } }, { agentId: 'default', sessionId: 'owner', metadata: {} }))
      expect(batch[0].rules).toContain('background')
      expect(wait).not.toHaveBeenCalled()
      release()
      expect((await host.managerPort.wait([result.id], 2_000)).completed[0]?.status).toBe('completed')
      expect(requests[0]?.tools?.map(tool => tool.function.name)).toContain('ReadFile')
    } finally { release(); wait.mockRestore(); await host.manager.shutdown() }
  })
})

test('maxTurns bounds output-length continuations and retains a resumable partial result', async () => {
  await fixture('---\nname: specialist\ndescription: Audit code\nmaxTurns: 1\ntools: []\n---\nAudit carefully.', async root => {
    const requests: CompletionRequest[] = []
    const llm: LlmClient = { async *stream(request) { requests.push(request); yield { content: requests.length === 1 ? 'First audit completed.' : 'Partial audit findings.', finishReason: requests.length === 1 ? 'stop' : 'length' } } }
    const tools = new ToolRegistry()
    tools.register({ type: 'function', function: { name: 'ReadFile', description: 'Read', parameters: { type: 'object', properties: {} } } }, () => 'read')
    const host = createNativeSubagentHost({ agentDefinitions: loadAgentDefinitions({ cwd: root, userDirectory: join(root, 'empty') }), cwd: root, eventBus: new DaemonSubagentEventBus(), llm, model: 'parent-model', permissionMode: 'accept-all', tools: tools.definitions(), toolExecutor: tools })
    try {
      const child = await host.managerPort.spawn({ creatorAgentId: 'default', promptProfile: 'specialist', message: 'Audit this' })
      const first = (await host.managerPort.wait([child.id], 2_000)).completed[0]!
      expect(first.status).toBe('completed')
      await host.retry(child.id, { message: 'Extend the audit' })
      const result = (await host.managerPort.wait([child.id], 2_000)).completed[0]!
      expect(requests).toHaveLength(2)
      expect(requests[0]?.tools ?? []).toEqual([])
      expect(result.status).toBe('error')
      expect(result.error).toContain('maxTurns=1')
      expect(result.lastOutput).toContain('Partial audit findings.')
      const resumed = await host.retry(child.id, { message: 'Continue' })
      expect(resumed.id).toBe(child.id)
      await host.managerPort.wait([child.id], 2_000)
      expect(requests).toHaveLength(3)
      expect(JSON.stringify(requests[2]?.messages)).toContain('Partial audit findings.')
    } finally { await host.manager.shutdown() }
  })
})

test('missing preloaded skills fail visibly before contacting the provider', async () => {
  await fixture('---\nname: specialist\ndescription: Audit code\nskills: [missing]\n---\nAudit carefully.', async root => {
    let calls = 0
    const llm: LlmClient = { async *stream() { calls++; yield { content: 'Must not run' } } }
    const tools = new ToolRegistry()
    const host = createNativeSubagentHost({ agentDefinitions: loadAgentDefinitions({ cwd: root, userDirectory: join(root, 'empty') }), cwd: root, eventBus: new DaemonSubagentEventBus(), llm, model: 'parent-model', permissionMode: 'accept-all', tools: [], toolExecutor: tools })
    try {
      const child = await host.managerPort.spawn({ creatorAgentId: 'default', promptProfile: 'specialist', message: 'Audit this' })
      const result = (await host.managerPort.wait([child.id], 2_000)).completed[0]!
      expect(result.status).toBe('error')
      expect(result.error).toContain("preloaded skill 'missing' is unavailable")
      expect(calls).toBe(0)
    } finally { await host.manager.shutdown() }
  })
})

test('inherited child uses its parent provider rather than the active host provider', async () => {
  await fixture('---\nname: specialist\ndescription: Review code\nmodel: inherit\n---\nReview carefully.', async root => {
    const calls: string[] = [];
    const host = createNativeSubagentHost({
      agentDefinitions: loadAgentDefinitions({ cwd:root, userDirectory:join(root,'empty') }),
      cwd:root, eventBus:new DaemonSubagentEventBus(), model:'kimi-for-coding', permissionMode:'accept-all', tools:[], toolExecutor:new ToolRegistry(),
      llm: { async *stream() { throw new Error('Wrong inherited Kimi transport'); yield {content:''}; } },
      resolveSourceProvider: source => { expect(source).toBe('codex-parent'); return 'codex'; },
      resolveProviderRoute: () => 'a'.repeat(64),
      resolveProviderProfile: profile => ({llm:{async *stream(request){calls.push(profile + '/' + request.model);yield {content:'reviewed'};}}}),
    });
    try {
      const child = await host.managerPort.spawn({ creatorAgentId:'default', promptProfile:'specialist', message:'Review this', sourceAgentId:'codex-parent', parentModel:'gpt-6-astra' });
      expect((await host.managerPort.wait([child.id],2000)).completed[0]?.status).toBe('completed');
      expect(calls).toEqual(['codex/gpt-6-astra']);
    } finally { await host.manager.shutdown(); }
  });
});
