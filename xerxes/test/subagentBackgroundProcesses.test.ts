// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { AgentDefinition } from '../src/agents/definitions.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { createNativeSubagentHost } from '../src/daemon/subagentHost.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { BackgroundCommandManager } from '../src/tools/backgroundCommands.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'
import { registerProcessTools } from '../src/tools/processTools.js'

const worker: AgentDefinition = { name: 'worker', description: 'background process test', source: 'test', model: '',
  systemPrompt: 'Start the server, then report.', allowedTools: null, excludeTools: [], tools: [], isolation: '', maxDepth: 3 }

/** Starts a long-running background job, then finishes its turn. */
class ServerStartingClient implements LlmClient {
  readonly toolResults: string[] = []
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    const result = request.messages.findLast(message => message.role === 'tool')
    if (result === undefined) {
      yield { toolCalls: [{ id: crypto.randomUUID(), type: 'function', function: {
        name: 'exec_command', arguments: { cmd: 'sleep', args: ['30'], run_in_background: true },
      } }] }
      return
    }
    this.toolResults.push(typeof result.content === 'string' ? result.content : JSON.stringify(result.content))
    yield { content: 'server started' }
  }
}

test('background commands a subagent started are stopped when its run ends', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-child-background-'))
  const background = new BackgroundCommandManager()
  const registry = new ToolRegistry()
  registerProcessTools(registry, new WorkspacePathResolver(directory), background)
  const client = new ServerStartingClient()
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([['worker', worker]]),
    cwd: directory,
    disposeProcessOwner: owner => background.disposeOwner(owner),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  try {
    const task = await host.managerPort.spawn({ message: 'start the dev server', promptProfile: 'worker', sourceAgentId: 'parent' })
    const done = await host.managerPort.wait([task.id], 10_000)
    expect(done.completed[0]?.lastOutput).toBe('server started')
    // The job really ran under the child's own owner id.
    expect(client.toolResults.join('')).toContain('procId')
    expect(background.listForOwner(task.historySessionId ?? task.id)).toEqual([])
  } finally {
    await host.manager.shutdown()
    await background.disposeAll()
    await rm(directory, { recursive: true, force: true })
  }
})
