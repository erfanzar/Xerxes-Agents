// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { ConfigurationError, FunctionExecutionError } from '../src/core/errors.js'
import { AgentMemory } from '../src/memory/agentMemory.js'
import { ContextualMemory } from '../src/memory/contextualMemory.js'
import { LongTermMemory } from '../src/memory/longTermMemory.js'
import { SimpleStorage } from '../src/memory/storage.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  BrowserSession,
  ComputerUseSession,
  HomeAssistantClient,
  InMemoryRLBackend,
  OutboundMessageRegistry,
  SearchHistoryTool,
  StaticAsker,
  WorkspacePathError,
  WorkspacePathResolver,
  registerCoreTools,
} from '../src/tools/index.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

async function inWorkspace(run: (workspace: string) => Promise<void>): Promise<void> {
  const workspace = await mkdtemp(join(tmpdir(), 'xerxes-bun-tools-'))
  try {
    await run(workspace)
  } finally {
    await rm(workspace, { force: true, recursive: true })
  }
}

function call(name: string, arguments_: JsonObject): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: 'function',
    function: { name, arguments: arguments_ },
  }
}

test('workspace resolver rejects lexical and symlink escapes', async () => {
  await inWorkspace(async workspace => {
    const outside = await mkdtemp(join(tmpdir(), 'xerxes-bun-outside-'))
    try {
      await Bun.write(join(outside, 'secret.txt'), 'outside')
      await symlink(outside, join(workspace, 'escape'))
      const paths = new WorkspacePathResolver(workspace)

      await expect(paths.resolve('../outside.txt')).rejects.toBeInstanceOf(WorkspacePathError)
      await expect(paths.resolve('escape/secret.txt')).rejects.toBeInstanceOf(WorkspacePathError)
      await expect(paths.resolve(join(outside, 'secret.txt'))).rejects.toBeInstanceOf(WorkspacePathError)
    } finally {
      await rm(outside, { force: true, recursive: true })
    }
  })
})

test('registered file tools read, write, edit, list, glob, and grep only within their workspace', async () => {
  await inWorkspace(async workspace => {
    const registry = new ToolRegistry()
    registerCoreTools(registry, { includeProcessTools: false, workspaceRoot: workspace })
    const context = { metadata: {} }

    const writeResult = await registry.execute(
      call('WriteFile', { file_path: 'notes/todo.txt', content: 'alpha\nbeta\n' }),
      context,
    )
    expect(writeResult).toContain('created')
    const firstLine = await registry.execute(call('ReadFile', { file_path: 'notes/todo.txt', limit: 1 }), context)
    expect(firstLine).toBe(
      'alpha\n\n\n[ReadFile] Showing lines 1-1 of 2. Continue with offset=1, limit=1. '
        + 'Use limit=-1 only when the whole file is intentionally required.',
    )

    expect(await registry.execute(call('AppendFile', { file_path: 'notes/todo.txt', lines: 'tail' }), context))
      .toContain('Appended 4 characters')

    expect(await registry.execute(call('FileEditTool', {
      file_path: 'notes/todo.txt',
      old_string: 'beta',
      new_string: 'gamma',
    }), context)).toContain('Applied 1 replacement')
    const updated = await registry.execute(call('ReadFile', { file_path: 'notes/todo.txt', limit: -1 }), context)
    expect(updated).toBe('alpha\ngamma\ntail\n')

    const listing = JSON.parse(
      await registry.execute(call('ListDir', { directory_path: '.', recursive: true }), context),
    ) as string[]
    expect(listing).toContain('notes/')
    expect(listing).toContain('notes/todo.txt')

    const globbed = JSON.parse(await registry.execute(call('GlobTool', { pattern: '**/*.txt' }), context)) as string[]
    expect(globbed).toEqual(['notes/todo.txt'])
    expect(await registry.execute(call('GrepTool', {
      pattern: 'gamma',
      glob: '**/*.txt',
      output_mode: 'content',
    }), context)).toBe('notes/todo.txt:2:gamma')

    await expect(registry.execute(call('WriteFile', { file_path: 'notes/todo.txt', content: 'replacement' }), context))
      .rejects.toBeInstanceOf(FunctionExecutionError)
    await expect(registry.execute(call('ReadFile', { file_path: '../outside.txt' }), context))
      .rejects.toBeInstanceOf(FunctionExecutionError)
  })
})

test('exec_command uses direct argv and returns bounded structured output', async () => {
  await inWorkspace(async workspace => {
    const registry = new ToolRegistry()
    registerCoreTools(registry, { workspaceRoot: workspace })
    const context = { metadata: {} }
    const executable = Bun.which('printf') ?? 'printf'

    const result = JSON.parse(await registry.execute(call('exec_command', {
      cmd: executable,
      args: ['hello'],
      workdir: '.',
    }), context)) as { command: string[]; cwd: string; exitCode: number; stdout: string; timedOut: boolean }
    expect(result.command).toEqual([executable, 'hello'])
    expect(result.cwd).toBe('.')
    expect(result.exitCode).toBe(0)
    expect(result.stdout).toBe('hello')
    expect(result.timedOut).toBeFalse()

    await expect(registry.execute(call('exec_command', { cmd: 'printf hello' }), context))
      .rejects.toBeInstanceOf(FunctionExecutionError)
  })
})

test('core registration includes the dependency-free AI, data, mathematics, and system surfaces', () => {
  const registry = new ToolRegistry()
  registerCoreTools(registry, { includeProcessTools: false })
  const names = new Set(registry.definitions().map(definition => definition.function.name))

  expect(names).toContain('JSONProcessor')
  expect(names).toContain('CSVProcessor')
  expect(names).toContain('TextProcessor')
  expect(names).toContain('TextEmbedder')
  expect(names).toContain('TextSimilarity')
  expect(names).toContain('Calculator')
  expect(names).toContain('StatisticalAnalyzer')
  expect(names).toContain('SystemInfo')
  expect(names).toContain('EnvironmentManager')
  expect(names).not.toContain('exec_command')
})

test('core registration accepts explicitly host-bound Claude-compatible adapters', () => {
  const registry = new ToolRegistry()
  registerCoreTools(registry, {
    includeProcessTools: false,
    claudeCompatibilityTools: {
      workflow: { workspaceRoot: process.cwd() },
    },
  })
  const names = new Set(registry.definitions().map(definition => definition.function.name))

  expect(names).toContain('TodoWriteTool')
  expect(names).toContain('PlanTool')
  expect(names).toContain('EnterWorktreeTool')
})

test('core registration exposes every supplied host-bound port without enabling it by accident', async () => {
  await inWorkspace(async workspace => {
    const registry = new ToolRegistry()
    const outbound = new OutboundMessageRegistry()
    outbound.register('test', () => ({ ok: true }))
    const memory = new AgentMemory({
      globalDirectory: join(workspace, 'global-memory'),
      projectDirectory: join(workspace, 'project-memory'),
    })
    const history = new SearchHistoryTool({ index: { search: () => [] } })
    registerCoreTools(registry, {
      workspaceRoot: workspace,
      agentMemoryTools: { memory },
      agentMetaTools: {},
      includeCodingTools: true,
      browserTools: { session: new BrowserSession() },
      clarifyTool: { asker: new StaticAsker({ answer: 'yes' }) },
      computerUseTool: { session: new ComputerUseSession() },
      historyTool: history,
      homeAssistantTools: {
        client: new HomeAssistantClient({ baseUrl: 'http://localhost:8123/', token: 'test-token' }),
      },
      memoryTools: {
        context: {
          memory: new ContextualMemory({ longTerm: new LongTermMemory({ storage: new SimpleStorage() }) }),
        },
      },
      rlTools: { backend: new InMemoryRLBackend() },
      sendMessageTool: { registry: outbound },
      workspaceMemoryTools: { workspaceRoot: workspace },
    })
    const names = new Set(registry.definitions().map(definition => definition.function.name))
    for (const name of [
      'agent_memory_status',
      'mixture_of_agents',
      'session_search',
      'skills_list',
      'skill_view',
      'browser_navigate',
      'clarify',
      'computer_use',
      'search_history',
      'ha_get_state',
      'save_memory',
      'rl_list_environments',
      'send_message',
      'skill_manage',
      'memory_add',
      'read_file',
    ]) {
      expect(names).toContain(name)
    }
  })
})

test('core registration rejects ambiguous skill_manage registrations', () => {
  const registry = new ToolRegistry()
  expect(() => registerCoreTools(registry, {
    agentMetaTools: {},
    skillManageTools: { authoredDirectory: '/tmp/xerxes-skills' },
  })).toThrow(ConfigurationError)
})
