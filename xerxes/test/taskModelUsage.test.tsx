// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Database } from 'bun:sqlite'
import { afterEach, expect, test } from 'bun:test'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { connect } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { formatUsageText, parseUsageReport, taskModelsOf, taskSplit } from '../src/auth/usageView.js'
import { ProfileStore } from '../src/bridge/profiles.js'
import { registerConfiguredAdvisor } from '../src/daemon/advisorWiring.js'
import { runtimeSettingsFile } from '../src/daemon/runMode.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { ModelSplitChipView, TaskModelsView } from '../src/desktop/renderer/TaskModels.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { closeModelUsageLedger, ModelUsageLedger, openModelUsageLedger, recordModelUsage } from '../src/runtime/modelUsageLedger.js'

afterEach(() => closeModelUsageLedger())

test('a ledger from before tasks gains task and role columns and keeps its rows', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-task-usage-'))
  try {
    const path = join(directory, 'model-usage.sqlite')
    const old = new Database(path, { create: true })
    old.run(`CREATE TABLE rounds (at INTEGER NOT NULL, model TEXT NOT NULL, profile TEXT NOT NULL DEFAULT '',
      input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, cache_read_tokens INTEGER NOT NULL DEFAULT 0, cache_write_tokens INTEGER NOT NULL DEFAULT 0)`)
    old.run("INSERT INTO rounds (at, model, profile, input_tokens, output_tokens) VALUES (5000, 'old-model', 'p', 10, 2)")
    old.close()

    const ledger = new ModelUsageLedger(path)
    ledger.record({ model: 'ternary-bonsai-2-27b', profile: 'bonsai-local', session: 's1', role: 'worker', inputTokens: 9000, outputTokens: 400, at: 6000 })
    ledger.record({ model: 'ternary-bonsai-2-27b', profile: 'bonsai-local', session: 's1', role: 'worker', inputTokens: 1000, outputTokens: 100, at: 7000 })
    ledger.record({ model: 'claude-code/default', profile: 'claude-code', session: 's1', role: 'advisor', inputTokens: 600, outputTokens: 200, bytesSent: 2400, at: 8000 })
    // An advisor whose provider reports no tokens still counts as a call: it sent something.
    ledger.record({ model: 'claude-code/default', profile: 'claude-code', session: 's1', role: 'advisor', inputTokens: 0, outputTokens: 0, bytesSent: 1600, at: 9000 })
    ledger.record({ model: 'ternary-bonsai-2-27b', profile: 'bonsai-local', session: 's2', role: 'worker', inputTokens: 5, outputTokens: 5, at: 9500 })

    expect(ledger.sessionTotals('s1')).toEqual([
      { model: 'ternary-bonsai-2-27b', profile: 'bonsai-local', role: 'worker', rounds: 2, inputTokens: 10_000, outputTokens: 500, cacheReadTokens: 0, cacheWriteTokens: 0, bytesSent: 0, lastAt: 7000 },
      { model: 'claude-code/default', profile: 'claude-code', role: 'advisor', rounds: 2, inputTokens: 600, outputTokens: 200, cacheReadTokens: 0, cacheWriteTokens: 0, bytesSent: 4000, lastAt: 9000 },
    ])
    expect(ledger.sessionTotals('')).toEqual([])
    // The machine totals still include the old row, which belongs to no task.
    expect(ledger.totals(0).map(row => row.model)).toContain('old-model')
    ledger.close()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

const bonsai = { name: 'bonsai-local', provider: 'custom', base_url: 'http://127.0.0.1:11581/v1', api_key: '', model: 'ternary-bonsai-2-27b', data_trust: 'local', model_capabilities: {}, sampling: {} }
const openrouter = { name: 'openrouter', provider: 'openrouter', base_url: 'https://openrouter.ai/api/v1', api_key: 'or-key', model: 'deepseek/deepseek-v4.1-flash', model_capabilities: {}, sampling: {} }

test('session.usage reports the task model by model, where each ran, and the advisor limits', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-task-usage-daemon-'))
  try {
    openModelUsageLedger(join(root, 'usage', 'model-usage.sqlite'))
    const profilesFile = join(root, 'profiles.json')
    await writeFile(profilesFile, JSON.stringify({ profiles: { 'bonsai-local': bonsai, openrouter } }))
    const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
    const socketPath = join(root, 'daemon.sock')
    const server = new DaemonServer({
      profileStore: new ProfileStore(profilesFile), runtime, socketPath,
      modeSettings: { file: runtimeSettingsFile(join(root, 'daemon', 'config.json')), loaded: { advisor: { profile: 'openrouter', max_calls: 4, max_usd: 2 } }, env: {} },
    })
    await server.start()
    const socket = connect({ path: socketPath })
    await new Promise<void>((resolve, reject) => { socket.once('connect', resolve); socket.once('error', reject) })
    socket.setEncoding('utf8')
    let buffer = '', id = 0
    const pending = new Map<number, (value: Record<string, unknown>) => void>()
    socket.on('data', chunk => {
      buffer += String(chunk)
      for (let newline = buffer.indexOf('\n'); newline >= 0; newline = buffer.indexOf('\n')) {
        const frame = JSON.parse(buffer.slice(0, newline)) as { id?: number; result?: Record<string, unknown> }
        buffer = buffer.slice(newline + 1)
        if (typeof frame.id === 'number') pending.get(frame.id)?.(frame.result ?? {})
      }
    })
    const call = (method: string, params: Record<string, unknown> = {}) => new Promise<Record<string, unknown>>(resolve => {
      id += 1; pending.set(id, resolve); socket.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
    })
    try {
      await call('initialize', { session_key: 'desktop-task' })
      const sessionId = String(((await call('session.status')).session as Record<string, unknown>).id)
      recordModelUsage({ model: 'ternary-bonsai-2-27b', profile: 'bonsai-local', session: sessionId, role: 'worker', inputTokens: 20_000, outputTokens: 1_000 })
      recordModelUsage({ model: 'deepseek/deepseek-v4.1-flash', profile: 'openrouter', session: sessionId, role: 'advisor', inputTokens: 1_500, outputTokens: 500, bytesSent: 6_000 })
      recordModelUsage({ model: 'some-model', profile: 'deleted-profile', session: sessionId, role: 'subagent', inputTokens: 100, outputTokens: 0 })
      recordModelUsage({ model: 'ternary-bonsai-2-27b', profile: 'bonsai-local', session: 'another-task', role: 'worker', inputTokens: 99_999, outputTokens: 1 })

      const usage = await call('session.usage')
      expect(usage.ok).toBe(true)
      expect(usage.advisor).toEqual({ profile: 'openrouter', max_calls: 4, max_usd: 2 })
      const models = usage.models as Array<Record<string, unknown>>
      expect(models.map(row => [row.model, row.role, row.local])).toEqual([
        ['ternary-bonsai-2-27b', 'worker', true],
        ['deepseek/deepseek-v4.1-flash', 'advisor', false],
        ['some-model', 'subagent', null],
      ])
      expect(models[1]).toMatchObject({ calls: 1, input_tokens: 1_500, output_tokens: 500, bytes_sent: 6_000 })

      // The same breakdown rides on usage.report's session, and the machine totals say where each model ran.
      const report = parseUsageReport(await call('usage.report'))
      expect(report.session?.models?.map(row => row.role)).toEqual(['worker', 'advisor', 'subagent'])
      expect(report.models?.find(row => row.model === 'ternary-bonsai-2-27b')?.local).toBe(true)
      const split = taskSplit(report.session!.models!)
      expect(split).toMatchObject({ localTokens: 21_000, cloudTokens: 2_000, unknownTokens: 100, advisorCalls: 1, bytesSent: 6_000 })
      expect(formatUsageText(report)).toContain('Advisor budget')
    } finally {
      socket.destroy()
      await server.stop()
    }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

const hybridTask = taskModelsOf({
  models: [
    { model: 'ternary-bonsai-2-27b', profile: 'bonsai-local', role: 'worker', local: true, calls: 14, input_tokens: 970_000, output_tokens: 30_000, cache_read_tokens: 0, cache_write_tokens: 0 },
    { model: 'claude-code/default', profile: 'claude-code', role: 'advisor', local: false, calls: 2, input_tokens: 25_000, output_tokens: 5_000, cache_read_tokens: 0, cache_write_tokens: 0, bytes_sent: 14_000, cost_usd: 0.12, cost_basis: 'subscription' },
  ],
  advisor: { profile: 'claude-code', max_calls: 5, max_usd: 1 },
})

test('the split counts tokens by where they ran and never invents a price', () => {
  const split = taskSplit(hybridTask.models!)
  expect(split).toMatchObject({ localTokens: 1_000_000, cloudTokens: 30_000, advisorCalls: 2, bytesSent: 14_000, cloudOnPlan: true, cloudUsd: undefined })
  expect(Math.round(split.localShare! * 100)).toBe(97)
  // A cloud model nobody prices makes the cloud total unknown rather than $0.
  const unpriced = taskSplit([...hybridTask.models!, { model: 'x', local: false, calls: 1, input: 10, output: 10, cacheRead: 0, cacheWrite: 0 }])
  expect(unpriced).toMatchObject({ cloudOnPlan: false, cloudUsd: undefined })
  expect(taskSplit([{ model: 'b', local: true, calls: 1, input: 5, output: 5, cacheRead: 0, cacheWrite: 0 }])).toMatchObject({ cloudUsd: 0, cloudOnPlan: false, localShare: 1 })
  expect(taskModelsOf({ ok: true })).toEqual({})
})

test('Hybrid shows the split in the composer and the full breakdown in Usage', () => {
  const task = { models: hybridTask.models!, advisor: hybridTask.advisor! }
  const chip = renderToStaticMarkup(createElement(ModelSplitChipView, { task, hybrid: true, onOpen: () => {} }))
  expect(chip).toContain('97% on this Mac')
  expect(chip).toContain('data-where="local"')
  expect(chip).toContain('data-where="cloud"')
  expect(chip).toContain('Advisor · claude-code/default (cloud)')

  const panel = renderToStaticMarkup(createElement(TaskModelsView, { task, hybrid: true }))
  for (const words of ['This task', 'On this Mac <b>97%</b>', 'Cloud <b>3%</b>', 'included in plan', 'Advisor calls', 'Sent to the advisor', '14 KB', '5 calls · $1.00']) {
    expect(panel).toContain(words)
  }

  // Outside Hybrid, a task that stayed on one side shows no chip; before any call, nothing at all.
  const localOnly = { models: [hybridTask.models![0]!] }
  expect(renderToStaticMarkup(createElement(ModelSplitChipView, { task: localOnly, hybrid: false, onOpen: () => {} }))).toBe('')
  expect(renderToStaticMarkup(createElement(ModelSplitChipView, { task: localOnly, hybrid: true, onOpen: () => {} }))).toContain('All on this Mac')
  expect(renderToStaticMarkup(createElement(ModelSplitChipView, { task: { models: [] }, hybrid: true, onOpen: () => {} }))).toBe('')
  expect(renderToStaticMarkup(createElement(TaskModelsView, { task: { models: [] }, hybrid: true }))).toContain('No model calls in this task yet.')
})

test('an advisor call through the configured provider is charged to its task as cloud advisor use', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-task-usage-advisor-'))
  // A stand-in for the cloud provider on this machine: an OpenAI-style endpoint that answers once.
  const server = Bun.serve({
    port: 0,
    async fetch(request) {
      const body = await request.json() as { stream?: boolean }
      const usage = { prompt_tokens: 300, completion_tokens: 50, total_tokens: 350 }
      if (!body.stream) return Response.json({ id: 'c', object: 'chat.completion', model: 'stub-advisor', choices: [{ index: 0, message: { role: 'assistant', content: 'Join on the invoice id.' }, finish_reason: 'stop' }], usage })
      const chunks = [
        { id: 'c', object: 'chat.completion.chunk', model: 'stub-advisor', choices: [{ index: 0, delta: { role: 'assistant', content: 'Join on the invoice id.' }, finish_reason: null }] },
        { id: 'c', object: 'chat.completion.chunk', model: 'stub-advisor', choices: [{ index: 0, delta: {}, finish_reason: 'stop' }], usage },
      ]
      return new Response(chunks.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`).join('') + 'data: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } })
    },
  })
  try {
    const ledger = openModelUsageLedger(join(directory, 'model-usage.sqlite'))!
    const tools = new ToolRegistry()
    const advisorProfile = { name: 'stub-cloud', provider: 'openai', base_url: `http://127.0.0.1:${server.port}/v1`, api_key: 'k', model: 'stub-advisor', data_trust: 'cloud', model_capabilities: {}, sampling: {} }
    expect(registerConfiguredAdvisor(tools, {
      runtime: { advisor: { profile: 'stub-cloud' } }, env: {}, workspaceRoot: directory,
      profiles: { get: name => name === 'stub-cloud' ? advisorProfile as never : undefined },
    })).toBe(true)
    const answer = await tools.execute({ id: 'a1', type: 'function', function: { name: 'AskAdvisor', arguments: { question: 'How do I match the rows?', context: 'two CSV files', kind: 'plan' } } }, { metadata: {}, sessionId: 'task-7' })
    expect(answer).toContain('Join on the invoice id.')
    const [row] = ledger.sessionTotals('task-7')
    expect(row).toMatchObject({ model: 'stub-advisor', profile: 'stub-cloud', role: 'advisor', rounds: 1, inputTokens: 300, outputTokens: 50 })
    expect(row!.bytesSent).toBeGreaterThan(100)
  } finally {
    server.stop(true)
    await rm(directory, { recursive: true, force: true })
  }
})
