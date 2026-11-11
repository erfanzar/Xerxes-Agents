// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { BUILTIN_AGENTS } from '../src/agents/definitions.js'
import { BoundaryRegistry, createSessionBoundaryFactory } from '../src/daemon/sessionBoundary.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { createNativeSubagentHost } from '../src/daemon/subagentHost.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { RunSafety } from '../src/runtime/runSafety.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import type { DataTrust } from '../src/security/boundaryTypes.js'
import { DataBoundary, parseDataBoundaryConfig, subagentBoundaryPort } from '../src/security/dataBoundary.js'

const lead = { data_boundary: 'hybrid', boundary: { private_paths: ['secret/**'], cloud_results: 'summary', summary_max_bytes: 300 } }

test('summary results: a cloud parent gets a marked, capped reply from a local agent that read private data', () => {
  const boundary = new DataBoundary(parseDataBoundaryConfig(lead, {}), '/work')
  boundary.markTainted('child-1', 'ReadFile touched secret/data.csv')
  const long = 'total 3 rows; '.repeat(60)
  const toCloud = boundary.filterSubagentResult(long, { childAgentId: 'child-1', childTrust: 'local', parentTrust: 'cloud' })
  expect(toCloud.withheld).toBe(false)
  expect(toCloud.forwardedBytes).toBeLessThanOrEqual(300)
  expect(toCloud.keptBytes).toBe(Buffer.byteLength(long) - toCloud.forwardedBytes!)
  expect(toCloud.text.split('\n')[0]).toContain('the rest stays on this machine')
  // A multi-byte character is never cut in half.
  const accents = boundary.filterSubagentResult('é'.repeat(400), { childAgentId: 'child-1', childTrust: 'local', parentTrust: 'cloud' })
  expect(accents.text).not.toContain('�')
  // A local parent sees everything; an untainted child passes untouched.
  expect(boundary.filterSubagentResult(long, { childAgentId: 'child-1', childTrust: 'local', parentTrust: 'local' }).text).toBe(long)
  expect(boundary.filterSubagentResult(long, { childAgentId: 'clean', childTrust: 'local', parentTrust: 'cloud' }).text).toBe(long)
  // The default still withholds.
  const strict = new DataBoundary(parseDataBoundaryConfig({ data_boundary: 'hybrid', boundary: { private_paths: ['secret/**'] } }, {}), '/work')
  strict.markTainted('child-1', 'read')
  expect(strict.filterSubagentResult(long, { childAgentId: 'child-1', childTrust: 'local', parentTrust: 'cloud' })).toMatchObject({ withheld: true })
})

test('bad cloud_results settings are refused with the key named', () => {
  expect(() => parseDataBoundaryConfig({ data_boundary: 'hybrid', boundary: { cloud_results: 'all' } }, {})).toThrow('boundary.cloud_results')
  expect(() => parseDataBoundaryConfig({ data_boundary: 'hybrid', boundary: { summary_max_bytes: 10 } }, {})).toThrow('boundary.summary_max_bytes')
  expect(() => parseDataBoundaryConfig({ data_boundary: 'hybrid', boundary: { summary_max_bytes: 1.5 } }, {})).toThrow('boundary.summary_max_bytes')
})

test('only a cloud conversation that may not open private paths is told to lead', () => {
  const boundary = new DataBoundary(parseDataBoundaryConfig(lead, {}), '/work')
  const text = boundary.leadInstruction('cloud')
  expect(text).toContain('secret/**')
  expect(text).toContain('local agents')
  expect(text).toContain('300 bytes')
  // The lead passes the spelling rule on, and states nothing about the files it was not told.
  expect(text).toContain('keep names, labels and IDs as the input spells them')
  expect(text).toContain('Never state a fact about them that no agent reported')
  expect(text).toContain('which currency or unit it uses')
  expect(boundary.leadInstruction('local')).toBe('')
  expect(new DataBoundary(parseDataBoundaryConfig({ data_boundary: 'hybrid' }, {}), '/work').leadInstruction('cloud')).toBe('')
  expect(new DataBoundary(parseDataBoundaryConfig({ ...lead, data_boundary: 'cloud' }, {}), '/work').leadInstruction('cloud')).toBe('')
  // The session gate carries it, from the registry the subagents share.
  const registry = new BoundaryRegistry()
  const port = createSessionBoundaryFactory({ runtime: () => lead, env: {}, trustOf: () => 'cloud' as const, registry })({ cwd: '/work', metadata: {} })
  expect(port?.instruction?.()).toBe(text)
  expect(registry.boundaryFor('/work', lead, {})).toBe(registry.boundaryFor('/work/', lead, {}))
})

/** A child that reads one file, then answers at length. */
class ReadThenAnswer implements LlmClient {
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    const results = request.messages.filter(message => message.role === 'tool')
    if (results.length === 0) {
      yield { toolCalls: [{ id: crypto.randomUUID(), type: 'function', function: { name: 'ReadFile', arguments: { file_path: 'secret/accounts.csv' } } }] }
      return
    }
    yield { content: `Seen: ${String(results[0]?.content).slice(0, 220)} ${'and more detail '.repeat(40)}` }
  }
}

async function runChild(childTrust: DataTrust): Promise<{ output: string; tainted: boolean; read: string }> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-hybrid-lead-'))
  await mkdir(join(root, 'secret'), { recursive: true })
  await writeFile(join(root, 'secret', 'accounts.csv'), 'id,balance\nA1,1200.50\n')
  const registry = new BoundaryRegistry()
  const tools = new ToolRegistry()
  let read = ''
  tools.register({ type: 'function', function: { name: 'ReadFile', description: 'read', parameters: { type: 'object', additionalProperties: true, properties: {} } } },
    () => { read = 'id,balance\nA1,1200.50'; return read })
  const host = createNativeSubagentHost({
    agentDefinitions: BUILTIN_AGENTS, cwd: root, eventBus: new DaemonSubagentEventBus(), llm: new ReadThenAnswer(), model: 'bonsai',
    permissionMode: 'accept-all', tools: tools.definitions(), toolExecutor: tools,
    resolveSourceWorkspace: () => root,
    dataBoundaryFor: path => registry.boundaryFor(path, lead, {}),
    trustOfProfile: () => childTrust,
    sourceTrust: () => 'cloud',
  })
  try {
    const child = await host.managerPort.spawn({ message: 'Count the rows in secret/accounts.csv', promptProfile: 'coder', title: 'Count', sourceAgentId: 'lead-session' })
    const done = await host.managerPort.wait([child.id], 2_000)
    return { output: String(done.completed[0]?.lastOutput ?? ''), tainted: registry.boundaryFor(root, lead, {})!.isTainted(child.id), read }
  } finally {
    await host.manager.shutdown()
    await rm(root, { recursive: true, force: true })
  }
}

test('a local child may read private files; its reply reaches the cloud lead marked and capped', async () => {
  const { output, tainted, read } = await runChild('local')
  expect(read).toContain('A1')
  expect(tainted).toBe(true)
  expect(output.startsWith('[Reply of local agent')).toBe(true)
  expect(output).toContain('the rest stays on this machine')
  expect(Buffer.byteLength(output)).toBeLessThan(600)
})

test('a cloud child is refused the private file, so nothing private reaches it or the lead', async () => {
  const { output, tainted, read } = await runChild('cloud')
  expect(read).toBe('')
  expect(tainted).toBe(false)
  expect(output).toContain('Data boundary')
  expect(output).not.toContain('[Reply of local agent')
})

test('a local child runs a script on private files without anyone to ask; a cloud child is refused it', () => {
  const boundary = new DataBoundary(parseDataBoundaryConfig(lead, {}), '/work')
  const script = { function: { name: 'exec_command', arguments: { cmd: 'python3', args: ['-I', '-'], stdin: "import csv\nprint(sum(1 for _ in open('secret/accounts.csv')))" } } }
  // The main conversation is asked first; a local subagent follows the normal permission mode.
  expect(boundary.toolDisposition(script, { agentId: 'main', trust: 'local' })).toBe('prompt')
  expect(subagentBoundaryPort(boundary, 'local', 'child-local').toolDisposition(script)).toBeUndefined()
  expect(boundary.isTainted('child-local')).toBe(true)
  expect(subagentBoundaryPort(boundary, 'cloud', 'child-cloud').toolDisposition(script)).toBe('deny')
  // Changing an input stays refused for every child.
  const edit = { function: { name: 'WriteFile', arguments: { file_path: 'secret/accounts.csv', content: 'x' } } }
  expect(subagentBoundaryPort(boundary, 'local', 'child-local').toolDisposition(edit)).toBe('deny')
})

/** A child that writes one new file, then reports what happened. */
class WriteThenReport implements LlmClient {
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    const results = request.messages.filter(message => message.role === 'tool')
    if (results.length === 0) {
      yield { toolCalls: [{ id: crypto.randomUUID(), type: 'function', function: { name: 'WriteFile', arguments: { file_path: 'out/summary.csv', content: 'region,total\nNorth,3\n' } } }] }
      return
    }
    yield { content: `Write said: ${String(results[0]?.content).slice(0, 120)}` }
  }
}

test.each([true, false])('a subagent in auto mode writes a new output file unasked only under its conversation safety floor (floor: %p)', async withFloor => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-child-floor-'))
  await writeFile(join(root, 'sales.csv'), 'region,units\nNorth,3\n')
  const tools = new ToolRegistry()
  let written = false
  tools.register({ type: 'function', function: { name: 'WriteFile', description: 'write', parameters: { type: 'object', additionalProperties: true, properties: {} } } },
    () => { written = true; return 'written' })
  const floor = withFloor ? await RunSafety.start({ outputFolder: root, sandboxUnavailable: 'test' }) : undefined
  const host = createNativeSubagentHost({
    agentDefinitions: BUILTIN_AGENTS, cwd: root, eventBus: new DaemonSubagentEventBus(), llm: new WriteThenReport(), model: 'bonsai',
    permissionMode: 'auto', tools: tools.definitions(), toolExecutor: tools,
    resolveSourceWorkspace: () => root,
    ...(floor ? { runSafetyFor: async () => floor } : {}),
  })
  try {
    const child = await host.managerPort.spawn({ message: 'Write the summary', promptProfile: 'coder', title: 'Write', sourceAgentId: 'lead-session' })
    const done = await host.managerPort.wait([child.id], 2_000)
    expect(written).toBe(withFloor)
    expect(String(done.completed[0]?.lastOutput)).toContain(withFloor ? 'written' : 'Permission denied')
  } finally {
    await host.manager.shutdown()
    await floor?.dispose?.()
    await rm(root, { recursive: true, force: true })
  }
})

test('with every path private, the output folder stays writable and nothing else does', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-boundary-writes-'))
  try {
    await writeFile(join(root, 'bank.csv'), 'txn_id\n')
    await mkdir(join(root, 'out'), { recursive: true })
    await writeFile(join(root, 'out', 'draft.csv'), 'x\n')
    const boundary = new DataBoundary(parseDataBoundaryConfig({ data_boundary: 'hybrid', boundary: { private_paths: ['**'] } }, {}), root)
    const write = (file_path: string) => boundary.toolDisposition({ function: { name: 'WriteFile', arguments: { file_path, content: 'y' } } }, { agentId: 'child', trust: 'local' })
    expect(write('out/discrepancies.csv')).toBeUndefined()
    expect(write('out/draft.csv')).toBeUndefined()
    expect(write('scripts/new.py')).toBe('deny')
    expect(write('bank.csv')).toBe('deny')
  } finally { await rm(root, { recursive: true, force: true }) }
})
