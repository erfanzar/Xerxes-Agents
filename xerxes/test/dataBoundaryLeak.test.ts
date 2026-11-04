// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Synthetic private-data leak suite: a canary in `secret/customers.csv` must
 * never reach a cloud-trust agent, its parent's context or the disclosure log
 * once a boundary mode is on, and nothing changes when the boundary is off.
 */

import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, test } from 'bun:test'

import type { DataTrust } from '../src/security/boundaryTypes.js'
import { DataBoundary, boundaryPermissionPort, parseDataBoundaryConfig } from '../src/security/dataBoundary.js'
import { DisclosureLog } from '../src/security/disclosureLog.js'
import { permissionDisposition } from '../src/streaming/permissions.js'
import { ClaudeAgentTools } from '../src/tools/claudeTools/index.js'
import type { SpawnAgentOptions, SpawnedAgentManagerPort, SpawnedAgentSnapshot } from '../src/operators/subagents.js'
import type { JsonObject } from '../src/types/toolCalls.js'

const CANARY = 'CANARY-7f3e9a-ACME-CUSTOMER-SSN-000-12-3456'
const roots: string[] = []
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function workspace(): string {
  const root = realpathSync(mkdtempSync(join(tmpdir(), 'xerxes-leak-')))
  roots.push(root)
  mkdirSync(join(root, 'secret'), { recursive: true })
  writeFileSync(join(root, 'secret', 'customers.csv'), `name,ssn\nalice,${CANARY}\n`)
  writeFileSync(join(root, 'README.md'), '# project\n')
  return root
}

function makeBoundary(root: string, mode: 'off' | 'hybrid' | 'private'): DataBoundary {
  const runtime = mode === 'off'
    ? { boundary: { private_paths: ['secret/**'] } }
    : { data_boundary: mode, boundary: { private_paths: ['secret/**'] } }
  return new DataBoundary(parseDataBoundaryConfig(runtime, {}), root)
}

const call = (name: string, args: JsonObject) => ({ function: { name, arguments: args } })
const LEAK_CALLS = [
  call('ReadFile', { file_path: 'secret/customers.csv' }),
  call('GrepTool', { pattern: 'CANARY', path: '.' }),
  call('GrepTool', { pattern: 'ssn', path: 'secret' }),
  call('GlobTool', { pattern: 'secret/*.csv' }),
  call('ListDir', { directory_path: 'secret' }),
  call('exec_command', { cmd: 'cat $(echo c2VjcmV0 | base64 -d)/*' }),
  call('exec_command', { cmd: 'cat', args: ['secret/customers.csv'] }),
  call('Bash', { command: 'grep -r ssn .' }),
  call('pty_open', { command: 'less secret/customers.csv' }),
]

test('hybrid: a cloud agent cannot read, search or shell into private paths even under accept-all', () => {
  const root = workspace()
  const boundary = makeBoundary(root, 'hybrid')
  const port = boundaryPermissionPort(boundary, () => 'cloud')
  for (const leak of LEAK_CALLS) {
    const disposition = permissionDisposition(leak, 'accept-all', undefined, 'parent', { boundary: port })
    expect({ tool: leak.function.name, args: leak.function.arguments, disposition })
      .toEqual({ tool: leak.function.name, args: leak.function.arguments, disposition: leak.function.arguments.cmd === 'cat $(echo c2VjcmV0 | base64 -d)/*' ? 'prompt' : 'deny' })
  }
  // The obfuscated command is unresolved: it prompts, never auto-allows.
  expect(boundary.toolVerdict(LEAK_CALLS[5]!, { trust: 'cloud' })?.reason).toContain('cannot be checked statically')
  // Unrelated reads stay unaffected.
  expect(permissionDisposition(call('ReadFile', { file_path: 'README.md' }), 'accept-all', undefined, 'parent', { boundary: port })).toBe('allow')
})

test('hybrid: a local agent may read private data and is tainted for it', () => {
  const root = workspace()
  const boundary = makeBoundary(root, 'hybrid')
  const port = boundaryPermissionPort(boundary, () => 'local')
  expect(permissionDisposition(LEAK_CALLS[0]!, 'accept-all', undefined, 'local-child', { boundary: port })).toBe('allow')
  expect(boundary.isTainted('local-child')).toBe(true)
  // A shell command that names a private path asks first for a local agent.
  expect(permissionDisposition(LEAK_CALLS[6]!, 'accept-all', undefined, 'other-local', { boundary: port })).toBe('prompt')
})

test('private mode: a cloud agent may not run any tool', () => {
  const root = workspace()
  const boundary = makeBoundary(root, 'private')
  const port = boundaryPermissionPort(boundary, () => 'cloud')
  expect(permissionDisposition(call('ReadFile', { file_path: 'README.md' }), 'accept-all', undefined, undefined, { boundary: port })).toBe('deny')
})

function snapshot(id: string, overrides: Partial<SpawnedAgentSnapshot> = {}): SpawnedAgentSnapshot {
  return {
    agentId: 'general-purpose',
    closed: false,
    createdAt: '2026-11-02T00:00:00.000Z',
    id,
    name: id,
    promptProfile: 'general-purpose',
    queueSize: 0,
    status: 'completed',
    title: 'Summarize customers',
    updatedAt: '2026-11-02T00:00:01.000Z',
    sourceAgentId: 'session-1',
    ...overrides,
  }
}

/** A manager whose child "reads" the secret through the boundary, then returns it. */
function leakyManager(boundary: DataBoundary, childTrust: DataTrust): SpawnedAgentManagerPort & { spawned: SpawnAgentOptions[] } {
  const handles: SpawnedAgentSnapshot[] = []
  const spawned: SpawnAgentOptions[] = []
  return {
    spawned,
    close: id => ({ ...handles.find(handle => handle.id === id)!, closed: true, previousStatus: 'completed', status: 'closed' }),
    listHandles: () => handles,
    resume: id => handles.find(handle => handle.id === id)!,
    sendInput: async id => handles.find(handle => handle.id === id)!,
    spawn: async options => {
      spawned.push(options ?? {})
      const id = `child-${handles.length + 1}`
      boundary.toolDisposition(call('ReadFile', { file_path: 'secret/customers.csv' }), { agentId: id, trust: childTrust })
      const content = readFileSync(join(boundary.root, 'secret', 'customers.csv'), 'utf8')
      const made = snapshot(id, {
        ...(options?.agent?.model ? { model: options.agent.model } : {}),
        ...(options?.agent?.providerProfile ? { providerProfile: options.agent.providerProfile } : {}),
        lastOutput: `Summary: ${content}`,
        completionSummary: `found ${CANARY}`,
      })
      handles.push(made)
      return made
    },
    wait: async targets => ({ completed: handles.filter(handle => targets.includes(handle.id)), pending: [] }),
  }
}

const trustOf = (model: string, profile?: string): DataTrust => (profile === 'local' || model.startsWith('local-') ? 'local' : 'cloud')
const context = { agentId: 'parent', metadata: {}, sessionId: 'session-1' }

test('hybrid: a tainted local child result never reaches a cloud parent on any read path', async () => {
  const root = workspace()
  const boundary = makeBoundary(root, 'hybrid')
  const manager = leakyManager(boundary, 'local')
  const tools = new ClaudeAgentTools({ manager, dataBoundary: { boundary, trustOf, parentTrust: () => 'cloud' } })
  const created = await tools.execute('AgentTool', { prompt: 'summarize', title: 'Summarize', model: 'local-llm', provider_profile: 'local', timeout: 1 }, context)
  const outputs = [
    JSON.stringify(created),
    String(await tools.execute('TaskOutputTool', { task_id: 'child-1' }, context)),
    JSON.stringify(await tools.execute('TaskGetTool', { task_id: 'child-1' }, context)),
    JSON.stringify(await tools.execute('AwaitAgents', { agent_ids: ['child-1'], timeout_seconds: 1 }, context)),
    JSON.stringify(await tools.execute('CheckAgentMessages', { peek: true }, context)),
    JSON.stringify(context.metadata),
  ]
  for (const output of outputs) expect(output).not.toContain(CANARY)
  expect(outputs[0]).toContain('withheld by the data boundary')
  expect(outputs[1]).toMatch(/\d+ bytes were not forwarded/)
})

test('hybrid: a clean child and a local parent both see the result; the local parent becomes tainted', async () => {
  const root = workspace()
  const boundary = makeBoundary(root, 'hybrid')
  const tools = new ClaudeAgentTools({ manager: leakyManager(boundary, 'local'), dataBoundary: { boundary, trustOf, parentTrust: () => 'local' } })
  const result = JSON.stringify(await tools.execute('AgentTool', { prompt: 'summarize', title: 'Summarize', model: 'local-llm', timeout: 1 }, context))
  expect(result).toContain(CANARY)
  // parentAgentId is not set by the fake manager, so taint goes through the
  // spawn request's parent id when the host reports it; verify the direct API.
  boundary.filterSubagentResult('x', { childAgentId: 'child-1', childTrust: 'local', parentTrust: 'local', parentAgentId: 'parent' })
  expect(boundary.isTainted('parent')).toBe(true)
})

test('hybrid: the tier lock refuses an explicit cloud model under a local tier', async () => {
  const root = workspace()
  const boundary = makeBoundary(root, 'hybrid')
  const manager = leakyManager(boundary, 'local')
  const tools = new ClaudeAgentTools({
    manager,
    intelligence: { default: 'light', light: { model: 'local-llm', provider_profile: 'local' } },
    dataBoundary: { boundary, trustOf, parentTrust: () => 'cloud' },
  })
  await expect(tools.execute('AgentTool', { prompt: 'p', title: 'T', model: 'gpt-5', provider_profile: 'openai' }, context))
    .rejects.toMatchObject({ code: 'DATA_BOUNDARY_TIER_LOCKED' })
  await expect(tools.execute('SpawnAgents', { agents: [{ prompt: 'p', title: 'T', intelligence: 'light', model: 'gpt-5' }] }, context))
    .rejects.toMatchObject({ code: 'DATA_BOUNDARY_TIER_LOCKED' })
  expect(manager.spawned).toHaveLength(0)
  // The local tier itself is fine.
  await tools.execute('AgentTool', { prompt: 'p', title: 'T', timeout: 1 }, context)
  expect(manager.spawned[0]?.agent?.model).toBe('local-llm')
})

test('private: outbound cloud requests are refused and the log never holds the text', async () => {
  const root = workspace()
  const boundary = makeBoundary(root, 'private')
  const path = join(root, 'home', 'boundary', 'disclosures.jsonl')
  const log = new DisclosureLog({ path })
  let asked = 0
  const gate = boundary.outboundGate({ approve: async () => { asked += 1; return true }, log })
  const text = `system...\nuser: summarize\ntool: ${CANARY}`
  const verdict = await gate.check({ provider: 'openai', model: 'gpt-5', trust: 'cloud', purpose: 'turn', sessionId: 's', text, bytes: Buffer.byteLength(text) })
  expect(verdict.allow).toBe(false)
  expect(asked).toBe(0)
  const records = log.read()
  expect(records).toHaveLength(1)
  expect(records[0]).toMatchObject({ decision: 'denied', provider: 'openai', trust: 'cloud' })
  expect(readFileSync(path, 'utf8')).not.toContain(CANARY)
  // Local requests pass and are recorded as sent.
  expect(await gate.check({ provider: 'llama', model: 'local', trust: 'local', purpose: 'turn', text, bytes: text.length })).toEqual({ allow: true, decision: 'sent' })
})

test('off: every one of these calls behaves exactly as today', async () => {
  const root = workspace()
  const boundary = makeBoundary(root, 'off')
  const port = boundaryPermissionPort(boundary, () => 'cloud')
  for (const leak of LEAK_CALLS) {
    for (const mode of ['accept-all', 'auto', 'manual', 'plan'] as const) {
      expect(permissionDisposition(leak, mode, undefined, 'parent', { boundary: port })).toBe(permissionDisposition(leak, mode, undefined, 'parent'))
    }
  }
  expect(boundary.isTainted('parent')).toBe(false)
  const manager = leakyManager(boundary, 'local')
  const tools = new ClaudeAgentTools({
    manager,
    intelligence: { default: 'light', light: { model: 'local-llm', provider_profile: 'local' } },
    dataBoundary: { boundary, trustOf, parentTrust: () => 'cloud' },
  })
  const result = JSON.stringify(await tools.execute('AgentTool', { prompt: 'p', title: 'T', model: 'gpt-5', timeout: 1 }, context))
  expect(result).toContain(CANARY)
  expect(manager.spawned[0]?.agent?.model).toBe('gpt-5')
  const log = new DisclosureLog({ path: join(root, 'off.jsonl') })
  const gate = boundary.outboundGate({ approve: async () => false, log })
  expect(await gate.check({ provider: 'openai', model: 'gpt', trust: 'cloud', purpose: 'turn', text: CANARY, bytes: CANARY.length })).toEqual({ allow: true, decision: 'sent' })
  expect(log.read()).toEqual([])
})
