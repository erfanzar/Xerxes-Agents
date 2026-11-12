// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, test } from 'bun:test'

import { ValidationError } from '../src/core/errors.js'
import { DataBoundaryError, type OutboundRequestSummary } from '../src/security/boundaryTypes.js'
import {
  DataBoundary,
  boundaryPermissionPort,
  deriveDataTrust,
  parseDataBoundaryConfig,
} from '../src/security/dataBoundary.js'
import { DisclosureLog } from '../src/security/disclosureLog.js'
import { permissionDisposition } from '../src/streaming/permissions.js'
import type { JsonObject } from '../src/types/toolCalls.js'

const roots: string[] = []
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function workspace(): string {
  const root = realpathSync(mkdtempSync(join(tmpdir(), 'xerxes-boundary-')))
  roots.push(root)
  mkdirSync(join(root, 'secret'), { recursive: true })
  mkdirSync(join(root, 'src'), { recursive: true })
  mkdirSync(join(root, 'out'), { recursive: true })
  writeFileSync(join(root, 'secret', 'customers.csv'), 'name\nalice\n')
  writeFileSync(join(root, 'src', 'main.ts'), 'export {}\n')
  writeFileSync(join(root, 'out', 'report.md'), '# old\n')
  return root
}

const call = (name: string, args: JsonObject) => ({ function: { name, arguments: args } })

function boundary(root: string, runtime: Record<string, unknown> = {}): DataBoundary {
  return new DataBoundary(parseDataBoundaryConfig({ data_boundary: 'hybrid', boundary: { private_paths: ['secret/**'] }, ...runtime }, {}), root)
}

test('config defaults keep the boundary off and the env var wins', () => {
  expect(parseDataBoundaryConfig({}, {})).toEqual({ mode: 'off', approveOutbound: 'off', readOnlyInputs: false, privatePaths: [], outputDir: 'out', cloudResults: 'withhold', summaryMaxBytes: 8000 })
  const hybrid = parseDataBoundaryConfig({ data_boundary: 'cloud' }, { XERXES_DATA_BOUNDARY: 'hybrid' })
  expect(hybrid.mode).toBe('hybrid')
  expect(hybrid.approveOutbound).toBe('first')
  expect(hybrid.readOnlyInputs).toBe(true)
  expect(parseDataBoundaryConfig({ data_boundary: 'private' }, {}).approveOutbound).toBe('off')
  expect(parseDataBoundaryConfig({ data_boundary: 'hybrid', boundary: { approve_outbound: 'every', read_only_inputs: false, output_dir: 'build/' } }, {}))
    .toMatchObject({ approveOutbound: 'every', readOnlyInputs: false, outputDir: 'build' })
})

test('invalid settings name the offending key', () => {
  const cases: Array<[Record<string, unknown>, Record<string, string>, string]> = [
    [{ data_boundary: 'loose' }, {}, 'data_boundary'],
    [{}, { XERXES_DATA_BOUNDARY: 'nope' }, 'XERXES_DATA_BOUNDARY'],
    [{ boundary: 'x' }, {}, 'boundary'],
    [{ boundary: { approve_outbound: 'sometimes' } }, {}, 'boundary.approve_outbound'],
    [{ boundary: { read_only_inputs: 'yes' } }, {}, 'boundary.read_only_inputs'],
    [{ boundary: { private_paths: 'secret' } }, {}, 'boundary.private_paths'],
    [{ boundary: { private_paths: ['../escape/**'] } }, {}, 'boundary.private_paths[0]'],
    [{ boundary: { output_dir: '/abs' } }, {}, 'boundary.output_dir'],
    [{ boundary: { surprise: true } }, {}, 'boundary.surprise'],
  ]
  for (const [runtime, env, field] of cases) {
    let caught: unknown
    try { parseDataBoundaryConfig(runtime, env) } catch (error) { caught = error }
    expect(caught).toBeInstanceOf(ValidationError)
    expect((caught as ValidationError).field).toBe(field)
  }
})

test('project data-policy.yaml adds private paths and can only tighten read_only_inputs', () => {
  const files: Record<string, string> = {}
  const project = { root: '/proj', readFile: (path: string) => files[path] }
  files['/proj/.xerxes/data-policy.yaml'] = 'private_paths:\n  - hr/**\nread_only_inputs: false\n'
  const config = parseDataBoundaryConfig({ data_boundary: 'hybrid', boundary: { private_paths: ['secret/**'], read_only_inputs: true } }, {}, project)
  expect(config.privatePaths).toEqual(['secret/**', 'hr/**'])
  expect(config.readOnlyInputs).toBe(true)
  files['/proj/.xerxes/data-policy.yaml'] = 'read_only_inputs: true\n'
  expect(parseDataBoundaryConfig({ data_boundary: 'cloud', boundary: { read_only_inputs: false } }, {}, project).readOnlyInputs).toBe(true)
  files['/proj/.xerxes/data-policy.yaml'] = 'unknown_key: 1\n'
  expect(() => parseDataBoundaryConfig({ data_boundary: 'hybrid' }, {}, project)).toThrow(ValidationError)
  // Off: the project file is never consulted.
  expect(parseDataBoundaryConfig({}, {}, project).privatePaths).toEqual([])
})

test('deriveDataTrust: explicit field wins, loopback and sockets are local, everything else cloud', () => {
  expect(deriveDataTrust({ base_url: 'http://localhost:11581/v1' })).toBe('local')
  expect(deriveDataTrust({ base_url: 'http://127.0.0.2:8080' })).toBe('local')
  expect(deriveDataTrust({ base_url: 'http://[::1]:8080/v1' })).toBe('local')
  expect(deriveDataTrust({ base_url: 'unix:/tmp/llama.sock' })).toBe('local')
  expect(deriveDataTrust({ base_url: '/tmp/llama.sock' })).toBe('local')
  expect(deriveDataTrust({ base_url: 'https://api.openai.com/v1' })).toBe('cloud')
  expect(deriveDataTrust({ base_url: 'http://127.example.com' })).toBe('cloud')
  expect(deriveDataTrust({ base_url: '' })).toBe('cloud')
  expect(deriveDataTrust({ base_url: 'claude-code://local' })).toBe('cloud')
  expect(deriveDataTrust(undefined)).toBe('cloud')
  expect(deriveDataTrust({ base_url: 'http://localhost:1', data_trust: 'cloud' })).toBe('cloud')
  expect(deriveDataTrust({ base_url: 'https://gpu.lan', data_trust: 'local' })).toBe('local')
})

test('isPrivatePath matches relative globs, ancestors and canonicalized paths', () => {
  const root = workspace()
  const gate = boundary(root)
  expect(gate.isPrivatePath('secret/customers.csv')).toBe(true)
  expect(gate.isPrivatePath(join(root, 'secret'))).toBe(true)
  expect(gate.isPrivatePath('./src/../secret/customers.csv')).toBe(true)
  expect(gate.isPrivatePath('secret/new-file.csv')).toBe(true)
  expect(gate.isPrivatePath('src/main.ts')).toBe(false)
  expect(gate.isPrivatePath('secretive.txt')).toBe(false)
})

test('read_only_inputs refuses edits to existing inputs but allows new files and the output dir', () => {
  const root = workspace()
  const gate = boundary(root)
  const local = { trust: 'local' as const }
  expect(gate.toolDisposition(call('FileEditTool', { file_path: 'src/main.ts', old_string: 'a', new_string: 'b' }), local)).toBe('deny')
  expect(gate.toolDisposition(call('WriteFile', { file_path: 'src/new.ts', content: 'x' }), local)).toBeUndefined()
  expect(gate.toolDisposition(call('WriteFile', { file_path: 'out/report.md', content: 'x', overwrite: true }), local)).toBeUndefined()
  expect(gate.toolDisposition(call('WriteFile', { file_path: 'secret/new.csv', content: 'x' }), local)).toBe('deny')
  const verdict = gate.toolVerdict(call('AppendFile', { file_path: 'src/main.ts', lines: 'x' }), local)
  expect(verdict?.code).toBe('DATA_BOUNDARY_READ_ONLY')
  const writable = boundary(root, { boundary: { private_paths: ['secret/**'], read_only_inputs: false } })
  expect(writable.toolDisposition(call('FileEditTool', { file_path: 'src/main.ts' }), local)).toBeUndefined()
})

test('non read-only shell prompts; read-only shell has no opinion', () => {
  const root = workspace()
  const gate = boundary(root)
  expect(gate.toolDisposition(call('exec_command', { cmd: 'git', args: ['status'] }), { trust: 'cloud' })).toBeUndefined()
  expect(gate.toolDisposition(call('exec_command', { cmd: 'rm', args: ['-rf', 'src'] }), { trust: 'local' })).toBe('prompt')
  expect(gate.toolDisposition(call('Bash', { command: 'ls src && touch x' }), { trust: 'cloud' })).toBe('prompt')
})

test('permission gate: deny wins, prompt beats allow, no opinion keeps the mode result', () => {
  const answers: Record<string, 'deny' | 'prompt' | undefined> = { A: 'deny', B: 'prompt', C: undefined }
  const options = { boundary: { toolDisposition: (candidate: { function: { name: string } }) => answers[candidate.function.name] } }
  expect(permissionDisposition(call('A', {}), 'accept-all', undefined, undefined, options)).toBe('deny')
  expect(permissionDisposition(call('B', {}), 'accept-all', undefined, undefined, options)).toBe('prompt')
  expect(permissionDisposition(call('C', {}), 'accept-all', undefined, undefined, options)).toBe('allow')
  expect(permissionDisposition(call('C', {}), 'manual', undefined, undefined, options)).toBe('prompt')
  // Policy deny still wins over a boundary prompt.
  expect(permissionDisposition(call('B', {}), 'accept-all', { check: () => 'deny' }, undefined, options)).toBe('deny')
  // A boundary prompt holds even for always-approval tools and safe tools.
  expect(permissionDisposition(call('ReadFile', { file_path: 'x' }), 'auto', undefined, undefined, { boundary: { toolDisposition: () => 'deny' } })).toBe('deny')
})

test('boundaryPermissionPort passes the agent id and its trust', () => {
  const root = workspace()
  const gate = boundary(root)
  const port = boundaryPermissionPort(gate, agentId => (agentId === 'local-child' ? 'local' : 'cloud'))
  const read = call('ReadFile', { file_path: 'secret/customers.csv' })
  expect(permissionDisposition(read, 'accept-all', undefined, 'parent', { boundary: port })).toBe('deny')
  expect(permissionDisposition(read, 'accept-all', undefined, 'local-child', { boundary: port })).toBe('allow')
  expect(gate.isTainted('local-child')).toBe(true)
  expect(gate.isTainted('parent')).toBe(false)
})

test('tier lock: hybrid refuses cloud over a local tier or from a tainted local parent; private refuses any cloud', () => {
  const root = workspace()
  const gate = boundary(root)
  expect(() => gate.assertTierAllowed({ requestedTrust: 'cloud', tierTrust: 'local', parentTrust: 'cloud' }))
    .toThrow(expect.objectContaining({ code: 'DATA_BOUNDARY_TIER_LOCKED' }))
  gate.assertTierAllowed({ requestedTrust: 'cloud', tierTrust: 'cloud', parentTrust: 'local', parentAgentId: 'clean' })
  gate.markTainted('dirty', 'test')
  expect(() => gate.assertTierAllowed({ requestedTrust: 'cloud', parentTrust: 'local', parentAgentId: 'dirty' })).toThrow(DataBoundaryError)
  gate.assertTierAllowed({ requestedTrust: 'local', tierTrust: 'local', parentTrust: 'local', parentAgentId: 'dirty' })
  const strict = new DataBoundary(parseDataBoundaryConfig({ data_boundary: 'private' }, {}), root)
  expect(() => strict.assertTierAllowed({ requestedTrust: 'cloud', parentTrust: 'local' }))
    .toThrow(expect.objectContaining({ code: 'DATA_BOUNDARY_NO_LOCAL_ROUTE' }))
  const off = new DataBoundary(parseDataBoundaryConfig({}, {}), root)
  off.assertTierAllowed({ requestedTrust: 'cloud', tierTrust: 'local', parentTrust: 'local' })
})

function summary(overrides: Partial<OutboundRequestSummary> = {}): OutboundRequestSummary {
  const text = overrides.text ?? 'hello'
  return { provider: 'openai', model: 'gpt', trust: 'cloud', purpose: 'turn', sessionId: 's1', text, bytes: text.length, ...overrides }
}

test('outbound gate: first asks once per session+provider, every asks each time, denial is logged', async () => {
  const root = workspace()
  const log = new DisclosureLog({ path: join(root, 'log', 'disclosures.jsonl') })
  const asked: string[] = []
  let answer = true
  const approve = async (request: OutboundRequestSummary) => { asked.push(`${request.sessionId}/${request.provider}`); return answer }
  const first = boundary(root).outboundGate({ approve, log })
  expect(await first.check(summary())).toEqual({ allow: true, decision: 'approved' })
  expect(await first.check(summary())).toEqual({ allow: true, decision: 'sent' })
  expect(await first.check(summary({ provider: 'anthropic' }))).toEqual({ allow: true, decision: 'approved' })
  expect(await first.check(summary({ trust: 'local', provider: 'llama' }))).toEqual({ allow: true, decision: 'sent' })
  expect(asked).toEqual(['s1/openai', 's1/anthropic'])
  answer = false
  const every = boundary(root, { boundary: { private_paths: ['secret/**'], approve_outbound: 'every' } }).outboundGate({ approve, log })
  const denied = await every.check(summary({ sessionId: 's2' }))
  expect(denied.allow).toBe(false)
  // A denial does not flip the gate into a looser mode: it asks again.
  expect((await every.check(summary({ sessionId: 's2' }))).allow).toBe(false)
  expect(asked.length).toBe(4)
  const decisions = log.read({ limit: 50 }).map(record => record.decision)
  expect(decisions).toEqual(['denied', 'denied', 'sent', 'approved', 'sent', 'approved'])
})

test('outbound gate off sends without logging; approval errors are logged and propagate', async () => {
  const root = workspace()
  const log = new DisclosureLog({ path: join(root, 'disclosures.jsonl') })
  const off = new DataBoundary(parseDataBoundaryConfig({}, {}), root).outboundGate({ approve: async () => false, log })
  expect(await off.check(summary())).toEqual({ allow: true, decision: 'sent' })
  expect(log.read()).toEqual([])
  const failing = boundary(root).outboundGate({ approve: async () => { throw new Error('aborted') }, log })
  await expect(failing.check(summary())).rejects.toThrow('aborted')
  expect(log.read()[0]?.decision).toBe('denied')
})
