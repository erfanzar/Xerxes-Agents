// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { DataBoundary, parseDataBoundaryConfig } from '../src/security/dataBoundary.js'
import { DisclosureLog } from '../src/security/disclosureLog.js'
import { captureDisclosure } from '../src/daemon/superagentCaptureAdmission.js'
import { validateCaptureImage } from '../src/hub/supervisor/abilities/captureImage.js'
import { captureEnvelope } from './support/captureImage.js'

const directories: string[] = []
afterEach(() => { for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true }) })

const source = { workspace: 'B', machine: 'B-machine', policyRevision: 'p1', viewId: 'a'.repeat(32) }
const destination = { machine: 'A-machine', profile: 'vision', model: 'fixture-model', revision: 'r1' }
function gate(mode: 'private' | 'cloud') {
  const config = parseDataBoundaryConfig({ data_boundary: mode }, {}, { root: '/fixture/b' })
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-capture-admit-')); directories.push(directory)
  return new DataBoundary(config, '/fixture/b').outboundGate({ approve: async () => false, log: new DisclosureLog({ path: join(directory, 'disclosures.jsonl') }) })
}
function request() { return { owner: 'owner', thread: 'thread', source, destination, image: validateCaptureImage(captureEnvelope()) } }

test('private B cannot disclose to a remote model merely because that model is loopback on A', async () => {
  const disclose = captureDisclosure({ owner: 'owner', thread: 'thread', source, destination, hubMachine: 'B-machine',
    enabled: true, vision: true, providerTrust: 'local', gate: gate('private'), current: () => true })
  await expect(disclose(request())).rejects.toThrow()
})

test('private B cannot release pixels to a remote Hub even when the model runs locally on B', async () => {
  const local = { ...destination, machine: 'B-machine' }
  const disclose = captureDisclosure({ owner: 'owner', thread: 'thread', source, destination: local, hubMachine: 'Hub-machine',
    enabled: true, vision: true, providerTrust: 'local', gate: gate('private'), current: () => true })
  await expect(disclose({ ...request(), destination: local })).rejects.toThrow()
})

test('missing vision opt-in, consent gate, source policy or current owner authority never grants disclosure', async () => {
  for (const vision of [undefined, false]) {
    const disclose = captureDisclosure({ owner: 'owner', thread: 'thread', source, destination, hubMachine: 'B-machine',
      enabled: true, vision, providerTrust: 'cloud', gate: gate('cloud'), current: () => true })
    await expect(disclose(request())).rejects.toThrow()
  }
  const denied = captureDisclosure({ owner: 'owner', thread: 'thread', source, destination, hubMachine: 'B-machine',
    enabled: true, vision: true, providerTrust: 'cloud', current: () => true })
  await expect(denied(request())).rejects.toThrow()
})

test('explicit same-machine private/local vision admission checks the exact image bytes with the existing gate', async () => {
  const local = { ...destination, machine: 'B-machine' }; const observed: string[] = []
  const policy = gate('private')
  const disclose = captureDisclosure({ owner: 'owner', thread: 'thread', source, destination: local, hubMachine: 'B-machine',
    enabled: true, vision: true, providerTrust: 'local', gate: { async check(summary, signal) { observed.push(summary.text); return policy.check(summary, signal) } }, current: () => true })
  const input = { ...request(), destination: local }
  await disclose(input)
  expect(observed).toHaveLength(1)
  expect(observed[0]).toContain(input.image.imageUrl)
})

test('source, owner, destination revision or post-await authority drift refuses an otherwise allowed route', async () => {
  let live = true
  const disclose = captureDisclosure({ owner: 'owner', thread: 'thread', source, destination, hubMachine: 'B-machine',
    enabled: true, vision: true, providerTrust: 'cloud', gate: gate('cloud'), current: () => live })
  for (const input of [{ ...request(), owner: 'other' }, { ...request(), source: { ...source, policyRevision: 'p2' } },
    { ...request(), destination: { ...destination, revision: 'r2' } }]) await expect(disclose(input)).rejects.toThrow()
  live = false; await expect(disclose(request())).rejects.toThrow()
  live = true
  const drift = captureDisclosure({ owner: 'owner', thread: 'thread', source, destination, hubMachine: 'B-machine',
    enabled: true, vision: true, providerTrust: 'cloud', gate: { async check() { live = false; return { allow: true, decision: 'sent' } } }, current: () => live })
  await expect(drift(request())).rejects.toThrow()
})

test('a vision capability never substitutes for explicit owner capture consent', async () => {
  const local = { ...destination, machine: 'B-machine' }
  const disclose = captureDisclosure({ owner: 'owner', thread: 'thread', source, destination: local, hubMachine: 'B-machine',
    enabled: false, vision: true, providerTrust: 'local', gate: gate('private'), current: () => true })
  await expect(disclose({ ...request(), destination: local })).rejects.toThrow()
})
