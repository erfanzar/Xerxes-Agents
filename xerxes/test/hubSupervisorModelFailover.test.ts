// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import type { AuthenticatedContext } from '../src/hub/identityPort.js'
import type { PoolCallResult, PoolMachine } from '../src/hub/runtimePool.js'
import { SupervisorModelControl, SupervisorModelStore } from '../src/hub/supervisor/modelControl.js'
import type { SupervisorModelRequest } from '../src/hub/supervisor/runner.js'

const cleanups: Array<() => void | Promise<void>> = []
afterEach(async () => { for (const cleanup of cleanups.splice(0).reverse()) await cleanup() })

const CHOICE = { machineId: 'mac', profile: 'work', model: 'same-model', effort: 'high' } as const
const REQUEST: SupervisorModelRequest = {
  system: 'You are XERXES.', messages: [{ role: 'user', content: 'What is running?' }], tools: [], maxTokens: 512,
}
const PARAMS = {
  profile: CHOICE.profile, model: CHOICE.model, effort: CHOICE.effort,
  system: REQUEST.system, messages: REQUEST.messages, tools: REQUEST.tools, max_tokens: REQUEST.maxTokens,
}

type FailoverChange = { readonly from: string; readonly to: string | undefined }
interface ModelCall {
  readonly machineId: string
  readonly method: string
  readonly params: Readonly<Record<string, unknown>>
  readonly signal?: AbortSignal
}

function machine(id: string, label: string, health: PoolMachine['health'] = 'online', modelsSupported = true): PoolMachine {
  return { id, label, health, modelsSupported, readOnly: false, workspaceIds: [] }
}
function accepted(text = 'answer'): PoolCallResult {
  return { state: 'accepted', result: { text, tool_calls: [], usage: { input_tokens: 5, output_tokens: 2 } } }
}
function rejected(code: string): PoolCallResult {
  return { state: 'rejected', error: { code, message: `machine returned ${code}` } }
}
function recipient(hubOwner: boolean): AuthenticatedContext {
  const id = hubOwner ? 'owner' : 'reader'
  return {
    principal: { kind: 'user', id, displayName: id, deviceId: `dev-${id}`, hubOwner },
    grants: [], browserSessionId: `session-${id}`,
    actor: { principal_kind: 'user', principal_id: id, display_name: id },
  }
}

async function fixture(
  initial: readonly PoolMachine[],
  respond: (call: ModelCall) => PoolCallResult | Promise<PoolCallResult> = () => accepted(),
  onFailover?: (change: FailoverChange) => void,
) {
  const dir = await mkdtemp(join(tmpdir(), 'xh-model-failover-'))
  cleanups.push(() => rm(dir, { recursive: true, force: true }))
  const databasePath = join(dir, 'hub.sqlite')
  const store = new SupervisorModelStore(databasePath)
  store.set(CHOICE, 'owner', 123)
  store.close()
  let rows = initial
  const calls: ModelCall[] = []
  const broadcasts: Readonly<Record<string, unknown>>[][] = []
  const changes: FailoverChange[] = []
  const errors: { error: unknown; context: string }[] = []
  const control = new SupervisorModelControl({
    databasePath,
    signer: { hubId: 'hub-x', sign: claims => ({ claims }) as never },
    audit: async () => {},
    machines: {
      machines: () => rows,
      async callModel(machineId, method, params, options) {
        const call = { machineId, method, params, ...(options?.signal === undefined ? {} : { signal: options.signal }) }
        calls.push(call)
        return respond(call)
      },
    },
    broadcast: build => { broadcasts.push([build(recipient(true)), build(recipient(false))]) },
    onFailover: change => { changes.push(change); onFailover?.(change) },
    onError: (error, context) => { errors.push({ error, context }) },
  })
  cleanups.push(() => control.close())
  return { control, calls, broadcasts, changes, errors, setMachines: (next: readonly PoolMachine[]) => { rows = next } }
}

describe('supervisor model machine failover', () => {
  test('an offline configured machine uses the same profile/model/effort elsewhere and announces the change once', async () => {
    const t = await fixture([machine('mac', 'Mac', 'offline'), machine('server', 'Server')])
    expect(t.control.describe()).toMatchObject({
      status: 'machine_offline', message: 'Mac is offline; XERXES will try the same model on another online machine.',
    })
    expect(t.control.view(true)).not.toHaveProperty('fallback')
    expect(await t.control.port.complete(REQUEST)).toEqual({ text: 'answer', toolCalls: [], usage: { inputTokens: 5, outputTokens: 2 } })
    expect(t.calls).toHaveLength(1)
    const { hub_auth: auth, ...params } = t.calls[0]!.params
    expect(params).toEqual(PARAMS)
    expect(t.calls[0]).toMatchObject({ machineId: 'server', method: 'hub.model.complete' })
    expect(auth).toMatchObject({ claims: { hub_id: 'hub-x', workspace_id: 'server', method: 'hub.model.complete', scopes: ['supervisor.model'] } })
    expect(t.control.stored()).toMatchObject(CHOICE)
    expect(t.control.describe()).toMatchObject({
      status: 'configured', config: { machineId: 'mac', machineLabel: 'Mac', profile: 'work', model: 'same-model', effort: 'high' },
      message: 'Mac is offline; XERXES is answering on Server with the same model.',
    })
    expect(t.control.view(true)).toMatchObject({ fallback: { machine_id: 'server', machine_label: 'Server' } })
    expect(t.broadcasts).toHaveLength(1)
    expect(t.broadcasts[0]).toEqual([t.control.view(true), t.control.view(false)])
    expect(t.changes).toEqual([{ from: 'Mac', to: 'Server' }])
    await t.control.port.complete(REQUEST)
    expect(t.broadcasts).toHaveLength(1)
    expect(t.changes).toHaveLength(1)
  })

  test.each(['missing', 'unsupported'] as const)('a %s configured machine is skipped', async kind => {
    const configured = kind === 'missing' ? [] : [machine('mac', 'Mac', 'online', false)]
    const t = await fixture([...configured, machine('server', 'Server')])
    await t.control.port.complete(REQUEST)
    expect(t.calls.map(call => call.machineId)).toEqual(['server'])
    expect(t.changes).toEqual([{ from: kind === 'missing' ? 'mac' : 'Mac', to: 'Server' }])
    if (kind === 'missing') {
      expect(t.control.describe()).toMatchObject({ status: 'configured', message: 'mac is offline; XERXES is answering on Server with the same model.' })
    }
  })

  test('an unavailable configured-machine call tries the next online machine', async () => {
    const t = await fixture([machine('mac', 'Mac'), machine('server', 'Server')], call => call.machineId === 'mac' ? rejected('unavailable') : accepted())
    await t.control.port.complete(REQUEST)
    expect(t.calls.map(call => call.machineId)).toEqual(['mac', 'server'])
    for (const call of t.calls) {
      const { hub_auth: _auth, ...params } = call.params
      expect(params).toEqual(PARAMS)
    }
    expect(t.changes).toEqual([{ from: 'Mac', to: 'Server' }])
  })

  test('candidates are tried by label, missing profiles are skipped, and the last successful fallback goes first next time', async () => {
    const t = await fixture([
      machine('mac', 'Mac', 'offline'), machine('beta', 'Beta'), machine('alpha', 'Alpha'), machine('gamma', 'Gamma'),
    ], call => call.machineId === 'alpha' ? rejected('unknown_profile') : accepted(call.machineId))
    expect((await t.control.port.complete(REQUEST)).text).toBe('beta')
    expect(t.calls.map(call => call.machineId)).toEqual(['alpha', 'beta'])
    expect((await t.control.port.complete(REQUEST)).text).toBe('beta')
    expect(t.calls.map(call => call.machineId)).toEqual(['alpha', 'beta', 'beta'])
    expect(t.changes).toEqual([{ from: 'Mac', to: 'Beta' }])
  })

  test.each(['not_found', 'invalid_request', 'unsupported', 'unavailable'])('a fallback rejection with %s moves to the next candidate', async code => {
    const t = await fixture([machine('mac', 'Mac', 'offline'), machine('alpha', 'Alpha'), machine('beta', 'Beta')], call => call.machineId === 'alpha' ? rejected(code) : accepted())
    await t.control.port.complete(REQUEST)
    expect(t.calls.map(call => call.machineId)).toEqual(['alpha', 'beta'])
  })

  test.each(['provider_failed', 'timeout', 'unknown_profile', 'not_found', 'invalid_request', 'unsupported'])('an online configured machine returning %s does not trigger failover', async code => {
    const t = await fixture([machine('mac', 'Mac'), machine('server', 'Server')], () => rejected(code))
    await expect(t.control.port.complete(REQUEST)).rejects.toThrow(`machine returned ${code}`)
    expect(t.calls.map(call => call.machineId)).toEqual(['mac'])
    expect(t.changes).toEqual([])
    expect(t.broadcasts).toEqual([])
  })

  test.each(['provider_failed', 'timeout'])('a fallback returning %s is a real error, not a reason to keep trying', async code => {
    const t = await fixture([machine('mac', 'Mac', 'offline'), machine('alpha', 'Alpha'), machine('beta', 'Beta')], () => rejected(code))
    await expect(t.control.port.complete(REQUEST)).rejects.toThrow(`machine returned ${code}`)
    expect(t.calls.map(call => call.machineId)).toEqual(['alpha'])
    expect(t.changes).toEqual([])
  })

  test('the next call returns to the configured machine when it comes online and clears failover', async () => {
    const t = await fixture([machine('mac', 'Mac', 'offline'), machine('server', 'Server')])
    await t.control.port.complete(REQUEST)
    t.setMachines([machine('mac', 'Mac'), machine('server', 'Server')])
    await t.control.port.complete(REQUEST)
    expect(t.calls.map(call => call.machineId)).toEqual(['server', 'mac'])
    expect(t.changes).toEqual([{ from: 'Mac', to: 'Server' }, { from: 'Mac', to: undefined }])
    expect(t.broadcasts).toHaveLength(2)
    expect(t.control.describe()).toMatchObject({ status: 'configured' })
    expect(t.control.describe()).not.toHaveProperty('message')
    expect(t.control.view(true)).not.toHaveProperty('fallback')
    await t.control.port.complete(REQUEST)
    expect(t.changes).toHaveLength(2)
  })

  test('no eligible candidate reports the unavailable profile and model without calling offline or unsupported machines', async () => {
    const t = await fixture([machine('mac', 'Mac', 'offline'), machine('stale', 'Stale', 'stale'), machine('old', 'Old', 'online', false)])
    await expect(t.control.port.complete(REQUEST)).rejects.toThrow('Mac is offline and no other online machine offers work/same-model')
    expect(t.calls).toEqual([])
    expect(t.control.view(true)).not.toHaveProperty('fallback')
  })

  test('exhausting candidates reports the configured profile and model', async () => {
    const t = await fixture([machine('mac', 'Mac', 'offline'), machine('server', 'Server')], () => rejected('unknown_profile'))
    await expect(t.control.port.complete(REQUEST)).rejects.toThrow('Mac is offline and no other online machine offers work/same-model')
    expect(t.calls.map(call => call.machineId)).toEqual(['server'])
  })

  test('a fallback that is no longer online is not advertised as configured', async () => {
    const t = await fixture([machine('mac', 'Mac', 'offline'), machine('server', 'Server')])
    await t.control.port.complete(REQUEST)
    t.setMachines([machine('mac', 'Mac', 'offline'), machine('server', 'Server', 'offline')])
    expect(t.control.describe()).toMatchObject({ status: 'machine_offline' })
    expect(t.control.view(true)).not.toHaveProperty('fallback')
  })

  test('an already aborted request does not try any machines', async () => {
    const t = await fixture([machine('mac', 'Mac'), machine('server', 'Server')])
    const reason = new Error('cancelled by person')
    await expect(t.control.port.complete({ ...REQUEST, signal: AbortSignal.abort(reason) })).rejects.toThrow(reason.message)
    expect(t.calls).toEqual([])
  })

  test.each(['mac', 'alpha'])('aborting during a call to %s stops before the next candidate', async first => {
    const abort = new AbortController()
    const t = await fixture([machine('mac', 'Mac', first === 'mac' ? 'online' : 'offline'), machine('alpha', 'Alpha'), machine('beta', 'Beta')], () => {
      abort.abort(new Error('stop trying'))
      return rejected(first === 'mac' ? 'unavailable' : 'unknown_profile')
    })
    await expect(t.control.port.complete({ ...REQUEST, signal: abort.signal })).rejects.toThrow('stop trying')
    expect(t.calls.map(call => call.machineId)).toEqual([first])
    expect(t.calls[0]!.signal).toBe(abort.signal)
    expect(t.changes).toEqual([])
  })

  test('onFailover errors are reported without losing the completion', async () => {
    const error = new Error('notice failed')
    const t = await fixture([machine('mac', 'Mac', 'offline'), machine('server', 'Server')], () => accepted(), () => { throw error })
    expect((await t.control.port.complete(REQUEST)).text).toBe('answer')
    expect(t.errors).toHaveLength(1)
    expect(t.errors[0]!.error).toBe(error)
    expect(t.errors[0]!.context).toContain('failover')
    expect(t.broadcasts).toHaveLength(1)
  })
})
