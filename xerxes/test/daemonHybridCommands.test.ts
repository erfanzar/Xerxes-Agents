// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtempSync, realpathSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, test } from 'bun:test'

import { PlanStore } from '../src/cortex/planStore.js'
import { calibrationCommand, plansCommand, routeCommand } from '../src/daemon/hybridCommands.js'
import { CalibrationStore } from '../src/runtime/calibration.js'

const roots: string[] = []
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function temp(): string {
  const root = realpathSync(mkdtempSync(join(tmpdir(), 'xerxes-hybrid-cmd-')))
  roots.push(root)
  return root
}

const ROUTES = [
  { id: 'local', profile: 'bonsai-local', model: 'bonsai-9b', trust: 'local', tier: 1 },
  { id: 'cloud', profile: 'opus', model: 'opus', trust: 'cloud', tier: 3, input_usd_per_mtok: 5, output_usd_per_mtok: 25 },
]

test('/route reports OFF by default and never claims to reroute', () => {
  const cwd = temp()
  const status = routeCommand('', { cwd, runtime: {}, env: {} })
  expect(status.ok).toBe(true)
  expect(status.report).toContain('Routing: OFF')
  const explain = routeCommand('explain reasoning', { cwd, runtime: { routing_routes: ROUTES }, env: {} })
  expect(explain.ok).toBe(true)
  expect(explain.report).toContain('Routing is OFF')
})

test('/route explain keeps private work local under a boundary and reports exclusions', () => {
  const cwd = temp()
  const runtime = { routing: 'rules', routing_routes: ROUTES, data_boundary: 'hybrid' }
  const result = routeCommand('explain reasoning --private', { cwd, runtime, env: {} })
  expect(result.ok).toBe(true)
  expect(result.report).toContain('route local')
  expect(result.report).toContain('excluded cloud')
  const privateMode = routeCommand('explain reasoning', { cwd, runtime: { ...runtime, data_boundary: 'private', routing_routes: [ROUTES[1]] }, env: {} })
  expect(privateMode.report).toContain('no allowed route')
})

test('/route rejects bad settings and bad arguments without throwing', () => {
  const cwd = temp()
  expect(routeCommand('', { cwd, runtime: { routing: 'always' }, env: {} }).ok).toBe(false)
  expect(routeCommand('explain --bogus', { cwd, runtime: {}, env: {} }).ok).toBe(false)
  expect(routeCommand('nope', { cwd, runtime: {}, env: {} }).ok).toBe(false)
})

test('/plans lists, shows and removes saved plans; OFF by default', async () => {
  const store = new PlanStore({ dir: temp() })
  const empty = await plansCommand('', { runtime: {}, env: {}, store })
  expect(empty.ok).toBe(true)
  expect(empty.report).toContain('Hybrid plans: OFF')
  expect(empty.report).toContain('No saved plans.')
  expect((await plansCommand('show missing', { runtime: {}, env: {}, store })).ok).toBe(false)
  expect((await plansCommand('remove missing', { runtime: {}, env: {}, store })).ok).toBe(false)
  expect((await plansCommand('frobnicate', { runtime: {}, env: {}, store })).ok).toBe(false)
  expect((await plansCommand('', { runtime: { hybrid_plans: 'maybe' }, env: {}, store })).ok).toBe(false)
})

test('/calibration needs a workload and reports per-strategy stats', async () => {
  const store = new CalibrationStore({ path: join(temp(), 'outcomes.jsonl') })
  for (let run = 0; run < 3; run += 1) {
    store.record({ workload: 'csv-reconcile', hardwareKey: 'hw-test', strategy: 'local-only', accepted: true, wallMs: 1000 + run })
  }
  const ports = { runtime: {}, env: {}, store, hardwareKey: async () => 'hw-test' }
  const usage = await calibrationCommand('', ports)
  expect(usage.ok).toBe(true)
  expect(usage.report).toContain('Xerxes does not record them yet')
  const report = await calibrationCommand('csv-reconcile', ports)
  expect(report.ok).toBe(true)
  expect(report.report).toContain('local-only')
  expect((await calibrationCommand('bad workload!', ports)).ok).toBe(false)
})
