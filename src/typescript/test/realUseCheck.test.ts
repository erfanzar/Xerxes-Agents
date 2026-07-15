// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  createRealUseChecks,
  formatRealUseReport,
  runRealUseChecks,
  type RealUseCheck,
} from '../scripts/realUseCheck.js'

test('native real-use runner executes local Bun surfaces and safely skips gated external probes', async () => {
  const report = await runRealUseChecks({
    allowNetwork: false,
    ports: { findExecutable: () => null },
  })

  expect(report.ok, formatRealUseReport(report)).toBeTrue()
  expect(report.failed).toBe(0)
  expect(report.results.filter(result => result.status === 'passed').map(result => result.plan)).toEqual(expect.arrayContaining([
    '03',
    '01',
    '02',
    '04',
    '05',
    '06',
    '08',
    '09',
    '10',
    '11',
    '12',
    '13',
    '14',
    '15',
    '16',
    '17',
    '18',
    '19',
    '20',
    '21',
    '22',
    '23',
    '24',
    '25',
    '26',
    '27/TUI',
    '28',
    '29',
    '30',
    '31',
    '32',
  ]))
  expect(report.results.filter(result => result.status === 'skipped').map(result => result.feature)).toEqual(expect.arrayContaining([
    'OSV public advisory query through the injected fetch port',
    'Git shadow snapshot rollback through the native snapshot manager',
    'Credentialed or hardware media probe supplied by the embedding host',
    'Public package-registry update probe supplied by the embedding host',
    'Live terminal UI probe supplied by the embedding host',
    'Credentialed provider probe supplied by the embedding host',
    'Live browser or hardware probe supplied by the embedding host',
  ]))
  expect(formatRealUseReport(report)).toContain('summary: passed=')
  expect(formatRealUseReport(report)).toContain('skipped=')
})

test('runner preserves passed/skipped/failed outcomes and redacts failure details', async () => {
  const checks: readonly RealUseCheck[] = [
    { plan: 'P', feature: 'pass', run: () => 'native local check' },
    { plan: 'S', feature: 'skip', skip: () => 'hardware unavailable', run: () => 'unreachable' },
    {
      plan: 'F',
      feature: 'fail',
      run: () => {
        throw new Error('api_key=sk-12345678901234567890')
      },
    },
  ]
  const report = await runRealUseChecks({ checks })
  const rendered = formatRealUseReport(report)

  expect(report).toMatchObject({ passed: 1, skipped: 1, failed: 1, ok: false })
  expect(report.results.map(result => result.status)).toEqual(['passed', 'skipped', 'failed'])
  expect(rendered).toContain('## PASSED (1)')
  expect(rendered).toContain('## SKIPPED (1)')
  expect(rendered).toContain('## FAILED (1)')
  expect(rendered).not.toContain('12345678901234567890')
  expect(rendered).toContain('[redacted]')
})

test('external checks run only through an explicit opt-in injected port', async () => {
  const checks = createRealUseChecks()
  const osv = checks.find(check => check.feature === 'OSV public advisory query through the injected fetch port')
  const provider = checks.find(check => check.feature === 'Credentialed provider probe supplied by the embedding host')
  if (!osv || !provider) throw new Error('expected explicit external checks')

  let osvRequests = 0
  const osvReport = await runRealUseChecks({
    allowNetwork: true,
    checks: [osv],
    ports: {
      networkProbe: async () => true,
      fetchImplementation: async () => {
        osvRequests += 1
        return Response.json({ vulns: [] })
      },
    },
  })
  expect(osvReport).toMatchObject({ passed: 1, skipped: 0, failed: 0 })
  expect(osvRequests).toBe(1)

  const skippedProvider = await runRealUseChecks({
    checks: [provider],
  })
  expect(skippedProvider.results[0]).toMatchObject({ status: 'skipped', detail: expect.stringContaining('no providerProbe supplied') })

  const passedProvider = await runRealUseChecks({
    checks: [provider],
    ports: { providerProbe: () => 'embedding host performed the controlled provider probe' },
  })
  expect(passedProvider).toMatchObject({ passed: 1, skipped: 0, failed: 0 })
})

test('credentialed, registry, and terminal checks do not run without their injected host ports', async () => {
  const checks = createRealUseChecks()
  const features = [
    'Credentialed or hardware media probe supplied by the embedding host',
    'Public package-registry update probe supplied by the embedding host',
    'Live terminal UI probe supplied by the embedding host',
  ]
  const external = checks.filter(check => features.includes(check.feature))
  expect(external).toHaveLength(3)

  const skipped = await runRealUseChecks({ checks: external })
  expect(skipped).toMatchObject({ passed: 0, skipped: 3, failed: 0 })

  const passed = await runRealUseChecks({
    checks: external,
    ports: {
      mediaProbe: () => 'host media probe completed',
      packageRegistryProbe: () => 'host registry probe completed',
      tuiProbe: () => 'host TUI probe completed',
    },
  })
  expect(passed).toMatchObject({ passed: 3, skipped: 0, failed: 0 })
})
