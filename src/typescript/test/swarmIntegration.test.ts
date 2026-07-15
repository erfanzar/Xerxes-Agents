// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { formatSwarmReport, runSwarmIntegration } from '../scripts/swarmIntegration.js'

test('native swarm integration exercise verifies orchestration, runtime, Cortex, and security boundaries', async () => {
  const report = await runSwarmIntegration()

  expect(report.ok, formatSwarmReport(report)).toBeTrue()
  expect(report.failed).toBe(0)
  expect(report.entries.filter(entry => !entry.error).map(entry => entry.category)).toEqual(expect.arrayContaining([
    'ORCHESTRATOR',
    'EXECUTOR',
    'XERXES',
    'CORTEX',
    'SUBAGENT',
    'SECURITY',
    'DEFINITIONS',
  ]))
  expect(formatSwarmReport(report)).toContain('Passed:')
  expect(formatSwarmReport(report)).toContain('Failed: 0')
})
