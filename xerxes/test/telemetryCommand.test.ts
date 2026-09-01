// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { runTelemetryCommand } from '../src/runtime/telemetryCommand.js'

test('telemetry command records, lists, benchmarks, and injects failures', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-telemetry-cmd-'))
  try {
    const recordResult = await runTelemetryCommand({ action: 'record', event: 'test', data: '{"ok":true}', directory })
    expect(recordResult.ok).toBeTrue()

    const listResult = await runTelemetryCommand({ action: 'list', directory })
    expect(listResult.ok).toBeTrue()
    expect(listResult.message).toContain('test')

    const benchmarkResult = await runTelemetryCommand({ action: 'benchmark', name: 'fs-bench', iterations: 3, target: 'fs', directory })
    expect(benchmarkResult.ok).toBeTrue()
    expect(benchmarkResult.message).toContain('p95:')

    const injectResult = await runTelemetryCommand({ action: 'inject', operation: 'write', mode: 'error', probability: 1, errorMessage: 'boom', directory })
    expect(injectResult.ok).toBeFalse()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
