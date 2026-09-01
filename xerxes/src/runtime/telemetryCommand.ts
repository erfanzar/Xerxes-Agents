// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'
import { Benchmark } from './benchmark.js'
import { FailureInjector, type FailureRule } from './failureInjector.js'

export type TelemetryCommandAction = 'record' | 'list' | 'benchmark' | 'inject'

export interface TelemetryCommandOptions {
  readonly action: TelemetryCommandAction
  readonly event?: string | undefined
  readonly data?: string | undefined
  readonly name?: string | undefined
  readonly iterations?: number | undefined
  readonly target?: 'sandbox' | 'fs' | undefined
  readonly operation?: string | undefined
  readonly mode?: 'error' | 'latency' | 'hang' | undefined
  readonly probability?: number | undefined
  readonly latencyMs?: number | undefined
  readonly errorMessage?: string | undefined
  readonly directory?: string | undefined
}

export interface TelemetryCommandResult {
  readonly ok: boolean
  readonly message?: string
  readonly error?: string
}

interface TelemetryEvent {
  readonly timestamp: number
  readonly event: string
  readonly data: unknown
}

export async function runTelemetryCommand(options: TelemetryCommandOptions): Promise<TelemetryCommandResult> {
  const directory = options.directory ?? join(xerxesHome(), 'telemetry')
  await mkdir(directory, { recursive: true })
  const logPath = join(directory, 'events.jsonl')

  switch (options.action) {
    case 'record': {
      if (!options.event) return { ok: false, error: 'record requires --event' }
      const event: TelemetryEvent = {
        timestamp: Date.now(),
        event: options.event,
        data: options.data === undefined ? null : JSON.parse(options.data) as unknown,
      }
      await writeFile(logPath, `${JSON.stringify(event)}\n`, { flag: 'a' })
      return { ok: true, message: `recorded ${options.event}` }
    }
    case 'list': {
      let events: TelemetryEvent[] = []
      try {
        const contents = await readFile(logPath, 'utf8')
        events = contents.split('\n').filter(Boolean).map(line => JSON.parse(line) as TelemetryEvent)
      } catch {
        // no events yet
      }
      const lines = events.map(e => `${new Date(e.timestamp).toISOString()}\t${e.event}\t${JSON.stringify(e.data)}`)
      return { ok: true, message: lines.join('\n') || 'no telemetry events' }
    }
    case 'benchmark': {
      const name = options.name ?? 'default'
      const iterations = options.iterations ?? 10
      const target = options.target ?? 'fs'
      const benchmark = new Benchmark({ name, iterations })
      const summary = await benchmark.run(async () => runBenchmarkTarget(target, directory))
      const lines = [
        `name: ${summary.name}`,
        `iterations: ${summary.iterations}`,
        `successes: ${summary.successes}`,
        `failures: ${summary.failures}`,
        `min: ${summary.minMs.toFixed(3)}ms`,
        `mean: ${summary.meanMs.toFixed(3)}ms`,
        `p95: ${summary.p95Ms.toFixed(3)}ms`,
        `p99: ${summary.p99Ms.toFixed(3)}ms`,
      ]
      return { ok: true, message: lines.join('\n') }
    }
    case 'inject': {
      if (!options.operation || !options.mode) return { ok: false, error: 'inject requires --operation and --mode' }
      const rule: FailureRule = {
        mode: options.mode,
        probability: options.probability ?? 1.0,
        ...(options.latencyMs === undefined ? {} : { latencyMs: options.latencyMs }),
        ...(options.errorMessage === undefined ? {} : { errorMessage: options.errorMessage }),
        match: op => op === options.operation,
      }
      const injector = new FailureInjector({ rules: [rule] })
      try {
        const result = await injector.inject(options.operation, async () => {
          await Bun.write(join(directory, 'inject-test.txt'), 'ok')
          return 'ok'
        })
        return { ok: true, message: `inject completed: ${result}` }
      } catch (error) {
        return { ok: false, error: error instanceof Error ? error.message : String(error) }
      }
    }
  }
}

async function runBenchmarkTarget(target: 'sandbox' | 'fs', directory: string): Promise<void> {
  if (target === 'fs') {
    const path = join(directory, 'bench.txt')
    await Bun.write(path, 'benchmark')
    await Bun.file(path).text()
  } else {
    // sandbox target would run sandboxed code; for now, fall back to fs
    const path = join(directory, 'bench.txt')
    await Bun.write(path, 'benchmark')
    await Bun.file(path).text()
  }
}
