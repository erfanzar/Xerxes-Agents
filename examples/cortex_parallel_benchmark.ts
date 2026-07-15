// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Native Cortex parallel inference benchmark with a safe deterministic default. */

import { createHash } from 'node:crypto'

import {
  Cortex,
  type CortexOutput,
  type LlmClient,
} from '../xerxes/src/index.js'
import { CortexAgent } from '../xerxes/src/cortex/agents/agent.js'
import { ProcessType } from '../xerxes/src/cortex/core/enums.js'
import {
  approximateTokens,
  divider,
  exampleLlm,
  optionValue,
  positiveIntegerOption,
  runMain,
  textOf,
  writeJsonWhenRequested,
} from './native_demo_support.js'

export interface BenchmarkOptions {
  readonly agents: number
  readonly contextBufferTokens: number
  readonly label: string
  readonly maxOutputTokens: number
  readonly minContextTokens: number
  readonly model: string
}

export interface BenchmarkWorkerResult {
  readonly contextTokens: number
  readonly elapsedMs: number
  readonly output: string
  readonly sentinel: string
  readonly valid: boolean
  readonly workerIndex: number
}

export interface BenchmarkReport {
  readonly aggregateTokensPerSecond: number
  readonly elapsedMs: number
  readonly outputTokens: number
  readonly requestedContextTokens: number
  readonly successfulWorkers: number
  readonly workers: readonly BenchmarkWorkerResult[]
}

export function buildSentinel(label: string, workerIndex: number): string {
  const digest = createHash('sha1').update(`${label}:${workerIndex}`).digest('hex').slice(0, 16)
  return `SENTINEL-${String(workerIndex).padStart(3, '0')}-${digest}`
}

export function buildLargeContext(options: BenchmarkOptions, workerIndex: number): { readonly text: string; readonly tokens: number; readonly sentinel: string } {
  const sentinel = buildSentinel(options.label, workerIndex)
  const target = options.minContextTokens + options.contextBufferTokens
  const pieces = [
    `Parallel inference benchmark for worker ${workerIndex}.`,
    'This synthetic context measures prompt ingestion and concurrent scheduling; it has no web or tool latency.',
  ]
  let block = 0
  while (approximateTokens(pieces.join('\n\n')) < target) {
    pieces.push(contextBlock(options.label, workerIndex, block))
    block += 1
  }
  pieces.push(`Return only this validation sentinel and nothing else:\n${sentinel}`)
  const text = pieces.join('\n\n')
  return { text, tokens: approximateTokens(text), sentinel }
}

/** Run concurrent Cortex agents. --live supplies the only path to a real endpoint. */
export async function runBenchmark(options: BenchmarkOptions, llm: LlmClient): Promise<BenchmarkReport> {
  const contexts = Array.from({ length: options.agents }, (_, workerIndex) => buildLargeContext(options, workerIndex))
  const workers = contexts.map((context, workerIndex) => new CortexAgent({
    id: `benchmark-${workerIndex}`,
    role: `Load Test Worker ${workerIndex}`,
    goal: 'Return the requested validation sentinel.',
    backstory: 'A synthetic benchmark worker that does not use tools or external state.',
    instructions: 'Read the task and return only the requested sentinel.',
    model: options.model,
    llm,
    maxTokens: options.maxOutputTokens,
    temperature: 0,
    memoryEnabled: false,
  }))
  const startedAt = performance.now()
  const output = await new Cortex({
    process: ProcessType.PARALLEL,
    maxParallel: options.agents,
    agents: workers,
    tasks: contexts.map((context, workerIndex) => ({
      id: `benchmark-${workerIndex}`,
      ...(workers[workerIndex] === undefined ? {} : { agentId: workers[workerIndex].id }),
      description: context.text,
      expectedOutput: context.sentinel,
    })),
  }).kickoff()
  const elapsedMs = performance.now() - startedAt
  return reportFromOutput(output, contexts, elapsedMs)
}

export function benchmarkOptions(args: readonly string[]): BenchmarkOptions {
  return {
    agents: positiveIntegerOption(args, '--agents', 8),
    contextBufferTokens: positiveIntegerOption(args, '--context-buffer-tokens', 64),
    label: optionValue(args, '--label') ?? 'native-cortex-benchmark',
    maxOutputTokens: positiveIntegerOption(args, '--max-output-tokens', 64),
    minContextTokens: positiveIntegerOption(args, '--min-context-tokens', 512),
    model: optionValue(args, '--model') ?? 'gpt-4o-mini',
  }
}

async function main(): Promise<void> {
  const args = Bun.argv.slice(2)
  const options = benchmarkOptions(args)
  const llm = exampleLlm(args, request => {
    const prompt = textOf(request.messages.at(-1)?.content)
    return prompt.match(/SENTINEL-[A-Za-z0-9-]+/)?.[0] ?? 'SENTINEL-MISSING'
  })
  divider('CORTEX PARALLEL BENCHMARK (native Bun)')
  console.log(`Workers: ${options.agents}\nTarget context: ${options.minContextTokens} estimated tokens\nMode: ${args.includes('--live') ? 'explicit live endpoint' : 'deterministic local LLM'}`)
  const report = await runBenchmark(options, llm)
  console.log(`\nCompleted ${report.successfulWorkers}/${options.agents} workers in ${report.elapsedMs.toFixed(1)}ms.`)
  console.log(`Estimated prompt throughput: ${report.aggregateTokensPerSecond.toFixed(1)} tokens/sec.`)
  for (const worker of report.workers) {
    console.log(`  worker ${worker.workerIndex}: ${worker.valid ? 'valid' : 'INVALID'} (${worker.contextTokens} estimated prompt tokens)`)
  }
  const written = await writeJsonWhenRequested(args, 'outputs/cortex_parallel_benchmark/report.json', report)
  if (written) console.log(`Wrote requested artifact: ${written}`)
}

function contextBlock(label: string, workerIndex: number, blockIndex: number): string {
  const trace = Array.from({ length: 16 }, (_, index) => `trace_${workerIndex}_${blockIndex}_${index}`).join(' ')
  return [
    `Benchmark ${label}; worker ${workerIndex}; block ${blockIndex}.`,
    'Synthetic telemetry discusses queue depth, KV cache pressure, prefill utilization, scheduler overlap, and decode variance.',
    `Trace terms: ${trace}`,
  ].join('\n')
}

function reportFromOutput(
  output: CortexOutput,
  contexts: readonly { readonly sentinel: string; readonly tokens: number }[],
  elapsedMs: number,
): BenchmarkReport {
  const workers = output.taskOutputs.map((task, workerIndex) => {
    const context = contexts[workerIndex]
    if (!context) throw new Error(`Missing benchmark context for worker ${workerIndex}`)
    return {
      workerIndex,
      contextTokens: context.tokens,
      sentinel: context.sentinel,
      output: task.output.trim(),
      valid: task.status === 'succeeded' && task.output.trim() === context.sentinel,
      elapsedMs: task.durationMs,
    }
  })
  const inputTokens = workers.reduce((total, worker) => total + worker.contextTokens, 0)
  return {
    workers,
    elapsedMs,
    requestedContextTokens: inputTokens,
    outputTokens: workers.reduce((total, worker) => total + approximateTokens(worker.output), 0),
    successfulWorkers: workers.filter(worker => worker.valid).length,
    aggregateTokensPerSecond: elapsedMs === 0 ? 0 : inputTokens / (elapsedMs / 1_000),
  }
}

if (import.meta.main) runMain(main)
