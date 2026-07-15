// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { resolve } from 'node:path'

const PROJECT_ROOT = resolve(import.meta.dir, '../../..')
const TYPESCRIPT_COMPILER = resolve(PROJECT_ROOT, 'src/typescript/node_modules/typescript/bin/tsc')

interface ExampleCommand {
  readonly args: readonly string[]
  readonly expected: string
  readonly path: string
}

const EXAMPLES: readonly ExampleCommand[] = [
  { path: 'examples/cortex_deepsearch_agent.ts', args: ['--topic', 'native test', '--researchers', '2'], expected: 'CORTEX DEEP-SEARCH' },
  { path: 'examples/cortex_parallel_benchmark.ts', args: ['--agents', '2', '--min-context-tokens', '16'], expected: 'Completed 2/2 workers' },
  { path: 'examples/deepsearch_agent_demo.ts', args: ['--topic', 'native test', '--queries', '2'], expected: 'Collected 3 illustrative results' },
  { path: 'examples/interactive_agent.ts', args: [], expected: 'INTERACTIVE BUN AGENT' },
  { path: 'examples/openclaw_capabilities_demo.ts', args: [], expected: 'OPENCLAW-CLASS CAPABILITIES' },
  { path: 'examples/scenario_1_conversational_assistant.ts', args: [], expected: 'CONVERSATIONAL ASSISTANT WITH MEMORY' },
  { path: 'examples/scenario_2_code_analyzer.ts', args: [], expected: 'INTELLIGENT CODE ANALYSIS AGENT' },
  { path: 'examples/scenario_3_multi_agent_collaboration.ts', args: [], expected: 'MULTI-AGENT COLLABORATION' },
  { path: 'examples/scenario_4_streaming_research_assistant.ts', args: [], expected: 'STREAMING RESEARCH ASSISTANT' },
  { path: 'examples/textual_tui.ts', args: [], expected: 'XERXES NATIVE TERMINAL UI' },
]

test('root Bun examples typecheck and run their offline defaults', async () => {
  const typecheck = await execute([process.execPath, TYPESCRIPT_COMPILER, '--noEmit', '-p', 'examples/tsconfig.json'])
  expect(typecheck.exitCode).toBe(0)
  expect(typecheck.stderr).toBe('')

  for (const example of EXAMPLES) {
    const result = await execute([process.execPath, example.path, ...example.args])
    expect(result.exitCode, `${example.path}\n${result.stderr}`).toBe(0)
    expect(result.stderr, example.path).toBe('')
    expect(result.stdout, example.path).toContain(example.expected)
  }
}, 60_000)

async function execute(command: readonly string[]): Promise<{ readonly exitCode: number; readonly stderr: string; readonly stdout: string }> {
  const process = Bun.spawn([...command], { cwd: PROJECT_ROOT, stderr: 'pipe', stdout: 'pipe' })
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(process.stdout).text(),
    new Response(process.stderr).text(),
    process.exited,
  ])
  return { stdout, stderr, exitCode }
}
