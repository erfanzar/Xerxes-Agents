// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** A Bun-native interactive diagnostics agent with explicit local commands. */

import { readFile } from 'node:fs/promises'
import { createInterface } from 'node:readline/promises'

import { ShortTermMemory, Xerxes } from '../src/typescript/src/index.js'
import { divider, exampleLlm, hasFlag, runMain } from './native_demo_support.js'

export class InteractiveWorkspace {
  readonly memory = new ShortTermMemory({ capacity: 50 })
  private readonly config = new Map<string, string>([
    ['model', 'gpt-4o-mini'],
    ['permission_mode', 'auto'],
    ['memory_capacity', '50'],
  ])
  private logCount = 0

  listImprovements(): string {
    return [
      'Native Bun improvements:',
      '- Explicit LLM, filesystem, browser, and sandbox ports.',
      '- Dependency-aware Cortex workflows and typed tool schemas.',
      '- No hidden Python subprocess or environment credential discovery.',
    ].join('\n')
  }

  testMemorySystem(action = 'demo', content = '', tags = ''): string {
    if (action === 'clear') {
      this.memory.clear()
      return 'Memory cleared.'
    }
    if (action === 'add') {
      this.memory.save(content || 'Example memory', { tags }, { agentId: 'interactive_agent', memoryType: 'interactive' })
      return `Saved memory: ${content || 'Example memory'}`
    }
    if (action === 'search') {
      return this.memory.search(content, 10).map(item => item.content).join('\n') || 'No matching memories.'
    }
    this.memory.save('Interactive memory demonstration', { tags: 'demo' }, { agentId: 'interactive_agent', memoryType: 'interactive' })
    return `Memory demo: ${JSON.stringify(this.memory.getStatistics())}`
  }

  testConfiguration(action = 'show', key = '', value = ''): string {
    if (action === 'set') {
      if (!key || !value) return 'Usage: config set <key> <value>'
      this.config.set(key, value)
      return `Configured ${key} = ${value}`
    }
    return [...this.config.entries()].map(([name, setting]) => `${name} = ${setting}`).join('\n')
  }

  testErrorHandling(scenario = 'timeout'): string {
    const demonstrations: Readonly<Record<string, string>> = {
      timeout: 'Timeout handling: a caller-owned AbortSignal terminates the request and reports cancellation.',
      validation: 'Validation handling: typed tool schemas reject malformed input before a handler runs.',
      provider: 'Provider handling: terminal stream failures are retained as explicit failed-turn state.',
    }
    return demonstrations[scenario] ?? `Unknown error scenario: ${scenario}`
  }

  testLoggingMetrics(message = 'Test log'): string {
    this.logCount += 1
    return `Recorded structured demo event #${this.logCount}: ${message}`
  }

  async analyzeCodeFile(path: string): Promise<string> {
    const source = await readFile(path, 'utf8')
    const functions = source.match(/\b(?:async\s+)?function\s+[A-Za-z_$]/g)?.length ?? 0
    const classes = source.match(/\bclass\s+[A-Za-z_$]/g)?.length ?? 0
    return `Code summary for ${path}: ${source.split('\n').length} lines, ${functions} functions, ${classes} classes.`
  }

  getSystemInfo(): string {
    return [
      `Runtime: Bun ${Bun.version}`,
      `Platform: ${process.platform}/${process.arch}`,
      `Memory entries: ${this.memory.size}`,
      `Working directory: ${process.cwd()}`,
    ].join('\n')
  }
}

export function createInteractiveRuntime(args: readonly string[]): Xerxes {
  return new Xerxes({
    model: 'gpt-4o-mini',
    coreTools: false,
    llm: exampleLlm(args, request => {
      const last = request.messages.at(-1)?.content
      return `Native interactive reply: ${typeof last === 'string' ? last.slice(0, 120) : 'received a multimodal prompt'}`
    }),
    systemPrompt: 'You are a concise Bun-native diagnostics assistant.',
  })
}

export async function executeInteractiveCommand(
  workspace: InteractiveWorkspace,
  runtime: Xerxes,
  line: string,
): Promise<string> {
  const [command = '', ...parts] = line.trim().split(/\s+/)
  const argument = parts.join(' ')
  switch (command.toLowerCase()) {
    case 'help':
      return 'Commands: help, improvements, memory [add|search|clear] ..., config [show|set] ..., error <kind>, log <text>, analyze <path>, system, ask <prompt>, quit'
    case 'improvements': return workspace.listImprovements()
    case 'memory': return workspace.testMemorySystem(parts[0], parts.slice(1).join(' '))
    case 'config': return workspace.testConfiguration(parts[0], parts[1], parts.slice(2).join(' '))
    case 'error': return workspace.testErrorHandling(argument)
    case 'log': return workspace.testLoggingMetrics(argument)
    case 'analyze': return argument ? workspace.analyzeCodeFile(argument) : 'Usage: analyze <path>'
    case 'system': return workspace.getSystemInfo()
    case 'ask': return (await runtime.run(argument || 'Give a native runtime status.', { freshSession: true })).output
    default: return command ? `Unknown command: ${command}. Type help.` : ''
  }
}

async function main(): Promise<void> {
  const args = Bun.argv.slice(2)
  const workspace = new InteractiveWorkspace()
  const runtime = createInteractiveRuntime(args)
  divider('INTERACTIVE BUN AGENT')
  if (!hasFlag(args, '--interactive')) {
    for (const command of ['improvements', 'memory demo', 'config show', 'error timeout', 'log startup', 'system', 'ask summarize native status']) {
      console.log(`> ${command}\n${await executeInteractiveCommand(workspace, runtime, command)}\n`)
    }
    console.log('Run again with --interactive for a readline command loop.')
    return
  }
  const reader = createInterface({ input: process.stdin, output: process.stdout })
  try {
    while (true) {
      const line = await reader.question('xerxes> ')
      if (['exit', 'quit'].includes(line.trim().toLowerCase())) break
      const output = await executeInteractiveCommand(workspace, runtime, line)
      if (output) console.log(output)
    }
  } finally {
    reader.close()
  }
}

if (import.meta.main) runMain(main)
