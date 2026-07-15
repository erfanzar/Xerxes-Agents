// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Scenario 2: concurrent TypeScript code analysis through native tool ports. */

import { ToolRegistry, type JsonObject, type ToolDefinition } from '../src/typescript/src/index.js'
import { divider, runMain } from './native_demo_support.js'

export interface CodeAnalysis {
  readonly complexity: number
  readonly issues: readonly string[]
  readonly statistics: Readonly<Record<string, number>>
  readonly suggestions: readonly string[]
}

export function analyzeTypeScriptCode(code: string): CodeAnalysis {
  const statistics = {
    classes: count(code, /\bclass\s+[A-Za-z_$]/g),
    conditions: count(code, /\bif\s*\(/g) + count(code, /\bswitch\s*\(/g),
    functions: count(code, /\b(?:async\s+)?function\s+[A-Za-z_$]/g) + count(code, /=>/g),
    imports: count(code, /\bimport\s/g),
    loops: count(code, /\b(?:for|while)\s*\(/g),
  }
  const issues: string[] = []
  const suggestions: string[] = []
  if (/\bany\b/.test(code)) issues.push('Avoid broad any; model the boundary with unknown or a concrete type.')
  if (/console\.log\(/.test(code)) suggestions.push('Consider structured logging for production paths.')
  if (/catch\s*\{/.test(code)) suggestions.push('Preserve a caught error when recovery needs context.')
  if (!/\b(?:describe|test|it)\s*\(/.test(code)) suggestions.push('Add focused tests for observable behavior.')
  return {
    statistics,
    complexity: statistics.loops + statistics.conditions,
    issues,
    suggestions,
  }
}

export function findSecurityIssues(code: string): string[] {
  const patterns: ReadonlyArray<readonly [RegExp, string]> = [
    [/\beval\s*\(/, 'eval() executes arbitrary code.'],
    [/\b(?:exec|spawn)\s*\([^)]*shell\s*:\s*true/, 'shell: true can introduce command injection.'],
    [/\b(?:apiKey|password|secret)\s*[:=]\s*['"][^'"]+['"]/i, 'Hard-coded credential-like value detected.'],
    [/\bfetch\s*\(\s*['"]http:/, 'Plain HTTP request may expose data in transit.'],
    [/\bnew Function\s*\(/, 'Dynamic Function construction executes arbitrary code.'],
  ]
  return patterns.filter(([pattern]) => pattern.test(code)).map(([, message]) => message)
}

export function suggestRefactoring(code: string): string[] {
  const suggestions: string[] = []
  if (code.split('\n').length > 40) suggestions.push('Split the large unit into small named helpers with testable inputs and outputs.')
  if (count(code, /\bif\s*\(/g) > 3) suggestions.push('Use early returns or a strategy map to reduce conditional nesting.')
  if (/\b(?:let|var)\b/.test(code)) suggestions.push('Prefer const for values that are not reassigned.')
  if (!suggestions.length) suggestions.push('The example is already small; keep the API boundary explicit.')
  return suggestions
}

export function generateTests(code: string): string {
  const names = [...code.matchAll(/\b(?:async\s+)?function\s+([A-Za-z_$][\w$]*)/g)].map(match => match[1]).filter(Boolean)
  const targets = names.length ? names : ['targetFunction']
  return targets.map(name => [
    `test('${name} returns the expected value', () => {`,
    `  expect(${name}(/* fixture */)).toEqual(/* expected */)`,
    '})',
  ].join('\n')).join('\n\n')
}

export function checkBestPractices(code: string): { readonly failed: readonly string[]; readonly passed: readonly string[] } {
  const passed: string[] = []
  const failed: string[] = []
  if (/\bimport\s/.test(code)) passed.push('Uses explicit module imports.')
  else failed.push('Use explicit module imports.')
  if (!/\bany\b/.test(code)) passed.push('Avoids broad any annotations.')
  else failed.push('Replace any with a concrete type or unknown at untrusted boundaries.')
  if (/\btry\s*\{/.test(code)) passed.push('Contains explicit error handling.')
  else failed.push('Add contextual error handling where operations can fail.')
  return { passed, failed }
}

export function createCodeAnalysisRegistry(): ToolRegistry {
  const registry = new ToolRegistry()
  for (const [definition, handler] of [
    [tool('analyze_typescript_code', 'Analyze TypeScript source for structural quality.'), (inputs: JsonObject) => analyzeTypeScriptCode(String(inputs.code ?? ''))],
    [tool('find_security_issues', 'Find obvious security hazards in TypeScript source.'), (inputs: JsonObject) => findSecurityIssues(String(inputs.code ?? ''))],
    [tool('suggest_refactoring', 'Suggest behavior-preserving refactoring steps.'), (inputs: JsonObject) => suggestRefactoring(String(inputs.code ?? ''))],
    [tool('generate_tests', 'Generate a small Bun test skeleton.'), (inputs: JsonObject) => generateTests(String(inputs.code ?? ''))],
    [tool('check_best_practices', 'Check a few portable TypeScript practices.'), (inputs: JsonObject) => checkBestPractices(String(inputs.code ?? ''))],
  ] as const) registry.register(definition, handler)
  return registry
}

export const SAMPLE_CODE = `
import { spawn } from 'node:child_process'

class DataProcessor {
  apiKey = 'example-not-a-real-key'
  process(data: number[]): number[] {
    return data.map(item => item > 100 ? item * 2 : item)
  }
  execute(command: string): void {
    spawn(command, { shell: true })
  }
}

function average(values: number[]): number {
  if (values.length === 0) return 0
  return values.reduce((total, value) => total + value, 0) / values.length
}
`.trim()

async function main(): Promise<void> {
  divider('SCENARIO 2: INTELLIGENT CODE ANALYSIS AGENT')
  const registry = createCodeAnalysisRegistry()
  const names = ['analyze_typescript_code', 'find_security_issues', 'suggest_refactoring', 'generate_tests', 'check_best_practices']
  const results = await Promise.all(names.map(name => registry.execute({
    id: `analysis-${name}`,
    type: 'function',
    function: { name, arguments: { code: SAMPLE_CODE } },
  }, { metadata: {} })))
  for (const [index, result] of results.entries()) {
    console.log(`\n${names[index]}\n${'-'.repeat(48)}\n${result}`)
  }
  console.log('\nAll analyses ran concurrently through the native ToolRegistry; no model or provider is needed.')
}

function tool(name: string, description: string): ToolDefinition {
  return {
    type: 'function',
    function: {
      name,
      description,
      parameters: {
        type: 'object',
        properties: { code: { type: 'string' } },
        required: ['code'],
      },
    },
  }
}

function count(value: string, expression: RegExp): number {
  return value.match(expression)?.length ?? 0
}

if (import.meta.main) runMain(main)
