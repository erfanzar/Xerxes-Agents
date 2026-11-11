// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ValidationError } from '../src/core/errors.js'
import { loadBuiltinAgentDefinitions } from '../src/agents/definitions.js'
import {
  LEAN_CORE_TOOL_NAMES,
  automaticPromptProfile,
  buildLeanSystemPrompt,
  leanProjectInstructions,
  leanPromptSize,
  parsePromptProfileConfig,
  selectLeanTools,
} from '../src/runtime/leanPrompt.js'
import { CSV_PROCESSOR_DEFINITION } from '../src/tools/dataTools.js'
import {
  FILE_EDIT_TOOL_DEFINITION,
  GLOB_TOOL_DEFINITION,
  GREP_TOOL_DEFINITION,
  LIST_DIR_DEFINITION,
  READ_FILE_DEFINITION,
  WRITE_FILE_DEFINITION,
} from '../src/tools/fileTools.js'
import { CHECK_COMMAND_DEFINITION, EXEC_COMMAND_DEFINITION } from '../src/tools/processTools.js'
import { CLAUDE_WORKFLOW_TOOL_DEFINITIONS } from '../src/tools/claudeTools/workflow.js'
import { XLSX_READER_DEFINITION, XLSX_WRITER_DEFINITION } from '../src/tools/xlsxTools.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

const askUser = CLAUDE_WORKFLOW_TOOL_DEFINITIONS.find(tool => tool.function.name === 'AskUserQuestionTool')
if (askUser === undefined) throw new Error('AskUserQuestionTool definition is missing')

const REAL_DEFINITIONS: readonly ToolDefinition[] = [
  READ_FILE_DEFINITION,
  WRITE_FILE_DEFINITION,
  LIST_DIR_DEFINITION,
  GLOB_TOOL_DEFINITION,
  GREP_TOOL_DEFINITION,
  FILE_EDIT_TOOL_DEFINITION,
  EXEC_COMMAND_DEFINITION,
  CHECK_COMMAND_DEFINITION,
  CSV_PROCESSOR_DEFINITION,
  XLSX_WRITER_DEFINITION,
  XLSX_READER_DEFINITION,
  askUser,
  // Present in the registry but not part of the lean core set.
  CLAUDE_WORKFLOW_TOOL_DEFINITIONS[0]!,
]

function advisor(): ToolDefinition {
  return {
    type: 'function',
    function: {
      name: 'AskAdvisor',
      description: 'Ask a stronger model for advice. Use only when stuck.',
      parameters: { type: 'object', properties: { question: { type: 'string', description: 'The problem.' } }, required: ['question'] },
    },
  }
}

test('parsePromptProfileConfig: profile wins, then env, then runtime, default full', () => {
  expect(parsePromptProfileConfig({}, {})).toBe('full')
  expect(parsePromptProfileConfig({ prompt_profile: 'lean' }, {})).toBe('lean')
  expect(parsePromptProfileConfig({ prompt_profile: 'lean' }, { XERXES_PROMPT_PROFILE: 'full' })).toBe('full')
  expect(parsePromptProfileConfig({}, { XERXES_PROMPT_PROFILE: ' LEAN ' })).toBe('lean')
  expect(parsePromptProfileConfig({ prompt_profile: 'full' }, { XERXES_PROMPT_PROFILE: 'full' }, 'lean')).toBe('lean')
  expect(parsePromptProfileConfig({ prompt_profile: null }, {})).toBe('full')
})

test('parsePromptProfileConfig rejects unknown values instead of defaulting', () => {
  expect(() => parsePromptProfileConfig({ prompt_profile: 'tiny' }, {})).toThrow(ValidationError)
  expect(() => parsePromptProfileConfig({}, { XERXES_PROMPT_PROFILE: 'tiny' })).toThrow('XERXES_PROMPT_PROFILE')
  expect(() => parsePromptProfileConfig({ prompt_profile: 3 }, {})).toThrow('prompt_profile')
})

test('selectLeanTools keeps only the core set and shortens descriptions without touching structure', () => {
  const selected = selectLeanTools([...REAL_DEFINITIONS, advisor()])
  const names = selected.map(tool => tool.function.name)
  expect(names).toContain('ReadFile')
  expect(names).toContain('exec_command')
  expect(names).toContain('check_command')
  expect(names).toContain('XlsxWriter')
  expect(names).toContain('AskAdvisor')
  expect(names).not.toContain('TodoWriteTool')

  for (const tool of selected) {
    expect(tool.function.description.length).toBeLessThanOrEqual(200)
    const original = REAL_DEFINITIONS.find(candidate => candidate.function.name === tool.function.name)
    if (original === undefined) continue
    const parameters = tool.function.parameters as { required?: string[]; properties: Record<string, { type?: unknown; description?: string }> }
    const originalParameters = original.function.parameters as typeof parameters
    expect(parameters.required).toEqual(originalParameters.required)
    for (const [key, property] of Object.entries(parameters.properties)) {
      expect(property.type).toEqual(originalParameters.properties[key]?.type)
      expect((property.description ?? '').length).toBeLessThanOrEqual(80)
    }
  }
  const exec = selected.find(tool => tool.function.name === 'exec_command')
  expect(exec?.function.description.startsWith('Run one executable directly from an argv list.')).toBe(true)
})

test('selectLeanTools omits AskAdvisor when it is not registered and honours extra and protected', () => {
  expect(selectLeanTools(REAL_DEFINITIONS).map(tool => tool.function.name)).not.toContain('AskAdvisor')
  const extra = selectLeanTools(REAL_DEFINITIONS, { extra: ['TodoWriteTool'] }).map(tool => tool.function.name)
  expect(extra).toContain('TodoWriteTool')
  const guarded = selectLeanTools(REAL_DEFINITIONS, { protected: true, extra: ['exec_command'] }).map(tool => tool.function.name)
  expect(guarded).not.toContain('exec_command')
  expect(guarded).toContain('ReadFile')
})

test('buildLeanSystemPrompt states the rules, adapts to tools, and clips project instructions', () => {
  const base = { cwd: '/work/proj', model: 'bonsai-8b', date: '2026-11-02' }
  const withAdvisor = buildLeanSystemPrompt({ ...base, toolNames: ['ReadFile', 'AskAdvisor', 'XlsxWriter'] })
  expect(withAdvisor).toContain('/work/proj')
  expect(withAdvisor).toContain('bonsai-8b')
  expect(withAdvisor).toContain('Read a file before you edit it')
  expect(withAdvisor).toContain('read-only')
  expect(withAdvisor).toContain('Never say you checked')
  // Grouping case variants must not rewrite the labels a person will read.
  expect(withAdvisor).toContain('Keep names, labels and IDs as the input spells them')
  expect(withAdvisor).toContain('AskAdvisor')
  expect(withAdvisor).toContain('XlsxWriter')

  const without = buildLeanSystemPrompt({ ...base, toolNames: ['ReadFile'] })
  expect(without).not.toContain('AskAdvisor')
  expect(without).not.toContain('exec_command')

  const long = buildLeanSystemPrompt({ ...base, toolNames: [], projectInstructions: 'x'.repeat(5000) })
  expect(long.length).toBeLessThan(without.length + 1700)
})

test('lean prompt plus real tool schemas, with a full instruction block, stays within 3,000 tokens', () => {
  const tools = selectLeanTools([...REAL_DEFINITIONS, advisor()])
  const prompt = buildLeanSystemPrompt({
    cwd: '/Users/someone/Documents/Projects/a-fairly-long-project-name',
    model: 'bonsai-8b-q4',
    date: '2026-11-02',
    toolNames: tools.map(tool => tool.function.name),
    projectInstructions: 'y'.repeat(1500),
  })
  const size = leanPromptSize(prompt, tools)
  console.log(`lean prompt: ${size.chars} chars, ~${size.estimatedTokens} tokens, ${tools.length} tools`)
  expect(size.estimatedTokens).toBeLessThanOrEqual(3000)
  expect(tools.map(tool => tool.function.name)).not.toContain('ListDir')
  expect(size.chars).toBe(prompt.length + JSON.stringify(tools).length)

  const full = leanPromptSize('', REAL_DEFINITIONS)
  expect(size.estimatedTokens - Math.ceil(prompt.length / 4)).toBeLessThan(full.estimatedTokens)
})

test('the default agent profile admits every lean core tool, so lean mode is not cut down by the profile', () => {
  const agent = loadBuiltinAgentDefinitions().get('default')
  if (agent === undefined) throw new Error('default agent profile is missing')
  const declared = new Set(agent.tools)
  const excluded = new Set(agent.excludeTools)
  const missing = LEAN_CORE_TOOL_NAMES.filter(name => !declared.has(name) || excluded.has(name))
  expect(missing).toEqual([])
})

test('with nothing configured, a Bonsai model or a known window of 32K or less picks lean', () => {
  // A Bonsai model gets lean wherever it runs, even with its 262K window.
  expect(automaticPromptProfile({ model: 'ternary-bonsai-2-27b', contextLimit: 262_144 })).toBe('lean')
  expect(automaticPromptProfile({ model: 'prism-ml/ternary-bonsai-2-27b' })).toBe('lean')
  // Other models: a small window gets lean; a big or unknown one keeps the full prompt.
  expect(automaticPromptProfile({ model: 'gpt-5.5', contextLimit: 262_144 })).toBe('full')
  expect(automaticPromptProfile({ contextLimit: 16_384 })).toBe('lean')
  expect(automaticPromptProfile({ contextLimit: 32_768 })).toBe('lean')
  expect(automaticPromptProfile({ contextLimit: 131_072 })).toBe('full')
  // An unknown window stays full: a loopback proxy can front a large cloud model.
  expect(automaticPromptProfile({})).toBe('full')
  // Anything configured wins over the route.
  expect(parsePromptProfileConfig({}, {}, 'full', { contextLimit: 16_384 })).toBe('full')
  expect(parsePromptProfileConfig({ prompt_profile: 'full' }, {}, undefined, { contextLimit: 16_384 })).toBe('full')
  expect(parsePromptProfileConfig({}, { XERXES_PROMPT_PROFILE: 'full' }, undefined, { contextLimit: 16_384 })).toBe('full')
  expect(parsePromptProfileConfig({}, {}, undefined, { contextLimit: 16_384 })).toBe('lean')
  expect(parsePromptProfileConfig({}, {})).toBe('full')
})

test('project instructions: one file as written, several labelled, a long block cut with the files to read', () => {
  expect(leanProjectInstructions([])).toBeUndefined()
  const agents = { name: 'AGENTS.md' as const, path: '/work/AGENTS.md', content: 'Use bun, not npm.\n' }
  expect(leanProjectInstructions([agents])).toBe('Use bun, not npm.')

  const claude = { name: 'CLAUDE.md' as const, path: '/work/CLAUDE.md', content: 'Write tests first.' }
  expect(leanProjectInstructions([agents, claude])).toBe('[AGENTS.md]\nUse bun, not npm.\n\n[CLAUDE.md]\nWrite tests first.')

  const long = leanProjectInstructions([{ ...agents, content: 'word '.repeat(1000) }, claude])
  expect(long?.length).toBeLessThanOrEqual(1500)
  expect(long).toContain('Read /work/AGENTS.md, /work/CLAUDE.md for the rest.')

  // Scanned like the full prompt's instruction files.
  expect(leanProjectInstructions([{ ...agents, content: 'Use\u200b bun.' }])).not.toContain('\u200b')
})
