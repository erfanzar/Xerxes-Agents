// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

import { buildBootstrapSystemPrompt } from '../src/runtime/bootstrap.js'
import { VERIFICATION_MANDATE_RULES } from '../src/runtime/verificationMandate.js'

const promptPath = join(import.meta.dir, '..', 'src', 'agents', 'default', 'system.md')

test('default agent prompt uses only current conditional tool names and stays bounded', async () => {
  const prompt = await readFile(promptPath, 'utf8')

  // The ceiling exists because this text is re-sent on every turn; raise it only for a rule
  // the tool surface actually depends on, never for restating something already stated.
  // Raised for the result-durability rule: the runtime prunes, off-loads and summarizes tool
  // results, and nothing else in the prompt tells the model that, so it plans to scroll back
  // to a value that is no longer there.
  expect(Buffer.byteLength(prompt, 'utf8')).toBeLessThan(6_000)
  expect(prompt).toContain('${ROLE_ADDITIONAL}')
  expect(prompt).toContain('provider-supplied tool list')
  expect(prompt).toContain('`exec_command`')
  expect(prompt).toContain('`write_stdin`')
  expect(prompt).toContain('`FileEditTool`')
  expect(prompt).toContain('`TaskListTool`')
  expect(prompt).toContain('`TaskOutputTool`')
  expect(prompt).toContain('`TaskStopTool`')
  expect(prompt).toContain('`SpawnAgents` accepts any number of agents')
  expect(prompt).toContain('without an artificial ceiling')
  expect(prompt).toContain('Track every cohort without user reminders')
  expect(prompt).toContain('Do not final-answer while required children are queued or running')
  expect(prompt).toContain('`AwaitAgents` with `wake_on: all`')
  expect(prompt).toContain('`PeekAgent`')
  expect(prompt).toContain('Do not busy-poll individual agents or retry stale targets')
  expect(prompt).toContain('retrieve every required omitted output before the final answer')
  expect(prompt).toContain('`SkillTool` is supplied')

  for (const staleName of ['Shell', 'TaskList', 'TaskOutput', 'TaskStop', 'StrReplaceFile']) {
    expect(prompt).not.toContain('`' + staleName + '`')
  }
})

test('default agent prompt forbids calculator processes and Python package guidance', async () => {
  const prompt = await readFile(promptPath, 'utf8')

  expect(prompt).toContain('simple arithmetic directly without tools')
  expect(prompt).toContain('Never launch Python, Node, Bun')
  expect(prompt).toContain('do not add a Python runtime')
  expect(prompt).not.toMatch(/\b(?:pip|virtualenv|venv)\b/iu)
  expect(prompt).not.toMatch(/Python packages?/iu)
  expect(prompt).not.toMatch(/install third-party (?:tools|packages)/iu)
})

test('default agent prompt states parallel-call and blast-radius rules the runtime enforces', async () => {
  const prompt = await readFile(promptPath, 'utf8')

  expect(prompt).toContain('Issue independent tool calls together in one response so they run in parallel')
  expect(prompt).toContain("when one call's input depends on another's result, call them in separate steps")
  expect(prompt).toContain('Judge blast radius before acting')
  expect(prompt).toContain('locally reversible work inside the workspace')
  expect(prompt).toContain('needs explicit user confirmation first')
})

test('the live bootstrap prompt ships the same verification mandate as the runtime prefix', () => {
  // The mandate previously existed only in PromptContextBuilder, which no shipping path calls;
  // asserting against the prompt the daemon actually sends is what keeps the two halves joined.
  const prompt = buildBootstrapSystemPrompt({
    cwd: '/workspace',
    date: '2026-07-27 Monday',
    model: 'test-model',
    platform: 'darwin',
  })

  expect(prompt).toContain('# Verification')
  for (const rule of VERIFICATION_MANDATE_RULES) {
    expect(prompt).toContain(rule)
  }
})
