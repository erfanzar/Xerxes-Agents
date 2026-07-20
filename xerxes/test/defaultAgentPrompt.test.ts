// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

const promptPath = join(import.meta.dir, '..', 'src', 'agents', 'default', 'system.md')

test('default agent prompt uses only current conditional tool names and stays bounded', async () => {
  const prompt = await readFile(promptPath, 'utf8')

  expect(Buffer.byteLength(prompt, 'utf8')).toBeLessThan(5_000)
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
