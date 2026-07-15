// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { AgentMemory } from '../src/memory/agentMemory.js'
import {
  WORKFLOW_MEMORY_FILE,
  captureUserWorkflowMemory,
  formatWorkflowMemoryNote,
  shouldCaptureWorkflowMemory,
} from '../src/runtime/workflowMemory.js'

test('workflow-memory detection is limited to explicit durable-memory requests', () => {
  expect(shouldCaptureWorkflowMemory('Remember that this repository uses Bun.')).toBeTrue()
  expect(shouldCaptureWorkflowMemory('I want you to understand this large project.')).toBeTrue()
  expect(shouldCaptureWorkflowMemory('Please inspect the project.')).toBeFalse()
  expect(shouldCaptureWorkflowMemory('')).toBeFalse()
})

test('workflow-memory capture writes the appropriate scope once and exposes a deterministic note', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-workflow-memory-'))
  try {
    const memory = new AgentMemory({ globalDirectory: join(root, 'global'), projectRoot: root })
    const message = 'Remember that deploys require a Bun test run.'
    const clock = () => new Date('2026-07-13T12:00:00.000Z')

    const first = await captureUserWorkflowMemory(message, memory, { clock, projectRoot: root })
    const duplicate = await captureUserWorkflowMemory(message, memory, { clock, projectRoot: root })
    const body = await memory.read('project', WORKFLOW_MEMORY_FILE)

    expect(first).toEqual({ captured: true, scope: 'project', path: WORKFLOW_MEMORY_FILE })
    expect(duplicate).toEqual({ captured: false, reason: 'duplicate', scope: 'project' })
    expect(body).toContain('2026-07-13T12:00:00.000Z')
    expect(body).toContain(message)
    expect(body.match(/deploys require/g)?.length).toBe(1)
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('workflow-memory capture preserves no-op reasons and handles a global-only memory store', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-workflow-global-'))
  try {
    const memory = new AgentMemory({ globalDirectory: join(root, 'global') })
    await expect(captureUserWorkflowMemory('ordinary request', memory)).resolves.toEqual({ captured: false, reason: 'no_signal' })
    await expect(captureUserWorkflowMemory('remember this', undefined)).resolves.toEqual({
      captured: false,
      reason: 'memory_unavailable',
    })
    expect(await captureUserWorkflowMemory('save this workflow', memory)).toMatchObject({ captured: true, scope: 'global' })
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('workflow-memory note validation resolves project context without hidden state', () => {
  expect(formatWorkflowMemoryNote('Keep this note.', {
    clock: () => new Date('2026-07-13T00:00:00.000Z'),
    projectRoot: '/tmp/example',
  })).toContain('**Project root:** `/tmp/example`')
  expect(() => formatWorkflowMemoryNote('   ')).toThrow('workflow instruction')
})
