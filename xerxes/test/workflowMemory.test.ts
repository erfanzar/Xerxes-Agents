// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { AgentMemory } from '../src/memory/agentMemory.js'
import {
  MAX_WORKFLOW_MEMORY_CHARACTERS,
  captureUserWorkflowMemory,
  formatWorkflowMemoryNote,
  shouldCaptureWorkflowMemory,
  workflowMemoryTopicPath,
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

    const topicPath = workflowMemoryTopicPath(message)
    const first = await captureUserWorkflowMemory(message, memory, { clock, projectRoot: root })
    const duplicate = await captureUserWorkflowMemory(message, memory, { clock, projectRoot: root })
    const body = await memory.read('project', topicPath)

    expect(topicPath).toStartWith('topics/workflow-remember-that-deploys-require-a-bun')
    expect(first).toEqual({ captured: true, scope: 'project', path: topicPath })
    expect(duplicate).toEqual({ captured: false, reason: 'duplicate', scope: 'project' })
    expect(body).toContain('2026-07-13T12:00:00.000Z')
    expect(body).toContain(message)
    expect(body.match(/deploys require/g)?.length).toBe(3) // frontmatter name and description plus the instruction

    // The topic reaches the prompt as a described manifest line, not as a body.
    const [entry] = (await memory.listFiles('project')).filter(file => file.path === topicPath)
    expect(entry?.type).toBe('workflow')
    expect(entry?.description).toContain('User workflow instruction')
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

test('explicit save requests are honored but never persist credentials or unbounded pastes', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-workflow-guards-'))
  try {
    const memory = new AgentMemory({ globalDirectory: join(root, 'global') })

    // An explicit request is still honored — that is the point of this path.
    const honored = await captureUserWorkflowMemory('Remember that erfan@prismml.com owns releases.', memory)
    expect(honored.captured).toBeTrue()
    expect(await memory.read('global', honored.path!)).toContain('erfan@prismml.com')

    const secret = 'Remember the deploy key: api_key=sk-live-9f2a7c4d1e6b8a305f7c'
    expect(await captureUserWorkflowMemory(secret, memory)).toEqual({ captured: false, reason: 'credential_blocked' })
    expect((await memory.listFiles('global')).some(file => file.path.includes('deploy-key'))).toBeFalse()

    const paste = 'Remember this log: ' + 'x'.repeat(MAX_WORKFLOW_MEMORY_CHARACTERS)
    expect(await captureUserWorkflowMemory(paste, memory)).toEqual({ captured: false, reason: 'too_long' })
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('workflow-memory note validation resolves project context without hidden state', () => {
  const note = formatWorkflowMemoryNote('Keep this note.', {
    clock: () => new Date('2026-07-13T00:00:00.000Z'),
    projectRoot: '/tmp/example',
  })
  expect(note).toContain('**Project root:** `/tmp/example`')
  expect(note).toStartWith('---\nname: workflow: Keep this note.\ndescription: User workflow instruction:')
  expect(() => formatWorkflowMemoryNote('   ')).toThrow('workflow instruction')
})
