// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { OrbEngine, wake } from '../src/desktop/renderer/AgentOrb.js'
import { toolOrbState } from '../src/desktop/renderer/activityPhrase.js'
import { orbStateOf } from '../src/desktop/renderer/RailStatus.js'
import type { Snapshot } from '../src/desktop/renderer/store.js'

/** A canvas with no drawing context: the clock and presets still run. */
function fakeCanvas(): HTMLCanvasElement {
  return { width: 0, height: 0, getContext: () => null } as unknown as HTMLCanvasElement
}

test('the canvas is sized in device pixels, capped at 2x', () => {
  const canvas = fakeCanvas()
  new OrbEngine(canvas, 20, { state: 'working', live: true }, 3)
  expect(canvas.width).toBe(40)
  expect(canvas.height).toBe(40)
})

test('a live orb settles into a still frame after its settle time', () => {
  const engine = new OrbEngine(fakeCanvas(), 64, { state: 'breathing', live: true, settleMs: 100 })
  expect(engine.wantsFrames()).toBe(true)
  engine.step(0.05)
  expect(engine.wantsFrames()).toBe(true)
  engine.step(0.06)
  expect(engine.wantsFrames()).toBe(false)
  // A new state starts its own settle window.
  engine.update({ state: 'solving', live: true, settleMs: 100 })
  expect(engine.wantsFrames()).toBe(true)
})

test('a still or off-screen orb never asks for frames', () => {
  const still = new OrbEngine(fakeCanvas(), 20, { state: 'working', live: false })
  expect(still.wantsFrames()).toBe(false)
  const hidden = new OrbEngine(fakeCanvas(), 20, { state: 'working', live: true })
  hidden.visible = false
  expect(hidden.wantsFrames()).toBe(false)
})

test('changing state switches to that state\'s tuned drawing', () => {
  const engine = new OrbEngine(fakeCanvas(), 20, { state: 'searching', live: true })
  expect(engine.mode).toBe('globe')
  engine.update({ state: 'shaping', live: true })
  expect(engine.mode).toBe('morph')
})

test('the ticker does not arm without a visible document', () => {
  expect(() => wake()).not.toThrow()
})

test('each kind of running tool picks a matching animation', () => {
  expect(toolOrbState({ name: 'GrepTool', verb: 'grep' })).toBe('searching')
  expect(toolOrbState({ name: 'ReadFile', verb: 'read' })).toBe('searching')
  expect(toolOrbState({ name: 'FileEditTool', verb: 'edit' })).toBe('shaping')
  expect(toolOrbState({ name: 'exec_command', verb: 'run' })).toBe('working')
  expect(toolOrbState({ name: 'AgentTool', verb: 'agent' })).toBe('connecting')
})

test('the orb follows the turn: tools, streaming reply, reasoning, waiting', () => {
  const base = { turnActive: true, blocks: [] } as unknown as Snapshot
  expect(orbStateOf({ ...base, blocks: [{ kind: 'tools', id: 1, running: true, items: [{ id: 't', name: 'GrepTool', verb: 'grep', arg: 'x', dur: '', state: 'working' }] }] } as unknown as Snapshot)).toBe('searching')
  expect(orbStateOf({ ...base, blocks: [{ kind: 'agent', id: 1, text: 'hi', streaming: true }] } as unknown as Snapshot)).toBe('composing')
  expect(orbStateOf(base)).toBe('solving')
  expect(orbStateOf({ ...base, approval: {} } as unknown as Snapshot)).toBe('breathing')
  expect(orbStateOf({ ...base, turnActive: false, submissionPending: true } as unknown as Snapshot)).toBe('connecting')
})
