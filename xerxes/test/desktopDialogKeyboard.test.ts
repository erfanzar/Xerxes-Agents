// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { handleDialogEscape } from '../src/desktop/renderer/dialogFocus.js'

function event(key: string) {
  let prevented = false, stopped = false
  return { key, preventDefault() { prevented = true }, stopImmediatePropagation() { stopped = true }, result: () => ({ prevented, stopped }) }
}

test('a parent sheet does not consume Escape owned by a nested confirmation', () => {
  let closed = 0
  const key = event('Escape')
  handleDialogEscape(key, () => { closed += 1 }, false)
  expect(closed).toBe(0)
  expect(key.result()).toEqual({ prevented: false, stopped: false })
})

test('the focused sheet consumes Escape and closes exactly once', () => {
  let closed = 0
  const key = event('Escape')
  handleDialogEscape(key, () => { closed += 1 }, true)
  expect(closed).toBe(1)
  expect(key.result()).toEqual({ prevented: true, stopped: true })
})

test('dialog Escape ownership never consumes Tab traversal', () => {
  const key = event('Tab')
  handleDialogEscape(key, () => { throw new Error('Unexpected close') }, true)
  expect(key.result()).toEqual({ prevented: false, stopped: false })
})
