// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { RunHistory } from '../src/desktop/renderer/RunHistory.js'
import { scheduleTime } from '../src/desktop/renderer/desktopRpc.js'

test('schedule display uses the selected timezone rather than the desktop timezone', () => {
  expect(scheduleTime('2026-09-14T08:00:00Z', 'UTC', 'en-US')).toBe('9/14/2026, 8:00:00 AM · UTC')
  expect(scheduleTime('2026-09-14T08:00:00Z', 'Europe/Istanbul', 'en-US')).toBe('9/14/2026, 11:00:00 AM · Europe/Istanbul')
  expect(scheduleTime('invalid', 'UTC')).toBe('Time unavailable')
  expect(scheduleTime('2026-09-14T08:00:00Z', 'invalid')).toContain('schedule timezone unavailable')
})

const actions = { busy: false, onClose() {}, onMore() {}, onInspect() {} }
test('schedule history exposes failed results and preserves output whitespace without raw envelopes', () => {
  const run = { id: 'run-1', state: 'failed', startedAt: 1000, error: 'Provider rejected configuration' }
  const output = 'checking configuration\n  missing model\n'
  const html = renderToStaticMarkup(createElement(RunHistory, {
    ...actions,
    page: { job: { prompt: 'Review the workspace' }, runs: [run], hasMore: true },
    result: { ...run, output, exitCode: 1, outputTruncated: true },
  }))
  expect(html).toContain('Provider rejected configuration')
  expect(html.match(/Provider rejected configuration/g)).toHaveLength(1)
  expect(html).toContain('aria-label="Run output">' + output)
  expect(html).toContain('Exit 1')
  expect(html).toContain('earlier output was truncated')
  expect(html).toContain('Load earlier runs')
  expect(html).not.toContain('"startedAt"')
  expect(html).not.toContain('run-1')
})
test('empty history and unavailable metadata do not imply success', () => {
  const empty = renderToStaticMarkup(createElement(RunHistory, { ...actions, page: { job: {}, runs: [], hasMore: false }, result: null }))
  expect(empty).toContain('has not run yet')
  const unknown = renderToStaticMarkup(createElement(RunHistory, { ...actions, page: { job: {}, runs: [{ id: 'unknown' }], hasMore: false }, result: { id: 'unknown' } }))
  expect(unknown).toContain('Start time unavailable')
  expect(unknown).toContain('Status unavailable')
  expect(unknown).toContain('No output recorded')
  expect(unknown).not.toContain('Exit 0')
})
