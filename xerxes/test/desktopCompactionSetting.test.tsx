// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { compactionStateOf, compactionSummary } from '../src/desktop/renderer/CompactionSetting.js'

test('the Auto-compact row reads the daemon reply and says when compaction happens', () => {
  const byDefault = compactionStateOf({ ok: true, threshold: 0.6, label: '60%', default_threshold: 0.6, default_label: '60%', saved: false })
  expect(byDefault).toEqual({ threshold: 0.6, label: '60%', defaultLabel: '60%', saved: false })
  expect(compactionSummary(byDefault)).toContain("fills 60% of the model's context window")

  const tokens = compactionStateOf({ ok: true, threshold: 400_000, label: '400K tokens', default_label: '60%', saved: true })
  expect(compactionSummary(tokens)).toContain('reaches 400K tokens')
  expect(compactionSummary(tokens)).toContain('compacts at 60% of it instead')

  const off = compactionStateOf({ ok: true, threshold: 0, label: 'off', default_label: '60%', saved: true })
  expect(compactionSummary(off)).toStartWith('Off')

  // A malformed reply never shows a made-up threshold.
  expect(compactionStateOf({ threshold: 'soon', saved: 'yes' })).toEqual({ threshold: 0, label: '', defaultLabel: '', saved: false })
})
