// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  compactionSettingsPath,
  compactionThresholdTokens,
  DEFAULT_AUTO_COMPACT_THRESHOLD,
  describeCompactionThreshold,
  optionalCompactionThreshold,
  parseCompactionThreshold,
  readCompactionSetting,
  saveCompactionSetting,
} from '../src/context/compactionThreshold.js'

test('the default compacts at 60% of the prompt budget', () => {
  expect(DEFAULT_AUTO_COMPACT_THRESHOLD).toBe(0.6)
  // A 1M-token model compacts at about 629K, well before the provider refuses.
  expect(compactionThresholdTokens(1_048_576, DEFAULT_AUTO_COMPACT_THRESHOLD)).toBe(629_145)
})

test('a threshold is a share of the window or an absolute token count', () => {
  expect(compactionThresholdTokens(1_048_576, 0.75)).toBe(786_432)
  expect(compactionThresholdTokens(1_048_576, 400_000)).toBe(400_000)
  // A token count past a smaller model's budget falls back to 60% of that budget,
  // instead of waiting until the window is already full.
  expect(compactionThresholdTokens(200_000, 400_000)).toBe(120_000)
  expect(compactionThresholdTokens(400_000, 400_000)).toBe(240_000)
  expect(compactionThresholdTokens(1_048_576, 0)).toBe(0)
  expect(compactionThresholdTokens(0, 0.6)).toBe(0)
  // An old fraction above 1 still means "only when the window is full".
  expect(compactionThresholdTokens(100_000, 4)).toBe(100_000)
})

test('people can type percentages, fractions, token counts or off', () => {
  expect(parseCompactionThreshold('60%')).toBe(0.6)
  expect(parseCompactionThreshold('75')).toBe(0.75)
  expect(parseCompactionThreshold(0.7)).toBe(0.7)
  expect(parseCompactionThreshold(' 400k ')).toBe(400_000)
  expect(parseCompactionThreshold('400K tokens')).toBe(400_000)
  expect(parseCompactionThreshold('1m')).toBe(1_000_000)
  expect(parseCompactionThreshold('1,048,576')).toBe(1_048_576)
  expect(parseCompactionThreshold(400_000)).toBe(400_000)
  expect(parseCompactionThreshold('off')).toBe(0)
  expect(parseCompactionThreshold(0)).toBe(0)
  for (const bad of ['', 'soon', '150%', '-5%', '500', 512, -1, '0.5k', null, undefined, {}, Number.NaN]) {
    expect(() => parseCompactionThreshold(bad)).toThrow('auto_compact_threshold')
    expect(optionalCompactionThreshold(bad)).toBeUndefined()
  }
})

test('thresholds read back as people wrote them', () => {
  expect(describeCompactionThreshold(0.6)).toBe('60%')
  expect(describeCompactionThreshold(0.725)).toBe('72.5%')
  expect(describeCompactionThreshold(400_000)).toBe('400K tokens')
  expect(describeCompactionThreshold(1_000_000)).toBe('1M tokens')
  expect(describeCompactionThreshold(1_048_576)).toBe('1,048,576 tokens')
  expect(describeCompactionThreshold(0)).toBe('off')
  // Every label parses back to the same threshold.
  for (const value of [0.6, 0.725, 400_000, 1_000_000, 1_048_576, 0]) {
    expect(parseCompactionThreshold(describeCompactionThreshold(value))).toBe(value)
  }
})

test('the saved threshold round-trips, and a bad value never reaches the file', async () => {
  const home = await mkdtemp(join(tmpdir(), 'xerxes-compaction-setting-'))
  try {
    const file = compactionSettingsPath(home)
    expect(file).toBe(join(home, 'daemon', 'compaction.json'))
    expect(readCompactionSetting(file)).toBeUndefined()
    expect(saveCompactionSetting(file, '400k')).toBe(400_000)
    expect(readCompactionSetting(file)).toBe(400_000)
    expect(() => saveCompactionSetting(file, 'whenever')).toThrow('auto_compact_threshold')
    expect(readCompactionSetting(file)).toBe(400_000)
    expect(JSON.parse(await readFile(file, 'utf8'))).toEqual({ version: 1, auto_compact_threshold: 400_000 })
    await writeFile(file, '{not json')
    expect(readCompactionSetting(file)).toBeUndefined()
  } finally {
    await rm(home, { recursive: true, force: true })
  }
})
