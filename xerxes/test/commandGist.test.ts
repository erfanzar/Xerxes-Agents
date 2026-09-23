// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { commandGist } from '../src/desktop/renderer/commandGist.js'

test('the rail shows what ran, not the wrapper it ran inside', () => {
  // The case this exists for: every background row in a JAX workspace began
  // `bash -c ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu XLA_FLAGS=…`, so at
  // rail width they all truncated to the same unreadable prefix and the one
  // token that told them apart was the one cut off.
  expect(commandGist('bash -c ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 pytest tests/'))
    .toBe('pytest tests/')
  expect(commandGist('bash -lc "FOO=1 npm run build"')).toBe('npm run build')
  expect(commandGist('/bin/sh -c BAR=2 ls -la')).toBe('ls -la')
  expect(commandGist('bash -c "cd /tmp && make -j8"')).toBe('cd /tmp && make -j8')
})

test('a plain command is left exactly as it is', () => {
  expect(commandGist('git status')).toBe('git status')
  expect(commandGist('  spaced   out   cmd  ')).toBe('spaced out cmd')
})

test('a command that is only environment assignments keeps all of them', () => {
  // Stripping greedily from the front would leave `BAR=2` and present that
  // as the command — worse than showing the whole thing.
  expect(commandGist('FOO=1 BAR=2')).toBe('FOO=1 BAR=2')
  expect(commandGist('PATH=/usr/bin')).toBe('PATH=/usr/bin')
})

test('nothing is ever reduced to an empty row', () => {
  // A blank row is unreadable in a different way; fall back to the original.
  expect(commandGist('bash -c ""')).toBe('bash -c ""')
  expect(commandGist('')).toBe('')
})
