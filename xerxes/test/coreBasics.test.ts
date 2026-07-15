// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  AGENTS_REGISTRY,
  BasicRegistryError,
  CLIENT_REGISTRY,
  REGISTRY,
  RegistryType,
  XERXES_REGISTRY,
  basicRegistry,
  prettyPrint,
} from '../src/core/basics.js'

test('prettyPrint preserves nested record indentation and handles empty records', () => {
  expect(prettyPrint({ key: 'value' })).toBe('key:\n  value')
  expect(prettyPrint({ outer: { inner: 'value' } })).toBe('outer:\n  inner:\n    value')
  expect(prettyPrint({})).toBe('')
  expect(prettyPrint({ key: 'value' }, 4)).toBe('    key:\n      value')
})

test('registry categories retain their stable backing registry objects', () => {
  expect(REGISTRY[RegistryType.CLIENT]).toBe(CLIENT_REGISTRY)
  expect(REGISTRY[RegistryType.AGENTS]).toBe(AGENTS_REGISTRY)
  expect(REGISTRY[RegistryType.XERXES]).toBe(XERXES_REGISTRY)
})

test('basicRegistry registers the same constructor and exposes public-field inspection', () => {
  class TestAgent {
    name = 'planner'
    _internal = 'hidden'
  }

  const name = 'core-basics-test-agent'
  const RegisteredAgent = basicRegistry(RegistryType.AGENTS, name)(TestAgent)
  try {
    const agent = new RegisteredAgent()
    expect(AGENTS_REGISTRY[name]).toBe(TestAgent)
    expect(agent.toDict()).toEqual({ name: 'planner' })
    expect(String(agent)).toContain('TestAgent')
    expect(String(agent)).toContain('name:')
  } finally {
    delete AGENTS_REGISTRY[name]
  }
})

test('basicRegistry rejects unknown categories and unsafe names', () => {
  expect(() => basicRegistry('invalid' as RegistryType, 'name')).toThrow(BasicRegistryError)
  expect(() => basicRegistry(RegistryType.CLIENT, '__proto__')).toThrow(BasicRegistryError)
})
