// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  PolicyAction,
  PolicyEngine,
  ToolPolicy,
  ToolPolicyViolation,
} from '../src/security/index.js'

test('tool-policy parity covers empty, deny-only, allow-only, and optional-tool admission rules', () => {
  expect(new ToolPolicy().evaluate('anything')).toBe(PolicyAction.ALLOW)

  const denyOnly = new ToolPolicy({ deny: ['execute_shell', 'delete_file'] })
  expect(denyOnly.evaluate('execute_shell')).toBe(PolicyAction.DENY)
  expect(denyOnly.evaluate('delete_file')).toBe(PolicyAction.DENY)
  expect(denyOnly.evaluate('search')).toBe(PolicyAction.ALLOW)

  const allowOnly = new ToolPolicy({ allow: ['search', 'read_file'] })
  expect(allowOnly.evaluate('search')).toBe(PolicyAction.ALLOW)
  expect(allowOnly.evaluate('read_file')).toBe(PolicyAction.ALLOW)
  expect(allowOnly.evaluate('execute_shell')).toBe(PolicyAction.DENY)

  const optional = new ToolPolicy({ optionalTools: ['dangerous_tool'] })
  expect(optional.evaluate('dangerous_tool')).toBe(PolicyAction.DENY)
  expect(new ToolPolicy({ allow: ['dangerous_tool'], optionalTools: ['dangerous_tool'] }).evaluate('dangerous_tool'))
    .toBe(PolicyAction.ALLOW)
})

test('policy-engine parity applies global and agent policies, notifies listeners, and supports dynamic removal', () => {
  const events: Array<readonly [string, string | undefined, PolicyAction]> = []
  const engine = new PolicyEngine({ globalPolicy: new ToolPolicy({ deny: ['execute_shell'] }) })
  engine.addListener((toolName, agentId, action) => events.push([toolName, agentId, action]))

  expect(engine.check('execute_shell')).toBe(PolicyAction.DENY)
  expect(engine.check('search', 'reader')).toBe(PolicyAction.ALLOW)
  engine.setAgentPolicy('coder', new ToolPolicy({ allow: ['execute_shell', 'search'] }))
  expect(engine.check('execute_shell', 'coder')).toBe(PolicyAction.ALLOW)
  expect(engine.check('execute_shell', 'reader')).toBe(PolicyAction.DENY)
  engine.enforce('search', 'coder')
  expect(() => engine.enforce('execute_shell', 'reader')).toThrow(ToolPolicyViolation)
  expect(() => engine.enforce('execute_shell', 'reader')).toThrow('execute_shell')

  engine.removeAgentPolicy('coder')
  expect(engine.check('execute_shell', 'coder')).toBe(PolicyAction.DENY)
  expect(events).toEqual([
    ['execute_shell', undefined, PolicyAction.DENY],
    ['search', 'reader', PolicyAction.ALLOW],
    ['execute_shell', 'coder', PolicyAction.ALLOW],
    ['execute_shell', 'reader', PolicyAction.DENY],
    ['search', 'coder', PolicyAction.ALLOW],
    ['execute_shell', 'reader', PolicyAction.DENY],
    ['execute_shell', 'reader', PolicyAction.DENY],
    ['execute_shell', 'coder', PolicyAction.DENY],
  ])
})
