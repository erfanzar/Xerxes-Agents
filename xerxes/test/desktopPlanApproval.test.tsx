// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { PlanApprovalCard, isPlanExitApproval } from '../src/desktop/renderer/App.js'

test('only an ExitPlanMode request is a plan decision', () => {
  expect(isPlanExitApproval({ toolName: 'ExitPlanModeTool', action: '' })).toBe(true)
  expect(isPlanExitApproval({ action: 'ExitPlanMode' })).toBe(true)
  expect(isPlanExitApproval({ toolName: 'exec_command', action: '' })).toBe(false)
  expect(isPlanExitApproval({ toolName: 'ExitPlanModeToolX', action: '' })).toBe(false)
})

test('the plan card asks like Claude Code: Ready to code?, Would you like to proceed?, three numbered answers, not Allow/Deny', () => {
  const html = renderToStaticMarkup(createElement(PlanApprovalCard, {
    approval: { id: 'r1', action: '', description: '', toolName: 'ExitPlanModeTool' },
  }))
  expect(html).toContain('Ready to code?')
  expect(html).toContain('Plan ready')
  expect(html).toContain('Would you like to proceed?')
  expect(html).toContain('Yes, and auto-accept edits')
  expect(html).toContain('Yes, manually approve edits')
  expect(html).toContain('No, keep planning')
  // The first answer is preselected, so ⌘↵ continues at once.
  expect(html).toMatch(/role="radio" aria-checked="true"[^>]*><kbd>1<\/kbd>/)
  expect(html).toContain('Continue with the plan')
  // Before the plan text loads, the card still opens the plan itself.
  expect(html).toContain('Open the plan')
  // The feedback box appears only once "keep planning" is picked.
  expect(html).not.toContain('aria-label="Refine the plan"')
  expect(html).not.toContain('Allow once')
  expect(html).not.toContain('Deny')
})
