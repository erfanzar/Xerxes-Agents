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

test('the plan card offers Continue with the plan and a refine box, not Allow/Deny', () => {
  const html = renderToStaticMarkup(createElement(PlanApprovalCard, {
    approval: { id: 'r1', action: '', description: '', toolName: 'ExitPlanModeTool' },
  }))
  expect(html).toContain('Plan ready')
  expect(html).toContain('Continue with the plan')
  expect(html).toContain('aria-label="Refine the plan"')
  expect(html).toContain('Open the plan')
  expect(html).not.toContain('Allow once')
  expect(html).not.toContain('Deny')
})
