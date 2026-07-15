// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Operator tools which do not mutate the workspace or start child processes. */
export const SAFE_OPERATOR_TOOLS: ReadonlySet<string> = new Set([
  'ask_user',
  'list_terminal_sessions',
  'parallel_tools',
  'web.time',
  'update_plan',
])

/** Operator tools which require an explicit high-power capability grant. */
export const HIGH_POWER_OPERATOR_TOOLS: ReadonlySet<string> = new Set([
  'exec_command',
  'write_stdin',
  'close_terminal_session',
  'apply_patch',
  'spawn_agent',
  'resume_agent',
  'send_input',
  'wait_agent',
  'close_agent',
  'view_image',
  'web.search_query',
  'web.image_query',
  'web.open',
  'web.click',
  'web.find',
  'web.screenshot',
  'web.weather',
  'web.finance',
  'web.sports',
])

export const ALL_OPERATOR_TOOLS: ReadonlySet<string> = new Set([
  ...SAFE_OPERATOR_TOOLS,
  ...HIGH_POWER_OPERATOR_TOOLS,
])

export interface OperatorRuntimeConfig {
  readonly allowedToolNames: ReadonlySet<string>
  readonly browserHeadless: boolean
  readonly browserScreenshotDirectory?: string
  readonly enabled: boolean
  readonly powerToolsEnabled: boolean
  readonly shellDefaultMaxOutputChars: number
  readonly shellDefaultWorkdir?: string
  readonly shellDefaultYieldMs: number
  readonly subagentDefaultPermissionMode: string
  readonly subagentDefaultProfile: string
  readonly subagentDefaultTimeoutMs: number
}

/** Create an immutable, session-scoped operator configuration. */
export function createOperatorRuntimeConfig(
  overrides: Partial<OperatorRuntimeConfig> = {},
): OperatorRuntimeConfig {
  const allowedToolNames = new Set(overrides.allowedToolNames ?? ALL_OPERATOR_TOOLS)
  return Object.freeze({
    enabled: false,
    powerToolsEnabled: true,
    browserHeadless: true,
    shellDefaultYieldMs: 1_000,
    shellDefaultMaxOutputChars: 4_000,
    subagentDefaultProfile: 'minimal',
    subagentDefaultPermissionMode: 'accept-all',
    subagentDefaultTimeoutMs: 30_000,
    ...overrides,
    allowedToolNames,
  })
}
