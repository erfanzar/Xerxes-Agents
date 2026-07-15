// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { SlashExecResponse } from '../gatewayTypes.js'

export function slashExecText(response: null | SlashExecResponse | undefined): null | string {
  const output = typeof response?.output === 'string' && response.output.trim() ? response.output : ''
  const warning = typeof response?.warning === 'string' ? response.warning.trim() : ''

  if (!output && !warning) {
    return null
  }

  if (warning && output) {
    return `warning: ${warning}\n${output}`
  }

  return warning ? `warning: ${warning}` : output
}
