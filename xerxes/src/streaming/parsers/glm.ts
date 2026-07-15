// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { parseToolCallBlocks } from './common.js'
import type { ParsedToolCall } from './types.js'

/** Parse GLM-4.5's XML-like tool-call blocks. */
export function parseGlm45ToolCalls(text: string): ParsedToolCall[] {
  return parseToolCallBlocks(text, '<tool_call>', '</tool_call>')
}

/** Parse GLM-4.7's XML-like function-call blocks. */
export function parseGlm47ToolCalls(text: string): ParsedToolCall[] {
  return parseToolCallBlocks(text, '<function_call>', '</function_call>')
}
