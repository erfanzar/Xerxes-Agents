// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { parseToolCallBlocks } from './common.js'
import type { ParsedToolCall } from './types.js'

/** Parse Qwen 2.5 chat XML-like tool-call blocks. */
export function parseQwenToolCalls(text: string): ParsedToolCall[] {
  return parseToolCallBlocks(text, '<tool_call>', '</tool_call>')
}

/** Parse Qwen3-Coder's sentinel-delimited function-call blocks. */
export function parseQwen3CoderToolCalls(text: string): ParsedToolCall[] {
  return parseToolCallBlocks(text, '|<function_call_start|>', '|<function_call_end|>')
}
