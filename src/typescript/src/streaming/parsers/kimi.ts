// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { parseToolCallBlocks } from './common.js'
import type { ParsedToolCall } from './types.js'

/** Parse Kimi-K2's sentinel-delimited tool-call blocks. */
export function parseKimiK2ToolCalls(text: string): ParsedToolCall[] {
  return parseToolCallBlocks(text, '<|tool_call|>', '<|/tool_call|>')
}
