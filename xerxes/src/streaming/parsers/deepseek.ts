// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { parseToolCallBlocks } from './common.js'
import type { ParsedToolCall } from './types.js'

export const DEEPSEEK_FULLWIDTH_BAR = '\uff5c'
export const DEEPSEEK_LOWER_EIGHTH_BLOCK = '\u2581'
export const DEEPSEEK_V3_OPEN_TAG = '<' + DEEPSEEK_FULLWIDTH_BAR + 'tool' + DEEPSEEK_LOWER_EIGHTH_BLOCK
  + 'call' + DEEPSEEK_LOWER_EIGHTH_BLOCK + 'begin' + DEEPSEEK_FULLWIDTH_BAR + '>'
export const DEEPSEEK_V3_CLOSE_TAG = '<' + DEEPSEEK_FULLWIDTH_BAR + 'tool' + DEEPSEEK_LOWER_EIGHTH_BLOCK
  + 'call' + DEEPSEEK_LOWER_EIGHTH_BLOCK + 'end' + DEEPSEEK_FULLWIDTH_BAR + '>'

/** Parse DeepSeek-V3's full-width-bar and lower-block tokenizer markers. */
export function parseDeepSeekV3ToolCalls(text: string): ParsedToolCall[] {
  return parseToolCallBlocks(text, DEEPSEEK_V3_OPEN_TAG, DEEPSEEK_V3_CLOSE_TAG)
}

/** Parse DeepSeek-V3.1's simplified XML-like tool blocks. */
export function parseDeepSeekV31ToolCalls(text: string): ParsedToolCall[] {
  return parseToolCallBlocks(text, '<tool>', '</tool>')
}
