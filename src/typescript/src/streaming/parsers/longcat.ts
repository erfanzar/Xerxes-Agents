// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { parseToolCallBlocks } from './common.js'
import type { ParsedToolCall } from './types.js'

/** Parse LongCat's namespaced XML-like tool-call blocks. */
export function parseLongCatToolCalls(text: string): ParsedToolCall[] {
  return parseToolCallBlocks(text, '<longcat:tool>', '</longcat:tool>')
}
