// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// Transcript row types shared by the UI transcript helpers. Extracted from the
// retired app/gatewayState.ts reducer so live modules stop importing legacy
// protocol code just to name these shapes.

import type { DisplayBlock } from '../gatewayTypes.js'

export type Role = 'user' | 'assistant' | 'tool' | 'system' | 'think'

export interface TranscriptRow {
  id: number
  role: Role
  text: string
  /** Structured tool-result panels (diff/todo/background_task/brief/generic). */
  blocks?: DisplayBlock[]
}
