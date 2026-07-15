// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useRef, useState } from 'react'

import * as inputHistory from '../lib/history.js'

export function useInputHistory() {
  const historyRef = useRef<string[]>(inputHistory.load())
  const [historyIdx, setHistoryIdx] = useState<number | null>(null)
  const historyDraftRef = useRef('')

  return { historyRef, historyIdx, setHistoryIdx, historyDraftRef, pushHistory: inputHistory.append }
}
