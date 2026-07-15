// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SubagentProgress } from '../types.js'

export function subagentElapsedSeconds(agent: SubagentProgress, now = Date.now()): number | null {
  if (typeof agent.durationSeconds === 'number') {
    return Math.max(0, agent.durationSeconds)
  }
  if ((agent.status === 'running' || agent.status === 'queued') && agent.startedAt) {
    return Math.max(0, (now - agent.startedAt) / 1000)
  }
  return null
}
