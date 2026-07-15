// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Execution strategies supported by a Cortex workflow. */
export const ProcessType = {
  SEQUENTIAL: 'sequential',
  HIERARCHICAL: 'hierarchical',
  PARALLEL: 'parallel',
  CONSENSUS: 'consensus',
  PLANNED: 'planned',
} as const

export type ProcessType = (typeof ProcessType)[keyof typeof ProcessType]

/** Linking strategies supported by a Cortex task chain. */
export const ChainType = {
  LINEAR: 'linear',
  BRANCHING: 'branching',
  LOOP: 'loop',
} as const

export type ChainType = (typeof ChainType)[keyof typeof ChainType]
