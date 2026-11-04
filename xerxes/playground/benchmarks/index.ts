// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createDabstepAdapter } from './dabstep.js'
import { createGdpvalAdapter } from './gdpval.js'
import { createSpreadsheetBenchAdapter } from './spreadsheetbench.js'
import { createSyntheticPrivateAdapter } from './syntheticPrivate.js'
import { createTerminalBenchAdapter } from './terminalBench.js'
import type { BenchmarkAdapter, BenchmarkId } from './types.js'

export * from './types.js'
export { createDabstepAdapter } from './dabstep.js'
export { createGdpvalAdapter } from './gdpval.js'
export { createSpreadsheetBenchAdapter } from './spreadsheetbench.js'
export { createSyntheticPrivateAdapter } from './syntheticPrivate.js'
export { createTerminalBenchAdapter } from './terminalBench.js'
export { materializeTaskFiles } from './util.js'

/** The built-in adapter for a benchmark id. */
export function createBenchmarkAdapter(id: BenchmarkId): BenchmarkAdapter {
  switch (id) {
    case 'dabstep':
      return createDabstepAdapter()
    case 'gdpval':
      return createGdpvalAdapter()
    case 'spreadsheetbench':
      return createSpreadsheetBenchAdapter()
    case 'synthetic-private':
      return createSyntheticPrivateAdapter()
    case 'terminal-bench':
      return createTerminalBenchAdapter()
  }
}
