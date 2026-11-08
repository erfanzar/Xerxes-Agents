// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Size llama-server's window from the memory that is free when it starts.
 *
 * llama-server allocates the whole KV cache for --ctx-size at load. A fixed
 * window chosen for a 24 GB machine is too large when other apps hold memory
 * (the load fails or the machine swaps) and wastes room on a larger one. So
 * the window is: half of what is free after the weights and a fixed reserve,
 * divided by the cache one token needs, capped at the model's own window. The
 * other half, and the reserve, stay for compute buffers and everything else.
 */
import { type CommandResult, type HardwareProfile } from './hardware.js'

const GIB = 1024 ** 3
const MIB = 1024 ** 2
/** Kept free for llama-server's compute buffers and the rest of the system. */
export const FIT_RESERVE_BYTES = 1 * GIB
/** Share of the remaining free memory the cache may take. */
export const FIT_CACHE_SHARE = 0.5
/** Smallest window worth starting: below it, a knowledge-work task cannot fit. */
export const MIN_FIT_WINDOW = 4096
const WINDOW_STEP = 1024

export interface WindowFitInput {
  readonly freeBytes: number
  readonly weightsBytes: number
  readonly kvBytesPerToken: number
  /** The model's own window; the fit never exceeds it. */
  readonly maxWindow: number
}

export type WindowFit =
  | { readonly fits: true; readonly ctxSize: number }
  /** Too little memory for MIN_FIT_WINDOW; `ctxSize` is that minimum, and loading may fail. */
  | { readonly fits: false; readonly ctxSize: number }

export function fitWindow(input: WindowFitInput): WindowFit {
  const spare = (input.freeBytes - input.weightsBytes - FIT_RESERVE_BYTES) * FIT_CACHE_SHARE
  const tokens = input.kvBytesPerToken > 0 ? spare / input.kvBytesPerToken : Number.POSITIVE_INFINITY
  const window = Math.min(input.maxWindow, Math.floor(tokens / WINDOW_STEP) * WINDOW_STEP)
  if (window < MIN_FIT_WINDOW) return { fits: false, ctxSize: Math.min(MIN_FIT_WINDOW, input.maxWindow) }
  return { fits: true, ctxSize: window }
}

/**
 * Memory free for a model now: an NVIDIA GPU's free VRAM, macOS's free,
 * inactive, speculative and purgeable pages, or Linux's MemAvailable.
 * Undefined when it cannot be read (the caller then says so and uses the
 * fixed window).
 */
export async function freeMemoryBytes(
  hw: HardwareProfile,
  run: (argv: string[]) => Promise<CommandResult>,
): Promise<number | undefined> {
  if (hw.gpu.vendor === 'nvidia') {
    const smi = await tryRun(run, ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'])
    const free = smi?.split('\n').map(line => Number.parseInt(line.trim(), 10)).filter(value => Number.isFinite(value) && value > 0)
    return free !== undefined && free.length > 0 ? Math.max(...free) * MIB : undefined
  }
  if (hw.platform === 'darwin') {
    const text = await tryRun(run, ['vm_stat'])
    return text === undefined ? undefined : parseVmStat(text)
  }
  if (hw.platform === 'linux') {
    const text = await tryRun(run, ['cat', '/proc/meminfo'])
    const match = text === undefined ? undefined : /^MemAvailable:\s+(\d+)\s+kB$/m.exec(text)
    return match ? Number(match[1]) * 1024 : undefined
  }
  return undefined
}

/** Free, inactive, speculative and purgeable pages from `vm_stat`, in bytes. */
export function parseVmStat(text: string): number | undefined {
  const pageSize = /page size of (\d+) bytes/.exec(text)
  if (!pageSize) return undefined
  let pages = 0
  let found = false
  for (const name of ['free', 'inactive', 'speculative', 'purgeable']) {
    const match = new RegExp(`^Pages ${name}:\\s+(\\d+)\\.`, 'm').exec(text)
    if (match) {
      pages += Number(match[1])
      found = true
    }
  }
  return found ? pages * Number(pageSize[1]) : undefined
}

async function tryRun(run: (argv: string[]) => Promise<CommandResult>, argv: string[]): Promise<string | undefined> {
  try {
    const result = await run(argv)
    return result.exitCode === 0 && result.stdout.trim() !== '' ? result.stdout : undefined
  } catch {
    return undefined
  }
}
