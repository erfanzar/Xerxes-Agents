// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { getHeapStatistics } from 'node:v8'

import { type HeapDumpResult, performHeapDump } from './memory.js'

export type MemoryLevel = 'critical' | 'high' | 'normal'

export interface MemorySnapshot {
  heapUsed: number
  level: MemoryLevel
  rss: number
}

export interface MemoryMonitorOptions {
  criticalBytes?: number
  highBytes?: number
  intervalMs?: number
  onCritical?: (snap: MemorySnapshot, dump: HeapDumpResult | null) => void
  onHigh?: (snap: MemorySnapshot, dump: HeapDumpResult | null) => void
  // Fired ONCE when heap growth looks abnormal while still far below the
  // critical exit threshold — the regime where the TUI used to die silently
  // A runaway render tree can OOM at a few hundred MB, well under
  // criticalBytes, so onCritical never fires and the gateway death otherwise
  // showed up only as a bare `stdin EOF`). A visible warning here makes that
  // class of death diagnosable instead of silent.
  onWarn?: (snap: MemorySnapshot) => void
  warnBytes?: number
}

const GB = 1024 ** 3
const MB = 1024 ** 2

// Resolve the exit / dump thresholds RELATIVE to the actual V8 heap ceiling
// (--max-old-space-size, 8GB for the TUI) instead of hardcoding 2.5GB. The old
// constant killed the process — and silently closed the gateway's stdin — at
// ~31% of an 8GB ceiling, treating a normal long-session heap as an OOM. We now
// exit only when genuinely near the ceiling (critical ~88%, high ~70%), and
// clamp to sane floors/ceilings so a tiny --max-old-space-size can't drive the
// thresholds below the warn watermark. Callers may still override explicitly.
function resolveThresholds(criticalBytes?: number, highBytes?: number) {
  let limit = 0
  try {
    limit = getHeapStatistics().heap_size_limit || 0
  } catch {
    limit = 0
  }

  // Fall back to the historical 8GB ceiling if V8 doesn't report one.
  const ceiling = limit > 0 ? limit : 8 * GB
  const critical = criticalBytes ?? Math.max(2 * GB, Math.round(ceiling * 0.88))
  const high = highBytes ?? Math.max(1 * GB, Math.min(critical - 256 * MB, Math.round(ceiling * 0.7)))

  return { critical, high }
}

// Keep this import deferred: the entry point intentionally initializes
// @opentui/core before @opentui/react, while this monitor is loaded earlier to
// cover startup. Cache release is only needed after a high-memory timer tick.
let releaseCaches: ((level: 'all' | 'half') => unknown) | null = null
let releaseCachesPromise: Promise<(level: 'all' | 'half') => unknown> | null = null

async function ensureReleaseCaches(): Promise<(level: 'all' | 'half') => unknown> {
  if (releaseCaches) {
    return releaseCaches
  }

  releaseCachesPromise ??= import('./terminalRuntime.opentui.js')
    .then(mod => {
      releaseCaches = mod.releaseTerminalCaches

      return releaseCaches
    })
    .catch(err => {
      releaseCachesPromise = null
      throw err
    })

  return releaseCachesPromise
}

export function startMemoryMonitor({
  criticalBytes,
  highBytes,
  intervalMs = 10_000,
  onCritical,
  onHigh,
  onWarn,
  warnBytes = 600 * MB
}: MemoryMonitorOptions = {}): () => void {
  const { critical, high } = resolveThresholds(criticalBytes, highBytes)
  const dumped = new Set<Exclude<MemoryLevel, 'normal'>>()
  const inFlight = new Set<Exclude<MemoryLevel, 'normal'>>()

  // Early-warning state (#34095): the silent-death regime is BELOW `high`, so
  // the level machine above never sees it. Track the previous sample and fire
  // onWarn at most once when heap both crosses a modest absolute floor AND is
  // climbing steeply (≥150MB between 10s ticks) — the signature of a render-
  // tree blowup — so the user gets a visible heads-up before Node OOMs under
  // the exit threshold. Re-armed only after heap falls back below the floor.
  // `lastHeap < 0` marks the un-seeded first sample so a cold start that opens
  // already-high can't be mistaken for sudden growth (growth = current - last).
  let lastHeap = -1
  let warned = false
  const WARN_GROWTH_STEP = 150 * MB

  // Cooldown prevents repeated auto dumps when heap oscillates around the
  // threshold (issue #21767). `dumped` alone is not enough — it clears on
  // every transition back to `normal`.
  const cooldownRaw = process.env.XERXES_AUTO_HEAPDUMP_COOLDOWN_MS?.trim()
  const cooldownParsed = cooldownRaw ? Number(cooldownRaw) : NaN
  const cooldownMs = Number.isFinite(cooldownParsed) && cooldownParsed >= 0 ? cooldownParsed : 600_000
  let lastAutoDumpAt = 0

  const tick = async () => {
    const { heapUsed, rss } = process.memoryUsage()

    // Sub-threshold abnormal-growth warning. Skip on the first (un-seeded)
    // sample — we need a prior reading to measure a delta against.
    if (heapUsed < high && lastHeap >= 0) {
      if (!warned && heapUsed >= warnBytes && heapUsed - lastHeap >= WARN_GROWTH_STEP) {
        warned = true
        onWarn?.({ heapUsed, level: 'normal', rss })
      } else if (heapUsed < warnBytes) {
        warned = false
      }
    }
    lastHeap = heapUsed

    const level: MemoryLevel = heapUsed >= critical ? 'critical' : heapUsed >= high ? 'high' : 'normal'

    if (level === 'normal') {
      dumped.clear()
      return
    }

    if (dumped.has(level) || inFlight.has(level)) {
      return
    }

    if (Date.now() - lastAutoDumpAt < cooldownMs) {
      return
    }

    inFlight.add(level)
    lastAutoDumpAt = Date.now()

    // Release renderer-owned caches before dump/exit — half on 'high'
    // (recoverable), full on 'critical' (post-dump RSS reduction).
    try {
      try {
        const release = await ensureReleaseCaches()
        release(level === 'critical' ? 'all' : 'half')
      } catch {
        // Best-effort: if the dynamic import fails for any reason we still
        // continue to the heap dump below so the user gets diagnostics.
      }

      dumped.add(level)
      const dump = await performHeapDump(level === 'critical' ? 'auto-critical' : 'auto-high').catch(() => null)
      const snap: MemorySnapshot = { heapUsed, level, rss }

      ;(level === 'critical' ? onCritical : onHigh)?.(snap, dump)
    } finally {
      inFlight.delete(level)
    }
  }

  const handle = setInterval(() => void tick(), intervalMs)

  handle.unref?.()

  return () => clearInterval(handle)
}
