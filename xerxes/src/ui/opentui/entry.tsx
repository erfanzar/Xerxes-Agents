// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// Canonical TUI entry point. It owns process/terminal lifecycle and mounts the
// production OpenTUI application through the native renderer.
import '../lib/forceTruecolor.js'

import { writeSync } from 'node:fs'

import { INLINE_MODE, TERMUX_TUI_MODE } from '../config/env.js'
import { $uiSessionId, getUiState } from '../app/uiStore.js'
import { GatewayClient } from '../gatewayClient.js'
import { formatExitHint } from '../lib/exitHint.js'
import { setupGracefulExit } from '../lib/gracefulExit.js'
import { formatBytes, type HeapDumpResult, performHeapDump } from '../lib/memory.js'
import { type MemorySnapshot, startMemoryMonitor } from '../lib/memoryMonitor.js'
import { recordParentLifecycle } from '../lib/parentLog.js'
import { resetTerminalModes } from '../lib/terminalModes.js'
import { DEFAULT_THEME, themeForMode } from '../theme.js'
import {
  clearActiveRenderer,
  destroyActiveRenderer,
  getActiveRenderer,
  installRendererRecovery,
  setActiveRenderer
} from './rendererSingleton.js'

if (!process.stdin.isTTY) {
  console.log('xerxes-tui: no TTY')
  process.exit(0)
}

resetTerminalModes()

let exitHintWritten = false
let lastKnownSessionId: null | string = null

$uiSessionId.subscribe(sessionId => {
  if (sessionId) {
    lastKnownSessionId = sessionId
  }
})

function writeExitHint() {
  if (exitHintWritten || !process.stdout.isTTY) {
    return
  }

  exitHintWritten = true

  try {
    writeSync(1, `\n${formatExitHint({ sessionId: getUiState().sid ?? lastKnownSessionId })}\n`)
  } catch {
    // The terminal can disappear before process exit handlers finish.
  }
}

process.on('exit', () => {
  destroyActiveRenderer()
  writeExitHint()
})

if (TERMUX_TUI_MODE) {
  process.stdout.write('\n')
} else {
  process.stdout.write('\x1b[2J\x1b[H\x1b[3J')
}

const gw = new GatewayClient({
  projectDir: process.env.XERXES_PROJECT_DIR || process.env.XERXES_CWD
})

const dumpNotice = (snap: MemorySnapshot, dump: HeapDumpResult | null) =>
  `xerxes-tui: ${snap.level} memory (${formatBytes(snap.heapUsed)}) — auto heap dump → ${dump?.heapPath ?? dump?.diagPath ?? '(failed)'}\n`

const notifyMemoryDiagnostic = (message: string) => {
  try {
    getActiveRenderer()?.triggerNotification(message.trim(), 'Xerxes memory')
  } catch {
    // Diagnostics are already persisted in the lifecycle log/sidecar. Never
    // fall back to a raw terminal write while OpenTUI owns the framebuffer.
  }
}

setupGracefulExit({
  cleanups: [
    () => {
      destroyActiveRenderer()

      return gw.kill('graceful-exit-cleanup')
    }
  ],
  onError: (scope, err) => {
    const message = err instanceof Error ? `${err.name}: ${err.message}\n${err.stack ?? ''}` : String(err)

    recordParentLifecycle(`${scope}: ${message.split('\n')[0]?.slice(0, 400) ?? ''}`)
    process.stderr.write(`xerxes-tui lifecycle ${scope}: ${message.slice(0, 2000)}\n`)
    destroyActiveRenderer()
  },
  onSignal: signal => {
    recordParentLifecycle(`graceful-exit received signal=${signal} → killing gateway`)
    destroyActiveRenderer()
    process.stderr.write(`xerxes-tui lifecycle: received ${signal}\n`)
  }
})

const stopMemoryMonitor = startMemoryMonitor({
  onCritical: (snap, dump) => {
    recordParentLifecycle(
      `memory-critical process.exit(137) heap=${formatBytes(snap.heapUsed)} rss=${formatBytes(snap.rss)} dump=${dump?.heapPath ?? 'failed'}`
    )
    destroyActiveRenderer()
    process.stderr.write(
      `xerxes-tui lifecycle: memory critical exit heap=${formatBytes(snap.heapUsed)} rss=${formatBytes(snap.rss)}\n`
    )
    process.stderr.write(dumpNotice(snap, dump))
    process.stderr.write('xerxes-tui: exiting to avoid OOM; restart to recover\n')
    process.exit(137)
  },
  onHigh: (snap, dump) => {
    const message = dumpNotice(snap, dump)

    recordParentLifecycle(message.trim())
    notifyMemoryDiagnostic(message)
  },
  onWarn: snap => {
    const breadcrumb = `memory-warning fast heap growth heap=${formatBytes(snap.heapUsed)} rss=${formatBytes(snap.rss)}`
    const message = `Xerxes heap is climbing fast (${formatBytes(snap.heapUsed)}); a large tool output or long session may be straining memory.`

    recordParentLifecycle(breadcrumb)
    notifyMemoryDiagnostic(message)
  }
})

if (process.env.XERXES_HEAPDUMP_ON_START === '1') {
  void performHeapDump('manual')
}

process.on('beforeExit', () => {
  stopMemoryMonitor()
  destroyActiveRenderer()
})

// Sequential, not Promise.all — @opentui/react's module-init touches
// @opentui/core state that must already be evaluated, and concurrent
// dynamic import() resolution order isn't guaranteed to respect that.
const { createCliRenderer } = await import('@opentui/core')
const { createRoot } = await import('@opentui/react')
const { AppOpenTui } = await import('./app.js')

const renderer = await createCliRenderer({
  backgroundColor: themeForMode(DEFAULT_THEME, 'code').color.statusBg,
  exitOnCtrlC: false,
  // Xerxes owns SIGINT/SIGTERM/SIGHUP and sequences renderer teardown before
  // gateway cleanup. A second OpenTUI signal handler can race that lifecycle.
  exitSignals: [],
  // INLINE_MODE stays in the main screen so terminal scrollback is preserved.
  screenMode: INLINE_MODE ? 'main-screen' : 'alternate-screen',
  useKittyKeyboard: { alternateKeys: true, disambiguate: true }
})

// Stash for imperative controller call sites.
setActiveRenderer(renderer)
const stopRendererRecovery = installRendererRecovery(renderer)

renderer.once('destroy', () => {
  stopRendererRecovery()
  clearActiveRenderer(renderer)
})

createRoot(renderer).render(<AppOpenTui gw={gw} />)
