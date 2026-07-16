// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
// OpenTUI-backed terminal services used by the renderer-independent
// controller. Keeping this boundary explicit prevents controller hooks from
// importing renderer internals or a compatibility package name.
import { EventEmitter } from 'node:events'
import { useEffect } from 'react'

import type { KeyEvent } from '@opentui/core'
import { useKeyboard } from '@opentui/react'

import {
  destroyActiveRenderer,
  forceRendererRepaint,
  getActiveRenderer
} from '../opentui/rendererSingleton.js'

// ── Pure utilities ───────────────────────────────────────────────────────
// Both are native under Bun (this app only ever runs under Bun).

export const stringWidth: (s: string) => number =
  typeof Bun !== 'undefined' && typeof Bun.stringWidth === 'function' ? Bun.stringWidth : s => s.length

const naiveWrap = (input: string, columns: number): string =>
  input
    .split('\n')
    .map(line => {
      const out: string[] = []

      for (let i = 0; i < line.length; i += columns) {
        out.push(line.slice(i, i + columns))
      }

      return out.join('\n') || line
    })
    .join('\n')

export const wrapAnsi: (
  input: string,
  columns: number,
  options?: { hard?: boolean; wordWrap?: boolean; trim?: boolean }
) => string = typeof Bun !== 'undefined' && typeof Bun.wrapAnsi === 'function' ? Bun.wrapAnsi : naiveWrap

export function isXtermJs(): boolean {
  return process.env.TERM_PROGRAM === 'vscode' || Boolean(process.env.VSCODE_PID)
}

export const scrollFastPathStats = {
  captured: 0,
  taken: 0,
  declined: { noPrevScreen: 0, heightDeltaMismatch: 0, other: 0 }
}

// ── Renderer-backed imperative helpers ─────────────────────────────────────

export function forceRedraw(_stdout?: NodeJS.WriteStream): boolean {
  return forceRendererRepaint()
}

export function releaseTerminalCaches(_level?: 'all' | 'half') {
  // OpenTUI's native core owns its own buffers; there are no JS-side
  // line-wrap/width caches to evict. Return the zeroed shape the caller
  // logs so nothing downstream divides by it.
  return { lineWidth: 0, slice: 0, width: 0, wrap: 0 }
}

interface SuspendableRenderer {
  requestRender: () => void
  resume: () => void
  suspend: () => void
}

let terminalSuspensionDepth = 0
let suspendedRenderer: SuspendableRenderer | null = null

export async function runWithTerminalSuspended(
  renderer: SuspendableRenderer | null,
  run: () => Promise<void>
): Promise<void> {
  const ownsSuspension = terminalSuspensionDepth === 0

  terminalSuspensionDepth++

  if (ownsSuspension) {
    suspendedRenderer = renderer

    try {
      renderer?.suspend()
    } catch (error) {
      terminalSuspensionDepth--
      suspendedRenderer = null
      throw error
    }
  }

  let runFailed = false

  try {
    await run()
  } catch (error) {
    runFailed = true
    throw error
  } finally {
    terminalSuspensionDepth--

    if (terminalSuspensionDepth === 0) {
      const resumeTarget = suspendedRenderer
      let restoreError: unknown
      let restoreFailed = false

      suspendedRenderer = null

      try {
        resumeTarget?.resume()
      } catch (error) {
        restoreError = error
        restoreFailed = true
      }

      try {
        resumeTarget?.requestRender()
      } catch (error) {
        if (!restoreFailed) {
          restoreError = error
          restoreFailed = true
        }
      }

      if (restoreFailed && !runFailed) {
        throw restoreError
      }
    }
  }
}

export async function withTerminalSuspended(run: () => Promise<void>): Promise<void> {
  await runWithTerminalSuspended(getActiveRenderer(), run)
}

// ── Hooks ──────────────────────────────────────────────────────────────────

export function useApp(): { readonly exit: (error?: Error) => void } {
  return {
    exit: (error?: Error) => {
      destroyActiveRenderer()
      process.exit(error ? 1 : 0)
    }
  }
}

// OpenTUI's native core owns the screen buffer and tracks the cursor itself.
// The reused controller writes raw control sequences straight to stdout
// (BRACKET_PASTE_ON/OFF in useMainApp). Under OpenTUI an out-of-band write
// desyncs the native renderer's cell/cursor
// model and corrupts subsequent frames (dropped spaces, bled glyphs). This
// proxy forwards columns/rows/isTTY/on/off/once (layout + resize still work)
// but swallows .write() so the controller can't scramble OpenTUI's buffer.
// OpenTUI enables its own bracketed-paste handling, so nothing is lost.
const guardedStdout = new Proxy(process.stdout, {
  get(target, prop, receiver) {
    if (prop === 'write') {
      return () => true
    }

    const value = Reflect.get(target, prop, receiver)

    return typeof value === 'function' ? value.bind(target) : value
  }
}) as NodeJS.WriteStream

export function useStdout(): { readonly stdout?: NodeJS.WriteStream } {
  return { stdout: guardedStdout }
}

export function useTerminalFocus(): boolean {
  // Focus-driven cursor dimming is cosmetic; OpenTUI's useFocus only
  // signals focus-GAINED (no blur), so a live boolean isn't cleanly
  // derivable. Assume focused — matches how the app behaves in practice.
  return true
}

export function useTerminalTitle(title: string | null): void {
  useEffect(() => {
    if (title) {
      getActiveRenderer()?.setTerminalTitle(title)
    }
  }, [title])
}

interface SelectionService {
  captureScrolledRows: (firstRow: number, lastRow: number, side: 'above' | 'below') => void
  clearSelection: () => void
  copySelection: () => Promise<string>
  copySelectionNoClear: () => Promise<string>
  getState: () => unknown
  hasSelection: () => boolean
  moveFocus: (move: unknown) => void
  setSelectionBgColor: (color: string) => void
  shiftAnchor: (deltaRows: number, minRow: number, maxRow: number) => void
  shiftSelection: (deltaRows: number, minRow: number, maxRow: number) => void
  subscribe: (listener: () => void) => () => void
  version: () => number
}

const NOOP_SELECTION: SelectionService = {
  copySelection: async () => '',
  copySelectionNoClear: async () => '',
  clearSelection: () => {},
  hasSelection: () => false,
  getState: () => null,
  version: () => 0,
  subscribe: () => () => {},
  shiftAnchor: () => {},
  shiftSelection: () => {},
  moveFocus: () => {},
  captureScrolledRows: () => {},
  setSelectionBgColor: () => {}
}

export function useSelection(): SelectionService {
  // OpenTUI owns native terminal selection. This stable no-op object keeps
  // the controller from installing a competing selection implementation.
  return NOOP_SELECTION
}

export function useHasSelection(): boolean {
  return false
}

const stdinEmitter = new EventEmitter()

export function useStdin() {
  // OpenTUI owns stdin raw mode; the controller only reads `querier`
  // (OSC-52 clipboard, which safely no-ops on null) and holds `stdin` for
  // paste flows. setRawMode is a no-op so the controller can't fight
  // OpenTUI's own input handling.
  return {
    stdin: process.stdin,
    setRawMode: () => {},
    isRawModeSupported: true,
    exitOnCtrlC: false,
    inputEmitter: stdinEmitter,
    querier: null
  }
}

interface ControllerKey {
  alt: boolean
  ctrl: boolean
  downArrow: boolean
  end: boolean
  escape: boolean
  fn: boolean
  home: boolean
  leftArrow: boolean
  meta: boolean
  pageDown: boolean
  pageUp: boolean
  return: boolean
  rightArrow: boolean
  shift: boolean
  super: boolean
  tab: boolean
  backspace: boolean
  delete: boolean
  upArrow: boolean
  wheelDown: boolean
  wheelUp: boolean
}

const ENTER_NAMES = new Set(['enter', 'kpenter', 'linefeed', 'return'])
const NON_PRINTABLE_NAMES = new Set([
  'backspace',
  'delete',
  'down',
  'end',
  'escape',
  'home',
  'left',
  'pagedown',
  'pageup',
  'right',
  'tab',
  'up',
  'wheeldown',
  'wheelup',
  ...ENTER_NAMES
])

/**
 * Recreate the controller's normalized printable input value.
 *
 * OpenTUI preserves the terminal control byte in `sequence` for modified
 * keys (Ctrl+C is `\x03`) while exposing the useful character as `name`
 * (`c`). The controller intentionally matches shortcuts against that decoded
 * character, so passing the raw sequence silently disables every Ctrl/Cmd/
 * Alt shortcut.  Ordinary printable input stays sequence-backed so case,
 * Unicode, and composed characters remain intact.
 */
function controllerInputFromKeyEvent(event: KeyEvent): string {
  const name = event.name ?? ''

  if (NON_PRINTABLE_NAMES.has(name)) {
    return ''
  }

  if (name === 'space') {
    return ' '
  }

  if (event.ctrl || event.meta || event.option || event.super) {
    // OpenTUI already decoded modifyOtherKeys / Kitty keyboard sequences into
    // `name`; using it also avoids leaking an ESC-prefixed Alt sequence.
    return name
  }

  const sequence = event.sequence ?? ''

  // Never forward terminal control sequences as composer text. Keep all
  // printable Unicode sequences, including surrogate-pair emoji.
  return sequence && sequence[0] !== '\x1b' && sequence.codePointAt(0)! >= 0x20 ? sequence : ''
}

function controllerKeyFromKeyEvent(event: KeyEvent): ControllerKey {
  const name = event.name ?? ''
  const superKey = Boolean(event.super)
  const optionKey = Boolean(event.option)

  return {
    // OpenTUI reports Alt/Option through `option` (and commonly `meta`). Keep
    // the explicit alias because Xerxes' configurable voice shortcut matcher
    // distinguishes Alt from Super.
    alt: optionKey || (Boolean(event.meta) && !superKey),
    backspace: name === 'backspace',
    ctrl: Boolean(event.ctrl),
    delete: name === 'delete',
    downArrow: name === 'down',
    end: name === 'end',
    escape: name === 'escape',
    fn: Boolean((event as KeyEvent & { fn?: boolean }).fn),
    home: name === 'home',
    leftArrow: name === 'left',
    // Treat Option and Escape as meta for compatibility with terminals that
    // encode Alt as an ESC prefix.
    meta: Boolean(event.meta) || optionKey || name === 'escape',
    pageDown: name === 'pagedown',
    pageUp: name === 'pageup',
    return: ENTER_NAMES.has(name),
    rightArrow: name === 'right',
    shift: Boolean(event.shift),
    super: superKey,
    tab: name === 'tab',
    upArrow: name === 'up',
    wheelDown: name === 'wheeldown',
    wheelUp: name === 'wheelup'
  }
}

/**
 * Global controller keys must not subsequently mutate the focused native
 * textarea. Do not blanket-block modified keys: OpenTUI owns useful editor
 * bindings such as Ctrl+A/E/W/K and Meta+Left/Right.
 */
function controllerOwnsTextareaKey(input: string, key: ControllerKey): boolean {
  // Escape does not mutate a textarea, and prompt/modal components need to
  // observe it after the controller listener. Preventing its default here
  // stops OpenTUI's later keyboard listeners and makes Esc unable to deny an
  // approval or close a component-owned overlay.
  if (key.tab || key.pageUp || key.pageDown || key.wheelUp || key.wheelDown) {
    return true
  }

  if (key.shift && (key.upArrow || key.downArrow)) {
    return true
  }

  const ch = input.toLowerCase()
  const actionModifier = process.platform === 'darwin' ? key.meta || key.super : key.ctrl

  // Ctrl+C and Ctrl+X are always owned by the global interrupt/session
  // handlers on every platform. Copy variants of C are global as well.
  if (ch === 'c' && (key.ctrl || key.meta || key.super)) {
    return true
  }

  if (ch === 'x' && key.ctrl) {
    return true
  }

  // These are unconditional action shortcuts in useInputHandlers. Alt+G is
  // also the documented VS Code/Cursor fallback for opening $EDITOR.
  if ((ch === 'd' || ch === 'l') && actionModifier) {
    return true
  }

  return ch === 'g' && (actionModifier || key.meta)
}

export function useInput(
  handler: (input: string, key: ControllerKey, event: unknown) => void,
  options?: { readonly isActive?: boolean }
): void {
  const isActive = options?.isActive ?? true

  useKeyboard(event => {
    if (!isActive) {
      return
    }

    const input = controllerInputFromKeyEvent(event)
    const key = controllerKeyFromKeyEvent(event)

    if (controllerOwnsTextareaKey(input, key)) {
      // Global OpenTUI listeners run before a focused renderable. Preventing
      // the default skips TextareaRenderable.handleKeyPress while still
      // allowing sibling global listeners (approval/confirm/clarify) to see
      // the key; stopPropagation() would incorrectly starve those handlers.
      event.preventDefault()
    }

    handler(input, key, event)
  })
}
