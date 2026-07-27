// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/**
 * Startup keystroke capture.
 *
 * Between process start and the first renderer frame the child performs two
 * sequential native-backed dynamic imports and clears the screen. Anything the
 * user types in that window is echoed by the terminal, wiped by the clear, and
 * then dropped — nobody reads stdin. Capturing those bytes here lets them seed
 * the composer instead of disappearing.
 *
 * The reducer is pure so the whole byte-to-text translation is testable without
 * a tty; the listener wrapper below is the only part that touches stdin.
 */

const ESC = '\x1b'
const BEL = '\x07'
const ETX = '\x03'
const BACKSPACE = '\x08'
const DEL = '\x7f'

/**
 * Bound on the seeded text. Startup typing is a sentence at most, while a
 * paste or an escape flood arriving in this window could be megabytes — a
 * composer seeded with that much text stalls the first frames it renders.
 */
const MAX_EARLY_INPUT_CHARS = 4096

/**
 * Bound on the carry buffer. A never-terminated OSC would otherwise accumulate
 * every subsequent byte as an unfinished escape sequence and swallow all real
 * typing after it.
 */
const MAX_PENDING_CHARS = 256

const DECODER = new TextDecoder()

const SEGMENTER =
  typeof Intl.Segmenter === 'function' ? new Intl.Segmenter(undefined, { granularity: 'grapheme' }) : null

export interface EarlyInputState {
  /** True once Ctrl-C was seen. Raw mode suppresses the tty's own SIGINT. */
  interrupt: boolean
  /** Escape-sequence bytes split across a chunk boundary, carried forward. */
  pending: string
  /** Typed text with escape sequences and control bytes already removed. */
  text: string
}

export const EMPTY_EARLY_INPUT: EarlyInputState = { interrupt: false, pending: '', text: '' }

interface EscapeScan {
  /** Characters the sequence occupies, valid only when `incomplete` is false. */
  consumed: number
  /** The chunk ended mid-sequence; the caller must carry the tail forward. */
  incomplete: boolean
}

/** Measure the escape sequence starting at `start` (which must hold an ESC). */
function scanEscape(input: string, start: number): EscapeScan {
  const next = input[start + 1]

  if (next === undefined) {
    return { consumed: 0, incomplete: true }
  }

  // CSI — parameters and intermediates until a final byte in 0x40..0x7e.
  if (next === '[') {
    for (let i = start + 2; i < input.length; i++) {
      const code = input.charCodeAt(i)

      if (code >= 0x40 && code <= 0x7e) {
        return { consumed: i - start + 1, incomplete: false }
      }
    }

    return { consumed: 0, incomplete: true }
  }

  // OSC/DCS/SOS/PM/APC — string sequences terminated by BEL or ST (ESC \).
  if (next === ']' || next === 'P' || next === 'X' || next === '^' || next === '_') {
    for (let i = start + 2; i < input.length; i++) {
      const ch = input[i]

      if (ch === BEL) {
        return { consumed: i - start + 1, incomplete: false }
      }

      if (ch === ESC) {
        if (input[i + 1] === undefined) {
          return { consumed: 0, incomplete: true }
        }

        if (input[i + 1] === '\\') {
          return { consumed: i - start + 2, incomplete: false }
        }
      }
    }

    return { consumed: 0, incomplete: true }
  }

  // SS3 — ESC O plus exactly one key byte (arrows in application cursor mode).
  if (next === 'O') {
    return input[start + 2] === undefined ? { consumed: 0, incomplete: true } : { consumed: 3, incomplete: false }
  }

  // Two-character escapes, including Alt-modified keys.
  return { consumed: 2, incomplete: false }
}

/**
 * Remove one user-perceived character. Code-unit slicing would split surrogate
 * pairs and tear emoji or combining marks apart, leaving unrenderable text in
 * the composer.
 */
export function dropLastGrapheme(value: string): string {
  if (!value) {
    return ''
  }

  if (SEGMENTER) {
    let start = 0

    for (const segment of SEGMENTER.segment(value)) {
      start = segment.index
    }

    return value.slice(0, start)
  }

  const last = value.codePointAt(value.length - 2)
  const isSurrogatePair = last !== undefined && last > 0xffff

  return value.slice(0, value.length - (isSurrogatePair ? 2 : 1))
}

const append = (value: string, addition: string): string =>
  value.length >= MAX_EARLY_INPUT_CHARS ? value : value + addition

/** Fold one stdin chunk into the accumulated startup input. Pure. */
export function reduceEarlyInput(previous: EarlyInputState, chunk: string | Uint8Array): EarlyInputState {
  const input = previous.pending + (typeof chunk === 'string' ? chunk : DECODER.decode(chunk))
  let text = previous.text
  let interrupt = previous.interrupt
  let pending = ''
  let i = 0

  while (i < input.length) {
    const ch = input[i]!

    if (ch === ESC) {
      const scan = scanEscape(input, i)

      if (scan.incomplete) {
        pending = input.slice(i)
        break
      }

      i += scan.consumed
      continue
    }

    i += 1

    if (ch === ETX) {
      interrupt = true
      continue
    }

    if (ch === '\r') {
      // A CRLF pair is one Enter press, not two blank lines.
      if (input[i] === '\n') {
        i += 1
      }

      text = append(text, '\n')
      continue
    }

    if (ch === '\n' || ch === '\t') {
      text = append(text, ch)
      continue
    }

    if (ch === BACKSPACE || ch === DEL) {
      text = dropLastGrapheme(text)
      continue
    }

    if (ch.charCodeAt(0) < 0x20) {
      continue
    }

    text = append(text, ch)
  }

  return { interrupt, pending: pending.length > MAX_PENDING_CHARS ? '' : pending, text }
}

/** Text worth seeding the composer with; Ctrl-C discards whatever preceded it. */
export function finalizeEarlyInputText(state: EarlyInputState): string {
  return state.interrupt ? '' : state.text.trimEnd()
}

type DataListener = (chunk: string | Uint8Array) => void

/** Structural stdin so tests can drive capture without a real tty. */
export interface EarlyInputStdin {
  isTTY?: boolean
  off: (event: 'data', listener: DataListener) => unknown
  on: (event: 'data', listener: DataListener) => unknown
  setRawMode?: (mode: boolean) => unknown
}

export interface EarlyInputCaptureOptions {
  /** Called once on Ctrl-C; raw mode means the tty will not deliver SIGINT. */
  onInterrupt?: () => void
  stdin?: EarlyInputStdin
}

export interface EarlyInputCapture {
  /** Detach the listener and return the reviewable text. */
  stop: () => string
}

/**
 * Read stdin until the renderer is about to take over.
 *
 * `stop` deliberately leaves raw mode enabled: the renderer claims stdin
 * immediately afterwards and toggling the mode back would flicker the cursor
 * and re-enable echo for the remainder of startup.
 */
export function startEarlyInputCapture(options: EarlyInputCaptureOptions = {}): EarlyInputCapture {
  const stdin = options.stdin ?? (process.stdin as unknown as EarlyInputStdin)

  if (!stdin.isTTY) {
    return { stop: () => '' }
  }

  let state = EMPTY_EARLY_INPUT
  let stopped = false
  let interruptReported = false

  const listener: DataListener = chunk => {
    state = reduceEarlyInput(state, chunk)

    if (state.interrupt && !interruptReported) {
      interruptReported = true
      options.onInterrupt?.()
    }
  }

  try {
    stdin.setRawMode?.(true)
  } catch {
    // Without raw mode the tty still buffers the line; capture what it gives.
  }

  stdin.on('data', listener)

  return {
    stop: () => {
      if (!stopped) {
        stopped = true

        try {
          stdin.off('data', listener)
        } catch {
          // A closed stdin cannot deliver more data anyway.
        }
      }

      return finalizeEarlyInputText(state)
    }
  }
}

let seededEarlyInput = ''

/** Hand captured startup text to the composer across the renderer import. */
export function setEarlyInputText(text: string): void {
  seededEarlyInput = text
}

/**
 * Consume the captured text. Reading clears it so a later composer remount
 * cannot resurrect keystrokes the user already saw and edited.
 */
export function takeEarlyInputText(): string {
  const text = seededEarlyInput

  seededEarlyInput = ''

  return text
}
