// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { writeSync } from 'node:fs'

export const TERMINAL_MODE_RESET =
  "\x1b[0'z" + // DEC locator reporting
  "\x1b[0'{" + // selectable locator events
  '\x1b[?2026l' + // synchronized output
  '\x1b[?2029l' + // passive mouse
  '\x1b[?1016l' + // SGR-pixels mouse
  '\x1b[?1015l' + // urxvt decimal mouse
  '\x1b[?1006l' + // SGR mouse
  '\x1b[?1005l' + // UTF-8 extended mouse
  '\x1b[?1003l' + // any-motion mouse
  '\x1b[?1002l' + // button-motion mouse
  '\x1b[?1001l' + // highlight mouse
  '\x1b[?1000l' + // click mouse
  '\x1b[?9l' + // X10 mouse
  '\x1b[?1004l' + // focus events
  '\x1b[?2004l' + // bracketed paste
  '\x1b[?1049l' + // alternate screen
  '\x1b[<u' + // kitty keyboard
  '\x1b[>4m' + // modifyOtherKeys
  '\x1b[0m' + // attributes
  '\x1b[?25h' // cursor visible

type ResettableStream = Pick<NodeJS.WriteStream, 'isTTY' | 'write'> & {
  fd?: number
}

export interface SttyResult {
  exitCode: number
  stdout: string
}

export type SttyRunner = (arguments_: readonly string[]) => SttyResult

export interface TerminalWatchdogOptions {
  /** Injectable for tests and hosts whose stdin is not `process.stdin`. */
  stdinIsTTY?: boolean
  /** Injectable so Windows behavior can be covered on a POSIX development host. */
  platform?: NodeJS.Platform
  /** Receives arguments after the `stty` executable name. */
  runStty?: SttyRunner
  /** Escape-protocol cleanup; defaults to the comprehensive reset below. */
  resetModes?: () => boolean
}

export interface TerminalRestoreResult {
  exactStateRestored: boolean
  fallbackStateRestored: boolean
  modesReset: boolean
}

export interface TerminalWatchdog {
  /** Exact opaque value returned by `stty -g`, when capture was possible. */
  readonly snapshot: string | null
  /** Idempotently restore termios first and terminal escape modes last. */
  restore: () => TerminalRestoreResult
}

export function resetTerminalModes(stream: ResettableStream = process.stdout): boolean {
  if (!stream.isTTY) {
    return false
  }

  const fd = typeof stream.fd === 'number' ? stream.fd : stream === process.stdout ? 1 : undefined
  if (fd !== undefined) {
    try {
      writeSync(fd, TERMINAL_MODE_RESET)

      return true
    } catch {
      // Fall through to stream.write for mocked or unusual TTY streams.
    }
  }

  try {
    stream.write(TERMINAL_MODE_RESET)

    return true
  } catch {
    return false
  }
}

const defaultSttyRunner: SttyRunner = arguments_ => {
  const result = Bun.spawnSync(['stty', ...arguments_], {
    stdin: 'inherit',
    stdout: 'pipe',
    stderr: 'ignore'
  })

  return {
    exitCode: result.exitCode,
    stdout: new TextDecoder().decode(result.stdout)
  }
}

const validSttySnapshot = (value: string): string | null => {
  const snapshot = value.trim()

  // `stty -g` is one opaque command argument. Reject control characters and
  // unreasonable output rather than ever treating diagnostic text as state.
  return snapshot && snapshot.length <= 4096 && !/[\0\r\n]/.test(snapshot) ? snapshot : null
}

/**
 * Capture the parent terminal before a TUI child starts and build a one-shot
 * restoration guard. The exact termios snapshot is preferred because `sane`
 * or hard-coded flags would erase a user's custom terminal configuration.
 *
 * Native crashes and SIGKILL cannot run cleanup in the renderer process, so
 * this guard is intentionally designed for the surviving CLI parent. Every
 * restoration path finishes by disabling OpenTUI's escape-protocol modes.
 */
export function createTerminalWatchdog(options: TerminalWatchdogOptions = {}): TerminalWatchdog {
  const stdinIsTTY = options.stdinIsTTY ?? Boolean(process.stdin.isTTY)
  const platform = options.platform ?? process.platform
  const runStty = options.runStty ?? defaultSttyRunner
  const resetModes = options.resetModes ?? (() => resetTerminalModes())
  const canUseStty = stdinIsTTY && platform !== 'win32'
  let snapshot: string | null = null

  if (canUseStty) {
    try {
      const captured = runStty(['-g'])

      if (captured.exitCode === 0) {
        snapshot = validSttySnapshot(captured.stdout)
      }
    } catch {
      // Missing `stty` is expected on some hosts. Escape cleanup still works.
    }
  }

  let restored: TerminalRestoreResult | null = null

  return {
    snapshot,
    restore: () => {
      if (restored) {
        return restored
      }

      let exactStateRestored = false
      let fallbackStateRestored = false
      let modesReset = false

      try {
        if (canUseStty && snapshot) {
          try {
            exactStateRestored = runStty([snapshot]).exitCode === 0
          } catch {
            exactStateRestored = false
          }
        }

        if (canUseStty && !exactStateRestored) {
          try {
            // `sane` explicitly restores canonical input, signals, and echo.
            // Merely applying `-raw` is not sufficient on every supported
            // `stty` implementation after a renderer dies in raw mode.
            fallbackStateRestored = runStty(['sane']).exitCode === 0
          } catch {
            fallbackStateRestored = false
          }
        }
      } finally {
        // Mouse, bracketed paste, alternate-screen, focus, and keyboard modes
        // are independent of termios. This must remain the final operation.
        try {
          modesReset = resetModes()
        } catch {
          modesReset = false
        }
      }

      restored = { exactStateRestored, fallbackStateRestored, modesReset }

      return restored
    }
  }
}

/** Run one child-lifecycle wait with guaranteed parent-terminal restoration. */
export async function withTerminalWatchdog<T>(
  run: () => Promise<T>,
  options: TerminalWatchdogOptions = {}
): Promise<T> {
  const watchdog = createTerminalWatchdog(options)

  try {
    return await run()
  } finally {
    watchdog.restore()
  }
}
