// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
interface SetupOptions {
  cleanups?: (() => Promise<void> | void)[]
  failsafeMs?: number
  onError?: (scope: 'uncaughtException' | 'unhandledRejection', err: unknown) => void
  onSignal?: (signal: NodeJS.Signals) => void
}

const SIGNAL_EXIT_CODE: Record<'SIGHUP' | 'SIGINT' | 'SIGTERM', number> = {
  SIGHUP: 129,
  SIGINT: 130,
  SIGTERM: 143
}

let wired = false

export function setupGracefulExit({ cleanups = [], failsafeMs = 4000, onError, onSignal }: SetupOptions = {}) {
  if (wired) {
    return
  }

  wired = true

  let shuttingDown = false

  const exit = (code: number, signal?: NodeJS.Signals) => {
    if (shuttingDown) {
      return
    }

    shuttingDown = true

    if (signal) {
      onSignal?.(signal)
    }

    setTimeout(() => process.exit(code), failsafeMs).unref?.()

    void Promise.allSettled(cleanups.map(fn => Promise.resolve().then(fn))).finally(() => process.exit(code))
  }

  const fatal = (scope: 'uncaughtException' | 'unhandledRejection', err: unknown) => {
    // Registering either process event suppresses the runtime's default fatal
    // exit. Logging alone would leave a half-failed React/OpenTUI tree alive,
    // often as a blank terminal that still accepts input. Report best-effort,
    // then use the same bounded cleanup path as signals.
    try {
      onError?.(scope, err)
    } catch {
      // A broken logger must not prevent terminal restoration and shutdown.
    } finally {
      exit(1)
    }
  }

  for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP'] as const) {
    process.on(sig, () => exit(SIGNAL_EXIT_CODE[sig], sig))
  }

  process.on('uncaughtException', err => fatal('uncaughtException', err))
  process.on('unhandledRejection', reason => fatal('unhandledRejection', reason))
}
