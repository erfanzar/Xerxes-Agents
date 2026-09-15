// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Native recovery stays available even when the renderer cannot paint. */
export function windowRecovery(port: {
  closed: () => boolean
  prompt: (detail: string) => Promise<boolean>
  reload: () => Promise<void>
  report: (error: unknown) => void
}): (detail: string) => Promise<void> {
  let pending = false
  return async detail => {
    if (pending || port.closed()) return
    pending = true
    try {
      if (await port.prompt(detail) && !port.closed()) await port.reload()
    } catch (error) {
      port.report(error)
    } finally {
      pending = false
    }
  }
}
