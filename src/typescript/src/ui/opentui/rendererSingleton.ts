// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
// Holds the active CliRenderer so imperative (non-hook) call sites in the
// controller — forceRedraw() from a slash command handler and
// withTerminalSuspended() during an $EDITOR shell-out — can reach it.
// entry.tsx sets it right after createCliRenderer(). Keeping this in a tiny
// module avoids pulling the React/OpenTUI hook surface into plain logic.
import type { CliRenderer } from '@opentui/core'

let active: CliRenderer | null = null

export function setActiveRenderer(renderer: CliRenderer): void {
  active = renderer
}

export function getActiveRenderer(): CliRenderer | null {
  return active
}
