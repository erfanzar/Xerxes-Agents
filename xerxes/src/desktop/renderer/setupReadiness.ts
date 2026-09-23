// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { connectionFailureKind } from './connectionFailure.js'

export function setupReadiness(snap: {cwd: string; connection: string; model: string; error?: string | null}) {
  const online = snap.connection === 'online'
  return {
    workspace: Boolean(snap.cwd),
    runtime: online,
    model: Boolean(snap.model.trim()),
    ready: Boolean(snap.cwd && online && snap.model.trim()),
    /**
     * Whether the runtime step is still making progress. Without this the
     * checklist said "Connecting to the shared runtime…" forever for an
     * auth or workspace rejection that `beat()` will never retry — a
     * spinner where an explanation belonged.
     */
    runtimeStalled: !online && snap.connection !== 'connecting' && connectionFailureKind(snap.error ?? null) !== 'transport',
  }
}
