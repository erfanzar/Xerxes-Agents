// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export function setupReadiness(snap: {cwd: string; connection: string; model: string}) {
  return {
    workspace: Boolean(snap.cwd),
    runtime: snap.connection === 'online',
    model: Boolean(snap.model.trim()),
    ready: Boolean(snap.cwd && snap.connection === 'online' && snap.model.trim()),
  }
}
