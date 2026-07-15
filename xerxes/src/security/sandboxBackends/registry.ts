// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { SandboxBackendAdapterConfigurationError } from './contracts.js'
import type { SandboxBackend } from '../sandbox.js'

/** One adapter registered by name in a caller-owned registry. */
export interface RegisteredSandboxBackend extends SandboxBackend {
  readonly name: string
}

/** Raised when a caller requests an adapter that has not been registered. */
export class SandboxBackendNotRegisteredError extends Error {
  readonly backendName: string

  constructor(backendName: string, available: readonly string[]) {
    super(`Unknown sandbox backend ${JSON.stringify(backendName)}. Available backends: ${available.join(', ') || '(none)'}`)
    this.name = 'SandboxBackendNotRegisteredError'
    this.backendName = backendName
  }
}

/**
 * Isolated adapter registry.
 *
 * The Python runtime used a module-global registry. The native port keeps the
 * registry caller-owned so tests, daemon sessions, and embedded hosts cannot
 * overwrite each other's backend selections.
 */
export class SandboxBackendRegistry {
  readonly #backends = new Map<string, RegisteredSandboxBackend>()

  constructor(backends: readonly RegisteredSandboxBackend[] = []) {
    for (const backend of backends) this.register(backend)
  }

  get(name: string): RegisteredSandboxBackend {
    const backend = this.#backends.get(normalizeName(name))
    if (backend === undefined) throw new SandboxBackendNotRegisteredError(name, this.list())
    return backend
  }

  list(): string[] {
    return [...this.#backends.keys()].sort()
  }

  register(backend: RegisteredSandboxBackend): void {
    if (backend === null || typeof backend !== 'object' || typeof backend.execute !== 'function') {
      throw new SandboxBackendAdapterConfigurationError('registry', 'backend must expose a name and execute(request)')
    }
    const name = normalizeName(backend.name)
    this.#backends.set(name, backend)
  }
}

function normalizeName(name: string): string {
  if (typeof name !== 'string' || !name.trim() || name.includes('\0')) {
    throw new SandboxBackendAdapterConfigurationError('registry', 'backend name must be a non-empty string without NUL bytes')
  }
  return name.trim()
}
