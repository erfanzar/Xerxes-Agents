// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export * from './base.js'
export * from './agentMemory.js'
export * from './agentSelfMemory.js'
export * from './compat.js'
export * from './contextFencing.js'
export * from './contextualMemory.js'
export * from './embedders.js'
export * from './entityMemory.js'
export * from './longTermMemory.js'
export * from './provider.js'
/**
 * Explicit external-memory provider registration entry point.
 *
 * Callers supply every environment, transport, and upstream dependency; the
 * memory subsystem never discovers credentials or starts external providers
 * during import.
 */
export {
  BUILTIN_MEMORY_PROVIDER_NAMES,
  createBuiltinMemoryProviders,
  registerBuiltinMemoryProviders,
  type BuiltinMemoryProviderDependencies,
  type BuiltinMemoryProviderName,
} from './plugins/builtins.js'
export * from './retrieval.js'
export * from './shortTermMemory.js'
export * from './storage.js'
export * from './turnIndexer.js'
export * from './userMemory.js'
export * from './userProfile.js'
export * from './vectorStorage.js'
