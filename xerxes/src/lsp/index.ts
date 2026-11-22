// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export { parseLspConfig, type LspServerConfig } from './config.js'
export { loadConfiguredLsp } from './configured.js'
export {
  discoverWorkspaceLspServers,
  mergeDiscoveredLspServers,
  type WorkspaceLspDiscoveryOptions,
} from './discovery.js'
export { LspHost, type LspCompletionItem, type LspHostOptions } from './host.js'
export { LspManager, type LspServerHealth } from './manager.js'
export { lspSettingsView, prepareLspSettingsEdit, saveLspSettings, type LspSettingsView } from './settings.js'
export { LspSettingsStore, type LspSettingsSnapshot } from './settingsStore.js'
