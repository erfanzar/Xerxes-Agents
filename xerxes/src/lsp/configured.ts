// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { join } from 'node:path'
import { LspManager } from './manager.js'
import { LspSettingsStore } from './settingsStore.js'

export {
  discoverWorkspaceLspServers,
  mergeDiscoveredLspServers,
  type WorkspaceLspDiscoveryOptions,
} from './discovery.js'

/** Only the user configuration directory supplies executable LSP settings. No project discovery. */
export function loadConfiguredLsp(home: string): LspManager {
  const snapshot = new LspSettingsStore(join(home, 'lsp.json')).read()
  return new LspManager({ servers: snapshot.servers })
}
