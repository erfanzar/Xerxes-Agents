// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { agentRetryCommands } from './commands/agentRetry.js'
import { attachCommands } from './commands/attach.js'
import { coreCommands } from './commands/core.js'
import { creditsCommands } from './commands/credits.js'
import { debugCommands } from './commands/debug.js'
import { integrationCommands } from './commands/integrations.js'
import { maintenanceCommands } from './commands/maintenance.js'
import { opsCommands } from './commands/ops.js'
import { presetCommands } from './commands/presets.js'
import { sessionCommands } from './commands/session.js'
import { setupCommands } from './commands/setup.js'
import type { SlashCommand } from './types.js'

export const SLASH_COMMANDS: SlashCommand[] = [
  ...coreCommands,
  ...creditsCommands,
  ...sessionCommands,
  ...maintenanceCommands,
  ...opsCommands,
  ...presetCommands,
  ...setupCommands,
  ...debugCommands,
  ...integrationCommands,
  ...attachCommands,
  // Registered last on purpose: the retry-aware `/agents` shadows the stock
  // dashboard command in the by-name map while delegating every non-retry
  // form to identical behavior.
  ...agentRetryCommands
]

const byName = new Map<string, SlashCommand>(
  SLASH_COMMANDS.flatMap(cmd => [cmd.name, ...(cmd.aliases ?? [])].map(name => [name, cmd] as const))
)

export const findSlashCommand = (name: string) => byName.get(name.toLowerCase())
