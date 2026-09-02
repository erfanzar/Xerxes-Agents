// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { launchXerxesCommand } from '../../../lib/externalCli.js'
import { withTerminalSuspended } from '../../../lib/terminalRuntime.opentui.js'
import { runExternalSetup } from '../../setupHandoff.js'
import type { SlashCommand } from '../types.js'

// Claude Code /init parity: one ordinary turn with a fixed curation prompt.
// Sent through transcript.send so it respects busy/steer/queue routing like
// any user message, and the agent — not client code — does the repo reading.
const INIT_PROMPT = [
  'Analyze this repository and write a XERXES.md file at the project root that helps',
  'future agent sessions work effectively here. Cover: what the project is and its',
  'purpose; how to build, test, lint, and run it (exact commands); code style and',
  'conventions worth enforcing; architecture map (key directories and what lives in',
  'them); anything unusual an agent would otherwise learn the hard way. Read package',
  'manifests, config files, CI workflows, existing docs, and the directory layout —',
  'do not guess. If a XERXES.md already exists, update it in place: keep accurate',
  'content, fix what has drifted, add what is missing. Keep the file tight and',
  'operational — an agent reads it at the start of every session.',
].join('\n')

export const setupCommands: SlashCommand[] = [
  {
    help: 'analyze the repo and write/update XERXES.md (Claude Code /init parity)',
    name: 'init',
    run: (_arg, ctx) => ctx.transcript.send(INIT_PROMPT)
  },
  {
    help: 'run full setup wizard (launches `xerxes setup`)',
    name: 'setup',
    run: (arg, ctx) =>
      void runExternalSetup({
        args: ['setup', ...arg.split(/\s+/).filter(Boolean)],
        ctx,
        done: 'setup complete — starting session…',
        launcher: launchXerxesCommand,
        suspend: withTerminalSuspended
      })
  }
]
