// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Test preload: every run gets a throwaway Xerxes home.
 *
 * A runtime built without an explicit directory falls back to the real
 * `~/.xerxes`. Two runner tests did exactly that, and every suite run left
 * fixture sessions ("Hello from the real loop.") in the developer's live
 * sidebar. Pointing XERXES_HOME at a temp directory for the whole process
 * makes forgetting a directory harmless instead of polluting real state.
 * A test that needs the unset default passes its own environment.
 */

import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const home = mkdtempSync(join(tmpdir(), 'xerxes-test-home-'))
process.env.XERXES_HOME = home
process.on('exit', () => { rmSync(home, { recursive: true, force: true }) })
