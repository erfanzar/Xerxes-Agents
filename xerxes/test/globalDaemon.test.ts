// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { join } from 'node:path'

test('desktop and TUI share one daemon, isolate concurrent turns, and survive client exit', async () => {
  const child = Bun.spawn([process.execPath, join(import.meta.dir, 'fixtures/daemon/globalClients.ts')], { stdout: 'pipe', stderr: 'pipe' })
  const [stdout, stderr, exit] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited])
  expect({ exit, stderr }).toEqual({ exit: 0, stderr: '' })
  expect(stdout).toContain('simultaneous production turns completed')
  expect(stdout).toContain('TUI exit leaves daemon alive')
}, 30000)
