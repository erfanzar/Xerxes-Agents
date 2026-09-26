// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { homedir, tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { xerxesHome } from '../src/daemon/paths.js'

test('test runs never resolve to the developer\'s real Xerxes home', () => {
  // bunfig.toml preloads test/support/isolatedHome.ts. Without it, a runtime
  // built with no session directory writes fixture chats into ~/.xerxes.
  expect(xerxesHome()).not.toBe(join(homedir(), '.xerxes'))
  expect(xerxesHome().startsWith(tmpdir()) || xerxesHome().startsWith('/private' + tmpdir())).toBe(true)
})
