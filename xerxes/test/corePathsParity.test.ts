// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import {
  agentsHome,
  agentsSubdirFor,
  xerxesHome,
  xerxesSubdirFor,
} from '../src/core/paths.js'

test('core home helpers honor default, blank, and tilde XERXES_HOME values without creating paths', () => {
  const defaultHome = join(homedir(), '.xerxes')
  const notCreated = join(tmpdir(), 'xerxes-core-path-parity-' + crypto.randomUUID())

  expect(xerxesHome({})).toBe(defaultHome)
  expect(xerxesHome({ XERXES_HOME: '' })).toBe(defaultHome)
  expect(xerxesHome({ XERXES_HOME: '   ' })).toBe(defaultHome)
  expect(xerxesHome({ XERXES_HOME: '~' })).toBe(homedir())
  expect(xerxesHome({ XERXES_HOME: '~/custom-xerxes' })).toBe(join(homedir(), 'custom-xerxes'))
  expect(xerxesHome({ XERXES_HOME: notCreated })).toBe(notCreated)
  expect(existsSync(notCreated)).toBe(false)
})

test('core subdirectory helpers retain independent Xerxes and agents homes', () => {
  expect(xerxesSubdirFor({ XERXES_HOME: '/tmp/custom-xerxes' })).toBe('/tmp/custom-xerxes')
  expect(xerxesSubdirFor({ XERXES_HOME: '/tmp/custom-xerxes' }, 'daemon', 'logs'))
    .toBe('/tmp/custom-xerxes/daemon/logs')
  expect(agentsHome('/tmp/home')).toBe('/tmp/home/.agents')
  expect(agentsSubdirFor('/tmp/home', 'skills')).toBe('/tmp/home/.agents/skills')
})
