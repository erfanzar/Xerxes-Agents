// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { chmod, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { delimiter, join } from 'node:path'

import { engineSearchPath, whichEngine } from '../src/local/enginePath.js'

test('an app opened from Finder still finds engines in the installers\' folders', async () => {
  const home = await mkdtemp(join(tmpdir(), 'xerxes-engine-path-'))
  try {
    // launchd's PATH for an app started from Finder or the Dock.
    const finder = { HOME: home, PATH: '/usr/bin:/bin:/usr/sbin:/sbin' }
    expect(engineSearchPath(finder).split(delimiter)).toEqual([
      '/usr/bin', '/bin', '/usr/sbin', '/sbin', '/opt/homebrew/bin', '/usr/local/bin', join(home, '.local', 'bin'), join(home, '.lmstudio', 'bin'),
    ])
    // A shell PATH that already has them keeps its order and gains no duplicates.
    expect(engineSearchPath({ HOME: home, PATH: '/opt/homebrew/bin:/usr/bin' }).split(delimiter).filter(dir => dir === '/opt/homebrew/bin')).toHaveLength(1)

    const bin = join(home, '.local', 'bin')
    await mkdir(bin, { recursive: true })
    await writeFile(join(bin, 'mlx_lm.server'), '#!/bin/sh\n')
    await chmod(join(bin, 'mlx_lm.server'), 0o755)
    expect(whichEngine('mlx_lm.server', finder)).toBe(join(bin, 'mlx_lm.server'))
    expect(whichEngine('no-such-engine-binary', finder)).toBeUndefined()
  } finally {
    await rm(home, { recursive: true, force: true })
  }
})
