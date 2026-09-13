// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { copyDesktopRuntimeAssets } from '../scripts/packageDesktopMac.js'

test('desktop runtime retains nested preset, skill and sandbox assets and rejects incomplete builds', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-desktop-assets-'))
  const source = join(root, 'build')
  const destination = join(root, 'runtime')
  const files = {
    'skills/review/SKILL.md': '# Review workspace',
    'default/agent.yaml': 'version: 1\nagent:\n  name: default\n',
    'default/subagents/reviewer.yaml': 'version: 1\nagent:\n  name: reviewer\n',
    'default/system.md': 'Built-in system prompt',
    'sandboxShim.ts': 'export const sandbox = true\n',
  }
  try {
    await mkdir(join(source, 'skills/review'), { recursive: true })
    await mkdir(join(source, 'default/subagents'), { recursive: true })
    for (const [file, content] of Object.entries(files)) await writeFile(join(source, file), content)
    await copyDesktopRuntimeAssets(source, destination)
    for (const [file, content] of Object.entries(files)) expect(await readFile(join(destination, file), 'utf8')).toBe(content)
    await rm(join(source, 'sandboxShim.ts'))
    await expect(copyDesktopRuntimeAssets(source, join(root, 'incomplete'))).rejects.toThrow()
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
