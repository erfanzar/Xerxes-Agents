// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { setupReadiness } from '../src/desktop/renderer/setupReadiness.js'
import { installerName } from '../scripts/buildDesktopInstaller.js'

test('setup requires a real workspace, connected runtime, and selected model', () => {
  expect(setupReadiness({ cwd: '/project', connection: 'online', model: 'model' }).ready).toBe(true)
  expect(setupReadiness({ cwd: '', connection: 'online', model: 'model' }).ready).toBe(false)
  expect(setupReadiness({ cwd: '/project', connection: 'offline', model: 'model' }).ready).toBe(
    false,
  )
  expect(setupReadiness({ cwd: '/project', connection: 'online', model: '   ' }).ready).toBe(false)
})
test('installer naming distinguishes CPU architecture and rejects unsafe paths', () => {
  expect(installerName('0.4.5', 'arm64')).toBe('Xerxes-Agents-0.4.5-macOS-arm64.dmg')
  expect(installerName('0.4.5', 'x64')).toContain('x64.dmg')
  expect(() => installerName('../escape', 'arm64')).toThrow('version')
  expect(() => installerName('0.4.5', 'unknown')).toThrow('architecture')
})

test('first workspace selection creates settings and atomically replaces them', async () => {
  const {mkdtemp,readFile,rm,writeFile} = await import('node:fs/promises')
  const {join} = await import('node:path')
  const {tmpdir} = await import('node:os')
  const {saveDesktopWorkspace} = await import('../src/desktop/main/workspaceSettings.js')
  const root = await mkdtemp(join(tmpdir(),'xerxes-setup-test-'))
  try {
    const file=join(root,'new-home','desktop.json')
    saveDesktopWorkspace(file,'/first-project')
    expect(JSON.parse(await readFile(file,'utf8')).workspace).toBe('/first-project')
    saveDesktopWorkspace(file,'/second-project')
    expect(JSON.parse(await readFile(file,'utf8')).workspace).toBe('/second-project')
    expect(() => saveDesktopWorkspace(file,'relative')).toThrow('absolute')
    expect(JSON.parse(await readFile(file,'utf8')).workspace).toBe('/second-project')
    await writeFile(join(root,'blocked'),'file')
    expect(() => saveDesktopWorkspace(join(root,'blocked','desktop.json'),'/project')).toThrow()
  } finally { await rm(root,{recursive:true,force:true}) }
})
