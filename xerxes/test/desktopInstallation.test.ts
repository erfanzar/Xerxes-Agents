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

test('workspace registry retains empty folders, deduplicates and migrates legacy settings', async () => {
  const {mkdtemp,rm,writeFile,readFile} = await import('node:fs/promises')
  const {loadDesktopWorkspaces,saveDesktopWorkspace} = await import('../src/desktop/main/workspaceSettings.js')
  const root=await mkdtemp('/tmp/xerxes-workspaces-'), file=root+'/desktop.json'
  try {
    expect(loadDesktopWorkspaces(file)).toEqual([])
    await writeFile(file,JSON.stringify({workspace:'/old'}))
    saveDesktopWorkspace(file,'/empty')
    saveDesktopWorkspace(file,'/old')
    expect(loadDesktopWorkspaces(file)).toEqual(['/old','/empty'])
    expect(JSON.parse(await readFile(file,'utf8')).workspace).toBe('/old')
    expect(()=>saveDesktopWorkspace(file,'relative')).toThrow()
    expect(loadDesktopWorkspaces(file)).toEqual(['/old','/empty'])
    await writeFile(file,'broken')
    expect(()=>saveDesktopWorkspace(file,'/new')).toThrow()
    expect(await readFile(file,'utf8')).toBe('broken')
  } finally {await rm(root,{recursive:true,force:true})}
})

// Reading this file happens on the window-restore path, before any window
// exists, so a torn write or a hand-edit must not be able to throw there.
// Refusing to CLOBBER unparseable content is a separate guarantee, asserted
// above, and must survive the read becoming total.
test('an unreadable workspace list degrades to empty instead of throwing', async () => {
  const {mkdtemp,rm,writeFile} = await import('node:fs/promises')
  const {loadDesktopWorkspaces,saveDesktopWorkspace} = await import('../src/desktop/main/workspaceSettings.js')
  const root=await mkdtemp('/tmp/xerxes-workspaces-corrupt-'), file=root+'/desktop.json'
  try {
    for (const body of ['', 'broken', '{"directories":', 'null', '[]', '"a string"', '{"directories":{"not":"an array"}}']) {
      await writeFile(file, body)
      expect(loadDesktopWorkspaces(file)).toEqual([])
    }
    // `null` parses cleanly but has no properties — reading `.directories` off
    // it threw just as hard as a syntax error did.
    await writeFile(file, 'null')
    expect(()=>loadDesktopWorkspaces(file)).not.toThrow()
    expect(()=>saveDesktopWorkspace(file,'/new')).toThrow()

    // A well-formed file with junk entries still yields only usable paths.
    await writeFile(file, JSON.stringify({workspace:42, directories:['/good', 'relative', 7, null, '/good']}))
    expect(loadDesktopWorkspaces(file)).toEqual(['/good'])
  } finally {await rm(root,{recursive:true,force:true})}
})
