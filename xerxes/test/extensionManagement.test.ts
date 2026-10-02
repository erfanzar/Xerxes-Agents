// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ManagedPlugins } from '../src/extensions/managedPlugins.js'
import { installLocalSkill } from '../src/extensions/localSkillInstall.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'

test('installed native tool modules execute, disable immediately, and retain settings across restart', async () => {
  const root = await mkdtemp(join(tmpdir(), 'managed-plugins-'))
  try {
    const path = join(root, 'fixture.ts')
    await writeFile(path, `export function register(registry) { registry.registerTool('fixture_echo', (...args) => args.join(' '), {name: 'fixture'}); }`)
    const manager = new ManagedPlugins(join(root, 'settings.json'))
    await manager.change('install', path)
    const tools = new ToolRegistry()
    manager.registerTools(tools)
    const call = { id: '1', type: 'function' as const, function: { name: 'plugin_fixture_echo', arguments: { args: ['hello', 'world'] } } }
    expect(await tools.execute(call, { metadata: {} })).toBe('hello world')
    await manager.change('disable', 'fixture')
    await expect(tools.execute(call, { metadata: {} })).rejects.toThrow('disabled')
    const restarted = new ManagedPlugins(join(root, 'settings.json'))
    await restarted.load()
    expect(restarted.inventory()[0]).toMatchObject({ name: 'fixture', enabled: false })
    await restarted.change('enable', 'fixture')
    const freshTools = new ToolRegistry()
    restarted.registerTools(freshTools)
    expect(await freshTools.execute(call, { metadata: {} })).toBe('hello world')
    await expect(restarted.change('install', path)).rejects.toThrow('already installed')
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('plugin registration errors are actionable and do not persist a successful install', async () => {
  const root = await mkdtemp(join(tmpdir(), 'managed-plugins-invalid-'))
  try {
    const path = join(root, 'bad.ts')
    await writeFile(path, `export function register(registry) { registry.registerHook('before', () => {}, {name: 'bad'}); }`)
    const manager = new ManagedPlugins(join(root, 'settings.json'))
    await expect(manager.change('install', path)).rejects.toThrow('embedding host')
    expect(manager.inventory()).toEqual([])
    expect(await Bun.file(join(root, 'settings.json')).exists()).toBe(false)
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('local skill installation preserves resources and rejects overwrites and symlinks', async () => {
  const root = await mkdtemp(join(tmpdir(), 'local-skill-install-'))
  try {
    const source = join(root, 'source')
    await mkdir(join(source, 'references'), { recursive: true })
    await writeFile(join(source, 'SKILL.md'), '---\nname: fixture\ndescription: Fixture guide\n---\nRead references/guide.md.')
    await writeFile(join(source, 'references/guide.md'), 'Detailed instructions.')
    const installed = await installLocalSkill(source, join(root, 'skills'), [])
    expect(await readFile(join(installed, 'references/guide.md'), 'utf8')).toBe('Detailed instructions.')
    await expect(installLocalSkill(source, join(root, 'skills'), [])).rejects.toThrow('already installed')
    await symlink(join(root, 'source/references/guide.md'), join(source, 'linked'))
    await expect(installLocalSkill(source, join(root, 'other-skills'), [])).rejects.toThrow('symlink')
    expect(await Bun.file(join(root, 'other-skills/fixture/SKILL.md')).exists()).toBe(false)
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('plugin mutations serialize across managers and repeated loads do not duplicate modules', async () => {
  const root = await mkdtemp(join(tmpdir(), 'managed-plugin-race-'))
  try {
    const manifest = join(root, 'plugins.json')
    const first = new ManagedPlugins(manifest)
    const second = new ManagedPlugins(manifest)
    for (const name of ['one', 'two']) await writeFile(join(root, `${name}.ts`), `export function register(r) { r.registerTool('${name}', () => '${name}', {name: '${name}'}); }`)
    await Promise.all([first.change('install', join(root, 'one.ts')), second.change('install', join(root, 'two.ts'))])
    await first.load(); await first.load()
    expect(first.inventory().map(item => item.name).sort()).toEqual(['one', 'two'])
    const tools = new ToolRegistry()
    first.registerTools(tools)
    await second.change('disable', 'one')
    await expect(tools.execute({ id: 'call', type: 'function', function: { name: 'plugin_one', arguments: { args: [] } } }, { metadata: {} })).rejects.toThrow('disabled')
    await first.load()
    expect(first.inventory().find(item => item.name === 'one')?.enabled).toBe(false)
    await writeFile(join(root, 'one.ts'), `export function register(r) { r.registerTool('one', () => 'changed', {name: 'one'}); }`)
    await expect(first.change('enable', 'one')).rejects.toThrow('Restart Xerxes')
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('managed plugin manifests reject relative paths and duplicate entries', async () => {
  const root = await mkdtemp(join(tmpdir(), 'managed-plugin-manifest-'))
  try {
    const manifest = join(root, 'plugins.json')
    await writeFile(manifest, JSON.stringify([{ path: './relative.ts', enabled: false, names: ['one'] }]))
    await expect(new ManagedPlugins(manifest).load()).rejects.toThrow('Invalid managed plugin entry')
    const entry = { path: join(root, 'one.ts'), enabled: false, names: ['one'] }
    await writeFile(manifest, JSON.stringify([entry, entry]))
    await expect(new ManagedPlugins(manifest).load()).rejects.toThrow('Duplicate')
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('concurrent skill installation never replaces an existing destination', async () => {
  const root = await mkdtemp(join(tmpdir(), 'skill-install-race-'))
  try {
    const source = join(root, 'source')
    await mkdir(source)
    await writeFile(join(source, 'SKILL.md'), '---\nname: fixture\ndescription: Guide\n---\nRead this guide.')
    const target = join(root, 'skills')
    const results = await Promise.allSettled([installLocalSkill(source, target, []), installLocalSkill(source, target, [])])
    expect(results.filter(result => result.status === 'fulfilled')).toHaveLength(1)
    expect(results.filter(result => result.status === 'rejected')).toHaveLength(1)
    const existing = join(root, 'other/fixture')
    await mkdir(existing, { recursive: true })
    await expect(installLocalSkill(source, join(root, 'other'), [])).rejects.toThrow('already installed')
    expect(await Bun.file(join(existing, 'SKILL.md')).exists()).toBe(false)
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('broken enabled modules can still be disabled without reading their edited or missing source', async () => {
  const root = await mkdtemp(join(tmpdir(), 'managed-plugin-recovery-'))
  try {
    for (const failure of ['edited', 'missing']) {
      const path = join(root, `${failure}.ts`)
      await writeFile(path, `export function register(r) { r.registerTool('${failure}', () => 'ok', {name: '${failure}'}); }`)
      const manager = new ManagedPlugins(join(root, `${failure}.json`))
      await manager.change('install', path)
      if (failure === 'edited') await writeFile(path, 'invalid module contents')
      else await rm(path)
      await expect(manager.change('disable', failure)).resolves.toContain('Disabled')
      expect(manager.inventory()[0]?.enabled).toBe(false)
      const restarted = new ManagedPlugins(join(root, `${failure}.json`))
      await expect(restarted.load()).resolves.toBeUndefined()
      expect(restarted.inventory()[0]?.enabled).toBe(false)
    }
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('a missing, broken or renamed enabled module does not stop startup and leaves the manifest untouched', async () => {
  const root = await mkdtemp(join(tmpdir(), 'managed-plugin-startup-'))
  try {
    const manifest = join(root, 'plugins.json')
    const good = join(root, 'good.ts')
    const broken = join(root, 'broken.ts')
    const renamed = join(root, 'renamed.ts')
    await writeFile(good, `export function register(r) { r.registerTool('good', () => 'good', {name: 'good'}); }`)
    await writeFile(broken, `throw new Error('import exploded')`)
    await writeFile(renamed, `export function register(r) { r.registerTool('other', () => 'x', {name: 'other'}); }`)
    const entries = [
      { path: join(root, 'gone.ts'), enabled: true, names: ['gone'] },
      { path: broken, enabled: true, names: ['broken'] },
      { path: renamed, enabled: true, names: ['renamed'] },
      { path: good, enabled: true, names: ['good'] },
    ]
    await writeFile(manifest, JSON.stringify(entries))
    const manager = new ManagedPlugins(manifest)
    await expect(manager.load()).resolves.toBeUndefined()
    const inventory = manager.inventory()
    expect(inventory.find(item => item.name === 'good')?.enabled).toBe(true)
    for (const name of ['gone', 'broken', 'renamed']) {
      const entry = inventory.find(item => item.name === name)
      expect(entry?.enabled).toBe(false)
      expect(entry?.loadError).toBeTruthy()
    }
    const tools = new ToolRegistry()
    manager.registerTools(tools)
    expect(await tools.execute({ id: '1', type: 'function', function: { name: 'plugin_good', arguments: { args: [] } } }, { metadata: {} })).toBe('good')
    expect(JSON.parse(await readFile(manifest, 'utf8'))).toEqual(entries)
    // Changing another module still works while one failed to load, and the
    // failed entry keeps its enabled intent in the manifest.
    await expect(manager.change('disable', 'good')).resolves.toContain('Disabled')
    expect(JSON.parse(await readFile(manifest, 'utf8')).find((item: { names: string[] }) => item.names[0] === 'gone').enabled).toBe(true)
    await expect(manager.change('disable', 'gone')).resolves.toContain('Disabled')
    await writeFile(join(root, 'gone.ts'), `export function register(r) { r.registerTool('gone', () => 'back', {name: 'gone'}); }`)
    await expect(manager.change('enable', 'gone')).resolves.toContain('Enabled')
    expect(manager.inventory().find(item => item.name === 'gone')).toMatchObject({ enabled: true })
  } finally { await rm(root, { recursive: true, force: true }) }
})
