// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, writeFile, readFile, rm, stat, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { McpSettingsStore, replaceMcpSettings } from '../src/mcp/settingsStore.js'
import { loadMcpConfig } from '../src/mcp/config.js'
import { MCPManager } from '../src/mcp/manager.js'

test('MCP settings save preserves siblings and rejects stale, malformed and linked files', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xm-settings-'))
  try {
    const path = join(dir, 'mcp.json'), store = new McpSettingsStore(path)
    await writeFile(path, JSON.stringify({ alpha: { command: 'old' }, beta: { command: 'other', env: { TOKEN: 'secret' } } }))
    const first = store.read()
    const saved = store.replace({ name: 'alpha', command: 'new' }, first.revision)
    expect(saved.revision).not.toBe(first.revision)
    expect(store.read()).toEqual(saved)
    expect(saved.servers[1]?.env).toEqual({ TOKEN: 'secret' })
    if (process.platform !== 'win32') expect((await stat(path)).mode & 0o777).toBe(0o600)
    expect(() => store.replace({ name: 'alpha', command: 'stale' }, first.revision)).toThrow('changed')
    expect(() => store.replace({ name: 'alpha', enabled: 'false' }, saved.revision)).toThrow()
    expect(store.read()).toEqual(saved)
    await writeFile(path, '{"bad":false}')
    expect(() => store.read()).toThrow('invalid entries')
    await rm(path); await symlink(join(dir, 'target'), path)
    expect(() => store.read()).toThrow()
  } finally { await rm(dir, { recursive: true, force: true }) }
})

test.each(['success', 'connection', 'stale', 'locked', 'cancel'] as const)('MCP settings coordinate disk and live state on %s', async mode => {
  const dir = await mkdtemp(join(tmpdir(), 'xm-settings-live-'))
  const path = join(dir, 'mcp.json'), store = new McpSettingsStore(path)
  const controller = new AbortController()
  const disconnected: string[] = []
  const manager = new MCPManager({ clientFactory: config => ({
    config, tools: [], resources: [], prompts: [],
    async connect() {
      if (config.command !== 'new') return
      if (mode === 'connection') throw new Error('private-token')
      if (mode === 'stale') await writeFile(path, JSON.stringify({ alpha: { command: 'external' } }))
      if (mode === 'locked') await writeFile(`${path}.settings-lock`, 'owned by another writer')
      if (mode === 'cancel') controller.abort('private-token')
    },
    async disconnect() { disconnected.push(config.command!) },
    async callTool() { return { content: [] } }, async readResource() { return { contents: [] } }, async getPrompt() { return { messages: [] } },
  }) })
  try {
    await writeFile(path, JSON.stringify({ alpha: { command: 'old' } }))
    const settings = store.read()
    await manager.addServer(settings.servers[0]!)
    const previous = manager.getServer('alpha')
    const updating = replaceMcpSettings(manager, store, { name: 'alpha', command: 'new', env: { TOKEN: 'private-token' } }, settings.revision, controller.signal)
    if (mode === 'success') {
      const saved = await updating
      expect(store.read()).toEqual(saved)
      expect(manager.getServer('alpha')?.config.command).toBe('new')
      expect(disconnected).toEqual(['old'])
    } else {
      await expect(updating).rejects.not.toThrow('private-token')
      expect(manager.getServer('alpha')).toBe(previous)
      expect(store.read().servers[0]?.command).toBe(mode === 'stale' ? 'external' : 'old')
      expect(disconnected).toEqual(['new'])
      if (mode === 'locked') expect(await readFile(`${path}.settings-lock`, 'utf8')).toBe('owned by another writer')
    }
  } finally { await manager.disconnectAll(); await rm(dir, { recursive: true, force: true }) }
})

test.each(['success', 'connection', 'shutdown', 'collision'] as const)('new MCP settings creation handles %s without partial publication', async mode => {
  const dir = await mkdtemp(join(tmpdir(), 'xm-settings-create-'))
  const store = new McpSettingsStore(join(dir, 'mcp.json'))
  let manager: MCPManager
  let cleaned = 0
  manager = new MCPManager({ clientFactory: config => ({
    config, tools: [], resources: [], prompts: [],
    async connect() {
      if (mode === 'connection') throw new Error('Cannot connect')
      if (mode === 'shutdown') await manager.disconnectAll()
    },
    async disconnect() { cleaned++ }, async callTool() { return { content: [] } },
    async readResource() { return { contents: [] } }, async getPrompt() { return { messages: [] } },
  }) })
  try {
    const empty = store.read()
    expect(empty.servers).toEqual([])
    if (mode === 'collision') await manager.addServer({ name: 'new', command: 'project', enabled: false })
    const pending = replaceMcpSettings(manager, store, { name: 'new', command: 'bun' }, empty.revision, undefined, true)
    if (mode === 'success') {
      const result = await pending
      expect(store.read()).toEqual(result)
      expect(manager.getServer('new')?.config.command).toBe('bun')
      expect((await stat(store.path)).nlink).toBe(1)
      if (process.platform !== 'win32') expect((await stat(store.path)).mode & 0o777).toBe(0o600)
      await expect(replaceMcpSettings(manager, store, { name: 'new', command: 'other' }, result.revision, undefined, true)).rejects.toThrow('already exists')
    } else {
      await expect(pending).rejects.toThrow()
      expect(store.read()).toEqual(empty)
      expect(manager.getServer('new')).toBeUndefined()
      expect(cleaned).toBe(mode === 'collision' ? 0 : 1)
    }
  } finally { await manager.disconnectAll(); await rm(dir, { recursive: true, force: true }) }
})

test.skipIf(process.platform === 'win32').each(['editor', 'startup'] as const)('MCP %s rejects a FIFO without waiting for a writer', async mode => {
  const dir = await mkdtemp(join(tmpdir(), 'xm-settings-fifo-'))
  let child: ReturnType<typeof Bun.spawn> | undefined
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    const path = join(dir, 'mcp.json')
    const fifo = Bun.spawn(['mkfifo', path], { stdout: 'pipe', stderr: 'pipe' })
    expect(await fifo.exited).toBe(0)
    const modulePath = new URL('../src/mcp/settingsStore.ts', import.meta.url).pathname
    const script = mode === 'startup'
      ? `import { loadMcpConfig } from ${JSON.stringify(new URL('../src/mcp/config.ts', import.meta.url).pathname)}; const result = loadMcpConfig(process.argv[1]); if (result.servers.length || result.warnings.length !== 1) process.exit(2)`
      : `import { McpSettingsStore } from ${JSON.stringify(modulePath)}; try { new McpSettingsStore(process.argv[1]).read(); process.exit(2) } catch (error) { if (!String(error).includes('regular file')) throw error }`
    child = Bun.spawn([process.execPath, '-e', script, path], { stdout: 'pipe', stderr: 'pipe' })
    let timedOut = false
    timer = setTimeout(() => { timedOut = true; child?.kill('SIGKILL') }, 2000)
    const code = await child.exited
    expect(timedOut).toBe(false)
    expect(code).toBe(0)
  } finally {
    if (timer) clearTimeout(timer)
    if (child) { child.kill(); await child.exited }
    await rm(dir, { recursive: true, force: true })
  }
})

test('MCP settings enforce the byte limit and reject invalid UTF-8', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xm-settings-bytes-'))
  try {
    const path = join(dir, 'mcp.json'), store = new McpSettingsStore(path)
    const source = '{"servers":[]}'
    await writeFile(path, source.padEnd(1_048_576))
    expect(store.read().servers).toEqual([])
    await writeFile(path, source.padEnd(1_048_577))
    expect(() => store.read()).toThrow('1 MiB')
    expect(loadMcpConfig(path).warnings).toHaveLength(1)
    expect(loadMcpConfig(path).servers).toEqual([])
    await writeFile(path, Buffer.concat([Buffer.from('{"alpha":{"command":"'), Buffer.from([0xff]), Buffer.from('"}}')]))
    expect(() => store.read()).toThrow('UTF-8')
    expect(loadMcpConfig(path).warnings).toHaveLength(1)
    expect(loadMcpConfig(path).servers).toEqual([])
  } finally { await rm(dir, { recursive: true, force: true }) }
})

test('MCP startup retains linked-file support while settings editing rejects links', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xm-settings-link-'))
  try {
    const target = join(dir, 'shared.json'), path = join(dir, 'mcp.json')
    await writeFile(target, JSON.stringify({ alpha: { command: 'bun', enabled: false } }))
    await symlink(target, path)
    expect(loadMcpConfig(path).warnings).toEqual([])
    expect(loadMcpConfig(path).servers[0]?.command).toBe('bun')
    expect(() => new McpSettingsStore(path).read()).toThrow()
  } finally { await rm(dir, { recursive: true, force: true }) }
})
