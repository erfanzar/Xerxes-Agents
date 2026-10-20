// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { ManagedPlugins, type ManagedPluginsOptions } from '../src/extensions/managedPlugins.js'
import { runPluginSecretCli } from '../src/extensions/pluginSecretCli.js'
import { PluginSecretStore } from '../src/extensions/pluginSecrets.js'
import { validateGitUrl, type GitRunner } from '../src/extensions/pluginShare.js'
import { argumentShape, detectRepeatedCommand, pluginSuggestionMessage } from '../src/extensions/pluginSuggestions.js'
import { permissionDescription } from '../src/streaming/permissions.js'
import { defaultPluginTestRunner, pluginManageTool, type PluginTestRunner } from '../src/tools/pluginManage.js'
import type { JsonObject } from '../src/types/toolCalls.js'

const key = async () => Buffer.alloc(32, 7)

async function fixture(prefix: string, options: Omit<ManagedPluginsOptions, 'secrets'> = {}) {
  const root = await mkdtemp(join(tmpdir(), prefix))
  const secretsPath = join(root, 'home', 'plugin-secrets.json')
  const manager = new ManagedPlugins(join(root, 'home', 'plugins.json'), {
    secrets: new PluginSecretStore(secretsPath, { key }),
    projectTrustFile: join(root, 'home', 'trust.json'),
    sourcesDirectory: join(root, 'home', 'sources'),
    ...options,
  })
  const tools = new ToolRegistry()
  manager.registerTools(tools)
  const call = (name: string, args: JsonObject = {}, signal?: AbortSignal) =>
    tools.execute({ id: '1', type: 'function', function: { name, arguments: args } }, { metadata: {} }, signal)
  const cleanup = async () => { manager.close(); await rm(root, { recursive: true, force: true }) }
  return { root, manager, tools, call, cleanup, secretsPath }
}

const structured = (name: string, body: string, extra = '', owner = 'iso') =>
  `r.registerTool('${name}', { description: '${name} tool', parameters: { type: 'object', properties: {} }, ${extra} execute: ${body} }, undefined, '${owner}')`

test('plugin code runs in its own process: declared env only, fetch refuses undeclared hosts, the daemon env is invisible', async () => {
  const { root, manager, call, cleanup } = await fixture('plugin-iso-env-', { load: { environment: { DECLARED: 'yes', HIDDEN: 'no' } } })
  try {
    const path = join(root, 'env.ts')
    await writeFile(path, `export function register(r) {
      r.registerPlugin({ name: 'iso', permissions: { env: ['DECLARED'], network: ['allowed.example'] } })
      ${structured('envs', `() => ({ declared: process.env.DECLARED ?? null, hidden: process.env.HIDDEN ?? null, home: process.env.HOME ?? null, pid: process.pid })`)}
      ${structured('blocked', `async () => { await fetch('https://blocked.invalid/x'); return 'reached' }`)}
    }`)
    await expect(manager.change('install', path)).resolves.toContain('iso declares: network allowed.example')
    const seen = JSON.parse(await call('plugin_envs'))
    expect(seen).toMatchObject({ declared: 'yes', hidden: null, home: null })
    expect(seen.pid).not.toBe(process.pid)
    await expect(call('plugin_blocked')).rejects.toThrow('refused')
    expect(manager.inventory()[0]?.permissions).toEqual({ network: ['allowed.example'], env: ['DECLARED'], filesystem: [], secrets: [] })
    await writeFile(join(root, 'bad.ts'), `export function register(r) { r.registerPlugin({ name: 'bad', permissions: { network: ['https://x.example/'] } }) }`)
    await expect(manager.change('install', join(root, 'bad.ts'))).rejects.toThrow('not a hostname')
  } finally { await cleanup() }
})

test('a hung, crashed or cancelled plugin call is stopped; the daemon survives and the next call gets a fresh process', async () => {
  const { root, manager, call, cleanup } = await fixture('plugin-iso-hang-')
  try {
    const path = join(root, 'hang.ts')
    await writeFile(path, `export function register(r) {
      r.registerPlugin({ name: 'iso' })
      ${structured('spin', `() => { while (true) {} }`, 'timeoutMs: 300,')}
      ${structured('crash', `() => process.exit(3)`)}
      ${structured('forever', `() => new Promise(() => {})`)}
      ${structured('ok', `() => 'fine ' + process.pid`)}
    }`)
    await manager.change('install', path)
    const started = Date.now()
    await expect(call('plugin_spin')).rejects.toThrow('timed out after 300 ms')
    expect(Date.now() - started).toBeLessThan(5_000)
    const first = await call('plugin_ok')
    expect(first).toStartWith('fine ')
    await expect(call('plugin_crash')).rejects.toThrow('exited with code 3')
    const second = await call('plugin_ok')
    expect(second).toStartWith('fine ')
    expect(second).not.toBe(first)
    const controller = new AbortController()
    const pending = call('plugin_forever', {}, controller.signal)
    setTimeout(() => controller.abort(), 50)
    await expect(pending).rejects.toThrow('cancelled')
    const usage = manager.inventory()[0]!.usage
    expect(usage.spin).toMatchObject({ calls: 1, failures: 1 })
    expect(usage.spin?.lastError).toContain('timed out')
    expect(usage.ok).toMatchObject({ calls: 2, failures: 0 })
    expect(usage.ok?.lastUsedAt).toBeString()
    expect(usage.forever?.failures).toBe(1)
  } finally { await cleanup() }
})

test('a restarted process refuses edited code until reload, and reload picks up edited helpers', async () => {
  const { root, manager, call, cleanup } = await fixture('plugin-iso-reload-')
  try {
    await writeFile(join(root, 'helper.ts'), `export const word = 'first'`)
    const path = join(root, 'index.ts')
    await writeFile(path, `import { word } from './helper.ts'\nexport function register(r) { r.registerPlugin({ name: 'iso' }); ${structured('word', `() => word`)}; ${structured('crash', `() => process.exit(1)`)} }`)
    await manager.change('install', path)
    expect(await call('plugin_word')).toBe('first')
    await writeFile(join(root, 'helper.ts'), `export const word = 'second'`)
    await expect(call('plugin_crash')).rejects.toThrow('exited')
    await expect(call('plugin_word')).rejects.toThrow('source changed')
    await manager.change('reload', 'iso')
    expect(await call('plugin_word')).toBe('second')
  } finally { await cleanup() }
})

test('plugin secrets are encrypted at 0600, delivered only when declared, and never echoed', async () => {
  const { root, manager, call, cleanup, secretsPath } = await fixture('plugin-iso-secrets-')
  try {
    const path = join(root, 'sec.ts')
    await writeFile(path, `export function register(r) {
      r.registerPlugin({ name: 'iso', permissions: { secrets: ['API_KEY', 'UNSET_KEY'] } })
      ${structured('secrets', `(input, context) => context.secrets ?? null`)}
    }`)
    await manager.change('install', path)
    // Declared but unset: an empty object, never another plugin's values.
    expect(JSON.parse(await call('plugin_secrets'))).toEqual({})
    await manager.setSecret('iso', 'API_KEY', 'hunter2-value')
    await manager.secrets.set('iso', 'NOT_DECLARED', 'other-value')
    expect(JSON.parse(await call('plugin_secrets'))).toEqual({ API_KEY: 'hunter2-value' })
    await expect(manager.setSecret('iso', 'NOT_DECLARED', 'x')).rejects.toThrow('does not declare secret NOT_DECLARED')
    await expect(manager.setSecret('missing', 'API_KEY', 'x')).rejects.toThrow('No plugin named missing')
    expect(await manager.secretStatus('iso')).toEqual({ plugin: 'iso', declared: ['API_KEY', 'UNSET_KEY'], set: ['API_KEY', 'NOT_DECLARED'] })
    const raw = await readFile(secretsPath, 'utf8')
    expect(raw).not.toContain('hunter2')
    expect((await stat(secretsPath)).mode & 0o777).toBe(0o600)
    expect(await manager.unsetSecret('iso', 'API_KEY')).toBe(true)
    expect(JSON.parse(await call('plugin_secrets'))).toEqual({})
    // A wrong key fails without revealing anything.
    await manager.setSecret('iso', 'API_KEY', 'hunter2-value')
    const other = new PluginSecretStore(secretsPath, { key: async () => Buffer.alloc(32, 9) })
    const failure = await other.values('iso', ['API_KEY']).catch(error => String(error))
    expect(failure).toContain('could not be decrypted')
    expect(failure).not.toContain('hunter2')
  } finally { await cleanup() }
})

test('the secret CLI reads the value hidden and prints names only', async () => {
  const root = await mkdtemp(join(tmpdir(), 'plugin-secret-cli-'))
  try {
    const lines: string[] = []
    const store = new PluginSecretStore(join(root, 'plugin-secrets.json'), { key })
    const io = { readSecret: async () => 'hunter2-value', print: (line: string) => lines.push(line), error: (line: string) => lines.push(line) }
    expect(await runPluginSecretCli(['secret', 'set', 'iso', 'API_KEY'], root, io, store)).toBe(0)
    expect(await runPluginSecretCli(['secret', 'list', 'iso'], root, io, store)).toBe(0)
    expect(await runPluginSecretCli(['secret', 'set', 'iso', 'lower'], root, io, store)).toBe(1)
    expect(await runPluginSecretCli(['secret'], root, io, store)).toBe(2)
    expect(lines.join('\n')).not.toContain('hunter2')
    expect(lines[1]).toContain('API_KEY')
    expect(await store.values('iso', ['API_KEY'])).toEqual({ API_KEY: 'hunter2-value' })
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('repeated command shapes in one turn produce one suggestion; varied or few calls do not', () => {
  const exec = (cmd: string, args: string[]) => ({ function: { name: 'exec_command', arguments: { cmd, args } } })
  const curls = [1, 22, 333].map(id => exec('curl', ['-s', `https://api.example.com/items/${id}?x=1`]))
  const found = detectRepeatedCommand([...curls, exec('ls', ['-la'])])
  expect(found).toMatchObject({ program: 'curl', pattern: 'https://api.example.com/items/…', count: 3 })
  expect(pluginSuggestionMessage(found!)).toContain('/plugin-creator')
  expect(pluginSuggestionMessage(found!)).toContain('Nothing was created')
  expect(detectRepeatedCommand(curls.slice(0, 2))).toBeUndefined()
  expect(detectRepeatedCommand([exec('curl', ['https://a.example/x']), exec('wget', ['https://a.example/x']), exec('git', ['status'])])).toBeUndefined()
  expect(detectRepeatedCommand([1, 2, 3].map(() => ({ function: { name: 'ReadFile', arguments: { file_path: 'a' } } })))).toBeUndefined()
  expect(detectRepeatedCommand([1, 2, 3].map(n => ({ function: { name: 'Bash', arguments: { command: `gh api repos/a/b/issues/${n}` } } })))).toMatchObject({ program: 'gh', count: 3 })
  expect(argumentShape('src/file12.ts')).toBe('src/…')
})

test('a model install runs the plugin tests first and refuses failing, missing or timed-out tests', async () => {
  const { root, manager, cleanup } = await fixture('plugin-iso-gate-')
  try {
    const folder = join(root, 'gated')
    await mkdir(folder)
    const path = join(folder, 'index.ts')
    await writeFile(path, `export function register(r) { r.registerPlugin({ name: 'gated', permissions: { network: ['api.example'] } }); ${structured('g', `() => 'g'`, '', 'gated')} }`)
    const runs: Array<{ file: string; cwd: string }> = []
    let result = { exitCode: 1, output: 'lots of output\n(fail) adds numbers\n1 fail' }
    const testRunner: PluginTestRunner = async (file, options) => { runs.push({ file, cwd: options.cwd }); return result }
    const run = (inputs: Record<string, unknown>) => pluginManageTool({ action: 'install', path: 'gated/index.ts', ...inputs }, { metadata: {} }, manager, () => root, { testRunner })
    expect(await run({})).toMatchObject({ ok: false, error: expect.stringContaining('No test file') })
    expect(runs).toEqual([])
    await writeFile(join(folder, 'index.test.ts'), `import { test } from 'bun:test'\ntest('x', () => {})`)
    expect(await run({})).toMatchObject({ ok: false, error: expect.stringContaining('tests failed (exit 1)'), test_output_tail: expect.stringContaining('(fail) adds numbers') })
    expect(runs[0]).toEqual({ file: join(folder, 'index.test.ts'), cwd: folder })
    expect(manager.inventory()).toEqual([])
    result = { exitCode: 0, output: '', timedOut: true } as typeof result
    expect(await run({})).toMatchObject({ ok: false, error: expect.stringContaining('timed out') })
    result = { exitCode: 0, output: '1 pass' }
    const installed = await run({})
    expect(installed).toMatchObject({ ok: true, tests: { passed: true }, plugins: [expect.objectContaining({ name: 'gated', permissions: expect.objectContaining({ network: ['api.example'] }) })] })
    const inspected = await pluginManageTool({ action: 'inspect', plugin: 'gated' }, { metadata: {} }, manager, () => root)
    expect(inspected).toMatchObject({ ok: true, plugin: { usage: { g: { calls: 0 } }, permissions: { enforcement: expect.stringContaining('not enforced') } } })
    await rm(join(folder, 'index.test.ts'))
    await manager.change('uninstall', 'gated')
    expect(await run({ skip_tests: true })).toMatchObject({ ok: true, tests: { skipped: true } })
    const describe = (inputs: Record<string, unknown>) => permissionDescription({ function: { name: 'PluginManageTool', arguments: { action: 'install', path: 'p.ts', ...inputs } } })
    expect(describe({ skip_tests: true })).toContain('WITHOUT TESTS')
    expect(describe({})).toContain('tests run first')
    expect(describe({})).toContain('not a sandbox')
  } finally { await cleanup() }
})

test('the default test runner runs bun test in the plugin folder and reports failures', async () => {
  const root = await mkdtemp(join(tmpdir(), 'plugin-test-runner-'))
  try {
    await writeFile(join(root, 'pass.test.ts'), `import { expect, test } from 'bun:test'\ntest('ok', () => expect(1).toBe(1))`)
    await writeFile(join(root, 'fail.test.ts'), `import { expect, test } from 'bun:test'\ntest('broken', () => expect(1).toBe(2))`)
    expect((await defaultPluginTestRunner(join(root, 'pass.test.ts'), { cwd: root, timeoutMs: 60_000 })).exitCode).toBe(0)
    const failed = await defaultPluginTestRunner(join(root, 'fail.test.ts'), { cwd: root, timeoutMs: 60_000 })
    expect(failed.exitCode).not.toBe(0)
    expect(failed.output).toContain('broken')
  } finally { await rm(root, { recursive: true, force: true }) }
}, 60_000)

test('project plugins load only after their exact content is trusted, and editing withdraws trust', async () => {
  const { root, manager, cleanup } = await fixture('plugin-iso-project-')
  try {
    const workspace = join(root, 'ws')
    const folder = join(workspace, '.xerxes', 'plugins', 'local')
    await mkdir(folder, { recursive: true })
    await writeFile(join(folder, 'index.ts'), `export function register(r) { r.registerPlugin({ name: 'local' }); r.registerTool('local_hi', { description: 'hi', parameters: { type: 'object' }, execute: () => 'hi from project' }, undefined, 'local') }`)
    const tools = new ToolRegistry()
    manager.project.registerTools(tools, workspace)
    await manager.project.refresh(workspace)
    const names = () => tools.definitions().map(definition => definition.function.name)
    expect(names()).not.toContain('plugin_local_hi')
    expect(manager.project.inventory(workspace)[0]).toMatchObject({ name: 'local', trusted: false, enabled: false, scope: 'project' })
    await expect(manager.project.trust(workspace, 'local')).resolves.toContain('plugin_local_hi')
    expect(names()).toContain('plugin_local_hi')
    expect(await tools.execute({ id: '1', type: 'function', function: { name: 'plugin_local_hi', arguments: {} } }, { metadata: {} })).toBe('hi from project')
    // Another workspace never sees it.
    const elsewhere = new ToolRegistry()
    manager.project.registerTools(elsewhere, join(root, 'other'))
    await manager.project.refresh(join(root, 'other'))
    expect(elsewhere.definitions().map(definition => definition.function.name)).not.toContain('plugin_local_hi')
    await writeFile(join(folder, 'index.ts'), `export function register(r) { r.registerPlugin({ name: 'local' }); r.registerTool('local_hi', { description: 'hi', parameters: { type: 'object' }, execute: () => 'edited' }, undefined, 'local') }`)
    await manager.project.refresh(workspace)
    expect(names()).not.toContain('plugin_local_hi')
    expect(manager.project.inventory(workspace)[0]).toMatchObject({ trusted: false })
    await expect(manager.project.trust(workspace, 'missing')).rejects.toThrow('No project plugin folder')
  } finally { await cleanup() }
})

test('every plugin process can run programs by name: PATH and TMPDIR are given, HOME still has to be declared', async () => {
  const { root, manager, call, cleanup } = await fixture('plugin-iso-path-')
  try {
    const path = join(root, 'which.ts')
    await writeFile(path, `export function register(r) {
      r.registerPlugin({ name: 'iso' })
      ${structured('which', `() => ({ path: process.env.PATH ?? null, home: process.env.HOME ?? null, bun: Bun.which('bun') !== null || Bun.which(process.execPath) !== null })`)}
    }`)
    await manager.change('install', path)
    const seen = JSON.parse(await call('plugin_which'))
    expect(seen.path).toBe(process.env.PATH ?? null)
    expect(seen.home).toBeNull()
  } finally { await cleanup() }
})

test('a turn can wait for the workspace scan: untrusted and trusted project plugins are known once it resolves', async () => {
  const { root, manager, cleanup } = await fixture('plugin-iso-ready-')
  try {
    const workspace = join(root, 'ws')
    const folder = join(workspace, '.xerxes', 'plugins', 'early')
    await mkdir(folder, { recursive: true })
    await writeFile(join(folder, 'index.ts'), `export function register(r) { r.registerPlugin({ name: 'early' }); r.registerTool('early_hi', { description: 'hi', parameters: { type: 'object' }, execute: () => 'early' }, undefined, 'early') }`)
    expect(manager.project.inventory(workspace)).toEqual([])
    await manager.project.whenReady(workspace)
    expect(manager.project.inventory(workspace)[0]).toMatchObject({ name: 'early', trusted: false })
    // Already scanned: resolves immediately, and an empty workspace resolves too.
    await manager.project.whenReady(workspace, 1)
    await expect(manager.project.whenReady(join(root, 'empty'))).resolves.toBeUndefined()
  } finally { await cleanup() }
})

test('git installs preview without running anything, need a one-time token, and reject unsafe URLs; export never overwrites', async () => {
  const gitCalls: string[][] = []
  const git: GitRunner = async args => {
    gitCalls.push([...args])
    const target = args[args.length - 1]!
    await mkdir(target, { recursive: true })
    await writeFile(join(target, 'index.ts'), `globalThis.loadedAtPreview = true\nexport function register(r) { r.registerPlugin({ name: 'remote', permissions: { network: ['api.remote.example'] } }); r.registerTool('remote_hi', { description: 'hi', parameters: { type: 'object' }, execute: () => 'remote' }, undefined, 'remote') }\n`)
    await writeFile(join(target, 'README.md'), '# Remote plugin\nSays hi.\n')
    return { exitCode: 0, output: '' }
  }
  const { root, manager, call, cleanup } = await fixture('plugin-iso-git-', { git })
  try {
    for (const bad of ['file:///tmp/x', 'http://example.com/a.git', '/local/path', 'ext::sh -c id', 'git://example.com/a', 'https://user:pw@example.com/a.git', '--upload-pack=x']) {
      expect(() => validateGitUrl(bad)).toThrow()
    }
    for (const good of ['https://github.com/a/b.git', 'git@github.com:a/b.git', 'ssh://git@example.com/a/b']) expect(validateGitUrl(good)).toBe(good)
    await expect(manager.previewSource('file:///tmp/x')).rejects.toThrow('Only https://')
    const { text, preview } = await manager.previewSource('https://example.com/remote.git')
    expect(gitCalls[0]).toEqual(expect.arrayContaining(['clone', '--depth', '1', '--', 'https://example.com/remote.git']))
    expect((globalThis as { loadedAtPreview?: boolean }).loadedAtPreview).toBeUndefined()
    expect(manager.inventory()).toEqual([])
    expect(text).toContain('index.ts · 2 lines')
    expect(text).toContain("network: ['api.remote.example']")
    expect(text).toContain('# Remote plugin')
    expect(text).toContain(`/plugins install --confirm ${preview.token}`)
    await expect(manager.installFromSource('not-a-token')).rejects.toThrow('Unknown or already used')
    await expect(manager.installFromSource(preview.token)).resolves.toContain('Source: https://example.com/remote.git')
    expect(await call('plugin_remote_hi')).toBe('remote')
    await expect(manager.installFromSource(preview.token)).rejects.toThrow('already used')
    // Export: sources and README, never .env files; never overwrites.
    await writeFile(join(preview.directory, '.env'), 'TOKEN=nope')
    const destination = join(root, 'shared')
    expect(await manager.exportPlugin('remote', destination)).toEqual(['README.md', 'index.ts'])
    expect(await Bun.file(join(destination, '.env')).exists()).toBe(false)
    await expect(manager.exportPlugin('remote', destination)).rejects.toThrow('already exists')
  } finally { await cleanup() }
})
