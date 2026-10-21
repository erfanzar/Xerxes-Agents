// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import { connect } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { lintDiscoveredHarness, lintHarness, renderLintReport } from '../src/extensions/harnessLint.js'
import { parseSkillMarkdown, SkillRegistry } from '../src/extensions/skills.js'
import { SkillBundleStore, registerAgentMetaTools, skillManageTool } from '../src/tools/agentMetaTools.js'

function markdown(name = 'lint-fixture', body = 'Review the changes.', fields = 'description: Review changes'): string {
  return `---\nname: ${name}\n${fields}\n---\n${body}`
}

function document(content = markdown(), path = '/virtual/lint-fixture/SKILL.md', bundleFiles: readonly string[] = []) {
  return { content, path, bundleFiles }
}

test('a healthy skill has no findings and a useful clean report', () => {
  expect(lintHarness([document()])).toEqual([])
  expect(renderLintReport([])).toBe('Harness lint: no problems found.')
})

test('missing and empty descriptions warn without making a skill fail', () => {
  for (const fields of ['', 'description:', 'description: "   "']) {
    const findings = lintHarness([document(markdown('lint-fixture', 'Review changes.', fields))])
    expect(findings).toEqual([expect.objectContaining({ code: 'description-missing', severity: 'warn' })])
    expect(findings[0]?.message).toContain('discoverable')
  }
})

test('built-in slash names and aliases fail', () => {
  for (const name of ['help', 'skills', 'quit']) {
    expect(lintHarness([document(markdown(name), `/virtual/${name}/SKILL.md`)])).toContainEqual(
      expect.objectContaining({ code: 'builtin-command-conflict', severity: 'fail' }),
    )
  }
})

test('all strict per-skill diagnoses are retained along with cross-skill failures', () => {
  const findings = lintHarness([document(markdown('help', 'bun scripts/missing.ts', 'description: Fine\nunknown_key: value'))])
  expect(findings.map(item => item.code)).toEqual(expect.arrayContaining([
    'name-directory-mismatch', 'frontmatter-unknown-key', 'builtin-command-conflict', 'script-missing',
  ]))
})

test('bundle scripts/ invocations must ship each file, including quoted and dot-relative paths', () => {
  const content = markdown('lint-fixture', [
    '```sh',
    'bun scripts/check.ts',
    'bash ./scripts/check.sh',
    'bun "scripts/path with spaces.ts"',
    // Not bundle scripts: the user's own project paths.
    'bun "./helpers/check.ts"',
    './bin/check',
    'bash helpers/check',
    '```',
  ].join('\n'))
  const findings = lintHarness([document(content)])
  expect(findings.filter(item => item.code === 'script-missing').map(item => item.message)).toEqual([
    expect.stringContaining('scripts/check.ts'),
    expect.stringContaining('./scripts/check.sh'),
    expect.stringContaining('scripts/path with spaces.ts'),
  ])
  expect(lintHarness([document(content, undefined, [
    'scripts/check.ts', 'scripts/check.sh', 'scripts/path with spaces.ts',
  ])])).toEqual([])
})

test('an installed skill with no scripts/ folder gets a warning, not a failure, for a project script path', async () => {
  const root = await mkdtemp(join(tmpdir(), 'harness-noscripts-'))
  try {
    const path = join(root, 'lint-fixture/SKILL.md')
    await mkdir(join(root, 'lint-fixture'), { recursive: true })
    await writeFile(path, markdown('lint-fixture', 'python scripts/export_onnx_model.py'))
    const findings = await lintDiscoveredHarness([parseSkillMarkdown(await readFile(path, 'utf8'), path)])
    expect(findings.filter(item => item.code === 'script-missing')).toEqual([
      expect.objectContaining({ severity: 'warn', message: expect.stringContaining("user's project") }),
    ])
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('ordinary project commands are not required bundle scripts; only scripts/ paths are', () => {
  const content = markdown('lint-fixture', [
    '```sh',
    'python manage.py migrate',
    'bun run build.ts',
    'node_modules/.bin/tsc --noEmit',
    'python convert_hf_to_gguf.py model/',
    'bun https://example.test/scripts/remote.ts',
    'bash /tmp/script.sh',
    '```',
  ].join('\n'))
  expect(lintHarness([document(content)])).toEqual([])
  expect(lintHarness([document(markdown('lint-fixture', 'Run `bun scripts/helper.ts`.'))])).toEqual([
    expect.objectContaining({ code: 'script-missing', message: expect.stringContaining('scripts/helper.ts') }),
  ])
})

test('uninvoked prose and output filenames are not mistaken for required bundled scripts', () => {
  const content = markdown('lint-fixture', 'Write `output/report.ts` and run `bun --version`. Read references/guide.md.')
  expect(lintHarness([document(content)])).toEqual([])
})

test('writing, mentioning, or printing a script path is not an invocation', () => {
  const content = markdown('lint-fixture', [
    'Write `scripts/generated.ts` later. The helper `./scripts/generated.sh` is an output.',
    '```sh',
    'echo scripts/check.ts',
    'cat ./scripts/generated.ts',
    '```',
  ].join('\n'))
  expect(lintHarness([document(content)])).toEqual([])
})

test('script references escaping the bundle are failures even if a supplied path claims to ship them', () => {
  expect(lintHarness([document(markdown('lint-fixture', 'Run `bun scripts/../../outside.ts`.'), undefined, ['../outside.ts'])])).toContainEqual(
    expect.objectContaining({ code: 'script-escapes-root', severity: 'fail' }),
  )
})

test('unknown agent tools warn only with an explicit known-tool set, respecting wildcard declarations', () => {
  const agents = [{ path: '/virtual/reviewer.yaml', tools: ['ReadFile', 'typo', 'plugin_*'], allowed_tools: ['typo2', '*'], allowedTools: ['ReadFile'] }]
  expect(lintHarness([], { agents })).toEqual([])
  const findings = lintHarness([], { agents, knownToolNames: new Set(['ReadFile', 'plugin_greeting']) })
  expect(findings).toEqual([
    expect.objectContaining({ code: 'agent-tool-unknown', severity: 'warn', message: expect.stringContaining('typo') }),
    expect.objectContaining({ code: 'agent-tool-unknown', severity: 'warn', message: expect.stringContaining('typo2') }),
  ])
  expect(lintHarness([], { agents: [{ path: 'agent.yaml', tools: ['ReadFile'] }], knownToolNames: new Set() })).toHaveLength(1)
})

test('duplicate names across roots warn for every distinct source, not repeated paths', () => {
  const first = document(markdown(), '/one/lint-fixture/SKILL.md')
  const second = document(markdown(), '/two/lint-fixture/SKILL.md')
  const findings = lintHarness([first, second, first])
  expect(findings).toHaveLength(2)
  expect(findings.every(item => item.code === 'skill-name-duplicate' && item.severity === 'warn')).toBeTrue()
  expect(findings.map(item => item.path)).toEqual([first.path, second.path])
})

test('one user copy over one bundled skill is a single info note on the copy; other duplicates still warn', () => {
  const bundled = document(markdown(), '/app/skills/lint-fixture/SKILL.md')
  const home = document(markdown(), '/home/u/.xerxes/skills/lint-fixture/SKILL.md')
  const override = lintHarness([home, bundled], { bundledRoot: '/app/skills' })
  expect(override).toEqual([expect.objectContaining({ code: 'skill-overrides-bundled', severity: 'info', path: home.path })])
  expect(override[0]!.message).toContain(bundled.path)
  expect(renderLintReport(override)).toStartWith('Harness lint: 0 failure(s), 0 warning(s), 1 note(s).')
  const project = document(markdown(), '/repo/.xerxes/skills/lint-fixture/SKILL.md')
  const clash = lintHarness([home, project, bundled], { bundledRoot: '/app/skills' })
  expect(clash).toHaveLength(3)
  expect(clash.every(item => item.code === 'skill-name-duplicate' && item.severity === 'warn')).toBeTrue()
})

test('the complete report groups paths and includes every code, message and fix hint without truncation', () => {
  const findings = lintHarness([
    document(markdown('help', 'bun scripts/missing.ts', 'description:\nunknown_key: bad')),
    document(markdown('help'), '/second/help/SKILL.md'),
  ])
  const report = renderLintReport(findings)
  for (const finding of findings) {
    expect(report).toContain(finding.code)
    expect(report).toContain(finding.message)
    expect(report).toContain(finding.fixHint)
  }
  expect(report.split('/virtual/lint-fixture/SKILL.md\n')).toHaveLength(2)
  const longHint = 'Keep the complete hint. '.repeat(10_000)
  expect(renderLintReport([{ code: 'long', severity: 'warn', path: 'long.md', message: longHint, fixHint: longHint }])).toContain(longHint)
})

test('disk lint reports unreadable skills and catches missing resources and symlinked scripts without executing them', async () => {
  const root = await mkdtemp(join(tmpdir(), 'harness-lint-'))
  try {
    const path = join(root, 'lint-fixture/SKILL.md')
    await mkdir(join(root, 'lint-fixture/scripts'), { recursive: true })
    await writeFile(path, markdown('lint-fixture', 'bun scripts/missing.ts\nbun scripts/link.ts'))
    await writeFile(join(root, 'outside.ts'), 'throw new Error("must never run")')
    await symlink(join(root, 'outside.ts'), join(root, 'lint-fixture/scripts/link.ts'))
    const findings = await lintDiscoveredHarness([parseSkillMarkdown(await readFile(path, 'utf8'), path)], {
      additionalPaths: [join(root, 'gone/SKILL.md')],
    })
    expect(findings.map(item => item.code)).toEqual(expect.arrayContaining(['script-missing', 'script-escapes-root', 'document-unreadable']))
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('skill_manage refuses all failures together before changing any bundle or live registry', async () => {
  const root = await mkdtemp(join(tmpdir(), 'harness-manage-'))
  try {
    const registry = new SkillRegistry()
    const store = new SkillBundleStore({ directory: root, registry })
    const result = await skillManageTool({ action: 'create', name: 'help', instructions: 'bun scripts/one.ts\nbash ./scripts/two.sh' }, { skillStore: store })
    expect(result).toMatchObject({ ok: false, error: 'harness_lint_failed' })
    expect(result.report).toEqual(expect.stringContaining('builtin-command-conflict'))
    expect(result.report).toEqual(expect.stringContaining('scripts/one.ts'))
    expect(result.report).toEqual(expect.stringContaining('./scripts/two.sh'))
    expect(result.report).toEqual(expect.stringContaining('description-missing'))
    expect(await Bun.file(join(root, 'help/SKILL.md')).exists()).toBeFalse()
    expect(registry.all()).toHaveLength(0)
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('skill_manage ships supplied files, returns warnings with success, and leaves saved content intact after a rejected update', async () => {
  const root = await mkdtemp(join(tmpdir(), 'harness-files-'))
  try {
    const store = new SkillBundleStore({ directory: root })
    const created = await skillManageTool({
      action: 'create', name: 'lint-fixture', instructions: 'bun scripts/check.ts', files: { 'scripts/check.ts': '// a real helper' },
    }, { skillStore: store })
    expect(created).toMatchObject({ ok: true, warnings: [expect.objectContaining({ code: 'description-missing', severity: 'warn' })] })
    expect(await readFile(join(root, 'lint-fixture/scripts/check.ts'), 'utf8')).toBe('// a real helper')
    const original = await readFile(join(root, 'lint-fixture/SKILL.md'), 'utf8')
    const rejected = await skillManageTool({ action: 'update', name: 'lint-fixture', instructions: 'bun scripts/missing.ts' }, { skillStore: store })
    expect(rejected).toMatchObject({ ok: false, error: 'harness_lint_failed' })
    expect(await readFile(join(root, 'lint-fixture/SKILL.md'), 'utf8')).toBe(original)
    expect(await skillManageTool({ action: 'update', name: 'lint-fixture', instructions: 'bun scripts/check.ts' }, { skillStore: store })).toMatchObject({ ok: true })
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('a bundle update that fails partway through restores every file it already replaced', async () => {
  const root = await mkdtemp(join(tmpdir(), 'harness-rollback-'))
  try {
    const store = new SkillBundleStore({ directory: root })
    await skillManageTool({
      action: 'create', name: 'lint-fixture', description: 'Review changes', instructions: 'bun scripts/check.ts', files: { 'scripts/check.ts': 'v1' },
    }, { skillStore: store })
    const original = await readFile(join(root, 'lint-fixture/SKILL.md'), 'utf8')
    // A directory where the update wants a file: its swap fails after check.ts and new.ts were swapped.
    await mkdir(join(root, 'lint-fixture/scripts/blocked.ts/inner'), { recursive: true })
    await expect(skillManageTool({
      action: 'update', name: 'lint-fixture', description: 'Review changes', instructions: 'bun scripts/check.ts and bun scripts/new.ts',
      files: { 'scripts/check.ts': 'v2', 'scripts/new.ts': 'new', 'scripts/blocked.ts': 'blocked' },
    }, { skillStore: store })).rejects.toThrow()
    expect(await readFile(join(root, 'lint-fixture/scripts/check.ts'), 'utf8')).toBe('v1')
    expect(await Bun.file(join(root, 'lint-fixture/scripts/new.ts')).exists()).toBeFalse()
    expect(await readFile(join(root, 'lint-fixture/SKILL.md'), 'utf8')).toBe(original)
    const { readdir } = await import('node:fs/promises')
    expect((await readdir(join(root, 'lint-fixture/scripts'))).filter(name => name.endsWith('.tmp'))).toEqual([])
    expect((await readdir(join(root, 'lint-fixture'))).filter(name => name.endsWith('.tmp'))).toEqual([])
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('supplied bundle paths cannot escape or overwrite SKILL.md, and every invalid path is reported', async () => {
  const root = await mkdtemp(join(tmpdir(), 'harness-paths-'))
  try {
    const store = new SkillBundleStore({ directory: root })
    const result = await skillManageTool({ action: 'create', name: 'lint-fixture', instructions: 'Review.', files: {
      '../escape.ts': 'bad', '/absolute.ts': 'bad', 'SKILL.md': 'bad',
    } }, { skillStore: store })
    expect(result).toMatchObject({ ok: false, error: 'harness_lint_failed' })
    expect(result.report).toEqual(expect.stringContaining('../escape.ts'))
    expect(result.report).toEqual(expect.stringContaining('/absolute.ts'))
    expect(result.report).toEqual(expect.stringContaining('SKILL.md'))
    expect(await Bun.file(join(root, 'lint-fixture/SKILL.md')).exists()).toBeFalse()
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('custom skill stores cannot bypass authoring lint and receive supplied file contents only after validation', async () => {
  const requests: unknown[] = []
  const store = { all: () => [], get: () => undefined, manage: (request: unknown) => { requests.push(request); return { ok: true } } }
  const rejected = await skillManageTool({ action: 'create', name: 'help', instructions: 'bun scripts/missing.ts' }, { skillStore: store })
  expect(rejected).toMatchObject({ ok: false, error: 'harness_lint_failed' })
  expect(requests).toHaveLength(0)
  expect(await skillManageTool({ action: 'create', name: 'lint-fixture', instructions: 'Review.' }, { skillStore: store })).toMatchObject({
    ok: true, warnings: [expect.objectContaining({ code: 'description-missing' })],
  })
  expect(requests).toHaveLength(1)
})

test('missing instructions do not hide the other authoring problems', async () => {
  const root = await mkdtemp(join(tmpdir(), 'harness-empty-'))
  try {
    const result = await skillManageTool({ action: 'create', name: 'help' }, { skillStore: new SkillBundleStore({ directory: root }) })
    expect(result).toMatchObject({ ok: false, findings: expect.arrayContaining([
      expect.objectContaining({ code: 'instructions-missing' }), expect.objectContaining({ code: 'builtin-command-conflict' }),
      expect.objectContaining({ code: 'description-missing' }),
    ]) })
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('invalid bundle-file content is rejected at the tool boundary', async () => {
  const store = { all: () => [], get: () => undefined, manage: () => ({ ok: true }) }
  await expect(skillManageTool({ action: 'create', name: 'lint-fixture', instructions: 'Review.', files: { 'script.ts': 42 } }, { skillStore: store })).rejects.toThrow('files')
})

test('the registered model tool returns the full retry feedback rather than committing a partial bundle', async () => {
  const root = await mkdtemp(join(tmpdir(), 'harness-tool-'))
  try {
    const tools = new ToolRegistry()
    registerAgentMetaTools(tools, { skillStore: new SkillBundleStore({ directory: root }) })
    const result = JSON.parse(await tools.execute({ id: 'lint', type: 'function', function: {
      name: 'skill_manage', arguments: { action: 'create', name: 'help', instructions: 'bun scripts/missing.ts', files: { 'scripts/other.ts': '// shipped' } },
    } }, { metadata: {} }))
    expect(result).toMatchObject({ ok: false, error: 'harness_lint_failed' })
    expect(result.report).toContain('builtin-command-conflict')
    expect(result.report).toContain('script-missing')
    expect(result.report).toContain('description-missing')
    expect(await Bun.file(join(root, 'help/scripts/other.ts')).exists()).toBeFalse()
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('skills lint RPC and completion include all problems, including shadowed sources, with ok true', async () => {
  const root = await mkdtemp(join(tmpdir(), 'harness-rpc-'))
  const first = join(root, 'one')
  const second = join(root, 'two')
  const server = new DaemonServer({
    socketPath: join(root, 'daemon.sock'), skillDirectory: first, skillDirectories: [first, second], exitLogPath: join(root, 'exits.jsonl'),
    runtime: new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') }),
  })
  await server.start()
  const socket = connect({ path: join(root, 'daemon.sock') })
  await new Promise<void>((resolveConnection, reject) => { socket.once('connect', resolveConnection); socket.once('error', reject) })
  socket.setEncoding('utf8')
  let buffer = ''
  let id = 0
  const pending = new Map<number, (value: Record<string, unknown>) => void>()
  socket.on('data', chunk => {
    buffer += String(chunk)
    let newline: number
    while ((newline = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, newline)
      buffer = buffer.slice(newline + 1)
      if (!line.trim()) continue
      const frame = JSON.parse(line) as { id?: number; result?: Record<string, unknown> }
      if (frame.id !== undefined) { pending.get(frame.id)?.(frame.result ?? {}); pending.delete(frame.id) }
    }
  })
  const request = (method: string, params: Record<string, unknown>): Promise<Record<string, unknown>> => {
    const current = ++id
    return new Promise(resolveResult => {
      pending.set(current, resolveResult)
      socket.write(JSON.stringify({ jsonrpc: '2.0', id: current, method, params }) + '\n')
    })
  }
  try {
    await request('initialize', { session_key: 'lint', project_dir: root })
    expect(await request('slash', { command: '/skills lint' })).toMatchObject({ ok: true, findings: [], report: 'Harness lint: no problems found.' })
    for (const directory of [first, second]) await mkdir(join(directory, 'help'), { recursive: true })
    await writeFile(join(first, 'help/SKILL.md'), markdown('help', 'bun scripts/missing.ts', 'description:'))
    await writeFile(join(second, 'help/SKILL.md'), markdown('help'))
    expect(await request('complete', { text: '/skills li' })).toMatchObject({ completions: expect.arrayContaining([
      expect.objectContaining({ value: '/skills lint ' }), expect.objectContaining({ value: '/skills list ' }),
    ]) })
    const result = await request('slash', { command: '/skills lint' })
    expect(result).toMatchObject({ ok: true, findings: expect.arrayContaining([
      expect.objectContaining({ code: 'builtin-command-conflict', severity: 'fail' }),
      expect.objectContaining({ code: 'description-missing', severity: 'warn' }),
      // Installed and shipping no scripts/ folder: the path most likely names the user's project.
      expect.objectContaining({ code: 'script-missing', severity: 'warn' }),
      expect.objectContaining({ code: 'skill-name-duplicate', path: join(first, 'help/SKILL.md') }),
      expect.objectContaining({ code: 'skill-name-duplicate', path: join(second, 'help/SKILL.md') }),
    ]) })
    expect(result.report).toEqual(expect.stringContaining(join(second, 'help/SKILL.md')))
  } finally { socket.destroy(); await server.stop(); await rm(root, { recursive: true, force: true }) }
}, 15_000)
