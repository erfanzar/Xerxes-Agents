// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { MAX_LOCKED_INPUTS, RunSafety } from '../src/runtime/runSafety.js'
import {
  bubblewrapArgs,
  findCommandSandbox,
  seatbeltProfile,
  type CommandSandbox,
  type CommandSandboxHost,
} from '../src/security/commandSandbox.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'
import { runTurn } from '../src/streaming/loop.js'
import { permissionDisposition, type PermissionMode } from '../src/streaming/permissions.js'
import { executeCommand, registerProcessTools, type ProcessResult } from '../src/tools/processTools.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

const folders: string[] = []
const runs: RunSafety[] = []
afterEach(async () => {
  await Promise.all(runs.splice(0).map(run => run.dispose()))
  for (const folder of folders.splice(0)) rmSync(folder, { recursive: true, force: true })
})

/** A task folder with one input file, data.csv. */
function taskFolder(): string {
  const folder = realpathSync(mkdtempSync(join(tmpdir(), 'xerxes-run-safety-')))
  folders.push(folder)
  writeFileSync(join(folder, 'data.csv'), 'region,units\nnorth,3\n')
  return folder
}

async function start(folder: string, sandbox?: CommandSandbox): Promise<RunSafety> {
  const run = await RunSafety.start({ outputFolder: folder, ...(sandbox ? { sandbox } : { sandboxUnavailable: 'none in this test' }) })
  runs.push(run)
  return run
}

function call(name: string, inputs: JsonObject): ToolCall {
  return { id: `call-${name}`, type: 'function', function: { name, arguments: inputs } }
}

const fakeSandbox: CommandSandbox = {
  runner: 'seatbelt',
  bind: policy => ({ runner: 'seatbelt', env: { TMPDIR: policy.tempFolder }, wrap: argv => ['sandboxed', ...argv] }),
}

function disposition(run: RunSafety, target: ToolCall, mode: PermissionMode = 'auto') {
  return permissionDisposition(target, mode, undefined, undefined, { runSafety: run })
}

test('auto asks before a write outside the output folder or to an input, and not for a new output file', async () => {
  const folder = taskFolder()
  const run = await start(folder)
  expect(run.inputs()).toEqual(['data.csv'])
  expect(disposition(run, call('WriteFile', { file_path: 'region_revenue.csv' }))).toBe('allow')
  expect(disposition(run, call('XlsxWriter', { path: 'out/report.xlsx' }))).toBe('allow')
  expect(disposition(run, call('WriteFile', { file_path: 'data.csv' }))).toBe('prompt')
  expect(disposition(run, call('FileEditTool', { file_path: 'data.csv' }))).toBe('prompt')
  expect(disposition(run, call('WriteFile', { file_path: '../elsewhere.csv' }))).toBe('prompt')
  expect(disposition(run, call('WriteFile', { file_path: '/etc/hosts' }))).toBe('prompt')
  // CSVProcessor reads freely, but writing over an input is a write like any other.
  expect(disposition(run, call('CSVProcessor', { operation: 'read', file_path: 'data.csv' }))).toBe('allow')
  expect(disposition(run, call('CSVProcessor', { operation: 'write', file_path: 'data.csv', overwrite: true }))).toBe('prompt')
  expect(disposition(run, call('CSVProcessor', { operation: 'write', file_path: 'clean.csv' }))).toBe('allow')
  // XlsxConvert reads `path` and writes `output_path`: only the output decides.
  expect(disposition(run, call('XlsxConvert', { path: 'data.csv', output_path: 'out/data.xlsx' }))).toBe('allow')
  expect(disposition(run, call('XlsxConvert', { path: 'out/data.xlsx', output_path: 'data.csv', overwrite: true }))).toBe('prompt')
})

test('without a sandbox a command that is not read-only still asks; with one it runs unless it asks to leave it', async () => {
  const folder = taskFolder()
  const python = call('exec_command', { cmd: 'python3', args: ['-'], stdin: 'print(1)' })
  expect(disposition(await start(folder), python)).toBe('prompt')
  const sandboxed = await start(folder, fakeSandbox)
  expect(disposition(sandboxed, python)).toBe('allow')
  expect(disposition(sandboxed, call('exec_command', { cmd: 'curl', args: ['https://example.com'], unsandboxed: true }))).toBe('prompt')
})

test('explicit accept-all and manual keep their meaning; the floor only acts in auto', async () => {
  const run = await start(taskFolder(), fakeSandbox)
  const overwrite = call('WriteFile', { file_path: 'data.csv' })
  expect(disposition(run, overwrite, 'accept-all')).toBe('allow')
  expect(disposition(run, call('WriteFile', { file_path: 'new.csv' }), 'manual')).toBe('prompt')
  expect(disposition(run, call('exec_command', { cmd: 'python3', args: ['x.py'] }), 'plan')).toBe('prompt')
})

test('a symlink inside the folder cannot carry a write outside it', async () => {
  const folder = taskFolder()
  const outside = realpathSync(mkdtempSync(join(tmpdir(), 'xerxes-outside-')))
  folders.push(outside)
  const { symlinkSync } = await import('node:fs')
  symlinkSync(outside, join(folder, 'link'))
  const run = await start(folder)
  expect(run.isOutputWrite('link/escape.csv')).toBe(false)
  expect(run.isOutputWrite('fresh/new.csv')).toBe(true)
})

test('an approved write releases that input, and a changed input is reported once', async () => {
  const folder = taskFolder()
  writeFileSync(join(folder, 'notes.txt'), 'keep me')
  const run = await start(folder, fakeSandbox)
  expect(run.commandSandbox()?.wrap(['python3'])).toEqual(['sandboxed', 'python3'])

  run.approved(call('FileEditTool', { file_path: 'notes.txt' }))
  expect(disposition(run, call('WriteFile', { file_path: 'notes.txt' }))).toBe('allow')
  writeFileSync(join(folder, 'notes.txt'), 'edited with approval')

  writeFileSync(join(folder, 'data.csv'), 'region,units\nnorth,4\n')
  const first = await run.inputFindings()
  expect(first).toEqual([expect.objectContaining({ code: 'input_modified', severity: 'error', path: 'data.csv' })])
  expect(await run.inputFindings()).toEqual([])
  rmSync(join(folder, 'data.csv'))
  expect((await run.inputFindings()).map(finding => finding.message)).toEqual(['An input file was deleted; inputs stay read-only.'])
})

test('.git and node_modules are never written without asking, and a huge folder gets no sandbox', async () => {
  const folder = taskFolder()
  mkdirSync(join(folder, '.git'))
  writeFileSync(join(folder, '.git', 'HEAD'), 'ref: refs/heads/main\n')
  const run = await start(folder, fakeSandbox)
  expect(run.inputs()).toEqual(['data.csv'])
  expect(run.isOutputWrite('.git/HEAD')).toBe(false)
  expect(run.isOutputWrite('.git/new-object')).toBe(false)

  const big = taskFolder()
  for (let index = 0; index <= MAX_LOCKED_INPUTS; index += 1) writeFileSync(join(big, `f${index}.txt`), '')
  const unlocked = await start(big, fakeSandbox)
  expect(unlocked.commandSandbox()).toBeUndefined()
  expect(unlocked.sandboxUnavailable).toContain('more than 2,000 files')
  expect(disposition(unlocked, call('exec_command', { cmd: 'python3', args: ['x.py'] }))).toBe('prompt')
})

test('the Seatbelt profile and bubblewrap arguments put the read-only rules after the writable ones', () => {
  const policy = {
    writable: ['/work', '/tmp/run'],
    readOnlyFolders: ['/work/.git'],
    readOnlyFiles: ['/work/data "1".csv'],
    tempFolder: '/tmp/run',
  }
  const profile = seatbeltProfile(policy).split('\n')
  expect(profile.slice(0, 4)).toEqual(['(version 1)', '(allow default)', '(deny network*)', '(deny mach-lookup)'])
  const allowWrites = profile.findIndex(line => line.startsWith('(allow file-write*'))
  expect(profile[allowWrites]).toContain('(subpath "/work")')
  expect(profile.findIndex(line => line.includes('(subpath "/work/.git")'))).toBeGreaterThan(allowWrites)
  expect(profile.at(-1)).toBe('(deny file-write* (literal "/work/data \\"1\\".csv"))')
  expect(profile.join('\n')).not.toContain('launchservicesd')

  const args = bubblewrapArgs(policy, ['python3', '-'])
  expect(args.slice(0, 4)).toEqual(['--die-with-parent', '--unshare-net', '--ro-bind', '/'])
  expect(args.indexOf('/work/data "1".csv')).toBeGreaterThan(args.indexOf('/tmp/run'))
  expect(args.slice(-3)).toEqual(['--', 'python3', '-'])
})

test('finding a sandbox needs a runner that really starts a command', async () => {
  const host = (platform: NodeJS.Platform, found: string | null, exit: number): CommandSandboxHost => ({
    platform, which: () => found, run: async () => exit,
  })
  expect((await findCommandSandbox(host('darwin', '/usr/bin/sandbox-exec', 0))).sandbox?.runner).toBe('seatbelt')
  expect(await findCommandSandbox(host('linux', null, 0))).toEqual({ reason: expect.stringContaining('bubblewrap (bwrap) is not installed') })
  expect(await findCommandSandbox(host('linux', '/usr/bin/bwrap', 1))).toEqual({ reason: expect.stringContaining('user namespaces') })
  expect((await findCommandSandbox(host('linux', '/usr/bin/bwrap', 0))).sandbox?.runner).toBe('bubblewrap')
  expect(await findCommandSandbox(host('win32', null, 0))).toEqual({ reason: 'no command sandbox is supported on win32' })
})

const onMac = process.platform === 'darwin' ? test : test.skip

onMac('a sandboxed Python step writes new files but not inputs, nothing outside the folder, and no network', async () => {
  const folder = taskFolder()
  const found = await findCommandSandbox()
  if (!found.sandbox) throw new Error(found.reason)
  const run = await start(folder, found.sandbox)
  const sandbox = run.commandSandbox()!
  const outside = join(tmpdir(), `xerxes-outside-${crypto.randomUUID()}.txt`)
  const script = [
    'import socket, tempfile',
    "open('region_revenue.csv', 'w').write('north,9\\n')",
    "results = []",
    "for name, action in [('input', lambda: open('data.csv', 'w').write('x')),",
    `                     ('outside', lambda: open(${JSON.stringify(outside)}, 'w').write('x')),`,
    "                     ('network', lambda: socket.create_connection(('1.1.1.1', 443), timeout=3)),",
    "                     ('temp', lambda: tempfile.NamedTemporaryFile().write(b'x'))]:",
    '    try:',
    '        action(); results.append(name + ":ok")',
    '    except Exception as error:',
    '        results.append(name + ":" + type(error).__name__)',
    "print(' '.join(results))",
  ].join('\n')
  const result = await executeCommand(
    { cmd: 'python3', args: ['-'], stdin: script },
    new WorkspacePathResolver(folder), undefined, undefined, undefined, undefined, sandbox,
  ) as ProcessResult
  expect(result.sandbox).toBe('seatbelt')
  expect(result.stdout.trim()).toBe('input:PermissionError outside:PermissionError network:PermissionError temp:ok')
  expect(readFileSync(join(folder, 'region_revenue.csv'), 'utf8')).toBe('north,9\n')
  expect(readFileSync(join(folder, 'data.csv'), 'utf8')).toBe('region,units\nnorth,3\n')
  expect(existsSync(outside)).toBe(false)
}, 30_000)

onMac('in auto mode the loop runs exec_command in the sandbox without asking', async () => {
  const folder = taskFolder()
  const found = await findCommandSandbox()
  if (!found.sandbox) throw new Error(found.reason)
  const run = await start(folder, found.sandbox)
  const registry = new ToolRegistry()
  registerProcessTools(registry, new WorkspacePathResolver(folder))
  let calls = 0
  const llm: LlmClient = {
    async *stream(_request: CompletionRequest): AsyncGenerator<LlmDelta> {
      calls += 1
      if (calls === 1) {
        yield { toolCalls: [call('exec_command', { cmd: 'python3', args: ['-'], stdin: "open('data.csv','w').write('x')" })] }
        yield { finishReason: 'tool_calls' }
        return
      }
      yield { content: 'done' }
    },
  }
  const state = createAgentState()
  const events: StreamEvent[] = []
  for await (const event of runTurn(
    { model: 'test', state, userMessage: 'overwrite the input', sessionId: 'run-safety', permissionMode: 'auto', tools: registry.definitions() },
    { llm, toolExecutor: registry, runSafety: run },
  )) events.push(event)
  expect(events.some(event => event.type === 'permission_request')).toBe(false)
  const execution = state.toolExecutions.at(-1)!
  expect(execution.permitted).toBe(true)
  expect(JSON.parse(execution.result)).toMatchObject({ sandbox: 'seatbelt', exitCode: 1 })
  expect(readFileSync(join(folder, 'data.csv'), 'utf8')).toBe('region,units\nnorth,3\n')
}, 30_000)

test('the lean prompt tells the model about the sandbox only when commands run in one', async () => {
  const { buildLeanSystemPrompt } = await import('../src/runtime/leanPrompt.js')
  const base = { cwd: '/work', model: 'ternary-bonsai-2-27b-mlx', date: '2026-11-07', toolNames: ['exec_command', 'WriteFile'] }
  expect(buildLeanSystemPrompt({ ...base, commandSandbox: true })).toContain('set unsandboxed: true; the user is asked first.')
  expect(buildLeanSystemPrompt(base)).not.toContain('unsandboxed')
})
