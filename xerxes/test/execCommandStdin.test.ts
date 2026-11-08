// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, test } from 'bun:test'

import { ValidationError } from '../src/core/errors.js'
import { selectLeanTools } from '../src/runtime/leanPrompt.js'
import { DataBoundary, parseDataBoundaryConfig } from '../src/security/dataBoundary.js'
import { DockerSandboxAdapter, SandboxBackendAdapterRequestError } from '../src/security/sandboxBackends/index.js'
import { SubprocessSandboxBackend } from '../src/security/subprocessSandbox.js'
import { permissionDescription, permissionDisposition } from '../src/streaming/permissions.js'
import { BackgroundCommandManager } from '../src/tools/backgroundCommands.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'
import { EXEC_COMMAND_DEFINITION, MAX_STDIN_CHARS, executeCommand } from '../src/tools/processTools.js'
import { ProcessRegistry } from '../src/runtime/processRegistry.js'
import type { JsonObject } from '../src/types/toolCalls.js'

const roots: string[] = []
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function workspace(): string {
  const root = realpathSync(mkdtempSync(join(tmpdir(), 'xerxes-stdin-')))
  roots.push(root)
  mkdirSync(join(root, 'secret'), { recursive: true })
  writeFileSync(join(root, 'secret', 'customers.csv'), 'name\nalice\n')
  writeFileSync(join(root, 'notes.txt'), 'one\ntwo\n')
  return root
}

const call = (args: JsonObject) => ({ function: { name: 'exec_command', arguments: args } })

test('stdin reaches the program in full, so an interpreter can run a script given as plain text', async () => {
  const paths = new WorkspacePathResolver(workspace())
  const script = 'total=0\nfor n in 3 4 5; do total=$((total + n)); done\necho "sum=$total"\n'
  const result = await executeCommand({ cmd: 'sh', args: ['-s'], stdin: script }, paths)
  expect(result).toMatchObject({ command: ['sh', '-s'], exitCode: 0, stdout: 'sum=12\n', timedOut: false })

  const echoed = await executeCommand({ cmd: 'cat', stdin: 'line 1\nline "2"\n' }, paths)
  expect(echoed).toMatchObject({ exitCode: 0, stdout: 'line 1\nline "2"\n' })
})

test('without stdin the program still reads an empty standard input', async () => {
  const paths = new WorkspacePathResolver(workspace())
  // `cat` with no file waits on stdin; it must see end-of-input at once.
  const result = await executeCommand({ cmd: 'cat', timeout_ms: 5_000 }, paths)
  expect(result).toMatchObject({ exitCode: 0, stdout: '', timedOut: false })
})

test('stdin is refused with an actionable error when it is too large or the command runs in the background', async () => {
  const paths = new WorkspacePathResolver(workspace())
  const background = new BackgroundCommandManager(new ProcessRegistry())
  await expect(executeCommand({ cmd: 'cat', stdin: 'x'.repeat(MAX_STDIN_CHARS + 1) }, paths)).rejects.toBeInstanceOf(ValidationError)
  const backgrounded = executeCommand({ cmd: 'cat', stdin: 'x', run_in_background: true }, paths, undefined, background)
  await expect(backgrounded).rejects.toThrow(/stdin.*run_in_background/)
})

test('a command given stdin is never auto-approved, and the approval shows the text', () => {
  const plain = call({ cmd: 'cat', args: ['notes.txt'] })
  const piped = call({ cmd: 'cat', args: ['-'], stdin: 'echo hi' })
  expect(permissionDisposition(plain, 'auto')).toBe('allow')
  expect(permissionDisposition(piped, 'auto')).toBe('prompt')
  expect(permissionDisposition(piped, 'plan')).toBe('prompt')
  expect(permissionDisposition(piped, 'accept-all')).toBe('allow')
  expect(permissionDescription(call({ cmd: 'python3', args: ['-'], stdin: 'import os\nprint(os.getcwd())' })))
    .toBe('Run: python3 - with 2 lines on stdin:\nimport os\nprint(os.getcwd())')
})

test('the data boundary scans stdin like a command and never treats it as read-only', () => {
  const root = workspace()
  const gate = new DataBoundary(parseDataBoundaryConfig({ data_boundary: 'hybrid', boundary: { private_paths: ['secret/**'] } }, {}), root)
  const reads = call({ cmd: 'python3', args: ['-'], stdin: "print(open('secret/customers.csv').read())" })
  expect(gate.toolDisposition(reads, { trust: 'cloud' })).toBe('deny')
  expect(gate.toolDisposition(reads, { trust: 'local' })).toBe('prompt')
  expect(gate.toolDisposition(call({ cmd: 'cat', args: ['notes.txt'] }), { trust: 'cloud' })).toBeUndefined()
  expect(gate.toolDisposition(call({ cmd: 'cat', args: ['notes.txt'], stdin: 'more' }), { trust: 'cloud' })).toBe('prompt')
})

test('the subprocess sandbox passes stdin through; a host-port sandbox refuses it rather than drop it', async () => {
  const root = workspace()
  const local = new SubprocessSandboxBackend({ allowedCommands: [process.execPath], workingDirectory: root })
  const output = JSON.parse(await local.execute({
    toolName: 'exec_command',
    arguments: { cmd: process.execPath, args: ['-e', 'process.stdout.write((await Bun.stdin.text()).toUpperCase())'], stdin: 'shout' },
    context: { metadata: {} },
  })) as { exitCode: number; stdout: string }
  expect(output).toMatchObject({ exitCode: 0, stdout: 'SHOUT' })

  let ran = false
  const docker = new DockerSandboxAdapter({
    config: { image: 'registry.example/xerxes:bun', workingDirectory: '/workspace' },
    host: { async runContainer() { ran = true; return { exitCode: 0, stderr: '', stdout: '', timedOut: false, truncated: false } } },
  })
  await expect(docker.execute({ toolName: 'exec_command', arguments: { cmd: 'python3', args: ['-'], stdin: 'print(1)' }, context: { metadata: {} } }))
    .rejects.toBeInstanceOf(SandboxBackendAdapterRequestError)
  expect(ran).toBe(false)
})

test('the lean tool set keeps stdin and its hint within the shortened schema', () => {
  const [lean] = selectLeanTools([EXEC_COMMAND_DEFINITION])
  const stdin = (lean!.function.parameters as { properties: Record<string, { description?: string; type?: string }> }).properties.stdin
  expect(stdin?.type).toBe('string')
  expect(stdin?.description).toBe('Text written to standard input, e.g. a script for python3 with args ["-"].')
})
