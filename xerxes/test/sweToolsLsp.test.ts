// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { chmodSync, existsSync, readFileSync, realpathSync, utimesSync, writeFileSync } from 'node:fs'
import { mkdir, mkdtemp, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, describe, expect, test } from 'bun:test'

import { loadBuiltinAgentDefinitions, type AgentDefinition } from '../src/agents/definitions.js'
import { FunctionExecutionError } from '../src/core/errors.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  LspManager,
  discoverWorkspaceLspServers,
  loadConfiguredLsp,
  mergeDiscoveredLspServers,
  parseLspConfig,
} from '../src/lsp/index.js'
import { DataBoundary, parseDataBoundaryConfig } from '../src/security/dataBoundary.js'
import { fileStateTracker, setFileFreshnessEnforcement } from '../src/tools/fileState.js'
import {
  APPLY_PATCH_TOOL_DEFINITION,
  FILE_OUTLINE_TOOL_DEFINITION,
  discoverRelatedTests,
  extractFileOutline,
  registerCoreTools,
} from '../src/tools/index.js'
import { LSP_RUNTIME_TOOL_NAME, addLspToolToBuiltinAgents } from '../src/tools/lspTools.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

const tempRoots: string[] = []

afterEach(async () => {
  fileStateTracker.clear()
  setFileFreshnessEnforcement(undefined)
  for (const root of tempRoots.splice(0)) {
    await rm(root, { force: true, recursive: true })
  }
})

async function makeWorkspace(prefix = 'xerxes-swe-tools-'): Promise<string> {
  const dir = realpathSync(await mkdtemp(join(tmpdir(), prefix)))
  tempRoots.push(dir)
  return dir
}

function toolCall(name: string, args: JsonObject): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: 'function',
    function: { name, arguments: args },
  }
}

describe('Core registration & capabilities contract', () => {
  test('registers ApplyPatchTool and FileOutlineTool with accurate capabilities', async () => {
    const workspace = await makeWorkspace()
    const registry = new ToolRegistry()
    registerCoreTools(registry, { includeProcessTools: false, workspaceRoot: workspace })

    const names = registry.definitions().map(def => def.function.name)
    expect(names).toContain('ApplyPatchTool')
    expect(names).toContain('FileOutlineTool')
    expect(APPLY_PATCH_TOOL_DEFINITION.function.name).toBe('ApplyPatchTool')
    expect(FILE_OUTLINE_TOOL_DEFINITION.function.name).toBe('FileOutlineTool')

    const patchCaps = registry.capabilities('ApplyPatchTool')
    expect(patchCaps.concurrencySafe).toBe(false)
    expect(patchCaps.destructive).toBe(true)
    expect(patchCaps.readOnly).toBe(false)
    expect(patchCaps.interruptBehavior).toBe('block')
    expect(patchCaps.openWorld).toBe(false)
    expect(patchCaps.defer).toBe(false)

    const outlineCaps = registry.capabilities('FileOutlineTool')
    expect(outlineCaps.concurrencySafe).toBe(true)
    expect(outlineCaps.destructive).toBe(false)
    expect(outlineCaps.readOnly).toBe(true)
    expect(outlineCaps.openWorld).toBe(false)
    expect(outlineCaps.defer).toBe(false)
  })
})

describe('ApplyPatchTool — V4A & Unified Diff atomic execution', () => {
  test('commits multi-file V4A patch (update multi-hunk, add, move+update, delete) and syncs fileStateTracker', async () => {
    const workspace = await makeWorkspace()
    await mkdir(join(workspace, 'src'), { recursive: true })
    await mkdir(join(workspace, 'test'), { recursive: true })

    writeFileSync(
      join(workspace, 'src', 'calc.ts'),
      [
        'export function add(a: number, b: number): number {',
        '  return a - b',
        '}',
        '',
        'export function multiply(a: number, b: number): number {',
        '  return a + b',
        '}',
        '',
      ].join('\n'),
    )
    writeFileSync(join(workspace, 'src', 'legacy.ts'), 'export const VERSION = 1\n')
    chmodSync(join(workspace, 'src', 'legacy.ts'), 0o755)
    writeFileSync(join(workspace, 'src', 'obsolete.ts'), 'export const UNUSED = true\n')
    writeFileSync(join(workspace, 'test', 'calc.test.ts'), 'import { add } from "../src/calc.js"\n')
    writeFileSync(join(workspace, 'bun.lock'), '')

    const registry = new ToolRegistry()
    const paths = registerCoreTools(registry, { includeProcessTools: false, workspaceRoot: workspace })
    const context = { metadata: {}, sessionId: 'sess-v4a' }

    // Seed fileStateTracker on obsolete.ts and legacy.ts to verify forget on delete/move-source
    await registry.execute(toolCall('ReadFile', { file_path: 'src/obsolete.ts' }), context)
    await registry.execute(toolCall('ReadFile', { file_path: 'src/legacy.ts' }), context)

    const patch = [
      '*** Begin Patch',
      '*** Update File: src/calc.ts',
      '@@ export function add',
      ' export function add(a: number, b: number): number {',
      '-  return a - b',
      '+  return a + b',
      ' }',
      '@@ export function multiply',
      ' export function multiply(a: number, b: number): number {',
      '-  return a + b',
      '+  return a * b',
      ' }',
      '*** Add File: src/nested/constants.ts',
      '+export const SCALE = 10',
      '+export const LABEL = "ok"',
      '*** Update File: src/legacy.ts',
      '*** Move to: src/version.ts',
      '@@',
      '-export const VERSION = 1',
      '+export const VERSION = 2',
      '*** Delete File: src/obsolete.ts',
      '*** End Patch',
    ].join('\n')

    const output = await registry.execute(toolCall('ApplyPatchTool', { patch }), context)
    expect(output).toContain('M src/calc.ts')
    expect(output).toContain('A src/nested/constants.ts')
    expect(output).toContain('R src/legacy.ts -> src/version.ts')
    expect(output).toContain('D src/obsolete.ts')
    expect(output).toContain('[related_tests] test/calc.test.ts')
    expect(output).toContain('bun test test/calc.test.ts')

    expect(readFileSync(join(workspace, 'src', 'calc.ts'), 'utf8')).toBe(
      [
        'export function add(a: number, b: number): number {',
        '  return a + b',
        '}',
        '',
        'export function multiply(a: number, b: number): number {',
        '  return a * b',
        '}',
        '',
      ].join('\n'),
    )
    expect(readFileSync(join(workspace, 'src', 'nested', 'constants.ts'), 'utf8')).toBe(
      'export const SCALE = 10\nexport const LABEL = "ok"\n',
    )
    expect(readFileSync(join(workspace, 'src', 'version.ts'), 'utf8')).toBe('export const VERSION = 2\n')
    expect(existsSync(join(workspace, 'src', 'legacy.ts'))).toBe(false)
    expect(existsSync(join(workspace, 'src', 'obsolete.ts'))).toBe(false)

    // Verify fileStateTracker was invalidated for deleted & moved-from paths and recorded for written paths.
    const obsoleteAbs = await paths.resolve('src/obsolete.ts')
    const legacyAbs = await paths.resolve('src/legacy.ts')
    const calcAbs = await paths.resolve('src/calc.ts')
    const versionAbs = await paths.resolve('src/version.ts')
    const addedAbs = await paths.resolve('src/nested/constants.ts')

    expect(fileStateTracker.peek('sess-v4a', obsoleteAbs)).toBeUndefined()
    expect(fileStateTracker.peek('sess-v4a', legacyAbs)).toBeUndefined()
    expect(fileStateTracker.peek('sess-v4a', calcAbs)).toBeDefined()
    expect(fileStateTracker.peek('sess-v4a', versionAbs)).toBeDefined()
    expect(fileStateTracker.peek('sess-v4a', addedAbs)).toBeDefined()

    // Subsequent FileEditTool on calc.ts and version.ts succeeds without re-reading!
    const followUp = await registry.execute(
      toolCall('FileEditTool', {
        file_path: 'src/version.ts',
        old_string: 'export const VERSION = 2',
        new_string: 'export const VERSION = 3',
      }),
      context,
    )
    expect(followUp).toContain('Applied 1 replacement(s)')
    expect(readFileSync(join(workspace, 'src', 'version.ts'), 'utf8')).toBe('export const VERSION = 3\n')
  })

  test('commits standard multi-file unified diff (modify, create via /dev/null, delete via /dev/null)', async () => {
    const workspace = await makeWorkspace()
    await mkdir(join(workspace, 'src'), { recursive: true })
    writeFileSync(
      join(workspace, 'src', 'service.ts'),
      'const retries = 1\nconst timeout = 100\nexport function run() {\n  return retries * timeout\n}\n',
    )
    writeFileSync(join(workspace, 'src', 'drop.ts'), 'const dead = true\n')

    const registry = new ToolRegistry()
    registerCoreTools(registry, { includeProcessTools: false, workspaceRoot: workspace })
    const context = { metadata: {}, sessionId: 'sess-unified' }

    const unified = [
      'diff --git a/src/service.ts b/src/service.ts',
      '--- a/src/service.ts',
      '+++ b/src/service.ts',
      '@@ -1,5 +1,5 @@',
      '-const retries = 1',
      '+const retries = 3',
      ' const timeout = 100',
      ' export function run() {',
      '-  return retries * timeout',
      '+  return retries * timeout + 1',
      ' }',
      'diff --git a/src/created.ts b/src/created.ts',
      'new file mode 100644',
      '--- /dev/null',
      '+++ b/src/created.ts',
      '@@ -0,0 +1,2 @@',
      '+export const CREATED = true',
      '+export const MODE = "strict"',
      'diff --git a/src/drop.ts b/src/drop.ts',
      'deleted file mode 100644',
      '--- a/src/drop.ts',
      '+++ /dev/null',
      '@@ -1 +0,0 @@',
      '-const dead = true',
    ].join('\n')

    const result = await registry.execute(toolCall('ApplyPatchTool', { patch: unified }), context)
    expect(result).toContain('M src/service.ts')
    expect(result).toContain('A src/created.ts')
    expect(result).toContain('D src/drop.ts')
    expect(readFileSync(join(workspace, 'src', 'service.ts'), 'utf8')).toBe(
      'const retries = 3\nconst timeout = 100\nexport function run() {\n  return retries * timeout + 1\n}\n',
    )
    expect(readFileSync(join(workspace, 'src', 'created.ts'), 'utf8')).toBe(
      'export const CREATED = true\nexport const MODE = "strict"\n',
    )
    expect(existsSync(join(workspace, 'src', 'drop.ts'))).toBe(false)
  })

  test('rolls back atomically (zero files modified) when hunk #2 of file #2 fails and returns pinpoint diagnostics', async () => {
    const workspace = await makeWorkspace()
    await mkdir(join(workspace, 'src'), { recursive: true })

    const originalFirst = 'export const alpha = 1\n'
    const originalSecond = [
      'export function stepOne() {',
      '  return "one"',
      '}',
      '',
      'export function stepTwo(input: number) {',
      '  const doubled = input * 2',
      '  return doubled + 1',
      '}',
      '',
    ].join('\n')
    const originalKeep = 'export const keepMe = true\n'

    writeFileSync(join(workspace, 'src', 'first.ts'), originalFirst)
    writeFileSync(join(workspace, 'src', 'second.ts'), originalSecond)
    writeFileSync(join(workspace, 'src', 'keep.ts'), originalKeep)

    const registry = new ToolRegistry()
    registerCoreTools(registry, { includeProcessTools: false, workspaceRoot: workspace })
    const context = { metadata: {}, sessionId: 'sess-rollback' }

    const failingPatch = [
      '*** Begin Patch',
      '*** Update File: src/first.ts',
      '@@',
      '-export const alpha = 1',
      '+export const alpha = 99',
      '*** Delete File: src/keep.ts',
      '*** Add File: src/brand_new/ghost.ts',
      '+export const ghost = true',
      '*** Update File: src/second.ts',
      '@@ export function stepOne',
      ' export function stepOne() {',
      '-  return "one"',
      '+  return "ONE"',
      ' }',
      '@@ export function stepTwo',
      ' export function stepTwo(input: number) {',
      '-  const doubled = input * 999',
      '   return doubled + 1',
      ' }',
      '*** End Patch',
    ].join('\n')

    let caught: unknown
    try {
      await registry.execute(toolCall('ApplyPatchTool', { patch: failingPatch }), context)
    } catch (error) {
      caught = error
    }

    expect(caught).toBeInstanceOf(FunctionExecutionError)
    const message = String(caught)
    // Pinpoint diagnostic: exact file, hunk #2 + header, and closest candidate line number!
    expect(message).toContain('src/second.ts')
    expect(message).toContain('hunk #2')
    expect(message).toContain('@@ export function stepTwo')
    expect(message).toContain('line 5')
    expect(message).toContain('zero files were modified')

    // Zero files modified or created on disk!
    expect(readFileSync(join(workspace, 'src', 'first.ts'), 'utf8')).toBe(originalFirst)
    expect(readFileSync(join(workspace, 'src', 'second.ts'), 'utf8')).toBe(originalSecond)
    expect(readFileSync(join(workspace, 'src', 'keep.ts'), 'utf8')).toBe(originalKeep)
    expect(existsSync(join(workspace, 'src', 'brand_new'))).toBe(false)
  })

  test('applies uniform leading indentation fallback when unique, and rejects ambiguous or non-uniform indentation', async () => {
    const workspace = await makeWorkspace()
    await mkdir(join(workspace, 'src'), { recursive: true })

    // 4-space base indentation on disk
    writeFileSync(
      join(workspace, 'src', 'indented.ts'),
      [
        'export class Worker {',
        '  run(items: number[]) {',
        '    if (items.length > 0) {',
        '      return items[0]!',
        '    }',
        '    return 0',
        '  }',
        '}',
        '',
      ].join('\n'),
    )

    const registry = new ToolRegistry()
    registerCoreTools(registry, { includeProcessTools: false, workspaceRoot: workspace })
    const context = { metadata: {} }

    // Patch supplies 0-space base indentation, preserving 2-space internal relative indent
    const uniqueIndentPatch = [
      '*** Begin Patch',
      '*** Update File: src/indented.ts',
      '@@',
      ' if (items.length > 0) {',
      '-  return items[0]!',
      '+  const head = items[0]!',
      '+  return head * 2',
      ' }',
      '*** End Patch',
    ].join('\n')

    const okResult = await registry.execute(toolCall('ApplyPatchTool', { patch: uniqueIndentPatch }), context)
    expect(okResult).toContain('uniform indentation normalization')
    expect(readFileSync(join(workspace, 'src', 'indented.ts'), 'utf8')).toBe(
      [
        'export class Worker {',
        '  run(items: number[]) {',
        '    if (items.length > 0) {',
        '      const head = items[0]!',
        '      return head * 2',
        '    }',
        '    return 0',
        '  }',
        '}',
        '',
      ].join('\n'),
    )

    // Ambiguous duplicate block under normalized indentation must be refused untouched!
    const duplicateContent = [
      'function a() {',
      '    if (ready) {',
      '      return 1',
      '    }',
      '}',
      'function b() {',
      '  if (ready) {',
      '    return 1',
      '  }',
      '}',
      '',
    ].join('\n')
    writeFileSync(join(workspace, 'src', 'dup.ts'), duplicateContent)

    const ambiguousPatch = [
      '*** Begin Patch',
      '*** Update File: src/dup.ts',
      '@@',
      ' if (ready) {',
      '-  return 1',
      '+  return 2',
      ' }',
      '*** End Patch',
    ].join('\n')

    await expect(registry.execute(toolCall('ApplyPatchTool', { patch: ambiguousPatch }), context)).rejects.toThrow(
      /matches 2 locations.*lines 2, 7/i,
    )
    expect(readFileSync(join(workspace, 'src', 'dup.ts'), 'utf8')).toBe(duplicateContent)

    // Non-uniform indentation mismatch must be refused with an explicit whitespace/indentation hint!
    const nonUniformPatch = [
      '*** Begin Patch',
      '*** Update File: src/indented.ts',
      '@@',
      ' if (items.length > 0) {',
      '-      const head = items[0]!',
      '+      const head = items[0] ?? 0',
      '   return head * 2',
      ' }',
      '*** End Patch',
    ].join('\n')

    await expect(registry.execute(toolCall('ApplyPatchTool', { patch: nonUniformPatch }), context)).rejects.toThrow(
      /Indentation\/whitespace mismatch near line 3/,
    )
  })

  test('enforces workspace path containment, symlink escape rejection, and fileStateTracker freshness when tracked', async () => {
    const workspace = await makeWorkspace()
    const outside = await makeWorkspace('xerxes-swe-outside-')
    writeFileSync(join(outside, 'secret.ts'), 'export const SECRET = 1\n')
    writeFileSync(join(workspace, 'good.ts'), 'export const GOOD = 1\n')
    await symlink(outside, join(workspace, 'link_out'))

    const registry = new ToolRegistry()
    registerCoreTools(registry, { includeProcessTools: false, workspaceRoot: workspace })
    const context = { metadata: {}, sessionId: 'sess-guard' }

    // Escape via `..` in second operation leaves `good.ts` untouched
    const traversalPatch = [
      '*** Begin Patch',
      '*** Update File: good.ts',
      '@@',
      '-export const GOOD = 1',
      '+export const GOOD = 2',
      '*** Update File: ../escape.ts',
      '@@',
      '-a',
      '+b',
      '*** End Patch',
    ].join('\n')
    await expect(registry.execute(toolCall('ApplyPatchTool', { patch: traversalPatch }), context)).rejects.toThrow(
      /escapes workspace root/,
    )
    expect(readFileSync(join(workspace, 'good.ts'), 'utf8')).toBe('export const GOOD = 1\n')

    // Symlink escape leaves `good.ts` untouched
    const symlinkPatch = [
      '*** Begin Patch',
      '*** Update File: good.ts',
      '@@',
      '-export const GOOD = 1',
      '+export const GOOD = 2',
      '*** Update File: link_out/secret.ts',
      '@@',
      '-export const SECRET = 1',
      '+export const SECRET = 2',
      '*** End Patch',
    ].join('\n')
    await expect(registry.execute(toolCall('ApplyPatchTool', { patch: symlinkPatch }), context)).rejects.toThrow(
      /resolves outside workspace root/,
    )
    expect(readFileSync(join(workspace, 'good.ts'), 'utf8')).toBe('export const GOOD = 1\n')

    // Tracked freshness check: read good.ts, simulate external edit on disk, then ApplyPatchTool refuses!
    await registry.execute(toolCall('ReadFile', { file_path: 'good.ts' }), context)
    writeFileSync(join(workspace, 'good.ts'), 'export const GOOD = 1\n// concurrent writer\n')
    const future = new Date(Date.now() + 5_000)
    utimesSync(join(workspace, 'good.ts'), future, future)

    const stalePatch = [
      '*** Begin Patch',
      '*** Update File: good.ts',
      '@@',
      '-export const GOOD = 1',
      '+export const GOOD = 3',
      '*** End Patch',
    ].join('\n')
    await expect(registry.execute(toolCall('ApplyPatchTool', { patch: stalePatch }), context)).rejects.toThrow(
      /changed on disk after you read it/,
    )
    expect(readFileSync(join(workspace, 'good.ts'), 'utf8')).toBe('export const GOOD = 1\n// concurrent writer\n')
  })
})

describe('DataBoundary enforcement for ApplyPatchTool & FileOutlineTool', () => {
  test('enforces private_paths, cloud denial, local tainting, and read_only_inputs across both tools', async () => {
    const workspace = await makeWorkspace('xerxes-swe-boundary-')
    await mkdir(join(workspace, 'secret'), { recursive: true })
    await mkdir(join(workspace, 'src'), { recursive: true })
    await mkdir(join(workspace, 'out'), { recursive: true })
    writeFileSync(join(workspace, 'secret', 'keys.ts'), 'export const KEY = "private"\n')
    writeFileSync(join(workspace, 'src', 'app.ts'), 'export const APP = 1\n')
    writeFileSync(join(workspace, 'out', 'generated.ts'), 'export const GEN = 1\n')

    const boundary = new DataBoundary(
      parseDataBoundaryConfig(
        {
          data_boundary: 'hybrid',
          boundary: { private_paths: ['secret/**'], read_only_inputs: true, output_dir: 'out' },
        },
        {},
      ),
      workspace,
    )

    const cloud = { trust: 'cloud' as const, agentId: 'cloud-lead' }
    const local = { trust: 'local' as const, agentId: 'local-worker' }

    // FileOutlineTool: cloud denied on private file, private dir, root dir overlapping private, and private glob
    expect(boundary.toolDisposition({ function: { name: 'FileOutlineTool', arguments: { file_path: 'secret/keys.ts' } } }, cloud)).toBe('deny')
    expect(boundary.toolDisposition({ function: { name: 'FileOutlineTool', arguments: { file_path: 'secret' } } }, cloud)).toBe('deny')
    expect(boundary.toolDisposition({ function: { name: 'FileOutlineTool', arguments: { file_path: '.' } } }, cloud)).toBe('deny')
    expect(boundary.toolDisposition({ function: { name: 'FileOutlineTool', arguments: { file_path: 'secret/*.ts' } } }, cloud)).toBe('deny')
    // Non-private file and non-private glob are allowed for cloud
    expect(boundary.toolDisposition({ function: { name: 'FileOutlineTool', arguments: { file_path: 'src/app.ts' } } }, cloud)).toBeUndefined()
    expect(boundary.toolDisposition({ function: { name: 'FileOutlineTool', arguments: { file_path: 'src/*.ts' } } }, cloud)).toBeUndefined()

    // Local worker outlining private file is allowed and marks taint
    expect(boundary.isTainted('local-worker')).toBe(false)
    expect(boundary.toolDisposition({ function: { name: 'FileOutlineTool', arguments: { file_path: 'secret/keys.ts' } } }, local)).toBeUndefined()
    expect(boundary.isTainted('local-worker')).toBe(true)

    // ApplyPatchTool: cloud denied when touching private path (even in unified diff or malformed patch)
    const privatePatch = '*** Begin Patch\n*** Update File: secret/keys.ts\n@@\n-export const KEY = "private"\n+export const KEY = "x"\n*** End Patch'
    const cloudVerdict = boundary.toolVerdict({ function: { name: 'ApplyPatchTool', arguments: { patch: privatePatch } } }, cloud)
    expect(cloudVerdict?.disposition).toBe('deny')
    expect(cloudVerdict?.code).toBe('DATA_BOUNDARY_DENIED')

    // ApplyPatchTool with read_only_inputs=true: modifying existing src/app.ts is denied, writing into out/ is allowed!
    const modifyInputPatch = '*** Begin Patch\n*** Update File: src/app.ts\n@@\n-export const APP = 1\n+export const APP = 2\n*** End Patch'
    const readOnlyVerdict = boundary.toolVerdict({ function: { name: 'ApplyPatchTool', arguments: { patch: modifyInputPatch } } }, local)
    expect(readOnlyVerdict?.disposition).toBe('deny')
    expect(readOnlyVerdict?.code).toBe('DATA_BOUNDARY_READ_ONLY')

    const outputPatch = '*** Begin Patch\n*** Update File: out/generated.ts\n@@\n-export const GEN = 1\n+export const GEN = 2\n*** Add File: src/new_helper.ts\n+export const NEW = 1\n*** End Patch'
    expect(boundary.toolDisposition({ function: { name: 'ApplyPatchTool', arguments: { patch: outputPatch } } }, local)).toBeUndefined()
  })
})

describe('FileOutlineTool — TS/JS, Python, Go, and Rust structural extraction', () => {
  test('extracts exact 1-based startLine-endLine spans across TypeScript, Python, Go, and Rust and supports batch & filters', async () => {
    const workspace = await makeWorkspace('xerxes-swe-outline-')
    await mkdir(join(workspace, 'src'), { recursive: true })

    // 1. TypeScript fixture (with comments, regexes, template literals containing braces)
    const tsSource = [
      'import { readFile } from "node:fs/promises"', // 1
      'import {',                                    // 2
      '  join,',                                     // 3
      '} from "node:path"',                          // 4
      '',                                            // 5
      'export interface Config {',                   // 6
      '  load(path: string): Promise<string>',       // 7
      '}',                                           // 8
      '',                                            // 9
      'export type Mode =',                          // 10
      '  | "fast"',                                  // 11
      '  | "safe"',                                  // 12
      '',                                            // 13
      'export enum Status {',                        // 14
      '  Idle = "idle",',                            // 15
      '  Ready = "ready",',                          // 16
      '}',                                           // 17
      '',                                            // 18
      'export class Engine {',                       // 19
      '  private secret = "{not a block}"',          // 20
      '',                                            // 21
      '  async start(root: string): Promise<boolean> {', // 22
      '    const tpl = `path: ${join(root, "a")}`',  // 23
      '    const rx = /\\{[a-z]+\\}/g',              // 24
      '    return tpl.length > 0 && Boolean(rx)',    // 25
      '  }',                                         // 26
      '',                                            // 27
      '  private helper(): number {',                // 28
      '    return 42',                               // 29
      '  }',                                         // 30
      '}',                                           // 31
      '',                                            // 32
      'function internalUtil(): void {',             // 33
      '  /* { ignored brace } */',                   // 34
      '}',                                           // 35
      '',                                            // 36
      'export const DEFAULT_TIMEOUT = 5000',         // 37
    ].join('\n') + '\n'
    writeFileSync(join(workspace, 'src', 'engine.ts'), tsSource)

    const tsOutline = extractFileOutline('src/engine.ts', tsSource)
    expect(tsOutline.language).toBe('typescript')
    expect(tsOutline.imports).toEqual({
      count: 2,
      startLine: 1,
      endLine: 4,
      modules: ['node:fs/promises', 'node:path'],
    })
    expect(tsOutline.symbols.map(s => [s.name, s.kind, s.startLine, s.endLine, s.exported])).toEqual([
      ['Config', 'interface', 6, 8, true],
      ['Mode', 'type', 10, 12, true],
      ['Status', 'enum', 14, 17, true],
      ['Engine', 'class', 19, 31, true],
      ['internalUtil', 'function', 33, 35, false],
      ['DEFAULT_TIMEOUT', 'constant', 37, 37, true],
    ])
    const engineClass = tsOutline.symbols.find(s => s.name === 'Engine')!
    expect(engineClass.children?.map(c => [c.name, c.startLine, c.endLine, c.exported])).toEqual([
      ['start', 22, 26, true],
      ['helper', 28, 30, false],
    ])

    // 2. Python fixture (decorators, multiline signature, triple-quoted docstring with fake `def`)
    const pySource = [
      'import os',                                   // 1
      'from typing import (',                        // 2
      '    Optional,',                               // 3
      ')',                                           // 4
      '',                                            // 5
      'MAX_RETRIES = 3',                             // 6
      '',                                            // 7
      '@dataclass',                                  // 8
      'class Runner:',                               // 9
      '    """',                                     // 10
      '    def fake_inside_docstring():',            // 11
      '        pass',                                // 12
      '    """',                                     // 13
      '    async def execute(',                      // 14
      '        self,',                               // 15
      '        task: str,',                          // 16
      '    ) -> Optional[str]:',                     // 17
      '        if not task:',                        // 18
      '            return None',                     // 19
      '        return os.path.basename(task)',       // 20
      '',                                            // 21
      'def _private_helper(x: int) -> int:',         // 22
      '    return x + 1',                            // 23
    ].join('\n') + '\n'
    writeFileSync(join(workspace, 'src', 'runner.py'), pySource)

    const pyOutline = extractFileOutline('src/runner.py', pySource)
    expect(pyOutline.imports).toEqual({
      count: 2,
      startLine: 1,
      endLine: 4,
      modules: ['os', 'typing'],
    })
    expect(pyOutline.symbols.map(s => [s.name, s.kind, s.startLine, s.endLine, s.exported])).toEqual([
      ['MAX_RETRIES', 'constant', 6, 6, true],
      ['Runner', 'class', 8, 20, true],
      ['_private_helper', 'function', 22, 23, false],
    ])
    expect(pyOutline.symbols[1]!.children?.map(c => [c.name, c.kind, c.startLine, c.endLine])).toEqual([
      ['execute', 'method', 14, 20],
    ])

    // 3. Go fixture
    const goSource = [
      'package server',                              // 1
      '',                                            // 2
      'import (',                                    // 3
      '  "context"',                                 // 4
      '  "net/http"',                                // 5
      ')',                                           // 6
      '',                                            // 7
      'const DefaultPort = 8080',                    // 8
      '',                                            // 9
      'type Handler interface {',                    // 10
      '  Serve(ctx context.Context) error',          // 11
      '}',                                           // 12
      '',                                            // 13
      'type Server struct {',                        // 14
      '  port int',                                  // 15
      '}',                                           // 16
      '',                                            // 17
      'func (s *Server) Start(ctx context.Context) error {', // 18
      '  _ = http.StatusOK',                         // 19
      '  return nil',                                // 20
      '}',                                           // 21
      '',                                            // 22
      'func dialInternal() error {',                 // 23
      '  return nil',                                // 24
      '}',                                           // 25
    ].join('\n') + '\n'
    writeFileSync(join(workspace, 'src', 'server.go'), goSource)

    const goOutline = extractFileOutline('src/server.go', goSource, { exportedOnly: true })
    expect(goOutline.symbols.map(s => [s.name, s.kind, s.startLine, s.endLine])).toEqual([
      ['DefaultPort', 'constant', 8, 8],
      ['Handler', 'interface', 10, 12],
      ['Server', 'struct', 14, 16],
    ])
    expect(goOutline.symbols[2]!.children?.map(c => [c.name, c.startLine, c.endLine])).toEqual([
      ['Server.Start', 18, 21],
    ])

    // 4. Rust fixture (including `'a` lifetimes and `r#"..."#` raw strings with braces!)
    const rsSource = [
      'use std::collections::HashMap;',              // 1
      '',                                            // 2
      '#[derive(Debug, Clone)]',                     // 3
      'pub struct Parser<\'a> {',                    // 4
      '    input: &\'a str,',                        // 5
      '}',                                           // 6
      '',                                            // 7
      'pub trait Tokenize {',                        // 8
      '    fn next_token(&mut self) -> Option<&str>;', // 9
      '}',                                           // 10
      '',                                            // 11
      'impl<\'a> Parser<\'a> {',                     // 12
      '    pub fn new(input: &\'a str) -> Self {',   // 13
      '        let _raw = r#"unbalanced { brace"#;', // 14
      '        Self { input }',                      // 15
      '    }',                                       // 16
      '',                                            // 17
      '    fn skip_ws(&self) -> usize {',            // 18
      '        self.input.len()',                    // 19
      '    }',                                       // 20
      '}',                                           // 21
      '',                                            // 22
      'pub fn parse_all<\'a>(src: &\'a str) -> HashMap<&\'a str, usize> {', // 23
      '    let _ = Parser::new(src).skip_ws();',     // 24
      '    HashMap::new()',                          // 25
      '}',                                           // 26
    ].join('\n') + '\n'
    writeFileSync(join(workspace, 'src', 'parser.rs'), rsSource)

    const rsOutline = extractFileOutline('src/parser.rs', rsSource)
    expect(rsOutline.symbols.map(s => [s.name, s.kind, s.startLine, s.endLine, s.exported])).toEqual([
      ['Parser', 'struct', 3, 6, true],
      ['Tokenize', 'trait', 8, 10, true],
      ["<'a> Parser<'a>", 'impl', 12, 21, true],
      ['parse_all', 'function', 23, 26, true],
    ])
    expect(rsOutline.symbols[2]!.children?.map(c => [c.name, c.startLine, c.endLine, c.exported])).toEqual([
      ['new', 13, 16, true],
      ['skip_ws', 18, 20, false],
    ])

    // End-to-end tool execution with directory batching & exported_only
    const registry = new ToolRegistry()
    registerCoreTools(registry, { includeProcessTools: false, workspaceRoot: workspace })
    const renderedBatch = await registry.execute(
      toolCall('FileOutlineTool', {
        file_path: 'src',
        exported_only: true,
        include_imports: false,
        max_depth: 2,
      }),
      { metadata: {} },
    )
    expect(renderedBatch).toContain('# src/engine.ts (typescript, 37 lines)')
    expect(renderedBatch).not.toContain('imports [')
    expect(renderedBatch).not.toContain('internalUtil')
    expect(renderedBatch).not.toContain('_private_helper')
    expect(renderedBatch).not.toContain('dialInternal')
    expect(renderedBatch).toContain('[19-31] export class Engine')
    expect(renderedBatch).toContain('[8-20] export class Runner')
    expect(renderedBatch).toContain('[23-26] export function parse_all')
  })
})

describe('discoverRelatedTests — multi-language mapping & runner detection', () => {
  test('maps TS/Vitest, TS/Bun, Python/pytest, Go/go test, and Rust/cargo test accurately', async () => {
    // 1. Vitest workspace
    const vitestWs = await makeWorkspace('xerxes-swe-vitest-')
    await mkdir(join(vitestWs, 'src', 'foo', '__tests__'), { recursive: true })
    await mkdir(join(vitestWs, 'test'), { recursive: true })
    writeFileSync(join(vitestWs, 'package.json'), JSON.stringify({ devDependencies: { vitest: '^2.0.0' } }))
    writeFileSync(join(vitestWs, 'src', 'foo', 'bar.ts'), 'export const bar = 1\n')
    writeFileSync(join(vitestWs, 'src', 'foo', '__tests__', 'bar.test.ts'), 'test("bar", () => {})\n')
    writeFileSync(join(vitestWs, 'test', 'bar.spec.ts'), 'test("spec", () => {})\n')

    const vitestRes = discoverRelatedTests(vitestWs, ['src/foo/bar.ts'])
    expect(vitestRes.relatedTests).toEqual(['src/foo/__tests__/bar.test.ts', 'test/bar.spec.ts'])
    expect(vitestRes.runner).toBe('vitest')
    expect(vitestRes.runnerCommand).toBe('vitest run src/foo/__tests__/bar.test.ts test/bar.spec.ts')

    // 2. Python / pytest workspace
    const pyWs = await makeWorkspace('xerxes-swe-pytest-')
    await mkdir(join(pyWs, 'pkg'), { recursive: true })
    await mkdir(join(pyWs, 'tests'), { recursive: true })
    writeFileSync(join(pyWs, 'pyproject.toml'), '[tool.pytest.ini_options]\n')
    writeFileSync(join(pyWs, 'pkg', 'bar.py'), 'def calc(): return 1\n')
    writeFileSync(join(pyWs, 'tests', 'test_bar.py'), 'def test_calc(): assert True\n')

    const pyRes = discoverRelatedTests(pyWs, ['pkg/bar.py'])
    expect(pyRes.relatedTests).toEqual(['tests/test_bar.py'])
    expect(pyRes.runner).toBe('pytest')
    expect(pyRes.runnerCommand).toBe('pytest tests/test_bar.py')

    // 3. Go workspace
    const goWs = await makeWorkspace('xerxes-swe-go-')
    await mkdir(join(goWs, 'pkg'), { recursive: true })
    writeFileSync(join(goWs, 'go.mod'), 'module example.com/demo\ngo 1.22\n')
    writeFileSync(join(goWs, 'pkg', 'bar.go'), 'package pkg\n')
    writeFileSync(join(goWs, 'pkg', 'bar_test.go'), 'package pkg\n')

    const goRes = discoverRelatedTests(goWs, ['pkg/bar.go'])
    expect(goRes.relatedTests).toEqual(['pkg/bar_test.go'])
    expect(goRes.runner).toBe('go')
    expect(goRes.runnerCommand).toBe('go test ./pkg')

    // 4. Rust workspace
    const rsWs = await makeWorkspace('xerxes-swe-rust-')
    await mkdir(join(rsWs, 'src'), { recursive: true })
    await mkdir(join(rsWs, 'tests'), { recursive: true })
    writeFileSync(join(rsWs, 'Cargo.toml'), '[package]\nname = "demo"\nversion = "0.1.0"\n')
    writeFileSync(join(rsWs, 'src', 'bar.rs'), 'pub fn bar() {}\n')
    writeFileSync(join(rsWs, 'tests', 'bar.rs'), '#[test] fn works() {}\n')

    const rsRes = discoverRelatedTests(rsWs, ['src/bar.rs'])
    expect(rsRes.relatedTests).toEqual(['tests/bar.rs'])
    expect(rsRes.runner).toBe('cargo')
    expect(rsRes.runnerCommand).toBe('cargo test --test bar')
  })
})

describe('Specialist LSP access & safe workspace LSP auto-discovery', () => {
  test('addLspToolToBuiltinAgents grants LSPTool to default, creator, coder, researcher, reviewer, and tester when configured', async () => {
    const baseDefinitions = new Map<string, AgentDefinition>(loadBuiltinAgentDefinitions())
    const customUserAgent: AgentDefinition = {
      ...baseDefinitions.get('coder')!,
      name: 'coder-custom',
      source: 'user',
    }
    baseDefinitions.set('coder-custom', customUserAgent)

    // Unconfigured manager is a strict no-op
    const unconfigured = new LspManager({ servers: [] })
    const unchanged = new Map(baseDefinitions)
    addLspToolToBuiltinAgents(unchanged, unconfigured)
    expect(unchanged.get('coder')?.allowedTools).not.toContain(LSP_RUNTIME_TOOL_NAME)
    await unconfigured.close()

    // Configured manager grants LSPTool on both `tools` and `allowedTools` for all 6 built-in profiles
    const configured = new LspManager({
      servers: [{ name: 'ts', command: 'typescript-language-server', args: ['--stdio'], languageId: 'typescript', extensions: ['.ts'] }],
    })
    try {
      const updated = new Map(baseDefinitions)
      addLspToolToBuiltinAgents(updated, configured)

      for (const specialist of ['default', 'creator', 'coder', 'researcher', 'reviewer', 'tester']) {
        const def = updated.get(specialist)
        expect(def).toBeDefined()
        expect(def!.tools).toContain(LSP_RUNTIME_TOOL_NAME)
        if (def!.allowedTools !== null) {
          expect(def!.allowedTools).toContain(LSP_RUNTIME_TOOL_NAME)
        }
      }
      // User-sourced profile is never mutated
      expect(updated.get('coder-custom')?.tools).not.toContain(LSP_RUNTIME_TOOL_NAME)
    } finally {
      await configured.close()
    }
  })

  test('discoverWorkspaceLspServers detects local & PATH servers safely without mutating explicit user configs or reading .lsp.json', async () => {
    const workspace = await makeWorkspace('xerxes-swe-lsp-disc-')
    const outside = await makeWorkspace('xerxes-swe-lsp-outside-')

    // Untrusted workspace .lsp.json MUST be ignored
    writeFileSync(
      join(workspace, '.lsp.json'),
      JSON.stringify({ servers: [{ name: 'evil', command: '/bin/sh', languageId: 'sh', extensions: ['.sh'] }] }),
    )

    // Create workspace markers & contained local binaries
    writeFileSync(join(workspace, 'tsconfig.json'), '{}')
    writeFileSync(join(workspace, 'pyproject.toml'), '[project]\nname = "demo"\n')
    writeFileSync(join(workspace, 'go.mod'), 'module example.com/m\n')
    writeFileSync(join(workspace, 'Cargo.toml'), '[package]\nname = "m"\n')

    await mkdir(join(workspace, 'node_modules', 'typescript-language-server'), { recursive: true })
    await mkdir(join(workspace, 'node_modules', '.bin'), { recursive: true })
    const realTsCli = join(workspace, 'node_modules', 'typescript-language-server', 'cli.js')
    writeFileSync(realTsCli, '#!/usr/bin/env node\n')
    const binSuffix = process.platform === 'win32' ? '.cmd' : ''
    const localTsBin = join(workspace, 'node_modules', '.bin', `typescript-language-server${binSuffix}`)
    await symlink(realTsCli, localTsBin)

    // Malicious escaping symlink in node_modules/.bin/pyright-langserver -> outside workspace!
    const outsidePyright = join(outside, 'evil-pyright')
    writeFileSync(outsidePyright, '#!/bin/sh\n')
    await symlink(outsidePyright, join(workspace, 'node_modules', '.bin', `pyright-langserver${binSuffix}`))

    // Contained .venv pyright binary
    const venvDir = process.platform === 'win32' ? 'Scripts' : 'bin'
    const venvExt = process.platform === 'win32' ? '.exe' : ''
    await mkdir(join(workspace, '.venv', venvDir), { recursive: true })
    const venvPyright = join(workspace, '.venv', venvDir, `pyright-langserver${venvExt}`)
    writeFileSync(venvPyright, '#!/usr/bin/env python3\n')

    const discovered = await discoverWorkspaceLspServers(workspace, {
      which: binary => {
        if (binary === 'gopls') return '/usr/local/bin/gopls'
        if (binary === 'rust-analyzer') return '/usr/local/bin/rust-analyzer'
        return null
      },
    })

    expect(discovered.map(s => s.name)).toEqual([
      'typescript-language-server',
      'pyright-langserver',
      'gopls',
      'rust-analyzer',
    ])
    expect(discovered.find(s => s.name === 'typescript-language-server')?.command).toBe(localTsBin)
    // Escaping symlink in node_modules/.bin was rejected; fell through to contained .venv binary!
    expect(discovered.find(s => s.name === 'pyright-langserver')?.command).toBe(venvPyright)
    expect(discovered.find(s => s.name === 'gopls')).toMatchObject({
      command: '/usr/local/bin/gopls',
      args: ['serve'],
      extensions: ['.go'],
    })
    expect(discovered.find(s => s.name === 'rust-analyzer')).toMatchObject({
      command: '/usr/local/bin/rust-analyzer',
      extensions: ['.rs'],
    })

    // Explicit user config preservation: existing servers are never mutated and keep ownership of their suffixes!
    const explicitServers = parseLspConfig({
      servers: [
        {
          name: 'custom-ts',
          command: '/opt/custom-ts-lsp',
          languageId: 'typescript',
          extensions: ['.ts', '.tsx'],
          enabled: true,
        },
        {
          name: 'gopls',
          command: '/opt/custom-gopls',
          languageId: 'go',
          extensions: ['.go'],
          enabled: false,
        },
      ],
    })
    const snapshotBefore = JSON.stringify(explicitServers)

    const nonConflicting = await discoverWorkspaceLspServers(workspace, {
      existingServers: explicitServers,
      which: binary => (binary === 'rust-analyzer' ? '/usr/local/bin/rust-analyzer' : null),
    })
    expect(JSON.stringify(explicitServers)).toBe(snapshotBefore)
    // `gopls` name existed -> skipped; `.ts` and `.tsx` claimed -> discovered TS server only claims remaining JS/MTS suffixes!
    expect(nonConflicting.find(s => s.name === 'gopls')).toBeUndefined()
    const tsRemaining = nonConflicting.find(s => s.name === 'typescript-language-server')
    expect(tsRemaining?.extensions).not.toContain('.ts')
    expect(tsRemaining?.extensions).not.toContain('.tsx')
    expect(tsRemaining?.extensions).toContain('.js')

    const merged = mergeDiscoveredLspServers(explicitServers, discovered)
    expect(JSON.stringify(explicitServers)).toBe(snapshotBefore)
    expect(merged.find(s => s.name === 'custom-ts')?.command).toBe('/opt/custom-ts-lsp')
    expect(merged.find(s => s.name === 'gopls')?.command).toBe('/opt/custom-gopls')

    // Confirm loadConfiguredLsp(home) still ignores workspace discovery by default
    const home = await makeWorkspace('xerxes-swe-lsp-home-')
    const manager = loadConfiguredLsp(home)
    expect(manager.configured).toBe(false)
    await manager.close()
  })
})
