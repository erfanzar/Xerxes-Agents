// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { existsSync } from 'node:fs'
import { basename, dirname, isAbsolute, join, resolve } from 'node:path'

import { PLUGIN_MANAGE_TOOL_NAME, PLUGIN_TOOL_PREFIX } from '../agents/toolPatterns.js'
import type { ToolExecutionContext, ToolRegistry } from '../executors/toolRegistry.js'
import { describePermissions, type ManagedPluginInventoryEntry, type ManagedPlugins } from '../extensions/managedPlugins.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'

export const PLUGIN_MANAGE_ACTIONS = ['list', 'inspect', 'install', 'reload', 'enable', 'disable', 'uninstall'] as const
export type PluginManageAction = (typeof PLUGIN_MANAGE_ACTIONS)[number]

/** Actions that only read; every other action changes what code runs and needs approval. */
export const PLUGIN_MANAGE_READ_ACTIONS: ReadonlySet<string> = new Set(['list', 'inspect'])

/** Outcome of running a plugin's Bun tests. */
export interface PluginTestRun {
  readonly exitCode: number
  readonly output: string
  readonly timedOut?: boolean
}

/** Runs one Bun test file. Injected so tests of this tool need not spawn processes. */
export type PluginTestRunner = (file: string, options: { readonly cwd: string; readonly timeoutMs: number }) => Promise<PluginTestRun>

export const PLUGIN_TEST_TIMEOUT_MS = 120_000
const TEST_OUTPUT_TAIL = 3_000

/** `bun test <file>` in the plugin's folder, with a minimal environment and a hard timeout. */
export const defaultPluginTestRunner: PluginTestRunner = async (file, options) => {
  const env: Record<string, string> = { NO_COLOR: '1' }
  for (const key of ['PATH', 'HOME', 'TMPDIR']) {
    const value = process.env[key]
    if (value) env[key] = value
  }
  const child = Bun.spawn([process.execPath, '--no-env-file', 'test', file], { cwd: options.cwd, env, stdin: 'ignore', stdout: 'pipe', stderr: 'pipe' })
  let timedOut = false
  const timer = setTimeout(() => { timedOut = true; try { child.kill('SIGKILL') } catch { /* gone */ } }, options.timeoutMs)
  try {
    const [stdout, stderr, exitCode] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited])
    return { exitCode, output: `${stdout}${stderr}`, ...(timedOut ? { timedOut: true } : {}) }
  } finally { clearTimeout(timer) }
}

/** The test file beside a plugin module: `index.test.ts`, `<name>.test.ts`, then `<name>.test.js`. */
export function findPluginTestFile(modulePath: string): string | undefined {
  const folder = dirname(modulePath)
  const stem = basename(modulePath).replace(/\.(?:[cm]?js|ts)$/, '')
  return [join(folder, 'index.test.ts'), join(folder, `${stem}.test.ts`), join(folder, `${stem}.test.js`)].find(existsSync)
}

export interface PluginManageToolOptions {
  readonly testRunner?: PluginTestRunner
  readonly testTimeoutMs?: number
}

export const PLUGIN_MANAGE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: PLUGIN_MANAGE_TOOL_NAME,
    description: [
      'List, inspect, install, reload, enable, disable or uninstall native Xerxes tool plugins.',
      'An installed plugin\'s tools appear as plugin_<name> and can be called from your next step.',
      'Every change asks the user for approval. Installs are temporary (this run of Xerxes only) unless scope is "permanent".',
      'install first runs the plugin\'s Bun tests (index.test.ts or <name>.test.ts beside the module) and refuses when they fail or are missing.',
    ].join(' '),
    parameters: {
      type: 'object',
      properties: {
        action: { type: 'string', enum: [...PLUGIN_MANAGE_ACTIONS] },
        path: { type: 'string', description: 'install: the plugin module (.ts/.js exporting register(registry)); relative paths resolve against the session directory.' },
        plugin: { type: 'string', description: 'inspect/reload/enable/disable/uninstall: plugin name or module path.' },
        scope: { type: 'string', enum: ['temporary', 'permanent'], description: 'install only. temporary (default): gone after Xerxes restarts. permanent: saved for every future session.' },
        skip_tests: { type: 'boolean', description: 'install only: install without a test file. The user sees this in the approval request; avoid it.' },
      },
      required: ['action'],
      additionalProperties: false,
    },
  },
}

const GUIDANCE = [
  'Use PluginManageTool when a task needs a capability no current tool offers and a small TypeScript tool would provide it.',
  'Write the plugin (structured tools with a JSON schema; see the authoring-tool-plugins skill) and its tests first, run the tests, then install it with scope "temporary". install reruns the tests and refuses on failure.',
  'Install only code you wrote in this session or the user pointed you to. Plugins run in their own process with the user\'s privileges (not a sandbox); the user approves every change.',
  'Ask for scope "permanent" only when the user wants the tool kept for future sessions.',
].join('\n')

function text(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function summary(entry: ManagedPluginInventoryEntry): Record<string, unknown> {
  return {
    name: entry.name,
    enabled: entry.enabled,
    scope: entry.scope,
    ...(entry.temporary ? { temporary: true } : {}),
    ...(entry.stale ? { stale: true } : {}),
    ...(entry.loadError ? { load_error: entry.loadError } : {}),
    module: entry.module,
    tools: entry.tools.map(tool => PLUGIN_TOOL_PREFIX + tool),
    commands: entry.commands.map(command => '/' + command),
  }
}

function permissionsPayload(entry: ManagedPluginInventoryEntry): Record<string, unknown> {
  return {
    ...entry.permissions,
    summary: describePermissions(entry.permissions) || 'none declared',
    enforcement: 'env is enforced (separate process with only the declared variables); network is a fetch guard rail only; filesystem is declared, not enforced. Not a security sandbox.',
  }
}

function tail(output: string): string {
  const trimmed = output.trim()
  return trimmed.length > TEST_OUTPUT_TAIL ? `…${trimmed.slice(-TEST_OUTPUT_TAIL)}` : trimmed
}

/** Run one PluginManageTool call against the managed plugin host. */
export async function pluginManageTool(
  inputs: JsonObject,
  context: Pick<ToolExecutionContext, 'metadata'>,
  plugins: ManagedPlugins,
  cwd: () => string,
  options: PluginManageToolOptions = {},
): Promise<Record<string, unknown>> {
  const action = text(inputs.action) as PluginManageAction
  if (!(PLUGIN_MANAGE_ACTIONS as readonly string[]).includes(action)) {
    throw new TypeError(`action must be one of ${PLUGIN_MANAGE_ACTIONS.join(', ')}`)
  }
  if (!PLUGIN_MANAGE_READ_ACTIONS.has(action) && context.metadata.permission_mode === 'plan') {
    throw new Error(`PluginManageTool ${action} is disabled in plan mode`)
  }
  if (action === 'list') return { ok: true, plugins: plugins.inventory().map(summary) }
  if (action === 'inspect') {
    const target = text(inputs.plugin)
    const entry = plugins.inventory().find(item => item.name === target || item.module === target)
    if (!entry) return { ok: false, error: 'plugin not found; use action "list"' }
    return {
      ok: true,
      plugin: {
        ...summary(entry), description: entry.description, version: entry.version, tool_descriptions: entry.toolDescriptions,
        permissions: permissionsPayload(entry), usage: entry.usage, running: entry.running,
      },
    }
  }
  if (action === 'install') {
    const raw = text(inputs.path)
    if (!raw) throw new TypeError('install needs path')
    const path = isAbsolute(raw) ? raw : resolve(cwd(), raw)
    const temporary = text(inputs.scope) !== 'permanent'
    const skipTests = inputs.skip_tests === true
    const testFile = findPluginTestFile(path)
    let tests: Record<string, unknown>
    if (!testFile) {
      if (!skipTests) {
        return { ok: false, error: `No test file beside ${path} (looked for index.test.ts, <name>.test.ts and <name>.test.js). Write tests and run them first, or pass skip_tests: true (the user sees this when approving).` }
      }
      tests = { skipped: true }
    } else {
      const run = await (options.testRunner ?? defaultPluginTestRunner)(testFile, { cwd: dirname(path), timeoutMs: options.testTimeoutMs ?? PLUGIN_TEST_TIMEOUT_MS })
      if (run.exitCode !== 0 || run.timedOut) {
        return {
          ok: false,
          error: `Install refused: ${run.timedOut ? 'the plugin tests timed out' : `the plugin tests failed (exit ${run.exitCode})`} — ${testFile}`,
          test_output_tail: tail(run.output),
        }
      }
      tests = { file: testFile, passed: true }
    }
    const message = await plugins.change('install', path, { temporary })
    const installed = plugins.inventory().filter(item => item.module === resolve(path))
    return {
      ok: true, message, tests,
      plugins: installed.map(entry => ({ ...summary(entry), permissions: permissionsPayload(entry) })),
    }
  }
  const target = text(inputs.plugin)
  if (!target) throw new TypeError(`${action} needs plugin`)
  const message = await plugins.change(action, target)
  return { ok: true, message, plugins: plugins.inventory().filter(item => item.name === target || item.module === target).map(summary) }
}

/** Expose plugin management to the model. Mutations always cross the approval gate (see permissions.ts). */
export function registerPluginManageTool(registry: ToolRegistry, plugins: ManagedPlugins, cwd: () => string, options: PluginManageToolOptions = {}): void {
  registry.register(
    PLUGIN_MANAGE_DEFINITION,
    (inputs, context) => pluginManageTool(inputs, context, plugins, cwd, options),
    'default',
    { concurrencySafe: false, destructive: true, openWorld: true, readOnly: false, interruptBehavior: 'block' },
    GUIDANCE,
  )
}
