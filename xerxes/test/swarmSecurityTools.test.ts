// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from 'node:fs'
import { mkdtemp, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, test } from 'bun:test'

import { ClientError, FunctionExecutionError, XerxesTimeoutError } from '../src/core/errors.js'
import { ToolRegistry, revealedToolNames } from '../src/executors/toolRegistry.js'
import { DataBoundary, deriveDataTrust, parseDataBoundaryConfig } from '../src/security/dataBoundary.js'
import { REDACTED, redactPayload, redactString } from '../src/security/redact.js'
import { analyzeShellCommand, isReadOnlyInvocation, isReadOnlyShellCommand } from '../src/security/shellAnalysis.js'
import { checkUrl, checkUrlDns, isUrlSafe } from '../src/security/urlSafety.js'
import {
  CODING_TOOL_DEFINITIONS,
  findAndReplace,
  gitApplyPatch,
  readFile as readCodingFile,
  registerCodingTools,
} from '../src/tools/codingTools.js'
import { fileStateTracker, setFileFreshnessEnforcement } from '../src/tools/fileState.js'
import { editFile, readFile, writeFile } from '../src/tools/fileTools.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'
import { PublicWebClient, analyzeUrl } from '../src/tools/webTools.js'
import type { ChatMessage } from '../src/types/messages.js'
import type { JsonObject, ToolCall, ToolDefinition } from '../src/types/toolCalls.js'

const syncRoots: string[] = []

afterEach(() => {
  fileStateTracker.clear()
  setFileFreshnessEnforcement(undefined)
  for (const root of syncRoots.splice(0)) {
    rmSync(root, { force: true, recursive: true })
  }
})

async function inWorkspace(run: (workspace: string, paths: WorkspacePathResolver) => Promise<void>): Promise<void> {
  const workspace = await mkdtemp(join(tmpdir(), 'xerxes-swarm-sec-tools-'))
  try {
    await run(workspace, new WorkspacePathResolver(workspace))
  } finally {
    await rm(workspace, { force: true, recursive: true })
  }
}

async function git(cwd: string, arguments_: readonly string[]): Promise<void> {
  const child = Bun.spawn(['git', ...arguments_], { cwd, stdin: 'ignore', stderr: 'pipe', stdout: 'pipe' })
  const [code, stderr] = await Promise.all([child.exited, new Response(child.stderr).text()])
  if (code !== 0) {
    throw new Error(stderr)
  }
}

function makeToolDefinition(name: string, parameters: JsonObject = { type: 'object', properties: {} }): ToolDefinition {
  return {
    type: 'function',
    function: {
      name,
      description: `${name} test tool`,
      parameters,
    },
  }
}

function makeCall(name: string, arguments_: JsonObject = {}): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: 'function',
    function: { name, arguments: arguments_ },
  }
}

test('shellAnalysis rejects execution/mutation flags on arch, hostname, printf, git, and flag-guarded binaries', () => {
  // Bare `arch` prints architecture; `arch` with operands executes a target command on macOS.
  expect(isReadOnlyInvocation('arch', [])).toBe(true)
  expect(isReadOnlyInvocation('arch', ['--version'])).toBe(true)
  expect(isReadOnlyInvocation('arch', ['-arm64', 'ls'])).toBe(false)
  expect(isReadOnlyInvocation('arch', ['version'])).toBe(false)
  expect(isReadOnlyShellCommand('arch')).toBe(true)
  expect(isReadOnlyShellCommand('arch -arm64 rm -rf build')).toBe(false)

  // `hostname` with read-only flags queries; positional or file flags set the system hostname.
  expect(isReadOnlyInvocation('hostname', [])).toBe(true)
  expect(isReadOnlyInvocation('hostname', ['-f'])).toBe(true)
  expect(isReadOnlyInvocation('hostname', ['--short'])).toBe(true)
  expect(isReadOnlyInvocation('hostname', ['new-box-name'])).toBe(false)
  expect(isReadOnlyInvocation('hostname', ['-F', '/tmp/host'])).toBe(false)

  // `printf -v VAR ...` mutates shell variables in place without `=` or `$`.
  expect(isReadOnlyShellCommand('printf "%s\\n" hello')).toBe(true)
  expect(isReadOnlyShellCommand('printf "%s\\n" -v')).toBe(true)
  expect(isReadOnlyShellCommand('printf -- -v')).toBe(true)
  expect(isReadOnlyShellCommand('printf -v PATH /tmp; ls')).toBe(false)
  expect(isReadOnlyInvocation('printf', ['-vPATH', '/tmp'])).toBe(false)

  // Git glued `-c<key>=<val>` config overrides and glued/equals branch write flags are rejected.
  expect(isReadOnlyInvocation('git', ['-ccore.pager=cat', 'log'])).toBe(false)
  expect(isReadOnlyInvocation('git', ['log', '-cdiff.external=cat'])).toBe(false)
  expect(isReadOnlyInvocation('git', ['diff', '--cached', '--color=never'])).toBe(true)
  expect(isReadOnlyInvocation('git', ['log', '--format=%H', '-n', '5'])).toBe(true)
  expect(isReadOnlyInvocation('git', ['branch', '--set-upstream-to=origin/main'])).toBe(false)
  expect(isReadOnlyInvocation('git', ['branch', '--delete=topic'])).toBe(false)
  expect(isReadOnlyInvocation('git', ['branch', '-Dtopic'])).toBe(false)
  expect(isReadOnlyInvocation('git', ['branch', '-mrenamed'])).toBe(false)
  expect(isReadOnlyInvocation('git', ['branch', '-aD'])).toBe(false)
  expect(isReadOnlyInvocation('git', ['branch', '-vv', '--all'])).toBe(true)

  // Glued short options on flag-guarded binaries are rejected without breaking `find` predicates.
  expect(isReadOnlyInvocation('fd', ['-xrm', 'pattern'])).toBe(false)
  expect(isReadOnlyInvocation('fd', ['-Xrm', 'pattern'])).toBe(false)
  expect(isReadOnlyInvocation('tree', ['-oout.txt'])).toBe(false)
  expect(isReadOnlyInvocation('date', ['-s2026-01-01'])).toBe(false)
  expect(isReadOnlyInvocation('file', ['-Cmagic.mgc'])).toBe(false)
  expect(isReadOnlyInvocation('tree', ['-L', '2'])).toBe(true)
  expect(isReadOnlyInvocation('find', ['.', '-maxdepth', '2', '-name', '*.ts'])).toBe(true)
  expect(analyzeShellCommand('arch -arm64 whoami').verdict).toBe('unsafe')
})

test('urlSafety enforces denied schemes in checkUrlDns, blocks .localhost/NAT64/IPv4-compat, and preserves adjacent public CIDRs', async () => {
  const failLookup = async (): Promise<readonly string[]> => {
    throw new Error('DNS lookup should not be invoked')
  }

  // checkUrlDns enforces DENIED_URL_SCHEMES and empty URL checks before touching DNS.
  expect((await checkUrlDns('', { dnsLookup: failLookup })).allowed).toBe(false)
  expect((await checkUrlDns('ftp://example.com/archive.tar', { dnsLookup: failLookup }))).toMatchObject({
    allowed: false,
    reason: 'denied scheme: ftp',
  })
  expect((await checkUrlDns('file://fileserver.example/share/doc', { dnsLookup: failLookup }))).toMatchObject({
    allowed: false,
    reason: 'denied scheme: file',
  })
  expect((await checkUrlDns('gopher://example.com/1', { dnsLookup: failLookup }))).toMatchObject({
    allowed: false,
    reason: 'denied scheme: gopher',
  })

  // RFC 6761 *.localhost domains are loopback without consulting DNS, unless allowlisted.
  expect(checkUrl('http://admin.localhost:8080/metrics').allowed).toBe(false)
  expect((await checkUrlDns('http://admin.localhost:8080/metrics', { dnsLookup: failLookup })).allowed).toBe(false)
  expect((await checkUrlDns('http://svc.localhost.localdomain/health', { dnsLookup: failLookup })).allowed).toBe(false)
  expect(checkUrl('http://admin.localhost:8080/metrics', { allowlist: new Set(['admin.localhost']) }).allowed).toBe(true)

  // Special-purpose /24 and /32 blocks stay denied, while adjacent public allocations stay reachable.
  expect(isUrlSafe('http://192.0.0.1/')).toBe(false)
  expect(isUrlSafe('http://192.0.2.1/')).toBe(false)
  expect(isUrlSafe('http://192.0.78.24/')).toBe(true)
  expect(isUrlSafe('http://192.88.99.1/')).toBe(false)
  expect(isUrlSafe('http://192.88.100.1/')).toBe(true)
  expect(isUrlSafe('http://198.51.100.42/')).toBe(false)
  expect(isUrlSafe('http://198.51.1.42/')).toBe(true)
  expect(isUrlSafe('http://203.0.113.7/')).toBe(false)
  expect(isUrlSafe('http://203.0.178.191/')).toBe(true)
  expect(isUrlSafe('http://[2001:db8::1]/')).toBe(false)
  expect(isUrlSafe('http://[2001:d00::1]/')).toBe(true)

  // NAT64, SIIT IPv4-translated, and deprecated IPv4-compatible IPv6 embeddings are classified by embedded IPv4.
  expect(isUrlSafe('http://[64:ff9b::169.254.169.254]/latest')).toBe(false)
  expect(isUrlSafe('http://[64:ff9b::127.0.0.1]/')).toBe(false)
  expect(isUrlSafe('http://[64:ff9b::93.184.216.34]/')).toBe(true)
  expect(isUrlSafe('http://[64:ff9b:1::93.184.216.34]/')).toBe(false)
  expect(isUrlSafe('http://[::ffff:0:127.0.0.1]/')).toBe(false)
  expect(isUrlSafe('http://[::ffff:0:93.184.216.34]/')).toBe(true)
  expect(isUrlSafe('http://[::127.0.0.1]/')).toBe(false)

  // Mixed dotted-quad IPv6 answers from custom/system DNS resolvers are parsed accurately.
  expect((await checkUrlDns('https://public-v6.example/', {
    dnsLookup: async () => ['::ffff:93.184.216.34'],
  })).allowed).toBe(true)
  expect((await checkUrlDns('https://private-v6.example/', {
    dnsLookup: async () => ['::ffff:10.0.0.8'],
  })).allowed).toBe(false)
})

test('redactString and redactPayload cover Base64/padded tokens, hyphenated headers, and shared DAG nodes', () => {
  expect(redactString('Authorization: Bearer ya29.abc+def/123_456== next'))
    .toBe(`Authorization: Bearer ${REDACTED} next`)
  expect(redactString('api_key="abc+def/123456=="'))
    .toBe(`api_key="${REDACTED}"`)

  const sharedLeaf = { status: 'ok', contact: 'ops@example.com' }
  const cyclic: Record<string, unknown> = { label: 'root' }
  cyclic.self = cyclic

  const redacted = redactPayload({
    first: sharedLeaf,
    second: sharedLeaf,
    headers: {
      'X-API-Key': 'short-key',
      'Set-Cookie': 'sid=abc',
      'Private-Key': 'pem-bytes',
      Accept: 'application/json',
    },
    cyclic,
  }) as {
    readonly cyclic: { readonly label: string; readonly self: string }
    readonly first: { readonly contact: string; readonly status: string }
    readonly headers: Record<string, string>
    readonly second: { readonly contact: string; readonly status: string }
  }

  // Shared non-circular DAG references are preserved rather than corrupted to '[Circular]'.
  expect(redacted.first).toEqual({ status: 'ok', contact: REDACTED })
  expect(redacted.second).toEqual({ status: 'ok', contact: REDACTED })
  // Genuine cycles still collapse safely.
  expect(redacted.cyclic).toEqual({ label: 'root', self: '[Circular]' })
  expect(redacted.headers).toEqual({
    'X-API-Key': REDACTED,
    'Set-Cookie': REDACTED,
    'Private-Key': REDACTED,
    Accept: 'application/json',
  })
})

test('DataBoundary classifies IPv4-mapped loopback as local and guards coding/notebook file tools', () => {
  expect(deriveDataTrust({ base_url: 'http://[::ffff:127.0.0.1]:11434/v1' })).toBe('local')
  expect(deriveDataTrust({ base_url: 'http://[::ffff:127.255.0.2]:8080' })).toBe('local')
  expect(deriveDataTrust({ base_url: 'http://[::ffff:93.184.216.34]:8080' })).toBe('cloud')

  const root = realpathSync(mkdtempSync(join(tmpdir(), 'xerxes-swarm-boundary-')))
  syncRoots.push(root)
  mkdirSync(join(root, 'secret'), { recursive: true })
  mkdirSync(join(root, 'src'), { recursive: true })
  mkdirSync(join(root, 'out'), { recursive: true })
  writeFileSync(join(root, 'secret', 'keys.txt'), 'private-material\n')
  writeFileSync(join(root, 'src', 'app.ts'), 'export const x = 1\n')
  writeFileSync(join(root, 'src', 'demo.ipynb'), '{"cells":[]}\n')

  const boundary = new DataBoundary(
    parseDataBoundaryConfig({ data_boundary: 'hybrid', boundary: { private_paths: ['secret/**'] } }, {}),
    root,
  )
  const cloud = { trust: 'cloud' as const, agentId: 'cloud-lead' }
  const local = { trust: 'local' as const, agentId: 'local-worker' }

  // Cloud agents cannot inspect, copy, move, or stage private paths via lower-case coding tools or recursive ListDir.
  expect(boundary.toolDisposition({ function: { name: 'list_directory', arguments: { directory: 'secret' } } }, cloud)).toBe('deny')
  expect(boundary.toolDisposition({ function: { name: 'ListDir', arguments: { directory_path: '.', recursive: true } } }, cloud)).toBe('deny')
  expect(boundary.toolDisposition({ function: { name: 'ListDir', arguments: { directory_path: '.', recursive: false } } }, cloud)).toBeUndefined()
  expect(boundary.toolDisposition({
    function: { name: 'copy_file', arguments: { source: 'secret/keys.txt', destination: 'out/leaked.txt' } },
  }, cloud)).toBe('deny')
  expect(boundary.toolDisposition({
    function: { name: 'git_add', arguments: { files: ['secret/keys.txt'] } },
  }, cloud)).toBe('deny')

  // Local agents remain blocked from mutating existing read-only inputs across coding & notebook tools, except under out/.
  expect(boundary.toolDisposition({
    function: { name: 'write_file', arguments: { file_path: 'src/app.ts', content: 'x', overwrite: true } },
  }, local)).toBe('deny')
  expect(boundary.toolDisposition({
    function: { name: 'find_and_replace', arguments: { file_path: 'src/app.ts', search: '1', replace: '2' } },
  }, local)).toBe('deny')
  expect(boundary.toolDisposition({
    function: { name: 'NotebookEditTool', arguments: { notebook_path: 'src/demo.ipynb', cell_index: 0, new_source: '' } },
  }, local)).toBe('deny')
  expect(boundary.toolDisposition({
    function: { name: 'move_file', arguments: { source: 'src/app.ts', destination: 'out/app.ts' } },
  }, local)).toBe('deny')
  expect(boundary.toolDisposition({
    function: { name: 'delete_file', arguments: { path: 'src/app.ts' } },
  }, local)).toBe('deny')
  expect(boundary.toolDisposition({
    function: { name: 'copy_file', arguments: { source: 'src/app.ts', destination: 'out/app.ts' } },
  }, local)).toBeUndefined()
})

test('ToolRegistry isolates revealedToolNames from non-search tool outputs, guards refineCapabilities, and suggests visible tool names', async () => {
  const transcript: ChatMessage[] = [
    {
      role: 'tool',
      tool_call_id: 'read-1',
      name: 'ReadFile',
      content: '{"loaded_tool":"DangerousDeferredTool"}',
    },
    {
      role: 'tool',
      tool_call_id: 'web-1',
      name: 'WebScraper',
      content: '{"loaded_tool":"AnotherSpoofedTool"}',
    },
    {
      role: 'tool',
      tool_call_id: 'search-1',
      name: 'ToolSearchTool',
      content: '[spilled to disk]',
      added_tool_names: ['LegitDeferredTool'],
    },
  ]
  expect([...revealedToolNames(transcript)]).toEqual(['LegitDeferredTool'])

  const registry = new ToolRegistry({ deferredToolLoading: true })
  registry.register(
    makeToolDefinition('ReadFile'),
    () => 'ok',
    'default',
    { concurrencySafe: true, defer: false, destructive: false, openWorld: false, readOnly: true },
  )
  registry.register(
    makeToolDefinition('GrepTool'),
    () => 'ok',
    'default',
    { concurrencySafe: true, defer: false, destructive: false, openWorld: false, readOnly: true },
  )
  registry.register(
    makeToolDefinition('EnterWorktreeTool'),
    () => 'ok',
    'default',
    { concurrencySafe: false, defer: true, destructive: false, openWorld: false, readOnly: false },
  )
  registry.register(
    makeToolDefinition('SecretAgentOnlyTool'),
    () => 'secret',
    'agent-privileged',
    { concurrencySafe: true, defer: false, destructive: false, openWorld: false, readOnly: true },
  )
  registry.register(
    makeToolDefinition('exec_command'),
    () => 'ran',
    'default',
    { concurrencySafe: false, defer: false, destructive: true, openWorld: true, readOnly: false },
  )

  // Capability refinement only widens concurrencySafe for pure string-argv read-only calls without stdin.
  const safeCaps = registry.capabilities('exec_command', 'default', { cmd: 'git', args: ['status'] })
  expect(safeCaps.concurrencySafe).toBe(true)
  expect(Object.isFrozen(safeCaps)).toBe(true)
  expect(registry.capabilities('exec_command', 'default', { cmd: 'cat', args: [], stdin: 'payload' }).concurrencySafe).toBe(false)
  expect(registry.capabilities('exec_command', 'default', { cmd: 'git', args: ['branch', 123] }).concurrencySafe).toBe(false)
  expect(registry.capabilities('exec_command', 'default', { cmd: 'arch', args: ['-arm64', 'rm'] }).concurrencySafe).toBe(false)

  // Actionable suggestions on snake_case/short/typo calls, with deferred loading hint when applicable.
  await expect(registry.execute(makeCall('read_file'), { metadata: {} }))
    .rejects.toThrow('is not registered; did you mean "ReadFile"?')
  await expect(registry.execute(makeCall('grep'), { metadata: {} }))
    .rejects.toThrow('is not registered; did you mean "GrepTool"?')
  await expect(registry.execute(makeCall('enter_worktree'), { metadata: {} }))
    .rejects.toThrow('is not registered; did you mean "EnterWorktreeTool" (load its schema first with ToolSearchTool)?')

  // Agent-private tools are never suggested to other agents.
  let unprivilegedError = ''
  try {
    await registry.execute(makeCall('secret_agent_only_tool'), { agentId: 'agent-guest', metadata: {} })
  } catch (error) {
    unprivilegedError = error instanceof Error ? error.message : String(error)
  }
  expect(unprivilegedError).toContain('is not registered')
  expect(unprivilegedError).not.toContain('SecretAgentOnlyTool')

  // Pre-aborted signal throws FunctionExecutionError cleanly.
  const preAborted = new AbortController()
  preAborted.abort()
  await expect(registry.execute(makeCall('ReadFile'), { metadata: {} }, preAborted.signal))
    .rejects.toBeInstanceOf(FunctionExecutionError)
})

test('codingTools capabilities, hunkless/symlink gitApplyPatch containment, and no-op findAndReplace backup safety', async () => {
  await inWorkspace(async (workspace, paths) => {
    const registry = new ToolRegistry()
    registerCodingTools(registry, paths)
    for (const definition of CODING_TOOL_DEFINITIONS) {
      expect(registry.hasDeclaredCapabilities(definition.function.name)).toBe(true)
    }
    expect(registry.capabilities('read_file')).toMatchObject({ readOnly: true, destructive: false, concurrencySafe: true })
    expect(registry.capabilities('git_status')).toMatchObject({ readOnly: true, destructive: false, concurrencySafe: true })
    expect(registry.capabilities('write_file')).toMatchObject({ readOnly: false, destructive: true, interruptBehavior: 'block' })

    // find_and_replace does not create or overwrite a .bak file when 0 matches occur.
    const targetPath = join(workspace, 'notes.txt')
    await Bun.write(targetPath, 'version-1\n')
    const session = { sessionId: 'coding-session' }
    await readCodingFile({ file_path: 'notes.txt' }, paths, session)

    const zeroMatch = await findAndReplace(
      { backup: true, file_path: 'notes.txt', search: 'absent', replace: 'changed' },
      paths,
      session,
    )
    expect(zeroMatch).toBe('Replaced 0 occurrence(s) in notes.txt')
    expect(await Bun.file(join(workspace, 'notes.txt.bak')).exists()).toBe(false)

    // Real replacement writes .bak; subsequent 0-match call preserves that original .bak.
    const oneMatch = await findAndReplace(
      { backup: true, file_path: 'notes.txt', search: 'version-1', replace: 'version-2' },
      paths,
      session,
    )
    expect(oneMatch).toContain('Replaced 1 occurrence(s) in notes.txt (backup saved as notes.txt.bak)')
    expect(await Bun.file(join(workspace, 'notes.txt.bak')).text()).toBe('version-1\n')
    await findAndReplace(
      { backup: true, file_path: 'notes.txt', search: 'still-absent', replace: 'version-3' },
      paths,
      session,
    )
    expect(await Bun.file(join(workspace, 'notes.txt.bak')).text()).toBe('version-1\n')

    // git_apply_patch blocks hunkless rename/copy escapes, symlink mode 120000, and refreshes session freshness.
    const outside = await mkdtemp(join(tmpdir(), 'xerxes-swarm-patch-outside-'))
    try {
      await git(workspace, ['init'])
      await git(workspace, ['config', 'user.email', 'test@example.invalid'])
      await git(workspace, ['config', 'user.name', 'Xerxes Test'])
      await git(workspace, ['config', 'core.autocrlf', 'false'])
      await Bun.write(join(workspace, 'inside.txt'), 'hello\n')
      await git(workspace, ['add', 'inside.txt'])
      await git(workspace, ['commit', '-m', 'init'])
      await symlink(outside, join(workspace, 'escape_dir'), 'dir')

      const renameEscapePatch = [
        'diff --git a/inside.txt b/escape_dir/leaked.txt',
        'similarity index 100%',
        'rename from inside.txt',
        'rename to escape_dir/leaked.txt',
        '',
      ].join('\n')
      await expect(gitApplyPatch({ patch_content: renameEscapePatch }, paths, session)).rejects.toThrow('workspace root')
      expect(await Bun.file(join(outside, 'leaked.txt')).exists()).toBe(false)

      const symlinkModePatch = [
        'diff --git a/planted_link b/planted_link',
        'new file mode 120000',
        'index 0000000..1111111',
        '--- /dev/null',
        '+++ b/planted_link',
        '@@ -0,0 +1 @@',
        `+${outside}`,
        '\\ No newline at end of file',
        '',
      ].join('\n')
      await expect(gitApplyPatch({ patch_content: symlinkModePatch }, paths, session)).rejects.toThrow('mode 120000')

      // Read inside.txt, apply a same-byte-length patch via gitApplyPatch, and verify stale snapshot is forgotten.
      await readFile({ file_path: 'inside.txt' }, paths, session)
      const validPatch = [
        '--- a/inside.txt',
        '+++ b/inside.txt',
        '@@ -1 +1 @@',
        '-hello',
        '+world',
        '',
      ].join('\n')
      expect(await gitApplyPatch({ patch_content: validPatch }, paths, session)).toBe('Patch applied successfully')
      await expect(writeFile({ file_path: 'inside.txt', content: 'clobber\n', overwrite: true }, paths, session))
        .rejects.toThrow('requires reading "inside.txt" first')
    } finally {
      await rm(outside, { force: true, recursive: true })
    }
  })
})

test('FileEditTool reports actionable line-prefix, indentation, and fresh-read diagnostics without mutating files', async () => {
  await inWorkspace(async (workspace, paths) => {
    const file = join(workspace, 'sample.ts')
    const initial = [
      'export function compute(input: number): number {',
      '  const doubled = input * 2',
      '  return doubled + 1',
      '}',
      '',
    ].join('\n')
    await Bun.write(file, initial)
    const context = { sessionId: 'edit-diagnostics' }
    await readFile({ file_path: 'sample.ts' }, paths, context)

    // 1. Line-number prefix copied from `read_file` output.
    await expect(editFile({
      file_path: 'sample.ts',
      old_string: '     2 |   const doubled = input * 2\n     3 |   return doubled + 1',
      new_string: '  return input * 3',
    }, paths, context)).rejects.toThrow('line-number prefixes')

    // 2. Indentation mismatch points to exact 1-based line numbers and quotes on-disk indentation.
    let indentationMessage = ''
    try {
      await editFile({
        file_path: 'sample.ts',
        old_string: '    const doubled = input * 2\n    return doubled + 1',
        new_string: '    return input * 3',
      }, paths, context)
    } catch (error) {
      indentationMessage = error instanceof Error ? error.message : String(error)
    }
    expect(indentationMessage).toContain('was not found exactly, but matches lines 2-3')
    expect(indentationMessage).toContain('  const doubled = input * 2')
    expect(await Bun.file(file).text()).toBe(initial)

    // 3. Completely absent string with a current whole-file read advises against redundant re-reading.
    await expect(editFile({
      file_path: 'sample.ts',
      old_string: 'nonexistentSymbol()',
      new_string: 'replacement()',
    }, paths, context)).rejects.toThrow('your last ReadFile is still current')
    expect(await Bun.file(file).text()).toBe(initial)
  })
})

test('PublicWebClient and analyzeUrl cancel unconsumed streams and enforce expired deadlines strictly', async () => {
  let headCancelled = false
  let fallbackGetCancelled = false
  const probeClient = new PublicWebClient({
    fetcher: async (_url, init) => {
      if (init.method === 'HEAD') {
        const body = new ReadableStream<Uint8Array>({
          cancel() {
            headCancelled = true
          },
        })
        return new Response(body, { status: 405 })
      }
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('<html><title>Probe</title></html>'))
        },
        cancel() {
          fallbackGetCancelled = true
        },
      })
      return new Response(body, { status: 200 })
    },
    urlSafety: { dnsLookup: async () => ['93.184.216.34'] },
  })

  const analysis = await analyzeUrl({
    url: 'https://example.com/resource',
    checkAvailability: true,
    extractMetadata: false,
  }, probeClient)
  expect(analysis.isAvailable).toBe(true)
  expect(headCancelled).toBe(true)
  expect(fallbackGetCancelled).toBe(true)

  // Oversized content-length cancels response.body immediately before throwing.
  let oversizedCancelled = false
  const cappedClient = new PublicWebClient({
    maxResponseBytes: 64,
    fetcher: async () => new Response(
      new ReadableStream<Uint8Array>({
        cancel() {
          oversizedCancelled = true
        },
      }),
      { headers: { 'content-length': '4096' }, status: 200 },
    ),
    urlSafety: { dnsLookup: async () => ['93.184.216.34'] },
  })
  const oversizedFetch = await cappedClient.fetch('https://example.com/large')
  await expect(cappedClient.text(oversizedFetch.response)).rejects.toBeInstanceOf(ClientError)
  expect(oversizedCancelled).toBe(true)

  // Once a fetch's total deadline has elapsed, neither a second redirect hop nor text() gets a free 1ms pass.
  let hopCalls = 0
  const expiringClient = new PublicWebClient({
    timeoutMs: 30,
    fetcher: async url => {
      hopCalls += 1
      if (url.endsWith('/first')) {
        await Bun.sleep(45)
        return new Response(null, { headers: { location: '/second' }, status: 302 })
      }
      return new Response('should-not-run', { status: 200 })
    },
    urlSafety: { dnsLookup: async () => ['93.184.216.34'] },
  })
  await expect(expiringClient.fetch('https://example.com/first')).rejects.toBeInstanceOf(XerxesTimeoutError)
  expect(hopCalls).toBe(1)
})
