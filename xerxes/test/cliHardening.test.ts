// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, readFile, realpath, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { version } from '../package.json' with { type: 'json' }

const CLI = join(import.meta.dir, '../src/cli.ts')

interface CliResult {
  readonly exitCode: number
  readonly stderr: string
  readonly stdout: string
}

async function runCli(args: readonly string[], options: { readonly config?: unknown } = {}): Promise<CliResult> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-bun-cli-hardening-'))
  try {
    const home = join(root, 'home')
    if (options.config !== undefined) {
      await mkdir(join(home, 'daemon'), { recursive: true })
      await writeFile(join(home, 'daemon', 'config.json'), JSON.stringify(options.config), 'utf8')
    }
    const child = Bun.spawn([process.execPath, CLI, ...args], {
      cwd: root,
      env: { ...process.env, XERXES_HOME: home },
      stderr: 'pipe',
      stdout: 'pipe',
    })
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
      child.exited,
    ])
    return { exitCode, stderr, stdout }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

test('-v prints the version and exits zero', async () => {
  const result = await runCli(['-v'])
  expect(result.exitCode).toBe(0)
  expect(result.stderr).toBe('')
  expect(result.stdout.trim()).toBe(version)
})

test('-V and --version still print the version', async () => {
  for (const flag of ['-V', '--version']) {
    const result = await runCli([flag])
    expect(result.exitCode).toBe(flag === '-V' || flag === '--version' ? 0 : result.exitCode)
    expect(result.stdout.trim()).toBe(version)
  }
})

test('an unrecognized top-level long flag is a usage error, never a prompt', async () => {
  // No provider config exists in this sandbox: if the catch-all forwarded the
  // tokens as a prompt, runOneShot would fail with the connection error instead
  // of this usage message, so the assertion also proves no provider was reached.
  const result = await runCli(['--unknownflag', 'hello'])
  expect(result.exitCode).toBe(1)
  expect(result.stderr).toContain("Unknown option '--unknownflag'")
  expect(result.stderr).toContain("'--'")
})

test('an unrecognized top-level short flag is a usage error too', async () => {
  const result = await runCli(['-x', 'hi'])
  expect(result.exitCode).toBe(1)
  expect(result.stderr).toContain("Unknown option '-x'")
})

test('--resume rejects dash-led prompt words instead of resuming with them', async () => {
  const result = await runCli(['--resume', 'abc123', '--model', 'gpt-4', 'hi'])
  expect(result.exitCode).toBe(1)
  expect(result.stderr).toContain("Unknown option '--model'")
  expect(result.stderr).toContain('xerxes --resume <session_id> -- --model')
  // A guard failure must render as guidance, never as an unhandled crash dump.
  expect(result.stderr).not.toMatch(/^\s+at\b/m)
})

test('-r rejects dash-led prompt words with the same guard', async () => {
  const result = await runCli(['-r', 'abc123', '--verbose'])
  expect(result.exitCode).toBe(1)
  expect(result.stderr).toContain("Unknown option '--verbose'")
})

test('--resume sends everything after the -- separator verbatim as the resumed prompt', async () => {
  // Unlike the bare one-shot form, '--resume' occupies argv's first slot, so a
  // single separator survives the bun launcher intact. With no provider
  // configured the daemon's foundation runner echoes the prompt back verbatim,
  // which is exactly the observable we need.
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-bun-cli-resume-sep-')))
  const home = join(root, 'home')
  const project = join(root, 'project')
  const sessions = join(home, 'sessions')
  const sessionId = 'beadcafe'
  try {
    await Promise.all([
      mkdir(join(home, 'daemon'), { recursive: true }),
      mkdir(project, { recursive: true }),
      mkdir(sessions, { recursive: true }),
    ])
    await writeFile(
      join(sessions, `${sessionId}.json`),
      JSON.stringify({
        session_id: sessionId,
        key: sessionId,
        cwd: project,
        agent_id: 'default',
        updated_at: '2026-07-14T00:00:00.000Z',
        turn_count: 1,
        messages: [
          { role: 'user', content: 'earlier context' },
          { role: 'assistant', content: 'kept.' },
        ],
      }),
      'utf8',
    )
    const child = Bun.spawn(
      [process.execPath, CLI, '--resume', sessionId, '--', '--model', 'gpt-4'],
      {
        cwd: project,
        env: { ...process.env, XERXES_HOME: home },
        stderr: 'pipe',
        stdout: 'pipe',
      },
    )
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
      child.exited,
    ])

    expect(exitCode).toBe(0)
    expect(stderr).toBe('')
    expect(stdout).toContain('--model gpt-4')
    expect(stdout).not.toContain('Unknown option')
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('a one-shot without any configured provider renders a clean two-line usage error', async () => {
  // Empty sandbox home: no profiles, no daemon config. The escaped-prompt form
  // passes the flag guards and reaches runOneShot, whose missing-connection
  // error must surface as the standard usage report, not a raw Bun stack.
  const result = await runCli(['--', '--', '--weird-prompt'])
  expect(result.exitCode).toBe(1)
  expect(result.stderr).toContain(
    'One-shot execution requires a configured runtime connection or active provider profile',
  )
  expect(result.stderr).toContain("run 'xerxes --help' for usage.")
  expect(result.stderr).not.toMatch(/^\s+at\b/m)
})

test('-- sends everything after the marker verbatim as the prompt', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-bun-cli-hardening-sep-')))
  const home = join(root, 'home')
  const project = join(root, 'project')
  const requests: Array<Record<string, unknown>> = []
  const server = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    async fetch(request) {
      requests.push((await request.json()) as Record<string, unknown>)
      return new Response(
        [
          'data: ' + JSON.stringify({
            choices: [{ delta: { content: 'verbatim received' }, finish_reason: 'stop' }],
          }) + '\n\n',
          'data: [DONE]\n\n',
        ].join(''),
        { headers: { 'Content-Type': 'text/event-stream' } },
      )
    },
  })
  try {
    await Promise.all([mkdir(join(home, 'daemon'), { recursive: true }), mkdir(project, { recursive: true })])
    await writeFile(
      join(home, 'daemon', 'config.json'),
      JSON.stringify({
        project_directory: project,
        runtime: {
          model: 'gpt-4o',
          provider: 'openai',
          base_url: `${server.url}v1`,
          api_key: 'test-key',
          permission_mode: 'accept-all',
        },
      }),
      'utf8',
    )
    // The bun launcher consumes one leading '--', so an escaped dash-led
    // prompt reaches cli.ts as ['--', ...rest] only when written twice.
    const escaped = Bun.spawn([process.execPath, CLI, '--', '--', '--weird', 'prompt'], {
      cwd: project,
      env: { ...process.env, XERXES_HOME: home },
      stderr: 'pipe',
      stdout: 'pipe',
    })
    const [escapedOut, escapedErr, escapedCode] = await Promise.all([
      new Response(escaped.stdout).text(),
      new Response(escaped.stderr).text(),
      escaped.exited,
    ])
    expect(escapedCode).toBe(0)
    expect(escapedErr).toBe('')
    expect(escapedOut).toBe('verbatim received\n')

    // A separator after prompt words survives a single hop and exempts the tail.
    const midPrompt = Bun.spawn([process.execPath, CLI, 'explain', '--', '--model'], {
      cwd: project,
      env: { ...process.env, XERXES_HOME: home },
      stderr: 'pipe',
      stdout: 'pipe',
    })
    const [midOut, midErr, midCode] = await Promise.all([
      new Response(midPrompt.stdout).text(),
      new Response(midPrompt.stderr).text(),
      midPrompt.exited,
    ])
    expect(midCode).toBe(0)
    expect(midErr).toBe('')
    expect(midOut).toBe('verbatim received\n')

    expect(requests).toHaveLength(2)
    const userContents = requests.map((request) => {
      const messages = Array.isArray(request?.messages) ? request.messages : []
      const user = messages.find((message): message is { role: string; content: string } => (
        typeof message === 'object' && message !== null
        && (message as { role?: unknown }).role === 'user'
        && typeof (message as { content?: unknown }).content === 'string'
      ))
      return user?.content ?? ''
    })
    expect(userContents).toContain('--weird prompt')
    expect(userContents).toContain('explain --model')
  } finally {
    server.stop(true)
    await rm(root, { recursive: true, force: true })
  }
})

test('a lone -- (consumed by the bun launcher) falls back to reading standard input', async () => {
  const result = await runCli(['--'])
  expect(result.exitCode).toBe(1)
  expect(result.stderr).toContain('No prompt was provided')
})

test('an empty prompt after the -- separator explains itself instead of prompting empty', async () => {
  const result = await runCli(['--', '--'])
  expect(result.exitCode).toBe(1)
  expect(result.stderr).toContain("Put prompt text after '--'")
})

test('daemon rejects an out-of-range websocket_port instead of silently using the default', async () => {
  const result = await runCli(['daemon'], { config: { control: { websocket_port: 99_999 } } })
  expect(result.exitCode).toBe(1)
  expect(result.stderr).toContain('control.websocket_port')
  expect(result.stderr).toContain('must be between 0 and 65535')
})

test('daemon rejects an unparseable websocket_port with the integer wording', async () => {
  const result = await runCli(['daemon'], { config: { control: { websocket_port: 'soon-ish' } } })
  expect(result.exitCode).toBe(1)
  expect(result.stderr).toContain('control.websocket_port')
  expect(result.stderr).toContain('must be a finite integer')
})

test('XERXES_AUDIT records survive process exit: queued turns land in the JSONL sink', async () => {
  // Script-level proof of the shutdown wiring: the CLI child enables the audit
  // sink, runs one resumed turn, and exits. The runtime shutdown hook must
  // close the collector so buffered records (turn_start through turn_end) are
  // on disk before the process is gone, not lost with an unref'd drain timer.
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-bun-cli-audit-')))
  const home = join(root, 'home')
  const project = join(root, 'project')
  const sessions = join(home, 'sessions')
  const sessionId = 'cafe1234'
  const server = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    async fetch() {
      return new Response(
        [
          'data: ' + JSON.stringify({
            choices: [{ delta: { content: 'audited' }, finish_reason: 'stop' }],
            usage: { prompt_tokens: 1, completion_tokens: 1 },
          }) + '\n\n',
          'data: [DONE]\n\n',
        ].join(''),
        { headers: { 'Content-Type': 'text/event-stream' } },
      )
    },
  })
  try {
    await Promise.all([
      mkdir(join(home, 'daemon'), { recursive: true }),
      mkdir(project, { recursive: true }),
      mkdir(sessions, { recursive: true }),
    ])
    await writeFile(
      join(home, 'daemon', 'config.json'),
      JSON.stringify({
        runtime: {
          model: 'gpt-4o',
          provider: 'openai',
          base_url: `${server.url}v1`,
          api_key: 'test-key',
          permission_mode: 'accept-all',
        },
      }),
      'utf8',
    )
    await writeFile(
      join(sessions, `${sessionId}.json`),
      JSON.stringify({
        session_id: sessionId,
        key: sessionId,
        cwd: project,
        agent_id: 'default',
        updated_at: '2026-07-14T00:00:00.000Z',
        turn_count: 1,
        messages: [
          { role: 'user', content: 'earlier context' },
          { role: 'assistant', content: 'kept.' },
        ],
      }),
      'utf8',
    )

    const child = Bun.spawn([process.execPath, CLI, '--resume', sessionId, 'audit me'], {
      cwd: project,
      env: { ...process.env, XERXES_AUDIT: '1', XERXES_HOME: home },
      stderr: 'pipe',
      stdout: 'pipe',
    })
    const [stdout, exitCode] = await Promise.all([
      new Response(child.stdout).text(),
      child.exited,
    ])

    expect(exitCode).toBe(0)
    expect(stdout).toContain('audited')
    const lines = (await readFile(join(home, 'audit', 'events.jsonl'), 'utf8'))
      .split('\n')
      .filter((line) => line.trim())
    const eventTypes = lines.map((line) => JSON.parse(line) as { event_type?: string }).map(
      (record) => record.event_type ?? '',
    )
    expect(eventTypes).toContain('turn_start')
    expect(eventTypes).toContain('turn_end')
  } finally {
    server.stop(true)
    await rm(root, { recursive: true, force: true })
  }
})

/**
 * Registry half of the daemon wiring seam: the CLI forwards call arguments as
 * `tools.capabilities(name, agentId, args)`. These assertions prove that call
 * widens concurrency exactly when the invocation is read-only, which is what
 * lets the loop parallelize read-only exec_command calls from the daemon.
 */
test('registry capabilities refine concurrency per invocation args, fail closed otherwise', () => {
  const registry = new ToolRegistry()
  registry.register({
    type: 'function',
    function: { name: 'exec_command', description: 'run a shell command', parameters: { properties: {}, type: 'object' } },
  }, () => 'ok')

  const baseline = registry.capabilities('exec_command')
  expect(baseline.concurrencySafe).toBeFalse()

  const readOnly = registry.capabilities('exec_command', undefined, { cmd: 'echo', args: ['hi'] })
  expect(readOnly.concurrencySafe).toBeTrue()
  // Only the scheduling axis may widen; permission-relevant axes stay put.
  expect(readOnly.readOnly).toBe(baseline.readOnly)
  expect(readOnly.destructive).toBe(baseline.destructive)

  const mutating = registry.capabilities('exec_command', undefined, { cmd: 'rm', args: ['-rf', 'build'] })
  expect(mutating.concurrencySafe).toBeFalse()
})
