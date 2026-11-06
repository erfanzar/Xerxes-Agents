// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { Xerxes } from '../src/xerxes.js'

const LEAN_MARKER = 'You are Xerxes, an assistant for code and data work'
const RETRY_MARKER = 'failed these checks'

/** The model's next move: a CSV-as-xlsx mistake, the fix after check feedback, or a final reply. */
function nextMove(messages: ReadonlyArray<{ readonly role?: string; readonly content?: unknown }>): 'mistake' | 'fix' | 'done' {
  const last = messages.at(-1)
  if (last?.role === 'tool') return 'done'
  return String(last?.content ?? '').includes(RETRY_MARKER) ? 'fix' : 'mistake'
}

const WRITE_CSV_INTO_XLSX = { name: 'WriteFile', arguments: { file_path: 'out.xlsx', content: 'region,revenue\nnorth,10\n' } }
const WRITE_WORKBOOK = { name: 'XlsxWriter', arguments: { path: 'out.xlsx', overwrite: true, sheets: [{ name: 'Revenue', rows: [['region', 'revenue'], ['north', 10]] }] } }

test('one-shot runs the cascade: a Bonsai run that writes CSV into .xlsx is retried in the same run and fixed', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-cascade-cli-'))
  const home = join(root, 'home')
  const seen: Array<{ system: string; tools: number; move: string }> = []
  const provider = Bun.serve({ hostname: '127.0.0.1', port: 0, async fetch(request) {
    const body = await request.json() as { tools?: unknown[]; messages?: Array<{ role?: string; content?: unknown }> }
    const messages = body.messages ?? []
    const move = nextMove(messages)
    seen.push({ system: String(messages.find(message => message.role === 'system')?.content ?? ''), tools: body.tools?.length ?? 0, move })
    const call = move === 'mistake' ? WRITE_CSV_INTO_XLSX : move === 'fix' ? WRITE_WORKBOOK : undefined
    const chunk = call
      ? { choices: [{ delta: { tool_calls: [{ index: 0, id: `call-${seen.length}`, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.arguments) } }] }, finish_reason: 'tool_calls' }] }
      : { choices: [{ delta: { content: 'Wrote out.xlsx.' }, finish_reason: 'stop' }] }
    return new Response(`data: ${JSON.stringify(chunk)}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
  } })
  try {
    await mkdir(join(home, 'daemon'), { recursive: true })
    await writeFile(join(home, 'daemon', 'config.json'), JSON.stringify({ project_directory: root, runtime: {
      model: 'ternary-bonsai-2-27b', provider: 'openai', base_url: `${provider.url}v1`, api_key: 'fixture-key', permission_mode: 'accept-all',
    } }))
    const child = Bun.spawn([process.execPath, join(import.meta.dir, '../src/cli.ts'), '--output-format', 'json', '--no-save', 'Make a revenue workbook.'], {
      cwd: root,
      env: { ...process.env, XERXES_HOME: home, XERXES_DEFERRED_TOOL_LOADING: '0', XERXES_EDIT_DIAGNOSTICS: '0' },
      stdout: 'pipe',
      stderr: 'pipe',
    })
    const [stdout, stderr, code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited])
    expect({ code, stderr: code === 0 ? '' : stderr }).toEqual({ code: 0, stderr: '' })
    const result = JSON.parse(stdout.trim().split('\n').at(-1)!) as { cascade?: Array<{ escalation?: string; retried?: boolean; findings?: Array<{ code?: string }> }> }
    // The checks caught the text-as-workbook mistake and the run retried inside one invocation.
    expect(result.cascade?.[0]).toMatchObject({ escalation: 'retry_local', retried: true })
    expect(result.cascade?.[0]?.findings?.map(finding => finding.code)).toContain('xlsx_is_text')
    expect(seen.map(entry => entry.move)).toEqual(['mistake', 'done', 'fix', 'done'])
    // Bonsai got the lean prompt and the lean tools.
    expect(seen[0]!.system).toContain(LEAN_MARKER)
    expect(seen[0]!.tools).toBeLessThanOrEqual(13)
    const bytes = new Uint8Array(await Bun.file(join(root, 'out.xlsx')).arrayBuffer())
    expect([bytes[0], bytes[1]]).toEqual([0x50, 0x4b])
  } finally {
    provider.stop(true)
    await rm(root, { recursive: true, force: true })
  }
}, 30_000)

/** An LLM that plays the same script as the mock provider. */
function scriptedLlm(): { llm: LlmClient; requests: CompletionRequest[] } {
  const requests: CompletionRequest[] = []
  const llm: LlmClient = {
    async *stream(request): AsyncGenerator<LlmDelta> {
      requests.push(request)
      const move = nextMove(request.messages as ReadonlyArray<{ role?: string; content?: unknown }>)
      const call = move === 'mistake' ? WRITE_CSV_INTO_XLSX : move === 'fix' ? WRITE_WORKBOOK : undefined
      if (call) {
        yield { toolCalls: [{ id: `call-${requests.length}`, type: 'function', function: { name: call.name, arguments: call.arguments } }] }
        return
      }
      yield { content: 'Wrote out.xlsx.' }
    },
  }
  return { llm, requests }
}

test('the SDK runs the cascade for a Bonsai model and reports each checked attempt on the result', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-cascade-sdk-'))
  try {
    const { llm, requests } = scriptedLlm()
    const xerxes = new Xerxes({ llm, model: 'ternary-bonsai-2-27b', workspaceRoot: root, permissionMode: 'accept-all' })
    const result = await xerxes.run('Make a revenue workbook.')
    expect(requests).toHaveLength(4)
    expect(result.cascade?.[0]?.escalation?.action).toBe('retry_local')
    expect(result.cascade?.[0]?.feedback).toContain(RETRY_MARKER)
    const system = requests[0]!.messages.filter(message => message.role === 'system').map(message => String(message.content)).join('\n')
    expect(system).toContain(LEAN_MARKER)
    const bytes = new Uint8Array(await Bun.file(join(root, 'out.xlsx')).arrayBuffer())
    expect([bytes[0], bytes[1]]).toEqual([0x50, 0x4b])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('the SDK keeps a non-Bonsai model on one attempt, still reporting the failed check', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-cascade-sdk-off-'))
  try {
    const { llm, requests } = scriptedLlm()
    const xerxes = new Xerxes({ llm, model: 'gpt-5.5', workspaceRoot: root, permissionMode: 'accept-all' })
    const result = await xerxes.run('Make a revenue workbook.')
    // Escalation is off by default for a cloud model: one attempt, the finding is reported.
    expect(requests).toHaveLength(2)
    expect(result.cascade?.[0]?.findings.map(finding => finding.code)).toContain('xlsx_is_text')
    expect(result.cascade?.[0]?.feedback).toBeUndefined()
    const system = requests[0]!.messages.filter(message => message.role === 'system').map(message => String(message.content)).join('\n')
    expect(system).not.toContain(LEAN_MARKER)
    // cascade: false turns the checks off entirely.
    const quiet = scriptedLlm()
    const plain = new Xerxes({ llm: quiet.llm, model: 'ternary-bonsai-2-27b', workspaceRoot: root, permissionMode: 'accept-all', cascade: false })
    const plainResult = await plain.run('Make a revenue workbook.')
    expect(quiet.requests).toHaveLength(2)
    expect(plainResult.cascade).toBeUndefined()
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})
