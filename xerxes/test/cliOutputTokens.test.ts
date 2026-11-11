// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ProfileStore } from '../src/bridge/profiles.js'

for (const [configured, expected] of [[undefined, 32_768], [-1, 65_536], [2_000, 2_000]] as const) {
  test(`one-shot OpenAI budget ${configured ?? 'default'} is not replaced by the model ceiling`, async () => {
    const directory = await mkdtemp(join(tmpdir(), 'xerxes-cli-output-'))
    const home = join(directory, 'home'), project = join(directory, 'project')
    const payloads: Record<string, unknown>[] = []
    const server = Bun.serve({ hostname: '127.0.0.1', port: 0, async fetch(request) {
      if (request.method === 'GET') return Response.json({ data: [{ id: 'offline-budget-fixture' }] })
      payloads.push(await request.json() as Record<string, unknown>)
      return new Response(`data: ${JSON.stringify({ choices: [{ delta: { content: 'done' }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`, { headers: { 'content-type': 'text/event-stream' } })
    } })
    try {
      await mkdir(join(home, 'daemon'), { recursive: true }); await mkdir(project)
      const profiles = new ProfileStore(join(home, 'profiles.json'))
      profiles.save({ name: 'offline', provider: 'openai', model: 'offline-budget-fixture', baseUrl: `${server.url}v1`, apiKey: 'test-key', setActive: true })
      profiles.replaceModelCapabilities('offline', { 'offline-budget-fixture': { max_output_tokens: 65_536, context_limit: 262_144 } })
      await Bun.write(join(home, 'daemon/config.json'), JSON.stringify({ project_directory: project, runtime: { permission_mode: 'accept-all', ...(configured === undefined ? {} : { max_tokens: configured }) } }))
      const child = Bun.spawn([process.execPath, join(import.meta.dir, '../src/cli.ts'), 'say done'], { cwd: project, env: { ...process.env, XERXES_HOME: home }, stdout: 'pipe', stderr: 'pipe' })
      const [stdout, stderr, exit] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited])
      expect({ exit, stderr, stdout }).toEqual({ exit: 0, stderr: '', stdout: 'done\n' })
      expect(payloads[0]?.max_tokens).toBe(expected)
    } finally { server.stop(true); await rm(directory, { recursive: true, force: true }) }
  }, 30_000)
}
