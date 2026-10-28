// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, spyOn, test } from 'bun:test'
import { createHash } from 'node:crypto'
import { mkdir, readdir, stat, unlink } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import {
  VOICE_MODELS, resolveVoiceModelFile, voiceModelFileUrlPath,
  type VoiceModel, type VoiceModelId, type ResolvedVoiceModelFile,
} from '../src/client/voiceModelManifest.js'
import { runHubCli } from '../src/hub/cli.js'
import { ensureHubDirectories, hubPaths } from '../src/hub/paths.js'
import { createHubService } from '../src/hub/service.js'
import { HubVoiceModelStore, type HubVoiceModelStoreOptions, type VoiceModelFetch } from '../src/hub/voice/modelStore.js'
import { FakeIdentity, disposeAll, registerCleanup, standardUsers, startFixture, tempDir, testSigner, until } from './hubHarness.js'

afterEach(disposeAll)
const bytes = new TextEncoder().encode('{"test":true}')
const hash = createHash('sha256').update(bytes).digest('hex')
const fixtureModel = (model: VoiceModel): VoiceModel => ({
  ...model, files: model.files.map(file => ({ ...file, size: bytes.length, sha256: hash })),
})
const models: Record<VoiceModelId, VoiceModel> = {
  vad: fixtureModel(VOICE_MODELS.vad), stt_base: fixtureModel(VOICE_MODELS.stt_base),
  stt_tiny: fixtureModel(VOICE_MODELS.stt_tiny), tts: fixtureModel(VOICE_MODELS.tts),
}
function resolveFixture(relative: string): ResolvedVoiceModelFile | undefined {
  const resolved = resolveVoiceModelFile(relative)
  if (!resolved) return undefined
  const model = models[resolved.id]
  const file = model.files.find(entry => entry.path === resolved.file.path)
  return file ? { id: resolved.id, model, file } : undefined
}
async function fixtureStore(options: Partial<HubVoiceModelStoreOptions> = {}) {
  const stateDir = await tempDir('xh-models-')
  const calls: string[] = []
  const store = new HubVoiceModelStore({
    stateDir, models, resolveFile: resolveFixture, environment: {},
    fetch: async url => { calls.push(url); return new Response(bytes) }, ...options,
  })
  registerCleanup(() => store.close())
  const resolved = store.files('all')[0]!
  const path = join(stateDir, 'voice-models', resolved.model.repo, resolved.model.revision, resolved.file.path)
  return { store, stateDir, calls, resolved, path }
}
const headers = { cookie: 'who=alice' }
const relativeOf = (entry: ResolvedVoiceModelFile, revision = entry.model.revision): string => `${entry.model.repo}/resolve/${revision}/${entry.file.path}`

async function publishedFiles(stateDir: string): Promise<string[]> {
  return await readdir(join(stateDir, 'voice-models'), { recursive: true }).catch(() => [])
}

describe('Hub voice model store', () => {
  test('streams verified chunks to private storage, then reuses the verified cache', async () => {
    const f = await fixtureStore({ fetch: async () => new Response(new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(bytes.slice(0, 3)); controller.enqueue(bytes.slice(3)); controller.close() },
    })) })
    expect(await f.store.get(f.resolved)).toBe(f.path)
    expect(new Uint8Array(await Bun.file(f.path).arrayBuffer())).toEqual(bytes)
    expect((await stat(dirname(f.path))).mode & 0o777).toBe(0o700)
    expect((await stat(f.path)).mode & 0o777).toBe(0o600)
    // A fresh process-equivalent instance must hash the stored bytes, not fetch them.
    const fresh = new HubVoiceModelStore({ stateDir: f.stateDir, models, resolveFile: resolveFixture, environment: {}, fetch: async () => { throw new Error('network must not be used') } })
    registerCleanup(() => fresh.close())
    expect(await fresh.get(f.resolved)).toBe(f.path)
    expect(await fresh.get(f.resolved)).toBe(f.path)
  })

  test('size overflow is refused while streaming and no file or temporary is published', async () => {
    const f = await fixtureStore({ fetch: async () => new Response(new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(bytes); controller.enqueue(new Uint8Array([1])); controller.close() },
    })) })
    await expect(f.store.get(f.resolved)).rejects.toThrow('size exceeded')
    expect(await Bun.file(f.path).exists()).toBe(false)
    expect((await publishedFiles(f.stateDir)).some(path => path.endsWith('.tmp'))).toBe(false)
  })

  test('verification counts decoded bytes instead of trusting an encoded Content-Length', async () => {
    const f = await fixtureStore({ fetch: async () => new Response(bytes, { headers: { 'content-length': '999', 'content-encoding': 'gzip' } }) })
    expect(await f.store.get(f.resolved)).toBe(f.path)
    expect(new Uint8Array(await Bun.file(f.path).arrayBuffer())).toEqual(bytes)
  })

  test('hash mismatch and short downloads publish nothing', async () => {
    for (const bad of [new Uint8Array(bytes.length), bytes.slice(1)]) {
      const f = await fixtureStore({ fetch: async () => new Response(bad) })
      await expect(f.store.get(f.resolved)).rejects.toThrow('verification')
      expect(await Bun.file(f.path).exists()).toBe(false)
      expect((await publishedFiles(f.stateDir)).some(path => path.endsWith('.tmp'))).toBe(false)
    }
  })

  test('a corrupt cached file is removed and refetched, including same-sized corruption', async () => {
    for (const bad of [new Uint8Array(bytes.length), bytes.slice(1)]) {
      const f = await fixtureStore()
      await mkdir(dirname(f.path), { recursive: true })
      await Bun.write(f.path, bad)
      expect(await f.store.get(f.resolved)).toBe(f.path)
      expect(f.calls).toHaveLength(1)
      expect(new Uint8Array(await Bun.file(f.path).arrayBuffer())).toEqual(bytes)
    }
  })

  test('N requests for one file share one fetch; main always downloads the pinned revision', async () => {
    const f = await fixtureStore({ fetch: async url => { seen.push(url); await Bun.sleep(15); return new Response(bytes) } })
    const seen: string[] = []
    const alias = f.store.resolve(relativeOf(f.resolved, 'main'))!
    const paths = await Promise.all(Array.from({ length: 12 }, () => f.store.get(alias)))
    expect(new Set(paths)).toEqual(new Set([f.path]))
    expect(seen).toEqual([`https://huggingface.co/${relativeOf(f.resolved)}`])
  })

  test('at most two distinct model downloads run simultaneously', async () => {
    let active = 0
    let peak = 0
    let total = 0
    const fetchPort: VoiceModelFetch = async () => {
      active += 1; total += 1; peak = Math.max(peak, active)
      await Bun.sleep(15)
      active -= 1
      return new Response(bytes)
    }
    const f = await fixtureStore({ fetch: fetchPort })
    await Promise.all(f.store.files('all').map(entry => f.store.get(entry)))
    expect(peak).toBe(2)
    expect(total).toBe(f.store.files('all').length)
  })

  test('uses the configured HTTPS mirror and only follows HTTPS redirects manually', async () => {
    const calls: Array<{ url: string; redirect: RequestRedirect | undefined }> = []
    const f = await fixtureStore({ environment: { XERXES_HUB_VOICE_MODELS_URL: 'https://mirror.example/models/' }, fetch: async (url, init) => {
      calls.push({ url, redirect: init.redirect })
      return calls.length === 1 ? new Response(null, { status: 302, headers: { location: 'https://cdn.example/pinned' } }) : new Response(bytes)
    } })
    await f.store.get(f.resolved)
    expect(calls).toEqual([
      { url: `https://mirror.example/models/${relativeOf(f.resolved)}`, redirect: 'manual' },
      { url: 'https://cdn.example/pinned', redirect: 'manual' },
    ])
  })

  test('refuses downgrade redirects without contacting the insecure target', async () => {
    const calls: string[] = []
    const f = await fixtureStore({ fetch: async url => { calls.push(url); return new Response(null, { status: 302, headers: { location: 'http://insecure.example/model' } }) } })
    await expect(f.store.get(f.resolved)).rejects.toThrow('not secure')
    expect(calls).toHaveLength(1)
    expect(await Bun.file(f.path).exists()).toBe(false)
  })

  test('bounds stalled fetches and body reads, cleans temporary files, and allows a retry', async () => {
    for (const port of [async () => new Promise<Response>(() => undefined), async () => new Response(new ReadableStream<Uint8Array>())]) {
      const f = await fixtureStore({ timeoutMs: 20, fetch: port })
      await expect(f.store.get(f.resolved)).rejects.toThrow('timed out')
      expect(await Bun.file(f.path).exists()).toBe(false)
      expect((await publishedFiles(f.stateDir)).some(path => path.endsWith('.tmp'))).toBe(false)
      await expect(f.store.get(f.resolved)).rejects.toThrow('timed out')
    }
  })

  test('the file deadline includes time waiting for a download slot', async () => {
    const f = await fixtureStore({ timeoutMs: 200, fetch: async () => { await Bun.sleep(120); return new Response(bytes) } })
    const selected = f.store.files('all').slice(0, 3)
    const outcomes = await Promise.allSettled(selected.map(entry => f.store.get(entry)))
    expect(outcomes.slice(0, 2).every(result => result.status === 'fulfilled')).toBe(true)
    expect(outcomes[2]?.status).toBe('rejected')
    if (outcomes[2]?.status === 'rejected') expect(String(outcomes[2].reason)).toContain('timed out')
    const last = selected[2]!
    expect(await Bun.file(join(f.stateDir, 'voice-models', last.model.repo, last.model.revision, last.file.path)).exists()).toBe(false)
  })

  test('a throwing reader releaseLock cannot skip fd close and temporary-file removal', async () => {
    const f = await fixtureStore({ fetch: async () => {
      const stream = new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(new Uint8Array(bytes.length + 1)); controller.close() } })
      const original = stream.getReader.bind(stream)
      Object.defineProperty(stream, 'getReader', { value: () => {
        const reader = original()
        const spy = spyOn(reader, 'releaseLock').mockImplementation(() => { throw new Error('synthetic lock error') })
        registerCleanup(() => spy.mockRestore())
        return reader
      } })
      return new Response(stream)
    } })
    await expect(f.store.get(f.resolved)).rejects.toThrow('size exceeded')
    expect(await Bun.file(f.path).exists()).toBe(false)
    expect((await publishedFiles(f.stateDir)).some(path => path.endsWith('.tmp'))).toBe(false)
  })

  test('a redirect without Location is invalid, not a redirect-limit error', async () => {
    const f = await fixtureStore({ fetch: async () => new Response(null, { status: 302 }) })
    await expect(f.store.get(f.resolved)).rejects.toThrow('invalid redirect')
    expect(await Bun.file(f.path).exists()).toBe(false)
  })

  test('failure observers receive sanitized causes for upstream, overflow and hash errors', async () => {
    for (const [response, cause] of [
      [new Response(null, { status: 503 }), 'upstream status 503'],
      [new Response(new Uint8Array(bytes.length + 1)), 'size exceeded'],
      [new Response(new Uint8Array(bytes.length)), 'hash mismatch'],
    ] as const) {
      const errors: Error[] = []
      const f = await fixtureStore({ fetch: async () => response, onError: error => { errors.push(error) } })
      await expect(f.store.get(f.resolved)).rejects.toThrow()
      expect(errors).toHaveLength(1)
      expect(errors[0]?.cause).toBe(cause)
      expect(JSON.stringify(errors)).not.toMatch(/https?:|\//u)
    }
  })

  test('close cancels active and queued downloads without publishing partial bytes', async () => {
    const signals: AbortSignal[] = []
    const f = await fixtureStore({ fetch: async (_url, init) => {
      signals.push(init.signal as AbortSignal)
      return new Response(new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(bytes.slice(0, 1)) } }))
    } })
    const jobs = f.store.files('all').slice(0, 6).map(entry => f.store.get(entry))
    const outcomes = Promise.allSettled(jobs)
    await until(() => signals.length === 2)
    await f.store.close()
    expect((await outcomes).every(row => row.status === 'rejected')).toBe(true)
    expect(signals.every(signal => signal.aborted)).toBe(true)
    expect((await publishedFiles(f.stateDir)).some(path => path.endsWith('.tmp') || path.endsWith('.onnx') || path.endsWith('.json'))).toBe(false)
  })
})

describe('Hub same-origin voice model HTTP', () => {
  test('requires a signed-in cookie for models and readiness; missing identity fails closed', async () => {
    const f = await fixtureStore()
    const hub = await startFixture({ server: { voiceModels: f.store } })
    for (const path of [`/voice-models/${relativeOf(f.resolved)}`, '/api/hub/v1/voice/models']) {
      expect((await fetch(hub.url(path))).status).toBe(401)
      expect((await fetch(hub.url(path), { headers: { cookie: 'who=nobody' } })).status).toBe(401)
    }
    expect(f.calls).toHaveLength(0)
    const closed = await startFixture({ identity: false, server: { voiceModels: f.store } })
    expect((await fetch(closed.url(`/voice-models/${relativeOf(f.resolved)}`), { headers })).status).toBe(503)
  })

  test('disables the HTTP idle timeout only after authentication and manifest resolution for GET or HEAD', async () => {
    const f = await fixtureStore()
    const timeouts: Array<{ method: string; path: string; seconds: number }> = []
    const hub = await startFixture({ server: { voiceModels: f.store, setRequestTimeout: (request, seconds) => {
      timeouts.push({ method: request.method, path: new URL(request.url).pathname, seconds })
    } } })
    const path = `/voice-models/${relativeOf(f.resolved)}`
    expect((await fetch(hub.url(path))).status).toBe(401)
    expect((await fetch(hub.url('/api/hub/v1/voice/models'), { headers })).status).toBe(200)
    expect((await fetch(hub.url('/voice-models/not-allowed'), { headers })).status).toBe(404)
    expect((await fetch(hub.url(path), { method: 'POST', headers })).status).toBe(405)
    expect(timeouts).toEqual([])
    expect((await fetch(hub.url(path), { headers })).status).toBe(200)
    expect((await fetch(hub.url(path), { method: 'HEAD', headers })).status).toBe(200)
    expect(timeouts).toEqual([{ method: 'GET', path, seconds: 0 }, { method: 'HEAD', path, seconds: 0 }])
  })

  test('retries ENOENT when the cache path disappears during Bun.file stat inside admission', async () => {
    const f = await fixtureStore()
    const probe = Bun.file(f.path)
    const prototype = Object.getPrototypeOf(probe) as { stat: typeof probe.stat }
    const originalStat = prototype.stat
    let removed = false
    const statSpy = spyOn(prototype, 'stat').mockImplementation(async function (this: typeof probe) {
      if (this.name === f.path && !removed) { removed = true; await unlink(f.path) }
      return originalStat.call(this)
    })
    registerCleanup(() => statSpy.mockRestore())
    const hub = await startFixture({ server: { voiceModels: f.store } })
    const response = await fetch(hub.url(`/voice-models/${relativeOf(f.resolved)}`), { headers })
    expect(response.status).toBe(200)
    expect(new Uint8Array(await response.arrayBuffer())).toEqual(bytes)
    expect(removed).toBe(true)
    expect(f.calls).toHaveLength(2)
  })

  test('main, pinned GET, HEAD and conditional GET serve exact bytes and private immutable headers', async () => {
    const f = await fixtureStore()
    const hub = await startFixture({ server: { voiceModels: f.store } })
    const url = hub.url(`/voice-models/${relativeOf(f.resolved, 'main')}`)
    const response = await fetch(url, { headers })
    expect(response.status).toBe(200)
    expect(new Uint8Array(await response.arrayBuffer())).toEqual(bytes)
    expect(response.headers.get('content-length')).toBe(String(bytes.length))
    expect(response.headers.get('content-type')).toBe('application/octet-stream')
    expect(response.headers.get('etag')).toBe(`"${hash}"`)
    expect(response.headers.get('cache-control')).toBe('private, max-age=31536000, immutable')
    expect(response.headers.get('x-content-type-options')).toBe('nosniff')
    const head = await fetch(hub.url(voiceModelFileUrlPath(f.resolved.model, f.resolved.file.path)), { method: 'HEAD', headers })
    expect(head.status).toBe(200)
    expect(head.headers.get('content-length')).toBe(String(bytes.length))
    expect(await head.text()).toBe('')
    const cached = await fetch(url, { headers: { ...headers, 'if-none-match': `"${hash}"` } })
    expect(cached.status).toBe(304)
    expect(await cached.text()).toBe('')
    const config = f.store.files('all').find(entry => entry.file.path === 'config.json')!
    expect((await fetch(hub.url(voiceModelFileUrlPath(config.model, config.file.path)), { headers })).headers.get('content-type')).toBe('application/json')
    expect(f.calls[0]).toBe(`https://huggingface.co/${relativeOf(f.resolved)}`)
    expect(f.calls).toHaveLength(2)
  })

  test('unlisted files, revisions and traversal attempts are 404 and never fetched', async () => {
    const f = await fixtureStore()
    const hub = await startFixture({ server: { voiceModels: f.store } })
    const base = `/voice-models/${f.resolved.model.repo}/resolve/main/`
    for (const path of [base + 'secret.bin', base + '../secret.bin', base + '%2e%2e/secret.bin', base + '..%2fsecret.bin', base + '%2e%2e%2fsecret.bin', base + 'onnx%5cmodel_int8.onnx', base + '%00', base + '%zz', `/voice-models/${relativeOf(f.resolved, 'a'.repeat(40))}`]) {
      expect((await fetch(hub.url(path), { headers })).status).toBe(404)
    }
    expect(f.calls).toHaveLength(0)
  })

  test('disabled models return safe 503 JSON; status remains available with enabled false', async () => {
    const f = await fixtureStore({ environment: { XERXES_HUB_VOICE_MODELS: 'off' } })
    const hub = await startFixture({ server: { voiceModels: f.store } })
    const response = await fetch(hub.url(`/voice-models/${relativeOf(f.resolved)}`), { headers })
    expect(response.status).toBe(503)
    expect(await response.json()).toEqual({ error: 'voice_models_unavailable', reason: 'Model downloads are disabled.' })
    const status = await fetch(hub.url('/api/hub/v1/voice/models'), { headers })
    expect(status.status).toBe(200)
    expect(await status.json()).toMatchObject({ enabled: false, tiers: { standard: { ready: false }, light: { ready: false } } })
    expect(f.calls).toHaveLength(0)
  })

  test('download failures return no URL, path or raw upstream message', async () => {
    const f = await fixtureStore({ fetch: async () => { throw new Error('https://secret.example/token /private/secret') } })
    const errors: Error[] = []
    const hub = await startFixture({ server: { voiceModels: f.store, onError: error => { if (error instanceof Error) errors.push(error) } } })
    const response = await fetch(hub.url(`/voice-models/${relativeOf(f.resolved)}`), { headers })
    expect(response.status).toBe(503)
    expect(await response.json()).toEqual({ error: 'voice_models_unavailable', reason: 'Model storage or download failed.' })
    expect(errors).toHaveLength(1)
    expect(errors[0]?.cause).toBe('storage or network failure')
    expect(JSON.stringify(errors)).not.toMatch(/secret|https?:|\//u)
  })

  test('HTTP overflow and hash mismatch fail with safe 503 and no cached bytes', async () => {
    for (const bad of [new Uint8Array(bytes.length + 1), new Uint8Array(bytes.length)]) {
      const f = await fixtureStore({ fetch: async () => new Response(bad) })
      const hub = await startFixture({ server: { voiceModels: f.store } })
      const response = await fetch(hub.url(`/voice-models/${relativeOf(f.resolved)}`), { headers })
      expect(response.status).toBe(503)
      const error = await response.json() as { error: string; reason: string }
      expect(error.error).toBe('voice_models_unavailable')
      expect(error.reason).not.toMatch(/https?:|\//u)
      expect(await Bun.file(f.path).exists()).toBe(false)
      expect((await publishedFiles(f.stateDir)).some(path => path.endsWith('.tmp'))).toBe(false)
    }
  })

  test('service mounts model routes without provider-backed voice and leaves an injected store caller-owned', async () => {
    const f = await fixtureStore()
    const paths = hubPaths({ XERXES_HOME: f.stateDir })
    await ensureHubDirectories(paths)
    const service = createHubService({
      paths, signer: testSigner(), identity: standardUsers(new FakeIdentity()), exposure: { port: 0 },
      voiceModels: f.store, registryPollMs: 0, onError: error => { throw error },
    })
    registerCleanup(() => service.stop())
    const info = await service.start()
    const url = `http://127.0.0.1:${info.port}`
    expect((await fetch(`${url}/api/hub/v1/voice/models`, { headers })).status).toBe(200)
    expect((await fetch(`${url}/voice-models/${relativeOf(f.resolved)}`, { headers })).status).toBe(200)
    await service.stop()
    expect(await f.store.get(f.resolved)).toBe(f.path)
    await f.store.close()
    await expect(f.store.get(f.resolved)).rejects.toThrow('closed')
  })

  test('service closes its own model store and forwards the configured model environment', async () => {
    const home = await tempDir('xh-owned-models-')
    const paths = hubPaths({ XERXES_HOME: home })
    await ensureHubDirectories(paths)
    const closeSpy = spyOn(HubVoiceModelStore.prototype, 'close')
    registerCleanup(() => closeSpy.mockRestore())
    const service = createHubService({
      paths, signer: testSigner(), identity: standardUsers(new FakeIdentity()), exposure: { port: 0 },
      voiceModelsEnvironment: { XERXES_HUB_VOICE_MODELS: 'off' }, registryPollMs: 0,
    })
    registerCleanup(() => service.stop())
    const info = await service.start()
    const status = await (await fetch(`http://127.0.0.1:${info.port}/api/hub/v1/voice/models`, { headers })).json()
    expect(status).toMatchObject({ enabled: false })
    expect(closeSpy).not.toHaveBeenCalled()
    await service.stop()
    expect(closeSpy).toHaveBeenCalledTimes(1)
  })

  test('readiness is size-only, reports exact tier totals, and never downloads', async () => {
    const f = await fixtureStore()
    const hub = await startFixture({ server: { voiceModels: f.store } })
    const before = await (await fetch(hub.url('/api/hub/v1/voice/models'), { headers })).json()
    expect(before).toMatchObject({ enabled: true, models: { vad: { ready: false, bytes: bytes.length } } })
    expect(f.calls).toHaveLength(0)
    await Promise.all(f.store.files('light').map(entry => f.store.get(entry)))
    const after = await (await fetch(hub.url('/api/hub/v1/voice/models'), { headers })).json()
    expect(after).toMatchObject({ tiers: { standard: { ready: false }, light: { ready: true, bytes: f.store.files('light').length * bytes.length } }, models: { vad: { ready: true }, stt_tiny: { ready: true }, tts: { ready: false } } })
    // Size-only inspection does not hash existing files; serving will still verify them.
    await Bun.write(f.path, new Uint8Array(bytes.length))
    expect((await f.store.status()).models.vad.ready).toBe(true)
  })

  test('phone CSP allows same-origin wasm and workers, without third-party origins or COEP', async () => {
    const f = await fixtureStore()
    const root = await tempDir('xh-model-shell-')
    await Bun.write(join(root, 'index.html'), '<html>phone</html>')
    const hub = await startFixture({ staticRoot: root, server: { voiceModels: f.store } })
    const response = await fetch(hub.url('/'))
    expect(response.status).toBe(200)
    const csp = response.headers.get('content-security-policy')!
    expect(csp).toContain("script-src 'self' 'wasm-unsafe-eval'")
    expect(csp).toContain("worker-src 'self'")
    expect(csp).toContain(`connect-src 'self' ws://127.0.0.1:${hub.port}`)
    expect(csp).not.toContain('https://')
    expect(csp).not.toContain('huggingface')
    expect(csp).not.toContain("'unsafe-eval'")
    expect(response.headers.get('permissions-policy')).toBe('microphone=(self)')
    expect(response.headers.get('cross-origin-opener-policy')).toBe('same-origin')
    expect(response.headers.has('cross-origin-embedder-policy')).toBe(false)
  })
})

describe('Hub voice-models CLI', () => {
  test('status --json emits exactly one object without touching the network', async () => {
    const f = await fixtureStore()
    const stdout: string[] = [], stderr: string[] = []
    const paths = hubPaths({ XERXES_HOME: f.stateDir })
    expect(await runHubCli(['voice-models', 'status', '--json'], { paths, voiceModels: f.store, stdout: line => stdout.push(line), stderr: line => stderr.push(line), lifecycle: { environment: {} } })).toBe(0)
    expect(stdout).toHaveLength(1)
    expect(JSON.parse(stdout[0]!)).toEqual(await f.store.status())
    expect(stderr).toHaveLength(0)
    expect(f.calls).toHaveLength(0)
  })

  test('fetch --tier light --json verifies every selected file, with progress only on stderr', async () => {
    const f = await fixtureStore()
    const stdout: string[] = [], stderr: string[] = []
    expect(await runHubCli(['voice-models', 'fetch', '--tier', 'light', '--json'], { paths: hubPaths({ XERXES_HOME: f.stateDir }), voiceModels: f.store, stdout: line => stdout.push(line), stderr: line => stderr.push(line), lifecycle: { environment: {} } })).toBe(0)
    expect(stdout).toHaveLength(1)
    expect(JSON.parse(stdout[0]!)).toMatchObject({ ok: true, tier: 'light', files: f.store.files('light').length, enabled: true, tiers: { light: { ready: true }, standard: { ready: false } } })
    expect(stderr).toHaveLength(f.store.files('light').length)
    expect(f.calls).toHaveLength(f.store.files('light').length)
  })

  test('invalid --json arguments still emit exactly one error object', async () => {
    const f = await fixtureStore()
    const stdout: string[] = []
    expect(await runHubCli(['voice-models', 'fetch', '--tier', 'unknown', '--json'], {
      paths: hubPaths({ XERXES_HOME: f.stateDir }), voiceModels: f.store, lifecycle: { environment: {} },
      stdout: line => stdout.push(line), stderr: () => undefined,
    })).toBe(1)
    expect(stdout).toHaveLength(1)
    expect(JSON.parse(stdout[0]!)).toEqual({ error: 'invalid_request', reason: 'Invalid voice-models arguments.' })
    expect(f.calls).toHaveLength(0)
  })

  test('failed fetch --json emits one safe error object and nonzero status', async () => {
    const f = await fixtureStore({ environment: { XERXES_HUB_VOICE_MODELS: 'off' } })
    const stdout: string[] = [], stderr: string[] = []
    expect(await runHubCli(['voice-models', 'fetch', '--json'], { paths: hubPaths({ XERXES_HOME: f.stateDir }), voiceModels: f.store, stdout: line => stdout.push(line), stderr: line => stderr.push(line), lifecycle: { environment: {} } })).toBe(1)
    expect(stdout).toHaveLength(1)
    expect(JSON.parse(stdout[0]!)).toEqual({ error: 'voice_models_unavailable', reason: 'Model downloads are disabled.' })
    expect(stderr).toEqual(['Model downloads are disabled.'])
  })
})
