// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { chmod, mkdir, mkdtemp, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import jpeg from 'jpeg-js'

import { parseChatCompletionRequest } from '../src/api-server/protocol.js'
import {
  AuditEmitter,
  CompositeCollector,
  ErrorEvent,
  InMemoryCollector,
  OTelCollector,
  ToolCallFailureEvent,
  TurnStartEvent,
  type AuditCollector,
  type OTelAttributes,
  type OTelSpan,
  type OTelTracer,
} from '../src/audit/index.js'
import { AnthropicOAuthSession } from '../src/auth/anthropicOAuth.js'
import { CopilotSession } from '../src/auth/copilotAuth.js'
import { refreshKimiCodeToken } from '../src/auth/kimiCodingOAuth.js'
import { removeOAuthCredentialFile, writeOAuthCredentialFile } from '../src/auth/oauthFlows.js'
import { ProviderAccountRegistry } from '../src/auth/providerAccountRegistry.js'
import { loginRadiusWithBrowser, RadiusOAuthSession } from '../src/auth/radiusOAuth.js'
import { ProfileStore, profileContextLimit } from '../src/bridge/profiles.js'
import { BridgeSlashRouter, type BridgeProviderProfile } from '../src/bridge/slashRouter.js'
import { imageSize } from '../src/core/imageFit.js'
import { coerceImage, downloadImage } from '../src/core/multimodal.js'
import { probeLoadedWindows } from '../src/local/loadedWindow.js'
import { fitWindow } from '../src/local/memoryFit.js'
import { readLogTail } from '../src/local/server.js'
import { createAgentState } from '../src/streaming/events.js'
import { collectGitDiff, parseUnifiedDiff } from '../src/workspace/gitDiff.js'
import { assertScmPath, GitScm } from '../src/workspace/gitScm.js'

async function withTempDir(prefix: string, run: (dir: string) => Promise<void>): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), prefix))
  try {
    await run(dir)
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
}

describe('auth: atomic owner-only credential persistence, cancellation, and account registry races', () => {
  test('writeOAuthCredentialFile enforces 0700/0600 modes, repairs loose permissions, and removeOAuthCredentialFile rethrows non-ENOENT errors', async () => {
    await withTempDir('xr-oauth-file-', async dir => {
      const authDir = join(dir, 'auth')
      const credPath = join(authDir, 'anthropic-oauth.json')

      // Pre-create with permissive mode to verify atomic overwrite repairs 0600.
      await mkdir(authDir, { recursive: true, mode: 0o755 })
      await writeFile(credPath, '{"stale":true}\n', { mode: 0o644 })
      await chmod(credPath, 0o644)

      await writeOAuthCredentialFile(credPath, { access: 'sk-ant-oat01-a', refresh: 'ref-1', expires: 1_900_000_000 })
      if (process.platform !== 'win32') {
        expect((await stat(credPath)).mode & 0o777).toBe(0o600)
      }

      const session = new AnthropicOAuthSession({ xerxesHome: dir, environment: {} })
      expect((await session.stored())?.access).toBe('sk-ant-oat01-a')
      expect(await session.logout()).toBe(true)
      expect(await session.logout()).toBe(false)

      // A directory at the credential path is EISDIR/EPERM, not ENOENT — must propagate.
      const dirAsFile = join(dir, 'directory-not-file')
      await mkdir(dirAsFile)
      await expect(removeOAuthCredentialFile(dirAsFile)).rejects.toThrow()
    })
  })

  test('loginRadiusWithBrowser aborts promptly on signal cancellation without leaking the callback server', async () => {
    const controller = new AbortController()
    let openedUrl = ''
    const loginPromise = loginRadiusWithBrowser(
      'https://gw.example.com',
      'https://gw.example.com/authorize',
      {
        callbackPort: 0,
        signal: controller.signal,
        openUrl: url => {
          openedUrl = url
          queueMicrotask(() => controller.abort(new Error('user cancelled')))
        },
      },
    )
    await expect(loginPromise).rejects.toThrow('Login cancelled')
    expect(openedUrl).toContain('redirect_uri=')
  })

  test('RadiusOAuthSession isolates concurrent credential flights per gateway and per caller AbortSignal', async () => {
    await withTempDir('xr-radius-flight-', async home => {
      await writeOAuthCredentialFile(join(home, 'auth', 'radius-oauth.json'), {
        gateway: 'https://gw.example.com',
        access: 'expired-access',
        refresh: 'rad-refresh-1',
        expires: 100,
      })

      let resolveFetch!: (response: Response) => void
      const fetchStarted = Promise.withResolvers<void>()
      const session = new RadiusOAuthSession({
        xerxesHome: home,
        environment: {},
        now: () => 1_000,
        fetchImplementation: async () => {
          fetchStarted.resolve()
          return new Promise<Response>(resolve => {
            resolveFetch = resolve
          })
        },
      })

      const abortCaller = new AbortController()
      const abortedPromise = session.credential('https://gw.example.com', abortCaller.signal)
      const survivorPromise = session.credential('https://gw.example.com')
      await fetchStarted.promise

      abortCaller.abort(new Error('caller-1-cancelled'))
      await expect(abortedPromise).rejects.toThrow('caller-1-cancelled')

      resolveFetch(new Response(JSON.stringify({
        access_token: 'fresh-rad-token',
        refresh_token: 'rad-refresh-2',
        expires_in: 3_600,
      })))
      const survivor = await survivorPromise
      expect(survivor.access).toBe('fresh-rad-token')
    })
  })

  test('CopilotSession.credential decouples caller cancellation from shared refresh and aborts device-flow sleep', async () => {
    await withTempDir('xr-copilot-cancel-', async home => {
      let releaseExchange!: () => void
      const exchangeEntered = Promise.withResolvers<void>()
      const copilot = new CopilotSession({
        xerxesHome: home,
        environment: { COPILOT_GITHUB_TOKEN: 'gho_shared' },
        now: () => 1_700_000_000,
        fetchImplementation: (async () => {
          exchangeEntered.resolve()
          await new Promise<void>(resolve => {
            releaseExchange = resolve
          })
          return Response.json({
            token: 'tid=1;proxy-ep=https://api.individual.githubcopilot.com',
            expires_at: 1_700_003_600,
          })
        }) as unknown as typeof fetch,
      })

      const firstAbort = new AbortController()
      const first = copilot.credential(firstAbort.signal)
      const second = copilot.credential()
      await exchangeEntered.promise

      firstAbort.abort(new Error('cancel-first-caller'))
      await expect(first).rejects.toThrow('cancel-first-caller')

      releaseExchange()
      const resolved = await second
      expect(resolved.refresh).toBe('gho_shared')
      if (process.platform !== 'win32') {
        expect((await stat(copilot.storedPath())).mode & 0o777).toBe(0o600)
      }

      // Device-flow poll sleep must abort immediately without waiting out a hung sleep.
      const loginAbort = new AbortController()
      const sleepEntered = Promise.withResolvers<void>()
      const deviceSession = new CopilotSession({
        xerxesHome: home,
        environment: {},
        sleep: () => {
          sleepEntered.resolve()
          return new Promise<void>(() => {})
        },
        fetchImplementation: (async () => Response.json({
          device_code: 'dc-1',
          user_code: 'CODE-1',
          verification_uri: 'https://github.com/login/device',
          interval: 15,
          expires_in: 900,
        })) as unknown as typeof fetch,
      })
      const pendingLogin = deviceSession.login(() => {}, loginAbort.signal)
      await sleepEntered.promise
      loginAbort.abort(new Error('stop-device-login'))
      await expect(pendingLogin).rejects.toThrow('stop-device-login')
    })
  })

  test('refreshKimiCodeToken does not enter retry backoff sleep after caller abort', async () => {
    const controller = new AbortController()
    let sleepCalls = 0
    const pending = refreshKimiCodeToken('https://auth.kimi.com', 'ref-tok', {
      signal: controller.signal,
      sleep: async () => {
        sleepCalls += 1
      },
      fetchImplementation: async (_url, init) => new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => reject(init.signal?.reason), { once: true })
        queueMicrotask(() => controller.abort(new Error('abort-in-flight')))
      }),
    })
    await expect(pending).rejects.toThrow('Kimi Code token refresh aborted')
    expect(sleepCalls).toBe(0)
  })

  test('ProviderAccountRegistry rejects path/control characters in labels and serializes concurrent account registrations', async () => {
    await withTempDir('xr-acct-race-', async dir => {
      const registry = new ProviderAccountRegistry(join(dir, 'accounts'), join(dir, 'profiles.json'))
      for (const bad of ['../escape', 'work/prod', 'work\\prod', 'bad\tlabel', 'bad\x7flabel']) {
        await expect(registry.create('claude', bad)).rejects.toThrow('Account label is required')
      }

      const created = await Promise.all([
        registry.create('claude', 'team'),
        registry.create('claude', 'team'),
        registry.create('codex', 'personal'),
        registry.create('codex', 'work'),
      ])
      expect(new Set(created.map(account => account.id)).size).toBe(4)
      expect(new Set(created.map(account => account.profileName)).size).toBe(4)

      const listed = await registry.list()
      expect(listed.map(account => account.id).sort()).toEqual(created.map(account => account.id).sort())
    })
  })
})

describe('workspace: unified diff hunk headers, paths with b/ and quotes, UTF-8/UTF-16 truncation, and SCM path guards', () => {
  test('parseUnifiedDiff preserves ++/-- code lines inside hunks, parses symmetric b/ and quoted paths, and avoids lone surrogates', () => {
    const emoji = '🚀'.repeat(2_100) // 4,200 UTF-16 code units; 4096 - 1 ('…') = 4095 lands mid-surrogate pair!
    const rawDiff = [
      'diff --git a/src/a b/b/index.ts b/src/a b/b/index.ts',
      'index 1111111..2222222 100644',
      '--- a/src/a b/b/index.ts',
      '+++ b/src/a b/b/index.ts',
      '@@ -10,3 +10,4 @@',
      ' context line',
      '---sql comment removed',
      '+++counter',
      `+${emoji}`,
      ' trailing context',
      'diff --git "a/dir/tab\\tname.ts" "b/dir/tab\\tname.ts"',
      'new file mode 100644',
      '--- /dev/null',
      '+++ "b/dir/tab\\tname.ts"',
      '@@ -0,0 +1,1 @@',
      '+export const ok = true',
    ].join('\n')

    const parsed = parseUnifiedDiff(rawDiff)
    expect(parsed.files).toBe(2)
    expect(parsed.deletions).toBe(1)
    expect(parsed.insertions).toBe(3)

    const fileRows = parsed.lines.filter(line => line.kind === 'file').map(line => line.text)
    expect(fileRows).toEqual(['src/a b/b/index.ts', 'dir/tab\tname.ts'])

    const sqlDeleted = parsed.lines.find(line => line.text === '---sql comment removed')
    expect(sqlDeleted).toEqual({ kind: 'del', text: '---sql comment removed', oldLine: 11 })

    const counterAdded = parsed.lines.find(line => line.text === '+++counter')
    expect(counterAdded).toEqual({ kind: 'add', text: '+++counter', newLine: 11 })

    const trailingContext = parsed.lines.find(line => line.text === ' trailing context')
    expect(trailingContext).toEqual({ kind: 'context', text: ' trailing context', oldLine: 12, newLine: 13 })

    const truncatedEmojiLine = parsed.lines.find(line => line.text.startsWith('+🚀'))
    expect(truncatedEmojiLine).toBeDefined()
    // Ensure no unpaired UTF-16 surrogates exist after truncation.
    expect(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/.test(truncatedEmojiLine!.text)).toBe(false)
  })

  test('assertScmPath and collectGitDiff reject dot, empty, backslash-traversal, and drive-prefixed paths, and commitContext slices UTF-8 cleanly', async () => {
    for (const bad of ['.', './a.ts', 'a/./b.ts', 'a//b.ts', '\\foo.ts', 'C:foo.ts', 'a\\..\\b.ts', 'a\\\\b.ts']) {
      expect(() => assertScmPath(bad)).toThrow()
    }

    await withTempDir('xr-scm-utf8-', async dir => {
      await Bun.spawn(['git', 'init', '-q', '-b', 'main', dir]).exited
      await Bun.spawn(['git', '-C', dir, 'config', 'user.name', 'Test']).exited
      await Bun.spawn(['git', '-C', dir, 'config', 'user.email', 'test@example.com']).exited
      await writeFile(join(dir, 'unicode.txt'), `${'空🚀'.repeat(400)}\n`, 'utf8')

      for (const bad of ['.', './unicode.txt', 'C:unicode.txt', 'sub\\..\\unicode.txt', '\\unicode.txt']) {
        expect((await collectGitDiff({ cwd: dir, includeUntracked: true, path: bad })).kind).toBe('error')
      }

      const scm = (await GitScm.open(dir))!
      // Cut right across multi-byte UTF-8 boundaries and verify zero U+FFFD replacement chars.
      for (const budget of [250, 251, 252, 253]) {
        const ctx = await scm.commitContext(budget)
        expect(ctx.truncated).toBe(true)
        expect(ctx.diff.includes('\uFFFD')).toBe(false)
        expect(Buffer.byteLength(ctx.diff, 'utf8')).toBeLessThanOrEqual(budget)
      }
    })
  })
})

describe('audit: secret redaction on failures/errors, loop-warning sessionId, OTel duplicate turnId cleanup, and fault-isolated CompositeCollector', () => {
  test('AuditEmitter redacts secrets in emitToolCallFailure and emitError and forwards sessionId to on_loop_warning hooks', () => {
    const memory = new InMemoryCollector()
    const seenPayloads: Array<Record<string, unknown>> = []
    const emitter = new AuditEmitter({
      collector: memory,
      sessionId: 'sess-audit-42',
      hookRunner: {
        hasHooks: point => point === 'on_loop_warning',
        run: (_point, payload) => {
          seenPayloads.push({ ...payload })
        },
      },
    })

    const secretKey = `sk-${'a'.repeat(32)}`
    emitter.emitToolCallFailure({
      toolName: 'WebFetch',
      errorType: 'HttpError',
      errorMessage: `Request failed with Authorization: Bearer ${secretKey}`,
    })
    emitter.emitError({
      errorType: 'ProviderError',
      errorMessage: `Upstream rejected key ${secretKey}`,
      context: `header Authorization: Bearer ${secretKey}`,
    })
    emitter.emitLoopWarning({ toolName: 'Bash', count: 4, pattern: 'repeat' })

    const events = memory.getEvents()
    const failure = events[0] as ToolCallFailureEvent
    const err = events[1] as ErrorEvent
    expect(failure.errorMessage).not.toContain(secretKey)
    expect(err.errorMessage).not.toContain(secretKey)
    expect(err.errorContext).not.toContain(secretKey)
    expect(seenPayloads).toEqual([
      expect.objectContaining({ toolName: 'Bash', count: 4, sessionId: 'sess-audit-42' }),
    ])
  })

  test('CompositeCollector delivers emit, flush, drain, and close to healthy sinks even when an earlier child throws', async () => {
    const healthy = new InMemoryCollector()
    let flushed = 0
    let drained = 0
    let closed = 0
    const broken: AuditCollector & { drain(): Promise<void>; close(): Promise<void> } = {
      emit() {
        throw new Error('broken-emit')
      },
      flush() {
        throw new Error('broken-flush')
      },
      async drain() {
        throw new Error('broken-drain')
      },
      async close() {
        throw new Error('broken-close')
      },
    }
    const observer: AuditCollector & { drain(): Promise<void>; close(): Promise<void> } = {
      emit(event) {
        healthy.emit(event)
      },
      flush() {
        flushed += 1
      },
      async drain() {
        drained += 1
      },
      async close() {
        closed += 1
      },
    }

    const composite = new CompositeCollector([broken, observer])
    expect(() => composite.emit(new ErrorEvent({ errorMessage: 'survived' }))).toThrow('broken-emit')
    expect(healthy.size).toBe(1)

    expect(() => composite.flush()).toThrow('broken-flush')
    expect(flushed).toBe(1)

    await expect(composite.drain()).rejects.toThrow('broken-drain')
    expect(drained).toBe(1)

    await expect(composite.close()).rejects.toThrow('broken-close')
    expect(closed).toBe(1)
  })

  test('OTelCollector ends a previous span when TurnStartEvent arrives twice with the same turnId', () => {
    const spans: Array<{ ended: boolean; name: string }> = []
    const tracer: OTelTracer = {
      startSpan(name: string, _opts?: Readonly<{ attributes?: OTelAttributes }>): OTelSpan {
        const record = { ended: false, name }
        spans.push(record)
        return {
          addEvent() {},
          end() {
            record.ended = true
          },
          setAttribute() {},
        }
      },
    }
    const collector = new OTelCollector({ tracer })
    collector.emit(new TurnStartEvent({ turnId: 'dup-turn' }))
    collector.emit(new TurnStartEvent({ turnId: 'dup-turn' }))
    expect(spans).toHaveLength(2)
    expect(spans[0]?.ended).toBe(true)
    expect(spans[1]?.ended).toBe(false)
    expect(collector.openTurnCount).toBe(1)
  })
})

describe('api-server, bridge, local, and core edge cases', () => {
  test('parseChatCompletionRequest accepts empty-string tool arguments, developer role, whitespace text parts, and text-part tool outputs', () => {
    const parsed = parseChatCompletionRequest({
      model: 'gpt-4o',
      messages: [
        { role: 'developer', content: 'Follow repository conventions.' },
        {
          role: 'user',
          content: [
            { type: 'text', text: '' },
            { type: 'text', text: 'Run status' },
          ],
        },
        {
          role: 'assistant',
          content: null,
          tool_calls: [
            { id: 'call_zero', type: 'function', function: { name: 'Status', arguments: '' } },
          ],
        },
        {
          role: 'tool',
          tool_call_id: 'call_zero',
          content: [
            { type: 'text', text: 'clean ' },
            { type: 'text', text: 'worktree' },
          ],
        },
      ],
      tools: [
        {
          type: 'function',
          function: { name: 'Status', description: 'Check status', parameters: { type: 'object', properties: {} } },
        },
      ],
    })

    expect(parsed.completion.messages[0]).toEqual({ role: 'system', content: 'Follow repository conventions.' })
    expect(parsed.completion.messages[2]).toEqual({
      role: 'assistant',
      content: '',
      tool_calls: [{ id: 'call_zero', type: 'function', function: { name: 'Status', arguments: {} } }],
    })
    expect(parsed.completion.messages[3]).toEqual({
      role: 'tool',
      tool_call_id: 'call_zero',
      content: 'clean worktree',
    })
  })

  test('ProfileStore model lookups ignore Object.prototype keys and BridgeSlashRouter clears stale sampling on /provider switch', async () => {
    await withTempDir('xr-bridge-prof-', async dir => {
      const store = new ProfileStore(join(dir, 'profiles.json'))
      const saved = store.save({
        name: 'custom',
        baseUrl: 'https://api.openai.com/v1',
        apiKey: 'sk-test',
        model: 'gpt-4o',
      })
      expect(profileContextLimit(saved, 'constructor')).toBeUndefined()
      expect(profileContextLimit(saved, 'openai/toString')).toBeUndefined()
    })

    const tuned: BridgeProviderProfile = {
      name: 'tuned',
      api_key: 'k1',
      base_url: 'https://api.openai.com/v1',
      model: 'gpt-4o',
      provider: 'openai',
      sampling: { temperature: 0.1, thinking: true, reasoning_effort: 'high' },
    }
    const clean: BridgeProviderProfile = {
      name: 'clean',
      api_key: 'k2',
      base_url: 'https://api.openai.com/v1',
      model: 'gpt-4.1',
      provider: 'openai',
      sampling: {},
    }
    let current = tuned
    const config: Record<string, unknown> = { model: 'gpt-4o' }
    const router = new BridgeSlashRouter({
      cwd: '/workspace',
      state: createAgentState(),
      config,
      host: {
        providers: {
          active: () => current,
          list: () => [tuned, clean],
          saveSampling: () => current,
          select: name => {
            current = name === 'clean' ? clean : tuned
            return current
          },
        },
      },
    })

    await router.dispatch('/provider tuned')
    expect(config).toMatchObject({ temperature: 0.1, thinking: true, reasoning_effort: 'high' })

    await router.dispatch('/provider clean')
    expect(config.model).toBe('gpt-4.1')
    expect(config.temperature).toBeUndefined()
    expect(config.thinking).toBeUndefined()
    expect(config.reasoning_effort).toBeUndefined()
  })

  test('fitWindow rejects negative spare memory when kvBytesPerToken is 0, probeLoadedWindows cancels oversized streams early, and readLogTail skips split UTF-8 continuation bytes', async () => {
    expect(fitWindow({
      freeBytes: 2 * 1024 ** 3,
      weightsBytes: 8 * 1024 ** 3,
      kvBytesPerToken: 0,
      maxWindow: 131_072,
    }).fits).toBe(false)

    let chunksProduced = 0
    const chunk = new Uint8Array(256 * 1024).fill(0x61) // 256 KiB
    const probed = await probeLoadedWindows(
      { baseUrl: 'http://127.0.0.1:1234/v1' },
      async () => new Response(new ReadableStream<Uint8Array>({
        pull(controller) {
          chunksProduced += 1
          if (chunksProduced > 20) {
            controller.close()
            return
          }
          controller.enqueue(chunk)
        },
      })),
    )
    expect(probed.ok).toBe(false)
    // MAX_RESPONSE_BYTES is 1 MiB (4 chunks); streaming abort stops well before all 20 chunks are pulled.
    expect(chunksProduced).toBeLessThanOrEqual(6)

    await withTempDir('xr-log-tail-', async dir => {
      const logFile = join(dir, 'engine.log')
      const content = `${'🚀'.repeat(40)}server ready\n`
      await writeFile(logFile, content, 'utf8')
      const fullLen = Buffer.byteLength(content, 'utf8')
      // Slice right inside a 4-byte emoji sequence and verify no leading U+FFFD appears.
      const tail = await readLogTail(logFile, fullLen - 2)
      expect(tail.includes('\uFFFD')).toBe(false)
      expect(tail).toContain('server ready')
    })
  })

  test('imageSize parses JPEGs with 0xFF fill bytes before markers, coerceImage accepts parameterized data URLs, and downloadImage aborts oversized chunked bodies', async () => {
    const rawJpeg = jpeg.encode({ data: Buffer.alloc(16 * 12 * 4, 200), width: 16, height: 12 }, 80).data
    // Inject ITU T.81 §B.1.1.2 0xFF fill bytes right after SOI (FF D8).
    const withFillBytes = new Uint8Array(rawJpeg.length + 3)
    withFillBytes.set(rawJpeg.subarray(0, 2), 0)
    withFillBytes.set([0xff, 0xff, 0xff], 2)
    withFillBytes.set(rawJpeg.subarray(2), 5)
    expect(imageSize(withFillBytes)).toEqual({ width: 16, height: 12, format: 'jpeg' })

    // Minimal valid 1x1 PNG base64 with RFC 2397 MIME parameter `;name=dot.png`.
    const png1x1 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
    const coerced = coerceImage(`data:image/png;name=dot.png;base64,${png1x1}`)
    expect(coerced.mediaType).toBe('image/png')
    expect(coerced.bytes.byteLength).toBeGreaterThan(20)

    let pulled = 0
    await expect(downloadImage('https://example.com/huge.png', {
      maxBytes: 1_024,
      fetchImplementation: async () => new Response(new ReadableStream<Uint8Array>({
        pull(controller) {
          pulled += 1
          if (pulled > 20) {
            controller.close()
            return
          }
          controller.enqueue(new Uint8Array(512).fill(0x89))
        },
      }), { status: 200, headers: { 'content-type': 'image/png' } }),
    })).rejects.toThrow('Image exceeds maximum size of 1024 bytes')
    expect(pulled).toBeLessThanOrEqual(4)
  })
})
