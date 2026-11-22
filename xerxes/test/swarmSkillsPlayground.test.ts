// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { createHash, randomBytes } from 'node:crypto'
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  BunWorkspaceModuleEvaluator,
  DaemonEvaluationSessionPort,
  NativeEvaluationAgent,
  classifyEvaluationFailure,
  sweepStaleEvaluationRuns,
  type EvaluationEvent,
  type EvaluationSessionPort,
} from '../playground/index.js'
import { readWorkbookCells } from '../playground/benchmarks/xlsxRead.js'
import type { DaemonEvent, DaemonRuntime } from '../src/daemon/runtime.js'
import { generateImagesViaOpenRouter, OPENROUTER_IMAGES_API } from '../src/images/openrouterImages.js'
import type { ImagesModel } from '../src/images/types.js'
import { parseArxivAtom } from '../src/skills/arxiv.js'
import {
  GoogleWorkspaceAuthorizationError,
  GoogleWorkspaceOAuthClient,
  googleWorkspaceRequestUrl,
  type GooglePendingAuthorization,
  type GoogleWorkspaceAuthorizationStorage,
  type GoogleWorkspaceToken,
} from '../src/skills/googleWorkspace/index.js'
import { extractMarkerDocument, MarkerExtractionError } from '../src/skills/ocrDocuments/marker.js'
import {
  BunOfficePackageDirectory,
  OfficePackageError,
  OfficePackageValidationError,
  condenseOoxmlXml,
  createOfficeZip,
  decodeXmlEntities,
  packOfficePackage,
  parseOoxmlTree,
  serializeOoxmlTree,
  setOoxmlAttribute,
  type OfficePackagePort,
} from '../src/skills/powerpoint/index.js'
import { normalizeOfficePartName } from '../src/skills/powerpoint/package.js'
import { parseYoutubeJson3Transcript, parseYoutubeTimedText } from '../src/skills/youtubeTranscript.js'
import { createSlackBroker } from '../src/slack-broker/broker.js'

class MemoryGoogleAuthStorage implements GoogleWorkspaceAuthorizationStorage {
  pending: GooglePendingAuthorization | undefined
  token: GoogleWorkspaceToken | undefined

  async loadPendingAuthorization(): Promise<GooglePendingAuthorization | undefined> {
    return this.pending
  }

  async loadToken(): Promise<GoogleWorkspaceToken | undefined> {
    return this.token
  }

  async removePendingAuthorization(): Promise<void> {
    this.pending = undefined
  }

  async removeToken(): Promise<void> {
    this.token = undefined
  }

  async savePendingAuthorization(pending: GooglePendingAuthorization): Promise<void> {
    this.pending = pending
  }

  async saveToken(token: GoogleWorkspaceToken): Promise<void> {
    this.token = token
  }
}

class MemoryOfficePackage implements OfficePackagePort {
  private readonly parts = new Map<string, Uint8Array>()

  constructor(entries: Readonly<Record<string, string | Uint8Array>>) {
    for (const [name, value] of Object.entries(entries)) {
      this.parts.set(
        normalizeOfficePartName(name),
        typeof value === 'string' ? new TextEncoder().encode(value) : new Uint8Array(value),
      )
    }
  }

  async deletePart(partName: string): Promise<void> {
    this.parts.delete(normalizeOfficePartName(partName))
  }

  async hasPart(partName: string): Promise<boolean> {
    return this.parts.has(normalizeOfficePartName(partName))
  }

  async listParts(): Promise<readonly string[]> {
    return [...this.parts.keys()].sort()
  }

  async readBytes(partName: string): Promise<Uint8Array> {
    const data = this.parts.get(normalizeOfficePartName(partName))
    if (data === undefined) throw new Error(`missing part: ${partName}`)
    return new Uint8Array(data)
  }

  async readText(partName: string): Promise<string> {
    return new TextDecoder().decode(await this.readBytes(partName))
  }

  async writeBytes(partName: string, contents: Uint8Array): Promise<void> {
    this.parts.set(normalizeOfficePartName(partName), new Uint8Array(contents))
  }

  async writeText(partName: string, contents: string): Promise<void> {
    await this.writeBytes(partName, new TextEncoder().encode(contents))
  }
}

const IMAGE_MODEL: ImagesModel = {
  api: OPENROUTER_IMAGES_API,
  baseUrl: 'https://openrouter.example/api/v1',
  cost: { cacheRead: 0, cacheWrite: 0, input: 1, output: 2 },
  id: 'google/gemini-3-pro-image',
  input: ['text', 'image'],
  name: 'Gemini image',
  output: ['image', 'text'],
  provider: 'openrouter',
}

describe('Wave 2 skills, images, Slack broker, and playground hardening', () => {
  test('Google Workspace OAuth requires matching callback state and redirect URI, and confines API paths', async () => {
    const storage = new MemoryGoogleAuthStorage()
    const oauth = new GoogleWorkspaceOAuthClient({
      authorizationEndpoint: 'https://accounts.google.test/o/oauth2/v2/auth',
      clientId: 'client-1',
      clientSecret: 'secret-1',
      redirectUri: 'http://127.0.0.1:4567/oauth/callback',
      scopes: ['https://www.googleapis.com/auth/gmail.readonly'],
      tokenEndpoint: 'https://accounts.google.test/token',
    }, {
      fetchImplementation: async () => Response.json({
        access_token: 'issued-access-token',
        expires_in: 3600,
        scope: 'https://www.googleapis.com/auth/gmail.readonly',
      }),
      storage,
    })

    const request = await oauth.beginAuthorization()

    // Callback URL without `state` must fail instead of bypassing CSRF verification.
    await expect(
      oauth.completeAuthorization('http://127.0.0.1:4567/oauth/callback?code=attacker-code'),
    ).rejects.toThrow(GoogleWorkspaceAuthorizationError)

    // Callback URL with wrong redirect path/origin or embedded credentials must fail.
    await expect(
      oauth.completeAuthorization(`http://127.0.0.1:4567/other?code=abc&state=${encodeURIComponent(request.state)}`),
    ).rejects.toThrow('redirect URI')
    await expect(
      oauth.completeAuthorization(`http://user:pass@127.0.0.1:4567/oauth/callback?code=abc&state=${encodeURIComponent(request.state)}`),
    ).rejects.toThrow('embedded credentials')

    // Valid callback succeeds; bare manual code flow also remains supported.
    const token = await oauth.completeAuthorization(
      `http://127.0.0.1:4567/oauth/callback?code=valid-code&state=${encodeURIComponent(request.state)}`,
    )
    expect(token.accessToken).toBe('issued-access-token')

    await oauth.beginAuthorization()
    const manual = await oauth.completeAuthorization('manual-oob-authorization-code')
    expect(manual.accessToken).toBe('issued-access-token')

    // Google REST URL builder blocks backslash origin escape, scheme smuggling, fragments, and %2e%2e traversal.
    const base = 'https://gmail.googleapis.com/gmail/v1/'
    for (const badPath of [
      '\\\\attacker.example\\steal',
      'users\\..\\..\\admin',
      'data:text/plain,hello',
      'users/me/messages#fragment',
      'users/%2e%2e/%2e%2e/admin',
      './users/me/messages',
    ]) {
      expect(() => googleWorkspaceRequestUrl(base, badPath)).toThrow('relative path')
    }
    const validSheetAppend = googleWorkspaceRequestUrl(
      'https://sheets.googleapis.com/v4/',
      'spreadsheets/sheet-1/values/Sheet1%21A1%3AB2:append',
      { valueInputOption: 'USER_ENTERED' },
    )
    expect(validSheetAppend.href).toBe(
      'https://sheets.googleapis.com/v4/spreadsheets/sheet-1/values/Sheet1%21A1%3AB2:append?valueInputOption=USER_ENTERED',
    )
  })

  test('PowerPoint OOXML blocks symlinks, DTDs, unmatched close tags, quoted > splits, and escaping .rels targets', async () => {
    const tempRoot = await mkdtemp(join(tmpdir(), 'xerxes-swarm-pptx-'))
    const outsideDir = join(tempRoot, 'outside')
    const packageRoot = join(tempRoot, 'pkg')
    try {
      await mkdir(outsideDir, { recursive: true })
      await mkdir(join(packageRoot, 'ppt', 'slides'), { recursive: true })
      const secretFile = join(outsideDir, 'secret.xml')
      await writeFile(secretFile, '<secret/>', 'utf8')
      await symlink(secretFile, join(packageRoot, 'ppt', 'slides', 'slide99.xml'))
      await symlink(outsideDir, join(packageRoot, 'ppt', 'linked'))

      const pkg = new BunOfficePackageDirectory(packageRoot)
      await expect(pkg.hasPart('ppt/slides/slide99.xml')).rejects.toThrow(OfficePackageError)
      await expect(pkg.readText('ppt/slides/slide99.xml')).rejects.toThrow(OfficePackageError)
      await expect(pkg.writeText('ppt/linked/overwrite.xml', '<x/>')).rejects.toThrow(OfficePackageError)
      await expect(pkg.deletePart('ppt/slides/slide99.xml')).rejects.toThrow(OfficePackageError)
      expect(await readFile(secretFile, 'utf8')).toBe('<secret/>')
      expect(() => normalizeOfficePartName('C:outside.xml')).toThrow(OfficePackageError)

      // Quoted `>` and `/>` inside XML attributes survive both condenseOoxmlXml and parseOoxmlTree.
      const trickyXml = `<?xml version="1.0"?>
        <p:sld xmlns:p="p" xmlns:a="a">
          <a:p attr="1 > 0" self="x/>">
            <a:t> keep > text </a:t>
          </a:p>
        </p:sld>`
      const condensed = condenseOoxmlXml(trickyXml)
      expect(condensed).toContain('<a:p attr="1 > 0" self="x/>"><a:t> keep > text </a:t></a:p>')

      const tree = parseOoxmlTree(trickyXml)
      const rootElement = tree.children.find(node => node.kind === 'element')
      if (!rootElement || rootElement.kind !== 'element') throw new Error('expected root element')
      setOoxmlAttribute(rootElement, 'data-check', 'a < b & "quoted"')
      const serialized = serializeOoxmlTree(tree)
      expect(serialized).toContain('data-check="a &lt; b & &quot;quoted&quot;"')

      // Forbidden DTDs and trailing unmatched closing tags are rejected.
      expect(() => condenseOoxmlXml('<!DOCTYPE foo [<!ENTITY xxe "boom">]><p:sld/>')).toThrow('DTD')
      expect(() => parseOoxmlTree('<!DOCTYPE foo><w:document/>')).toThrow('DTD')
      expect(() => condenseOoxmlXml('<p:sld></p:sld></p:extra>')).toThrow(OfficePackageError)
      expect(decodeXmlEntities('&#0; &#xD800; &#65;')).toBe('&#0; &#xD800; A')

      // Structural .rels validation catches path-escaping and unflagged external targets.
      const escapingRels = new MemoryOfficePackage({
        '[Content_Types].xml': '<Types/>',
        '_rels/.rels': '<Relationships><Relationship Id="rId1" Type="officeDocument" Target="ppt/presentation.xml"/></Relationships>',
        'ppt/_rels/presentation.xml.rels': '<Relationships><Relationship Id="rId9" Type="slide" Target="../../../outside.xml"/></Relationships>',
        'ppt/presentation.xml': '<p:presentation xmlns:p="p"/>',
      })
      await expect(packOfficePackage(escapingRels, { kind: 'pptx' })).rejects.toThrow(OfficePackageValidationError)

      const unflaggedExternal = new MemoryOfficePackage({
        '[Content_Types].xml': '<Types/>',
        '_rels/.rels': '<Relationships><Relationship Id="rId1" Type="officeDocument" Target="ppt/presentation.xml"/></Relationships>',
        'ppt/_rels/presentation.xml.rels': '<Relationships><Relationship Id="rId9" Type="hyperlink" Target="https://attacker.example/leak"/></Relationships>',
        'ppt/presentation.xml': '<p:presentation xmlns:p="p"/>',
      })
      await expect(packOfficePackage(unflaggedExternal, { kind: 'pptx' })).rejects.toThrow('TargetMode="External"')
    } finally {
      await rm(tempRoot, { force: true, recursive: true })
    }
  })

  test('arXiv, YouTube, xlsxRead, and OCR marker handle malformed entities and unsafe filenames safely', async () => {
    const atom = parseArxivAtom(`<?xml version="1.0"?>
      <feed>
        <entry>
          <id>http://arxiv.org/abs/2502.00001v2</id>
          <title>Entity &amp;lt;tag&amp;gt; &#99999999; &#xD800; &#x3B1;</title>
          <summary>Literal &amp;#60; stays literal.</summary>
          <published>2026-02-01T00:00:00Z</published>
          <updated>2026-02-02T00:00:00Z</updated>
          <author><name>Grace</name></author>
        </entry>
      </feed>`)
    expect(atom.papers[0]?.title).toBe('Entity &lt;tag&gt; &#99999999; &#xD800; α')
    expect(atom.papers[0]?.abstract).toBe('Literal &#60; stays literal.')

    const timed = parseYoutubeTimedText('<transcript><text start="1.5" dur="2.0">&amp;lt;ok&amp;gt; &#x110000; &#39;hi&#39;</text></transcript>')
    expect(timed).toEqual([{ duration: 2, start: 1.5, text: "&lt;ok&gt; &#x110000; 'hi'" }])

    const json3 = parseYoutubeJson3Transcript(JSON.stringify({
      events: [{ dDurationMs: 500, segs: [{ utf8: '&amp;#60;safe &#9999999;' }], tStartMs: 1000 }],
    }))
    expect(json3).toEqual([{ duration: 0.5, start: 1, text: '&#60;safe &#9999999;' }])

    const workbookBytes = createOfficeZip([
      { data: new TextEncoder().encode('<workbook><sheets><sheet name="Main &#0; &#xD800;" sheetId="1" r:id="rId1"/></sheets></workbook>'), name: 'xl/workbook.xml' },
      { data: new TextEncoder().encode('<Relationships><Relationship Id="rId1" Target="worksheets\\sheet1.xml"/></Relationships>'), name: 'xl/_rels/workbook.xml.rels' },
      { data: new TextEncoder().encode('<worksheet><sheetData><row r="1"><c r="A1" t="str"><v>42</v></c></row></sheetData></worksheet>'), name: 'xl/worksheets/sheet1.xml' },
    ])
    const workbook = readWorkbookCells(workbookBytes)
    expect(workbook.sheets[0]?.name).toBe('Main &#0; &#xD800;')
    expect(workbook.sheets[0]?.cells.get('A1')).toBe('42')

    for (const unsafeName of ['   ', 'bad\0.png', 'C:evil.png']) {
      await expect(extractMarkerDocument(
        'doc.pdf',
        { convert: async () => ({ images: { [unsafeName]: new Uint8Array([1]) }, markdown: 'ok' }) },
        { outputDirectory: 'out' },
        { ensureDirectory: async () => {}, join: (dir, name) => `${dir}/${name}`, writeFile: async () => {} },
      )).rejects.toThrow(MarkerExtractionError)
    }
  })

  test('Slack broker releases denied flow slots on redeem and rate-limits unauthenticated floods without starving Hub starts', async () => {
    const origin = 'https://broker.example.test'
    const enrollmentToken = randomBytes(32).toString('base64url')
    let nowMs = 50_000
    const broker = createSlackBroker({
      clientId: '123.456',
      clientSecret: 'secret-abc',
      enrollmentToken,
      fetch: async () => {
        throw new Error('unexpected Slack API call')
      },
      now: () => nowMs,
      publicOrigin: origin,
    })

    const send = (path: string, init?: RequestInit): Promise<Response> => {
      const headers = new Headers(init?.headers)
      headers.set('host', 'broker.example.test')
      return broker.handle(new Request(new URL(path, origin), { ...init, headers }))
    }

    // 200 unauthenticated /v1/slack/start attempts return 401 without consuming the 128/min authenticated start budget.
    const verifier = randomBytes(32).toString('base64url')
    const challenge = createHash('sha256').update(verifier).digest('base64url')
    for (let index = 0; index < 200; index += 1) {
      const unauth = await send('/v1/slack/start', {
        body: JSON.stringify({ challenge }),
        headers: { authorization: `Bearer ${randomBytes(32).toString('base64url')}`, 'content-type': 'application/json' },
        method: 'POST',
      })
      expect(unauth.status).toBe(401)
    }
    const legitimateStart = await send('/v1/slack/start', {
      body: JSON.stringify({ challenge }),
      headers: { authorization: `Bearer ${enrollmentToken}`, 'content-type': 'application/json' },
      method: 'POST',
    })
    expect(legitimateStart.status).toBe(200)
    const started = await legitimateStart.json() as { browser_url: string; id: string; pairing_code: string }

    // Sub-second remaining TTL still produces Max-Age >= 1 rather than Max-Age=0.
    nowMs += 10 * 60_000 - 200
    const visitResp = await send(started.browser_url)
    expect(visitResp.headers.get('set-cookie')).toContain('Max-Age=1')
    const cookie = visitResp.headers.get('set-cookie')?.split(';')[0] ?? ''

    // Deny via Slack callback and verify redeem frees the slot immediately (subsequent redeem is expired).
    const pairResp = await send(started.browser_url, {
      body: new URLSearchParams({ code: started.pairing_code }),
      headers: { 'content-type': 'application/x-www-form-urlencoded', cookie, origin },
      method: 'POST',
    })
    expect(pairResp.status).toBe(303)
    const slackPage = await (await send(started.browser_url, { headers: { cookie } })).text()
    const authHref = /href="([^"]+)"/.exec(slackPage)?.[1]?.replaceAll('&amp;', '&') ?? ''
    const state = new URL(authHref).searchParams.get('state') ?? ''
    const cbResp = await send(`/v1/slack/callback?state=${encodeURIComponent(state)}&error=access_denied`, { headers: { cookie } })
    expect(cbResp.status).toBe(303)

    const firstRedeem = await send('/v1/slack/redeem', {
      body: JSON.stringify({ id: started.id, verifier }),
      headers: { 'content-type': 'application/json' },
      method: 'POST',
    })
    expect(await firstRedeem.json()).toEqual({ status: 'denied' })

    const secondRedeem = await send('/v1/slack/redeem', {
      body: JSON.stringify({ id: started.id, verifier }),
      headers: { 'content-type': 'application/json' },
      method: 'POST',
    })
    expect(await secondRedeem.json()).toEqual({ status: 'expired' })

    // Unauthenticated floods past 2048 requests in a 60s window hit 429 rate_limited.
    nowMs += 60_000
    let lastStatus = 0
    for (let index = 0; index < 2049; index += 1) {
      const flood = await send('/v1/slack/start', {
        body: '{}',
        headers: { 'content-type': 'application/json' },
        method: 'POST',
      })
      lastStatus = flood.status
    }
    expect(lastStatus).toBe(429)
  })

  test('OpenRouter images parses parameterized/multiline data URLs, cancels retry bodies, and aborts retry sleep immediately', async () => {
    const rawPng = Buffer.from([0x89, 0x50, 0x4e, 0x47]).toString('base64')
    const wrappedBase64 = `${rawPng.slice(0, 4)}\r\n${rawPng.slice(4)}`

    const parsedResult = await generateImagesViaOpenRouter(IMAGE_MODEL, { input: [{ text: 'draw', type: 'text' }] }, {
      apiKey: 'sk-test',
      fetch: async () => Response.json({
        choices: [{
          message: {
            content: 'done',
            images: [
              { image_url: { url: `DATA:IMAGE/PNG;charset=binary;BASE64,${wrappedBase64}\n` } },
            ],
          },
        }],
      }),
    })
    expect(parsedResult.stopReason).toBe('stop')
    expect(parsedResult.output).toEqual([
      { text: 'done', type: 'text' },
      { data: rawPng, mimeType: 'image/png', type: 'image' },
    ])

    // Retryable 429 cancels unconsumed body and aborts immediately when signal fires mid-backoff.
    let retryBodyCancelled = false
    const controller = new AbortController()
    const stream429 = new ReadableStream<Uint8Array>({
      cancel() {
        retryBodyCancelled = true
      },
      start(streamController) {
        streamController.enqueue(new TextEncoder().encode('rate limited'))
      },
    })

    const started = Date.now()
    const abortedRetry = await generateImagesViaOpenRouter(IMAGE_MODEL, { input: [] }, {
      apiKey: 'sk-test',
      fetch: async () => new Response(stream429, { headers: { 'retry-after': '30' }, status: 429 }),
      maxRetries: 2,
      signal: controller.signal,
      sleep: async () => {
        setTimeout(() => controller.abort(), 10)
        await new Promise<void>(() => {})
      },
    })
    expect(Date.now() - started).toBeLessThan(1_000)
    expect(retryBodyCancelled).toBeTrue()
    expect(abortedRetry.stopReason).toBe('aborted')
  })

  test('DaemonEvaluationSessionPort preserves streaming whitespace and cancels unfinished turns on reset or break', async () => {
    const cancelledKeys: string[] = []
    const evictedKeys: string[] = []
    let opened = 0
    let pendingTurnResolve: (() => void) | undefined

    const fakeRuntime = {
      cancelTurn(sessionKey: string) {
        cancelledKeys.push(sessionKey)
        pendingTurnResolve?.()
      },
      evictSession(sessionKey: string) {
        evictedKeys.push(sessionKey)
      },
      async openSession(sessionKey: string) {
        opened += 1
        return { model: 'eval-daemon-model', sessionKey }
      },
      reload() {},
      async submitTurn(_sessionKey: string, prompt: string, emit: (event: DaemonEvent) => void) {
        if (prompt === 'stream-whitespace') {
          for (const delta of ['Hello', ' ', 'world', '\n', '  const x = 1;']) {
            emit({ payload: { text: delta }, seq: 1, session_id: 's', timestamp: 0, type: 'text_part' } as unknown as DaemonEvent)
          }
          emit({ payload: {}, seq: 2, session_id: 's', timestamp: 0, type: 'turn_end' } as unknown as DaemonEvent)
          return
        }
        emit({ payload: { text: 'partial' }, seq: 1, session_id: 's', timestamp: 0, type: 'text_part' } as unknown as DaemonEvent)
        await new Promise<void>(resolve => {
          pendingTurnResolve = resolve
        })
      },
    } as unknown as DaemonRuntime

    const port = new DaemonEvaluationSessionPort(fakeRuntime)
    await port.start({ homeDirectory: '/tmp/h', permissionMode: 'accept-all', workspaceDirectory: '/tmp/w' })

    const chunks: string[] = []
    for await (const event of port.submit({ prompt: 'stream-whitespace', signal: new AbortController().signal, timeoutMs: 5_000 })) {
      if (event.type === 'text') chunks.push(event.text)
    }
    expect(chunks.join('')).toBe('Hello world\n  const x = 1;')

    // Breaking out of an active turn iterator early cancels the daemon turn immediately.
    for await (const event of port.submit({ prompt: 'hang', signal: new AbortController().signal, timeoutMs: 5_000 })) {
      if (event.type === 'text') break
    }
    expect(cancelledKeys.length).toBeGreaterThanOrEqual(1)

    await port.reset()
    expect(opened).toBe(2)
    await port.close()
    expect(evictedKeys).toHaveLength(2)
  })

  test('BunWorkspaceModuleEvaluator blocks symlinks, tolerates console.log + undefined returns, and survives hung grandchild pipes', async () => {
    const root = await mkdtemp(join(tmpdir(), 'xerxes-swarm-eval-'))
    const workspace = join(root, 'workspace')
    const outside = join(root, 'outside')
    try {
      await mkdir(workspace, { recursive: true })
      await mkdir(outside, { recursive: true })
      await writeFile(join(outside, 'secret.ts'), 'export const leak = () => "secret"\n', 'utf8')
      await symlink(join(outside, 'secret.ts'), join(workspace, 'linked.ts'))

      await writeFile(
        join(workspace, 'noisy.ts'),
        [
          'console.log("top-level noisy stdout banner")',
          'export function compute(a: number, b: number): { sum: number } {',
          '  console.log("per-call debug log", a, b)',
          '  return { sum: a + b }',
          '}',
          'export function noop(): void {',
          '  console.log("void return")',
          '}',
        ].join('\n'),
        'utf8',
      )

      await writeFile(
        join(workspace, 'hangPipe.ts'),
        [
          'export async function wedge(): Promise<number> {',
          '  Bun.spawn({ cmd: [process.execPath, "-e", "setTimeout(() => {}, 30000)"], stdout: "inherit", stderr: "inherit" })',
          '  await new Promise(() => {})',
          '  return 1',
          '}',
        ].join('\n'),
        'utf8',
      )

      const fastEvaluator = new BunWorkspaceModuleEvaluator({ timeoutMs: 5_000 })
      await expect(
        fastEvaluator.invoke({ args: [], exportName: 'leak', modulePath: 'linked.ts', workspaceDirectory: workspace }),
      ).rejects.toThrow('inside the workspace')

      expect(
        await fastEvaluator.invoke({ args: [19, 23], exportName: 'compute', modulePath: 'noisy.ts', workspaceDirectory: workspace }),
      ).toEqual({ sum: 42 })

      expect(
        await fastEvaluator.invoke({ args: [], exportName: 'noop', modulePath: 'noisy.ts', workspaceDirectory: workspace }),
      ).toBeUndefined()

      const shortEvaluator = new BunWorkspaceModuleEvaluator({ timeoutMs: 150 })
      const started = Date.now()
      await expect(
        shortEvaluator.invoke({ args: [], exportName: 'wedge', modulePath: 'hangPipe.ts', workspaceDirectory: workspace }),
      ).rejects.toThrow('timed out after 150ms')
      expect(Date.now() - started).toBeLessThan(2_500)

      // Stale run sweep removes symlinked `run-*` entries without deleting the symlink target.
      const runsRoot = join(root, 'runs')
      await mkdir(runsRoot, { recursive: true })
      await symlink(outside, join(runsRoot, 'run-symlinked'))
      await sweepStaleEvaluationRuns(runsRoot)
      expect(await Bun.file(join(outside, 'secret.ts')).exists()).toBeTrue()
    } finally {
      await rm(root, { force: true, recursive: true })
    }
  })

  test('NativeEvaluationAgent supports per-turn AbortSignal, bounds wedged iterator cleanup, and classifies failure categories', async () => {
    let returnCalled = 0
    const wedgedTransport: EvaluationSessionPort = {
      approve: async () => new Promise<void>(() => {}),
      close: async () => {},
      reset: async () => {},
      start: async () => ({ model: 'wedged-model' }),
      submit: () => ({
        [Symbol.asyncIterator](): AsyncIterator<EvaluationEvent> {
          let emitted = false
          return {
            async next() {
              if (!emitted) {
                emitted = true
                return { done: false, value: { id: 'req-1', type: 'approval_request' } }
              }
              return new Promise<IteratorResult<EvaluationEvent>>(() => {})
            },
            async return() {
              returnCalled += 1
              return new Promise<IteratorResult<EvaluationEvent>>(() => {})
            },
          }
        },
      }),
    }

    const agent = new NativeEvaluationAgent({
      retryDelayMs: () => 5_000,
      sleep: async () => new Promise<void>(() => {}),
      start: { homeDirectory: '/tmp/h', permissionMode: 'accept-all', workspaceDirectory: '/tmp/w' },
      transport: wedgedTransport,
    })
    await agent.start()

    // Hung approve() + hung iterator.return() still resolves promptly on timeoutMs.
    const timedStart = Date.now()
    const timedOut = await agent.turn('check timeout', { retries: 0, timeoutMs: 60 })
    expect(timedOut.error).toBe('timeout')
    expect(returnCalled).toBe(1)
    expect(Date.now() - timedStart).toBeLessThan(1_500)

    // Caller AbortSignal cancels in-flight turn or retry sleep without exhausting retries.
    const abortController = new AbortController()
    setTimeout(() => abortController.abort(), 15)
    const cancelled = await agent.turn('check cancel', { retries: 2, signal: abortController.signal, timeoutMs: 5_000 })
    expect(cancelled.error).toBe('cancelled')

    // Deterministic failure taxonomy separates harness/provider errors from model assertion misses.
    expect(classifyEvaluationFailure({ detail: 'passed', ok: true })).toBeUndefined()
    expect(classifyEvaluationFailure({ detail: 'ERROR: timeout', ok: false, turnError: 'timeout' })).toBe('timeout')
    expect(classifyEvaluationFailure({ detail: 'grader timed out (> 90s)', ok: false })).toBe('timeout')
    expect(classifyEvaluationFailure({ detail: 'ERROR: cancelled', ok: false, turnError: 'cancelled' })).toBe('cancelled')
    expect(classifyEvaluationFailure({ detail: 'grader raised: TypeError: boom', ok: false })).toBe('grader_exception')
    expect(classifyEvaluationFailure({ detail: 'expected 42', ok: false, reply: '[Error: 503 overloaded]' })).toBe('transport_error')
    expect(classifyEvaluationFailure({ detail: 'expected 42, got 41', ok: false, reply: '41' })).toBe('assertion_failed')
  })
})
