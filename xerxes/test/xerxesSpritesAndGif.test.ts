// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { join } from 'node:path'
import { createElement, type ReactElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import {
  downloadGifBytes,
  encodeAnimatedGif,
  type GifFrame,
} from '../src/desktop/renderer/gifEncoder.js'

type SimurghSpriteRole = 'coordinator' | 'worker' | 'verifier' | 'narrator' | 'monitor'
type SimurghSpriteState = 'working' | 'waiting' | 'stopped' | 'killed' | 'error'
type AgentBrandProvider = 'xerxes' | 'claude-code' | 'antigravity' | 'codex' | 'acp'

interface SimurghSpriteProps {
  readonly role: SimurghSpriteRole
  readonly state?: SimurghSpriteState
  readonly size?: number
  readonly animate?: boolean
  readonly className?: string
  readonly title?: string
}

interface XerxesSpritesModule {
  readonly SIMURGH_ROLES: readonly SimurghSpriteRole[]
  readonly SimurghSprite: (props: SimurghSpriteProps) => ReactElement
  readonly XerxesBrandEmblem: (props: {
    readonly size?: number
    readonly className?: string
    readonly animate?: boolean
  }) => ReactElement
  readonly XerxesThinkingMark: (props: {
    readonly size?: number
    readonly active?: boolean
    readonly className?: string
  }) => ReactElement
  readonly AgentBrandGlyph: (props: {
    readonly provider: AgentBrandProvider
    readonly size?: number
    readonly animate?: boolean
    readonly className?: string
  }) => ReactElement
  readonly drawSimurghSpriteOnCanvas: (
    ctx: CanvasRenderingContext2D,
    role: SimurghSpriteRole,
    state: SimurghSpriteState,
    x: number,
    y: number,
    size: number,
    frameIndex?: number,
  ) => void
  readonly resolveSimurghRole: (name: string, role?: string) => SimurghSpriteRole
}

// Runtime import keeps `tsconfig.json` (which excludes JSX compilation for
// `.ts` runtime tests) cleanly green while executing the real TSX module in Bun.
const SPRITES_MODULE_URL = new URL('../src/desktop/renderer/XerxesSprites.tsx', import.meta.url).href
const {
  AgentBrandGlyph,
  SIMURGH_ROLES,
  SimurghSprite,
  XerxesBrandEmblem,
  XerxesThinkingMark,
  drawSimurghSpriteOnCanvas,
  resolveSimurghRole,
} = (await import(SPRITES_MODULE_URL)) as XerxesSpritesModule

const BRAND_DIR = join(import.meta.dir, '..', '..', 'assets', 'brand')

// ---------------------------------------------------------------------------
// Independent GIF89a + LZW decoder used to verify binary spec compliance
// ---------------------------------------------------------------------------

interface DecodedGifFrame {
  readonly width: number
  readonly height: number
  readonly delayCs: number
  readonly indices: Uint8Array
}

interface DecodedGif {
  readonly header: string
  readonly width: number
  readonly height: number
  readonly hasGlobalColorTable: boolean
  readonly globalColorTable: Uint8Array
  readonly netscapeLoopCount: number | null
  readonly frames: readonly DecodedGifFrame[]
  readonly trailer: number
}

function decodeLzwStream(data: Uint8Array, minCodeSize: number, expectedPixels: number): Uint8Array {
  const clearCode = 1 << minCodeSize
  const eoiCode = clearCode + 1

  let bitPos = 0
  const readBits = (count: number): number => {
    let val = 0
    for (let i = 0; i < count; i += 1) {
      const byteIdx = bitPos >> 3
      const bitIdx = bitPos & 7
      const byteVal = data[byteIdx] ?? 0
      val |= ((byteVal >> bitIdx) & 1) << i
      bitPos += 1
    }
    return val
  }

  let codeSize = minCodeSize + 1
  let nextCode = eoiCode + 1
  const prefix = new Int32Array(4096)
  const suffix = new Uint8Array(4096)
  const stack = new Uint8Array(4096)
  for (let i = 0; i < clearCode; i += 1) {
    prefix[i] = -1
    suffix[i] = i
  }

  const out = new Uint8Array(expectedPixels)
  let outPos = 0
  let prevCode = -1
  let firstChar = 0

  while (bitPos + codeSize <= data.length * 8) {
    const code = readBits(codeSize)
    if (code === clearCode) {
      codeSize = minCodeSize + 1
      nextCode = eoiCode + 1
      prevCode = -1
      continue
    }
    if (code === eoiCode) {
      break
    }

    let cur = code
    let stackTop = 0
    if (cur === nextCode) {
      if (prevCode < 0) {
        throw new Error('Invalid LZW KwKwK sequence before first literal code.')
      }
      stack[stackTop++] = firstChar
      cur = prevCode
    } else if (cur > nextCode) {
      throw new Error(`Invalid LZW code ${String(cur)} > nextCode ${String(nextCode)}.`)
    }

    while (cur >= clearCode) {
      stack[stackTop++] = suffix[cur] ?? 0
      cur = prefix[cur] ?? -1
      if (cur < 0 || stackTop >= 4096) {
        throw new Error('Corrupt LZW prefix chain.')
      }
    }

    firstChar = suffix[cur] ?? 0
    stack[stackTop++] = firstChar

    while (stackTop > 0) {
      stackTop -= 1
      if (outPos < out.length) {
        out[outPos++] = stack[stackTop] ?? 0
      } else {
        throw new Error('Decoded LZW pixels exceeded frame pixel count.')
      }
    }

    if (prevCode >= 0 && nextCode < 4096) {
      prefix[nextCode] = prevCode
      suffix[nextCode] = firstChar
      nextCode += 1
      if (nextCode === 1 << codeSize && codeSize < 12) {
        codeSize += 1
      }
    }

    prevCode = code
  }

  if (outPos !== expectedPixels) {
    throw new Error(`Decoded ${String(outPos)} pixels, expected ${String(expectedPixels)}.`)
  }
  return out
}

function decodeGif89a(bytes: Uint8Array): DecodedGif {
  const header = Buffer.from(bytes.subarray(0, 6)).toString('ascii')
  const width = (bytes[6] ?? 0) | ((bytes[7] ?? 0) << 8)
  const height = (bytes[8] ?? 0) | ((bytes[9] ?? 0) << 8)
  const packed = bytes[10] ?? 0
  const hasGlobalColorTable = (packed & 0x80) !== 0
  const gctSize = hasGlobalColorTable ? 3 * (1 << ((packed & 0x07) + 1)) : 0
  let pos = 13
  const globalColorTable = bytes.slice(pos, pos + gctSize)
  pos += gctSize

  let netscapeLoopCount: number | null = null
  let pendingDelayCs = 0
  const frames: DecodedGifFrame[] = []
  let trailer = 0

  while (pos < bytes.length) {
    const intro = bytes[pos++] ?? 0
    if (intro === 0x3b) {
      trailer = intro
      break
    }

    if (intro === 0x21) {
      const label = bytes[pos++] ?? 0
      if (label === 0xf9) {
        const blockSize = bytes[pos++] ?? 0
        expect(blockSize).toBe(4)
        pos += 1 // packed GCE flags
        pendingDelayCs = (bytes[pos] ?? 0) | ((bytes[pos + 1] ?? 0) << 8)
        pos += 2
        pos += 1 // transparent index
        const term = bytes[pos++] ?? -1
        expect(term).toBe(0x00)
      } else if (label === 0xff) {
        const appBlockSize = bytes[pos++] ?? 0
        const appId = Buffer.from(bytes.subarray(pos, pos + appBlockSize)).toString('ascii')
        pos += appBlockSize
        const subLen = bytes[pos++] ?? 0
        if (appId === 'NETSCAPE2.0' && subLen === 3 && bytes[pos] === 0x01) {
          netscapeLoopCount = (bytes[pos + 1] ?? 0) | ((bytes[pos + 2] ?? 0) << 8)
        }
        pos += subLen
        while (pos < bytes.length) {
          const nextLen = bytes[pos++] ?? 0
          if (nextLen === 0) break
          pos += nextLen
        }
      } else {
        while (pos < bytes.length) {
          const subLen = bytes[pos++] ?? 0
          if (subLen === 0) break
          pos += subLen
        }
      }
      continue
    }

    if (intro === 0x2c) {
      pos += 4 // left, top
      const fw = (bytes[pos] ?? 0) | ((bytes[pos + 1] ?? 0) << 8)
      const fh = (bytes[pos + 2] ?? 0) | ((bytes[pos + 3] ?? 0) << 8)
      pos += 4
      const imgPacked = bytes[pos++] ?? 0
      if ((imgPacked & 0x80) !== 0) {
        pos += 3 * (1 << ((imgPacked & 0x07) + 1))
      }

      const minCodeSize = bytes[pos++] ?? 8
      const chunks: Uint8Array[] = []
      let totalLen = 0
      while (pos < bytes.length) {
        const subLen = bytes[pos++] ?? 0
        if (subLen === 0) break
        const chunk = bytes.subarray(pos, pos + subLen)
        chunks.push(chunk)
        totalLen += subLen
        pos += subLen
      }
      const compressed = new Uint8Array(totalLen)
      let offset = 0
      for (const chunk of chunks) {
        compressed.set(chunk, offset)
        offset += chunk.length
      }

      const indices = decodeLzwStream(compressed, minCodeSize, fw * fh)
      frames.push({ width: fw, height: fh, delayCs: pendingDelayCs, indices })
      pendingDelayCs = 0
      continue
    }

    throw new Error(`Unexpected GIF block introducer 0x${intro.toString(16)} at offset ${String(pos - 1)}.`)
  }

  return {
    header,
    width,
    height,
    hasGlobalColorTable,
    globalColorTable,
    netscapeLoopCount,
    frames,
    trailer,
  }
}

function makeSolidFrame(
  width: number,
  height: number,
  r: number,
  g: number,
  b: number,
  delayMs?: number,
): GifFrame {
  const rgba = new Uint8ClampedArray(width * height * 4)
  for (let i = 0; i < rgba.length; i += 4) {
    rgba[i] = r
    rgba[i + 1] = g
    rgba[i + 2] = b
    rgba[i + 3] = 255
  }
  return delayMs === undefined ? { width, height, rgba } : { width, height, rgba, delayMs }
}

// ---------------------------------------------------------------------------
// 1. Pure TypeScript GIF89a Encoder Tests
// ---------------------------------------------------------------------------

describe('gifEncoder', () => {
  const origDocument = globalThis.document
  const origUrl = globalThis.URL

  afterEach(() => {
    Object.defineProperty(globalThis, 'document', {
      value: origDocument,
      configurable: true,
      writable: true,
    })
    Object.defineProperty(globalThis, 'URL', {
      value: origUrl,
      configurable: true,
      writable: true,
    })
  })

  test('encodes a 1x1 GIF89a with exact Xerxes brand palette fidelity', () => {
    // Amethyst brand color #a855f7 -> (168, 85, 247)
    const frame = makeSolidFrame(1, 1, 168, 85, 247, 120)
    const bytes = encodeAnimatedGif([frame], { loop: 0 })

    expect(Buffer.from(bytes.subarray(0, 6)).toString('ascii')).toBe('GIF89a')
    expect(bytes[bytes.length - 1]).toBe(0x3b)

    const decoded = decodeGif89a(bytes)
    expect(decoded.header).toBe('GIF89a')
    expect(decoded.width).toBe(1)
    expect(decoded.height).toBe(1)
    expect(decoded.hasGlobalColorTable).toBe(true)
    expect(decoded.netscapeLoopCount).toBe(0)
    expect(decoded.trailer).toBe(0x3b)
    expect(decoded.frames).toHaveLength(1)

    const f0 = decoded.frames[0]
    expect(f0).toBeDefined()
    expect(f0?.delayCs).toBe(12)
    const colorIdx = f0?.indices[0] ?? 0
    expect(decoded.globalColorTable[colorIdx * 3]).toBe(168)
    expect(decoded.globalColorTable[colorIdx * 3 + 1]).toBe(85)
    expect(decoded.globalColorTable[colorIdx * 3 + 2]).toBe(247)
  })

  test('encodes multi-frame animations with NETSCAPE2.0 loop and accurate delays', () => {
    const frames: GifFrame[] = [
      makeSolidFrame(8, 8, 13, 14, 17, 80), // #0d0e11 obsidian
      makeSolidFrame(8, 8, 251, 191, 36, 150), // #fbbf24 solar gold
      makeSolidFrame(8, 8, 52, 211, 153, 200), // #34d399 emerald
      makeSolidFrame(8, 8, 56, 189, 248, 90), // #38bdf8 cyan
    ]

    const bytes = encodeAnimatedGif(frames, { loop: 3 })
    const decoded = decodeGif89a(bytes)

    expect(decoded.header).toBe('GIF89a')
    expect(decoded.netscapeLoopCount).toBe(3)
    expect(decoded.frames).toHaveLength(4)
    expect(decoded.frames.map((f) => f.delayCs)).toEqual([8, 15, 20, 9])

    const expectedRgb: readonly (readonly [number, number, number])[] = [
      [13, 14, 17],
      [251, 191, 36],
      [52, 211, 153],
      [56, 189, 248],
    ]
    for (let i = 0; i < expectedRgb.length; i += 1) {
      const f = decoded.frames[i]
      const rgb = expectedRgb[i]
      expect(f).toBeDefined()
      expect(rgb).toBeDefined()
      const idx = f?.indices[0] ?? 0
      expect([
        decoded.globalColorTable[idx * 3],
        decoded.globalColorTable[idx * 3 + 1],
        decoded.globalColorTable[idx * 3 + 2],
      ]).toEqual([rgb?.[0], rgb?.[1], rgb?.[2]])
    }
  })

  test('handles high-entropy multi-color frames requiring 12-bit LZW table resets', () => {
    const width = 96
    const height = 96
    const rgba = new Uint8ClampedArray(width * height * 4)
    let seed = 0x12345678
    for (let px = 0; px < width * height; px += 1) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
      rgba[px * 4] = (seed >>> 24) & 0xff
      rgba[px * 4 + 1] = (seed >>> 16) & 0xff
      rgba[px * 4 + 2] = (seed >>> 8) & 0xff
      rgba[px * 4 + 3] = 255
    }

    const bytes = encodeAnimatedGif([{ width, height, rgba }], { loop: 0 })
    const decoded = decodeGif89a(bytes)

    expect(decoded.frames).toHaveLength(1)
    expect(decoded.frames[0]?.indices.length).toBe(width * height)
    expect(decoded.trailer).toBe(0x3b)
  })

  test('handles exact 9-to-10-bit EOI boundary cleanly when frame ends at code 512', () => {
    // 255 distinct consecutive pixels produce 254 table insertions (258..511),
    // leaving nextCode === 512 at the final pixel before EOI.
    const width = 255
    const height = 1
    const rgba = new Uint8ClampedArray(width * height * 4)
    for (let x = 0; x < width; x += 1) {
      rgba[x * 4] = x
      rgba[x * 4 + 1] = (x * 7) & 0xff
      rgba[x * 4 + 2] = (x * 13) & 0xff
      rgba[x * 4 + 3] = 255
    }

    const bytes = encodeAnimatedGif([{ width, height, rgba }])
    const decoded = decodeGif89a(bytes)
    expect(decoded.frames).toHaveLength(1)
    expect(decoded.frames[0]?.indices.length).toBe(255)
  })

  test('omits NETSCAPE2.0 extension when loop is negative', () => {
    const bytes = encodeAnimatedGif([makeSolidFrame(2, 2, 168, 85, 247)], { loop: -1 })
    const decoded = decodeGif89a(bytes)
    expect(decoded.netscapeLoopCount).toBeNull()
    expect(decoded.trailer).toBe(0x3b)
  })

  test('validates empty frames, invalid dimensions, mismatches, and short RGBA buffers', () => {
    expect(() => encodeAnimatedGif([])).toThrow(/at least one frame/i)
    expect(() =>
      encodeAnimatedGif([{ width: 0, height: 4, rgba: new Uint8ClampedArray(0) }]),
    ).toThrow(/invalid gif frame dimensions/i)
    expect(() =>
      encodeAnimatedGif([
        makeSolidFrame(4, 4, 0, 0, 0),
        makeSolidFrame(5, 4, 0, 0, 0),
      ]),
    ).toThrow(/do not match/i)
    expect(() =>
      encodeAnimatedGif([{ width: 4, height: 4, rgba: new Uint8ClampedArray(10) }]),
    ).toThrow(/smaller than required/i)
  })

  test('downloadGifBytes returns false in headless Bun and true when browser DOM APIs exist', () => {
    const gif = encodeAnimatedGif([makeSolidFrame(2, 2, 168, 85, 247)])
    expect(downloadGifBytes(gif, 'headless.gif')).toBe(false)

    let clicked = false
    let downloadName = ''
    let createdHref = ''
    const fakeLink = {
      href: '',
      download: '',
      rel: '',
      style: { display: '' },
      parentNode: null as unknown,
      click() {
        clicked = true
        downloadName = this.download
        createdHref = this.href
      },
    }
    const fakeBody = {
      appendChild(node: typeof fakeLink) {
        node.parentNode = fakeBody
        return node
      },
      removeChild(node: typeof fakeLink) {
        node.parentNode = null
        return node
      },
    }

    Object.defineProperty(globalThis, 'document', {
      value: {
        body: fakeBody,
        createElement: () => fakeLink,
      },
      configurable: true,
      writable: true,
    })
    Object.defineProperty(globalThis, 'URL', {
      value: {
        createObjectURL: () => 'blob:xerxes-test-gif',
        revokeObjectURL: () => {},
      },
      configurable: true,
      writable: true,
    })

    expect(downloadGifBytes(gif, 'swarm-replay')).toBe(true)
    expect(clicked).toBe(true)
    expect(downloadName).toBe('swarm-replay.gif')
    expect(createdHref).toBe('blob:xerxes-test-gif')
  })
})

// ---------------------------------------------------------------------------
// 2. Cosmic Simurgh Sprites, Emblem, Thinking Mark & Brand Glyphs Tests
// ---------------------------------------------------------------------------

describe('XerxesSprites', () => {
  test('resolveSimurghRole maps explicit roles and naming patterns accurately', () => {
    expect(resolveSimurghRole('coordinator')).toBe('coordinator')
    expect(resolveSimurghRole('coord-main')).toBe('coordinator')
    expect(resolveSimurghRole('custom', 'coordinator')).toBe('coordinator')

    expect(resolveSimurghRole('graph-verifier', 'worker')).toBe('verifier')
    expect(resolveSimurghRole('security-audit')).toBe('verifier')
    expect(resolveSimurghRole('code-reviewer')).toBe('verifier')
    expect(resolveSimurghRole('flaky-test')).toBe('verifier')
    expect(resolveSimurghRole('check-build')).toBe('verifier')
    expect(resolveSimurghRole('qa-runner')).toBe('verifier')
    expect(resolveSimurghRole('custom-agent', 'verifier')).toBe('verifier')

    expect(resolveSimurghRole('narrator', 'worker')).toBe('narrator')
    expect(resolveSimurghRole('swarm-narrator')).toBe('narrator')

    expect(resolveSimurghRole('monitor', 'worker')).toBe('monitor')
    expect(resolveSimurghRole('health-monitor')).toBe('monitor')

    expect(resolveSimurghRole('brand-sprites', 'worker')).toBe('worker')
    expect(resolveSimurghRole('swarm-graph')).toBe('worker')
  })

  test('SimurghSprite renders crisp 24×24 monochrome 1.5px dev-tool line icons (zero medallions, crowns, or hex literals) across all 5 roles & 5 states', () => {
    const states: readonly SimurghSpriteState[] = ['working', 'waiting', 'stopped', 'killed', 'error']

    for (const role of SIMURGH_ROLES) {
      for (const state of states) {
        const html = renderToStaticMarkup(
          createElement(SimurghSprite, { role, state, size: 16, animate: true }),
        )
        expect(html).toContain('<svg')
        expect(html).toContain('viewBox="0 0 24 24"')
        expect(html).toContain('stroke-width="1.5"')
        expect(html).toContain(`data-simurgh-role="${role}"`)
        expect(html).toContain(`data-simurgh-state="${state}"`)
        // Pure monochrome line icons: zero text, zero pixel-art rects, zero RPG polygons/medallions, zero SMIL <animate>
        expect(html).not.toContain('<text')
        expect(html).not.toContain('shape-rendering="crispEdges"')
        expect(html).not.toContain('<rect')
        expect(html).not.toContain('<polygon')
        expect(html).not.toContain('<animate')
        expect(html).not.toContain('r="14.5"')

        // Zero hardcoded #RRGGBB hex literals — 100% theme-dynamic var(--x-*)
        expect(html.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])
        expect(html).toContain('var(--x-')

        // All 5 role icons stay strictly static (never spin/orbit)
        expect(html).not.toContain('xerxes-simurgh-orbit')

        if (state === 'killed' || state === 'stopped') {
          expect(html).toContain('var(--x-meta')
        }

        if (state === 'error') {
          expect(html).toContain('var(--x-failed')
        }
      }
    }

    // animate={false} also stays strictly static
    const staticWorking = renderToStaticMarkup(
      createElement(SimurghSprite, { role: 'coordinator', state: 'working', animate: false }),
    )
    expect(staticWorking).not.toContain('xerxes-simurgh-orbit')
    expect(staticWorking).not.toContain('<animate')
  })

  test('each role has a distinct clean line-icon silhouette', () => {
    const silhouettes = new Set<string>()
    for (const role of SIMURGH_ROLES) {
      const markup = renderToStaticMarkup(
        createElement(SimurghSprite, { role, state: 'working', size: 24, animate: false }),
      )
      expect(markup).not.toContain('<text')
      // Normalize role-specific class/attr/title/stroke so we compare actual SVG geometry
      const geometry = markup
        .replace(/aria-label="[^"]*"/, '')
        .replace(/class="[^"]*"/, '')
        .replace(/data-simurgh-role="[^"]*"/, '')
        .replace(/stroke="[^"]*"/, '')
        .replace(/<title>[^<]*<\/title>/, '')
      silhouettes.add(geometry)
    }
    expect(silhouettes.size).toBe(5)
  })

  test('drawSimurghSpriteOnCanvas strokes crisp anti-aliased vector paths for all 5 roles', () => {
    const ops: string[] = []
    let saved = 0
    let restored = 0
    interface MockVectorCanvasContext {
      fillStyle: string
      strokeStyle: string
      lineWidth: number
      lineCap: string
      lineJoin: string
      globalAlpha: number
      save(): void
      restore(): void
      beginPath(): void
      closePath(): void
      moveTo(x: number, y: number): void
      lineTo(x: number, y: number): void
      quadraticCurveTo(cpx: number, cpy: number, x: number, y: number): void
      arc(x: number, y: number, r: number, s: number, e: number): void
      setLineDash(segments: number[]): void
      stroke(): void
      fill(): void
    }
    const mockCtx: MockVectorCanvasContext = {
      fillStyle: '',
      strokeStyle: '',
      lineWidth: 1,
      lineCap: 'butt',
      lineJoin: 'miter',
      globalAlpha: 1,
      save() {
        saved += 1
      },
      restore() {
        restored += 1
      },
      beginPath() {
        ops.push('begin')
      },
      closePath() {
        ops.push('close')
      },
      moveTo(x, y) {
        ops.push(`M:${Math.round(x)},${Math.round(y)}`)
      },
      lineTo(x, y) {
        ops.push(`L:${Math.round(x)},${Math.round(y)}`)
      },
      quadraticCurveTo() {
        ops.push('Q')
      },
      arc() {
        ops.push('A')
      },
      setLineDash() {},
      stroke() {
        ops.push(`S:${mockCtx.strokeStyle}`)
      },
      fill() {
        ops.push(`F:${mockCtx.fillStyle}`)
      },
    }

    const signatures = new Set<string>()
    for (const role of SIMURGH_ROLES) {
      ops.length = 0
      drawSimurghSpriteOnCanvas(mockCtx as unknown as CanvasRenderingContext2D, role, 'working', 10, 20, 24, 0)
      expect(mockCtx.lineCap).toBe('round')
      expect(mockCtx.lineJoin).toBe('round')
      expect(ops.filter((op) => op.startsWith('S:')).length).toBeGreaterThanOrEqual(1)
      signatures.add(ops.join(';'))
    }
    expect(signatures.size).toBe(5)
    expect(saved).toBe(5)
    expect(restored).toBe(5)

    // Zero or negative size safely no-ops
    const before = ops.length
    drawSimurghSpriteOnCanvas(mockCtx as unknown as CanvasRenderingContext2D, 'worker', 'killed', 0, 0, 0, 1)
    expect(ops.length).toBe(before)
  })

  test('XerxesBrandEmblem, XerxesThinkingMark, and AgentBrandGlyph render Xerxes phoenix icon and clean provider marks', () => {
    const emblem = renderToStaticMarkup(createElement(XerxesBrandEmblem, { size: 40, animate: true }))
    expect(emblem).toContain('<img')
    expect(emblem).toContain('src="./logo-128.png"')
    expect(emblem).toContain('data-brand-provider="xerxes"')
    expect(emblem).not.toContain('<svg')
    expect(emblem).not.toContain('data-emblem="xerxes-mark"')
    expect(emblem.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])

    const staticEmblem = renderToStaticMarkup(createElement(XerxesBrandEmblem, { size: 32, animate: false }))
    expect(staticEmblem).toContain('src="./logo-128.png"')
    expect(staticEmblem).not.toContain('<svg')
    expect(staticEmblem.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])

    const activeMark = renderToStaticMarkup(createElement(XerxesThinkingMark, { size: 15, active: true }))
    expect(activeMark).toContain('data-thinking-active="true"')
    expect(activeMark).toContain('data-emblem="xerxes-progress"')
    expect(activeMark).toContain('xerxes-mark-spin')
    expect(activeMark).toContain('xerxes-dot-grid')
    expect(activeMark.match(/<circle /g)?.length).toBe(4)
    expect(activeMark.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])

    const settledMark = renderToStaticMarkup(createElement(XerxesThinkingMark, { size: 15, active: false }))
    expect(settledMark).toContain('data-thinking-active="false"')
    expect(settledMark).not.toContain('xerxes-mark-spin')
    expect(settledMark).toContain('<circle')
    expect(settledMark).toContain('r="3.5"')
    expect(settledMark).not.toContain('<path')
    expect(settledMark.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])

    const providers: readonly AgentBrandProvider[] = ['xerxes', 'claude-code', 'antigravity', 'codex', 'acp']
    const glyphMarkups = new Set<string>()
    for (const provider of providers) {
      const markup = renderToStaticMarkup(createElement(AgentBrandGlyph, { provider, size: 16, animate: true }))
      expect(markup).toContain(`data-brand-provider="${provider}"`)
      // Brand icons never spin or blink/breathe
      expect(markup).not.toContain('xerxes-glyph-spin')
      expect(markup).not.toContain('xerxes-emblem-breath')
      expect(markup).not.toContain('<animate')
      if (provider === 'xerxes') {
        expect(markup).toContain('<img')
        expect(markup).toContain('src="./logo-128.png"')
        expect(markup).not.toContain('<svg')
        expect(markup).not.toContain('data-emblem="xerxes-mark"')
      } else if (provider === 'claude-code') {
        expect(markup).toContain('<svg')
        expect(markup).toContain('m4.714 15.956')
        expect(markup).toContain('rgb(217, 119, 87)')
      } else if (provider === 'antigravity') {
        expect(markup).toContain('<svg')
        expect(markup).toContain('M21.751 22.607')
        expect(markup).toContain('rgb(49, 134, 255)')
        expect(markup).toContain('rgb(0, 185, 92)')
        expect(markup).toContain('rgb(251, 188, 4)')
        expect(markup).toContain('rgb(252, 65, 61)')
      } else if (provider === 'codex') {
        expect(markup).toContain('<svg')
        expect(markup).toContain('M22.282 9.821')
      }
      expect(markup.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])
      glyphMarkups.add(markup)
    }
    expect(glyphMarkups.size).toBe(providers.length)
  })
})

// ---------------------------------------------------------------------------
// 3. Standalone Brand SVG & GIF Pack Verification (assets/brand/*)
// ---------------------------------------------------------------------------

describe('assets/brand standalone pack', () => {
  const expectedSvgs: readonly string[] = [
    'xerxes-derafsh-kaviani.svg',
    'xerxes-simurgh-emblem.svg',
    'xerxes-swarm-constellation.svg',
    'xerxes-thinking-halo.svg',
    'xerxes-sprites-sheet.svg',
  ]

  const expectedGifs: ReadonlyArray<readonly [string, number, number, number]> = [
    ['xerxes-derafsh-kaviani.gif', 280, 320, 20],
    ['xerxes-simurgh-coordinator.gif', 160, 160, 12],
    ['xerxes-simurgh-worker.gif', 160, 160, 12],
    ['xerxes-simurgh-emblem.gif', 280, 280, 16],
    ['xerxes-immortals-battlefield.gif', 640, 400, 24],
    ['xerxes-swarm-hive.gif', 640, 400, 24],
    ['xerxes-working.gif', 96, 96, 12],
    ['xerxes-celebrate.gif', 128, 128, 16],
  ]

  test('all 5 standalone brand SVGs exist and contain valid self-contained SVG markup', async () => {
    for (const filename of expectedSvgs) {
      const file = Bun.file(join(BRAND_DIR, filename))
      expect(await file.exists()).toBe(true)
      const content = await file.text()
      expect(content.length).toBeGreaterThan(400)
      expect(content).toContain('<svg')
      expect(content).toContain('</svg>')
      expect(content).toContain('viewBox=')
      expect(content).not.toContain('@import')
      expect(content).not.toContain('fonts.googleapis.com')
    }

    const kavianiSvg = await Bun.file(join(BRAND_DIR, 'xerxes-derafsh-kaviani.svg')).text()
    expect(kavianiSvg).toContain('viewBox="0 0 260 300"')
    expect(kavianiSvg).toContain('XERXES')
    expect(kavianiSvg.match(/<path\b/g)?.length).toBe(4)

    const thinkingSvg = await Bun.file(join(BRAND_DIR, 'xerxes-thinking-halo.svg')).text()
    expect(thinkingSvg).toContain('<animateTransform')

    // Header titles & subtitles share one <text><tspan> flow so font metric
    // variance never collides adjacent text elements.
    const constellationSvg = await Bun.file(join(BRAND_DIR, 'xerxes-swarm-constellation.svg')).text()
    expect(constellationSvg).toContain('<tspan')
    expect(constellationSvg).toContain('XERXES IMMORTALS')
    expect(constellationSvg).not.toContain('<polygon')
    expect(constellationSvg).not.toContain('x="218"')

    const sheetSvg = await Bun.file(join(BRAND_DIR, 'xerxes-sprites-sheet.svg')).text()
    expect(sheetSvg).toContain('<tspan')
    expect(sheetSvg).toContain('stroke-width="1.5"')
    expect(sheetSvg).not.toContain('x="274"')
  })

  test('all 8 animated brand GIFs exist, start with GIF89a, end with 0x3B, and decode cleanly', async () => {
    for (const [filename, width, height, expectedFrames] of expectedGifs) {
      const file = Bun.file(join(BRAND_DIR, filename))
      expect(await file.exists()).toBe(true)
      const bytes = new Uint8Array(await file.arrayBuffer())
      expect(bytes.length).toBeGreaterThan(1024)
      expect(Buffer.from(bytes.subarray(0, 6)).toString('ascii')).toBe('GIF89a')
      expect(bytes[bytes.length - 1]).toBe(0x3b)

      const decoded = decodeGif89a(bytes)
      expect(decoded.width).toBe(width)
      expect(decoded.height).toBe(height)
      expect(decoded.netscapeLoopCount).toBe(0)
      expect(decoded.frames).toHaveLength(expectedFrames)
      expect(decoded.trailer).toBe(0x3b)
    }
  })
})
