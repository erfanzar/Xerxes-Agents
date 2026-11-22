// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Pure TypeScript GIF89a animated encoder with zero external dependencies.
 *
 * Builds a hybrid 256-color global palette (curated Xerxes dark-UI and cosmic
 * brand ramps combined with adaptive frequent colors sampled across frames),
 * compresses each frame using standard variable-bit-width GIF LZW (8-bit
 * minimum code size, clear code 256, EOI 257, 12-bit reset), and emits a
 * browser-compatible looping GIF89a binary with `NETSCAPE2.0` extension.
 */

export interface GifFrame {
  readonly width: number
  readonly height: number
  readonly rgba: Uint8ClampedArray | Uint8Array
  readonly delayMs?: number
}

export interface GifEncodeOptions {
  readonly loop?: number
  readonly defaultDelayMs?: number
}

const MAX_GIF_DIMENSION = 65535
const DEFAULT_DELAY_MS = 100
const MIN_CODE_SIZE = 8
const CLEAR_CODE = 1 << MIN_CODE_SIZE
const EOI_CODE = CLEAR_CODE + 1
const MAX_LZW_CODES = 4096
const LZW_TABLE_SIZE = 8192
const LZW_TABLE_MASK = LZW_TABLE_SIZE - 1
const MAX_EXACT_HISTOGRAM = 8192
const MAX_EXACT_CACHE = 16384
const MAX_PALETTE_SAMPLES = 360_000

/**
 * Curated anchor colors that stay permanently in the global palette so both
 * dark Xerxes UI surfaces and Cosmic Simurgh pixel sprites render crisply
 * without muddy quantization or banding.
 */
const ANCHOR_HEX_COLORS: readonly string[] = [
  // Obsidian & dark UI neutral ramp (index 0 is pure black)
  '#000000', '#06070a', '#090a0e', '#0d0e11', '#111318', '#151820', '#1a1e28', '#222733',
  '#27272a', '#2c3240', '#3a4152', '#3f3f46', '#52525b', '#71717a', '#9ca3af', '#a1a1aa',
  '#bac2d0', '#d4d8e2', '#e5e7eb', '#f4f4f5', '#ffffff',

  // Amethyst / Royal Purple / Violet plumage ramp
  '#120822', '#160b29', '#1e1035', '#22103f', '#33175c', '#3b1c6e', '#4c1d95', '#5b21b6',
  '#6d28d9', '#7e22ce', '#9333ea', '#a855f7', '#b983ff', '#c084fc', '#d8b4fe', '#e9d5ff',
  '#f3e8ff',

  // Solar Gold / Amber crown & starlight ramp
  '#3b2206', '#5c350a', '#78350f', '#92400e', '#b45309', '#d97706', '#f59e0b', '#fbbf24',
  '#fcd34d', '#fde68a', '#fef3c7', '#fffbeb',

  // Emerald verification ramp
  '#052e16', '#064e3b', '#065f46', '#047857', '#059669', '#10b981', '#34d399', '#6ee7b7',
  '#a7f3d0', '#d1fae5',

  // Cyan / Sky celestial highlight ramp
  '#082f49', '#0c4a6e', '#0369a1', '#0284c7', '#0ea5e9', '#38bdf8', '#7dd3fc', '#bae6fd',
  '#e0f2fe',

  // Crimson / Rose error ramp
  '#4c0519', '#881337', '#9f1239', '#be123c', '#e11d48', '#f43f5e', '#fb7185', '#fda4af',
  '#ffe4e6',

  // Deep Indigo / Teal / Scribe & Sentinel accents
  '#0f172a', '#1e1b4b', '#312e81', '#3730a3', '#4338ca', '#4f46e5', '#6366f1', '#818cf8',
  '#0f766e', '#14b8a6', '#2dd4bf', '#99f6e4', '#fb923c',
]

class ByteBuffer {
  private data: Uint8Array
  private pos = 0

  constructor(initialCapacity: number) {
    this.data = new Uint8Array(Math.max(64, initialCapacity))
  }

  private ensure(extra: number): void {
    const required = this.pos + extra
    if (required <= this.data.length) return
    let nextCap = this.data.length
    while (nextCap < required) nextCap *= 2
    const next = new Uint8Array(nextCap)
    next.set(this.data.subarray(0, this.pos))
    this.data = next
  }

  writeByte(b: number): void {
    this.ensure(1)
    this.data[this.pos++] = b & 0xff
  }

  writeUint16LE(value: number): void {
    this.ensure(2)
    this.data[this.pos++] = value & 0xff
    this.data[this.pos++] = (value >> 8) & 0xff
  }

  writeBytes(bytes: Uint8Array): void {
    if (bytes.length === 0) return
    this.ensure(bytes.length)
    this.data.set(bytes, this.pos)
    this.pos += bytes.length
  }

  writeAscii(text: string): void {
    this.ensure(text.length)
    for (let i = 0; i < text.length; i += 1) {
      this.data[this.pos++] = text.charCodeAt(i) & 0x7f
    }
  }

  toUint8Array(): Uint8Array {
    return this.data.slice(0, this.pos)
  }
}

function parseHexRgb(hex: string): readonly [number, number, number] {
  const clean = hex.startsWith('#') ? hex.slice(1) : hex
  const value = Number.parseInt(clean, 16)
  return [(value >> 16) & 0xff, (value >> 8) & 0xff, value & 0xff]
}

function rgbKey(r: number, g: number, b: number): number {
  return ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff)
}

function premultiplyPixel(r: number, g: number, b: number, a: number): readonly [number, number, number] {
  if (a >= 255) return [r, g, b]
  if (a <= 0) return [0, 0, 0]
  return [
    Math.round((r * a) / 255),
    Math.round((g * a) / 255),
    Math.round((b * a) / 255),
  ]
}

function colorDistSq(r1: number, g1: number, b1: number, r2: number, g2: number, b2: number): number {
  const dr = r1 - r2
  const dg = g1 - g2
  const db = b1 - b2
  return 2 * dr * dr + 4 * dg * dg + 3 * db * db
}

interface ColorCandidate {
  readonly r: number
  readonly g: number
  readonly b: number
  readonly count: number
}

function buildGlobalPalette(frames: readonly GifFrame[], width: number, height: number): Uint8Array {
  const palette = new Uint8Array(256 * 3)
  const seen = new Set<number>()
  let count = 0

  const pushColor = (r: number, g: number, b: number): boolean => {
    if (count >= 256) return false
    const key = rgbKey(r, g, b)
    if (seen.has(key)) return false
    seen.add(key)
    const offset = count * 3
    palette[offset] = r
    palette[offset + 1] = g
    palette[offset + 2] = b
    count += 1
    return true
  }

  for (const hex of ANCHOR_HEX_COLORS) {
    const [r, g, b] = parseHexRgb(hex)
    pushColor(r, g, b)
  }

  const pixelsPerFrame = width * height
  const totalPixels = pixelsPerFrame * frames.length
  const stride = Math.max(1, Math.floor(totalPixels / MAX_PALETTE_SAMPLES))

  const exactCounts = new Map<number, number>()
  let exactOverflow = false
  const bucketR = new Float64Array(32768)
  const bucketG = new Float64Array(32768)
  const bucketB = new Float64Array(32768)
  const bucketN = new Uint32Array(32768)

  for (const frame of frames) {
    const rgba = frame.rgba
    for (let px = 0; px < pixelsPerFrame; px += stride) {
      const i = px * 4
      const r0 = rgba[i] ?? 0
      const g0 = rgba[i + 1] ?? 0
      const b0 = rgba[i + 2] ?? 0
      const a0 = rgba[i + 3] ?? 255
      const [r, g, b] = premultiplyPixel(r0, g0, b0, a0)
      const key = rgbKey(r, g, b)
      if (seen.has(key)) continue

      if (!exactOverflow) {
        const prev = exactCounts.get(key)
        if (prev !== undefined) {
          exactCounts.set(key, prev + 1)
        } else if (exactCounts.size < MAX_EXACT_HISTOGRAM) {
          exactCounts.set(key, 1)
        } else {
          exactOverflow = true
        }
      }

      const bIdx = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)
      bucketR[bIdx] = (bucketR[bIdx] ?? 0) + r
      bucketG[bIdx] = (bucketG[bIdx] ?? 0) + g
      bucketB[bIdx] = (bucketB[bIdx] ?? 0) + b
      bucketN[bIdx] = (bucketN[bIdx] ?? 0) + 1
    }
  }

  const candidates: ColorCandidate[] = []
  if (!exactOverflow) {
    for (const [key, freq] of exactCounts) {
      candidates.push({
        r: (key >> 16) & 0xff,
        g: (key >> 8) & 0xff,
        b: key & 0xff,
        count: freq,
      })
    }
  } else {
    for (let bIdx = 0; bIdx < 32768; bIdx += 1) {
      const n = bucketN[bIdx] ?? 0
      if (n === 0) continue
      candidates.push({
        r: Math.round((bucketR[bIdx] ?? 0) / n),
        g: Math.round((bucketG[bIdx] ?? 0) / n),
        b: Math.round((bucketB[bIdx] ?? 0) / n),
        count: n,
      })
    }
  }

  candidates.sort((a, b) => b.count - a.count)

  // If all distinct exact frame colors fit directly into remaining slots,
  // include every single one verbatim for lossless pixel-art fidelity.
  if (!exactOverflow && candidates.length <= 256 - count) {
    for (const c of candidates) {
      pushColor(c.r, c.g, c.b)
    }
  } else {
    const thresholds = [600, 160, 0]
    for (const minDistSq of thresholds) {
      if (count >= 256) break
      for (const c of candidates) {
        if (count >= 256) break
        const key = rgbKey(c.r, c.g, c.b)
        if (seen.has(key)) continue
        if (minDistSq > 0) {
          let tooClose = false
          for (let j = 0; j < count; j += 1) {
            const off = j * 3
            const d = colorDistSq(
              c.r,
              c.g,
              c.b,
              palette[off] ?? 0,
              palette[off + 1] ?? 0,
              palette[off + 2] ?? 0,
            )
            if (d < minDistSq) {
              tooClose = true
              break
            }
          }
          if (tooClose) continue
        }
        pushColor(c.r, c.g, c.b)
      }
    }
  }

  // Fill any remaining palette entries with a balanced 4×4×4 color cube so the
  // 256-color Global Color Table has full gamut coverage.
  const cubeSteps = [0, 85, 170, 255]
  for (const r of cubeSteps) {
    for (const g of cubeSteps) {
      for (const b of cubeSteps) {
        if (count >= 256) break
        pushColor(r, g, b)
      }
    }
  }

  return palette
}

class PaletteMapper {
  private readonly palette: Uint8Array
  private readonly exactMap = new Map<number, number>()
  private readonly coarseCache = new Int16Array(262144)

  constructor(palette: Uint8Array) {
    this.palette = palette
    this.coarseCache.fill(-1)
    for (let idx = 0; idx < 256; idx += 1) {
      const off = idx * 3
      const key = rgbKey(palette[off] ?? 0, palette[off + 1] ?? 0, palette[off + 2] ?? 0)
      if (!this.exactMap.has(key)) {
        this.exactMap.set(key, idx)
      }
    }
  }

  mapRgb(r: number, g: number, b: number): number {
    const key = rgbKey(r, g, b)
    const exact = this.exactMap.get(key)
    if (exact !== undefined) return exact

    const useCoarse = this.exactMap.size >= MAX_EXACT_CACHE
    const coarseIdx = ((r >> 2) << 12) | ((g >> 2) << 6) | (b >> 2)
    if (useCoarse) {
      const cached = this.coarseCache[coarseIdx] ?? -1
      if (cached >= 0) return cached
    }

    let bestIdx = 0
    let bestDist = Number.POSITIVE_INFINITY
    for (let idx = 0; idx < 256; idx += 1) {
      const off = idx * 3
      const dist = colorDistSq(
        r,
        g,
        b,
        this.palette[off] ?? 0,
        this.palette[off + 1] ?? 0,
        this.palette[off + 2] ?? 0,
      )
      if (dist < bestDist) {
        bestDist = dist
        bestIdx = idx
        if (dist === 0) break
      }
    }

    if (useCoarse) {
      this.coarseCache[coarseIdx] = bestIdx
    } else {
      this.exactMap.set(key, bestIdx)
    }
    return bestIdx
  }

  quantizeFrame(rgba: Uint8ClampedArray | Uint8Array, pixelCount: number): Uint8Array {
    const indices = new Uint8Array(pixelCount)
    for (let px = 0; px < pixelCount; px += 1) {
      const i = px * 4
      const r0 = rgba[i] ?? 0
      const g0 = rgba[i + 1] ?? 0
      const b0 = rgba[i + 2] ?? 0
      const a0 = rgba[i + 3] ?? 255
      const [r, g, b] = premultiplyPixel(r0, g0, b0, a0)
      indices[px] = this.mapRgb(r, g, b)
    }
    return indices
  }
}

function writeLzwImageData(out: ByteBuffer, indices: Uint8Array): void {
  out.writeByte(MIN_CODE_SIZE)

  const tableKeys = new Int32Array(LZW_TABLE_SIZE)
  const tableVals = new Int16Array(LZW_TABLE_SIZE)
  const tableStamps = new Int32Array(LZW_TABLE_SIZE)
  let generation = 1

  const subBlock = new Uint8Array(255)
  let subLen = 0
  let bitBuf = 0
  let bitCount = 0

  const flushSubBlock = (): void => {
    if (subLen === 0) return
    out.writeByte(subLen)
    out.writeBytes(subBlock.subarray(0, subLen))
    subLen = 0
  }

  const pushEncodedByte = (byteVal: number): void => {
    subBlock[subLen++] = byteVal & 0xff
    if (subLen === 255) flushSubBlock()
  }

  const writeCode = (code: number, widthBits: number): void => {
    bitBuf |= code << bitCount
    bitCount += widthBits
    while (bitCount >= 8) {
      pushEncodedByte(bitBuf & 0xff)
      bitBuf >>>= 8
      bitCount -= 8
    }
  }

  let codeSize = MIN_CODE_SIZE + 1
  let nextCode = EOI_CODE + 1
  let codeLimit = 1 << codeSize

  writeCode(CLEAR_CODE, codeSize)

  if (indices.length > 0) {
    let prefix = indices[0] ?? 0

    for (let i = 1; i < indices.length; i += 1) {
      const pixel = indices[i] ?? 0
      const key = (prefix << 8) | pixel
      let slot = ((key * 2654435761) >>> 19) & LZW_TABLE_MASK
      let found = -1

      while (tableStamps[slot] === generation) {
        if (tableKeys[slot] === key) {
          found = tableVals[slot] ?? -1
          break
        }
        slot = (slot + 1) & LZW_TABLE_MASK
      }

      if (found >= 0) {
        prefix = found
      } else {
        writeCode(prefix, codeSize)

        if (nextCode < MAX_LZW_CODES) {
          tableStamps[slot] = generation
          tableKeys[slot] = key
          tableVals[slot] = nextCode
          nextCode += 1
          if (nextCode > codeLimit && codeSize < 12) {
            codeSize += 1
            codeLimit = 1 << codeSize
          }
        } else {
          writeCode(CLEAR_CODE, codeSize)
          generation += 1
          nextCode = EOI_CODE + 1
          codeSize = MIN_CODE_SIZE + 1
          codeLimit = 1 << codeSize
        }

        prefix = pixel
      }
    }

    writeCode(prefix, codeSize)
    if (nextCode >= codeLimit && codeSize < 12) {
      codeSize += 1
    }
  }

  writeCode(EOI_CODE, codeSize)

  if (bitCount > 0) {
    pushEncodedByte(bitBuf & 0xff)
  }

  flushSubBlock()
  out.writeByte(0x00)
}

function validateFrames(frames: readonly GifFrame[]): { readonly width: number; readonly height: number } {
  if (frames.length === 0) {
    throw new Error('encodeAnimatedGif requires at least one frame.')
  }

  const first = frames[0]
  if (first === undefined) {
    throw new Error('encodeAnimatedGif requires at least one frame.')
  }

  const { width, height } = first
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1 ||
    width > MAX_GIF_DIMENSION ||
    height > MAX_GIF_DIMENSION
  ) {
    throw new Error(`Invalid GIF frame dimensions: ${String(width)}x${String(height)}.`)
  }

  const minBytes = width * height * 4
  for (let i = 0; i < frames.length; i += 1) {
    const frame = frames[i]
    if (frame === undefined) {
      throw new Error(`Missing GIF frame at index ${String(i)}.`)
    }
    if (frame.width !== width || frame.height !== height) {
      throw new Error(
        `GIF frame ${String(i)} dimensions ${String(frame.width)}x${String(frame.height)} do not match ${String(width)}x${String(height)}.`,
      )
    }
    if (frame.rgba.length < minBytes) {
      throw new Error(
        `GIF frame ${String(i)} RGBA buffer length ${String(frame.rgba.length)} is smaller than required ${String(minBytes)} bytes.`,
      )
    }
  }

  return { width, height }
}

/**
 * Encodes one or more RGBA frames into a looping GIF89a binary.
 */
export function encodeAnimatedGif(frames: readonly GifFrame[], options?: GifEncodeOptions): Uint8Array {
  const { width, height } = validateFrames(frames)
  const defaultDelayMs = options?.defaultDelayMs ?? DEFAULT_DELAY_MS
  const loopCount = options?.loop ?? 0
  const pixelCount = width * height

  const palette = buildGlobalPalette(frames, width, height)
  const mapper = new PaletteMapper(palette)

  const out = new ByteBuffer(1024 + Math.ceil((pixelCount * frames.length) / 3))

  // GIF89a Header
  out.writeAscii('GIF89a')

  // Logical Screen Descriptor: 256-color Global Color Table, 8-bit color resolution
  out.writeUint16LE(width)
  out.writeUint16LE(height)
  out.writeByte(0xf7)
  out.writeByte(0x00)
  out.writeByte(0x00)

  // Global Color Table (768 bytes)
  out.writeBytes(palette)

  // NETSCAPE2.0 Looping Application Extension
  if (loopCount >= 0) {
    const clampedLoop = Math.min(0xffff, Math.floor(loopCount))
    out.writeByte(0x21)
    out.writeByte(0xff)
    out.writeByte(0x0b)
    out.writeAscii('NETSCAPE2.0')
    out.writeByte(0x03)
    out.writeByte(0x01)
    out.writeUint16LE(clampedLoop)
    out.writeByte(0x00)
  }

  // Per-frame Graphic Control Extension + Image Descriptor + LZW stream
  for (const frame of frames) {
    const delayMs = frame.delayMs ?? defaultDelayMs
    const delayCentiseconds = Math.max(0, Math.min(0xffff, Math.round(delayMs / 10)))

    // Graphic Control Extension (disposal method = 1: do not dispose, opaque frame)
    out.writeByte(0x21)
    out.writeByte(0xf9)
    out.writeByte(0x04)
    out.writeByte(0x04)
    out.writeUint16LE(delayCentiseconds)
    out.writeByte(0x00)
    out.writeByte(0x00)

    // Image Descriptor (full screen, uses Global Color Table)
    out.writeByte(0x2c)
    out.writeUint16LE(0)
    out.writeUint16LE(0)
    out.writeUint16LE(width)
    out.writeUint16LE(height)
    out.writeByte(0x00)

    const indices = mapper.quantizeFrame(frame.rgba, pixelCount)
    writeLzwImageData(out, indices)
  }

  // GIF Trailer
  out.writeByte(0x3b)

  return out.toUint8Array()
}

/**
 * Triggers a browser download for encoded GIF bytes via a temporary object URL.
 * Returns `false` in headless/SSR environments where DOM APIs are unavailable.
 */
export function downloadGifBytes(bytes: Uint8Array, filename: string): boolean {
  if (
    typeof document === 'undefined' ||
    typeof Blob === 'undefined' ||
    typeof URL === 'undefined' ||
    typeof URL.createObjectURL !== 'function'
  ) {
    return false
  }

  const safeName = filename.trim().length > 0
    ? (filename.toLowerCase().endsWith('.gif') ? filename : `${filename}.gif`)
    : 'xerxes-swarm.gif'

  const copy = new Uint8Array(bytes.byteLength)
  copy.set(bytes)
  const blob = new Blob([copy], { type: 'image/gif' })
  const objectUrl = URL.createObjectURL(blob)

  const link = document.createElement('a')
  link.href = objectUrl
  link.download = safeName
  link.rel = 'noopener'
  link.style.display = 'none'

  const parent = document.body
  if (parent) {
    parent.appendChild(link)
  }
  link.click()
  if (parent && link.parentNode === parent) {
    parent.removeChild(link)
  }

  if (typeof URL.revokeObjectURL === 'function') {
    if (typeof setTimeout === 'function') {
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1000)
    } else {
      URL.revokeObjectURL(objectUrl)
    }
  }

  return true
}
