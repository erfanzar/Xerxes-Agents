// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Small native QR Code encoder: byte mode, error correction level M,
 * versions 1-10 (up to 213 bytes), automatic mask selection by penalty score.
 * Produces a module matrix, an SVG and a terminal (Unicode half-block) string.
 */

interface VersionSpec {
  readonly ecPerBlock: number
  /** `[blockCount, dataCodewordsPerBlock]` groups, shorter blocks first. */
  readonly groups: readonly (readonly [number, number])[]
}

const SPECS: Readonly<Record<number, VersionSpec>> = {
  1: { ecPerBlock: 10, groups: [[1, 16]] },
  2: { ecPerBlock: 16, groups: [[1, 28]] },
  3: { ecPerBlock: 26, groups: [[1, 44]] },
  4: { ecPerBlock: 18, groups: [[2, 32]] },
  5: { ecPerBlock: 24, groups: [[2, 43]] },
  6: { ecPerBlock: 16, groups: [[4, 27]] },
  7: { ecPerBlock: 18, groups: [[4, 31]] },
  8: { ecPerBlock: 22, groups: [[2, 38], [2, 39]] },
  9: { ecPerBlock: 22, groups: [[3, 36], [2, 37]] },
  10: { ecPerBlock: 26, groups: [[4, 43], [1, 44]] },
}

const ALIGNMENT: Readonly<Record<number, readonly number[]>> = {
  1: [],
  2: [6, 18],
  3: [6, 22],
  4: [6, 26],
  5: [6, 30],
  6: [6, 34],
  7: [6, 22, 38],
  8: [6, 24, 42],
  9: [6, 26, 46],
  10: [6, 28, 50],
}

export const QR_MAX_VERSION = 10
/** Largest payload (bytes) that fits version 10 at level M. */
export const QR_MAX_BYTES = 213

export interface QrVersionInfo {
  readonly version: number
  readonly size: number
  readonly ecCodewordsPerBlock: number
  readonly blocks: readonly number[]
  readonly dataCodewords: number
  readonly totalCodewords: number
}

export function qrVersionInfo(version: number): QrVersionInfo {
  const spec = SPECS[version]
  if (!spec) throw new RangeError(`QR version ${version} is not supported`)
  const blocks: number[] = []
  for (const [count, length] of spec.groups) for (let i = 0; i < count; i += 1) blocks.push(length)
  const dataCodewords = blocks.reduce((sum, length) => sum + length, 0)
  return {
    version,
    size: 17 + 4 * version,
    ecCodewordsPerBlock: spec.ecPerBlock,
    blocks,
    dataCodewords,
    totalCodewords: dataCodewords + spec.ecPerBlock * blocks.length,
  }
}

/** Data modules available for codewords and remainder bits (function patterns excluded). */
export function rawDataModules(version: number): number {
  let result = (16 * version + 128) * version + 64
  if (version >= 2) {
    const alignments = Math.floor(version / 7) + 2
    result -= (25 * alignments - 10) * alignments - 55
    if (version >= 7) result -= 36
  }
  return result
}

export interface QrCode {
  readonly version: number
  readonly size: number
  readonly mask: number
  /** `modules[row][column]`, true = dark. */
  readonly modules: readonly (readonly boolean[])[]
  /** True for finder, timing, alignment, format and version modules. */
  readonly isFunction: readonly (readonly boolean[])[]
}

function gfMultiply(x: number, y: number): number {
  let z = 0
  for (let i = 7; i >= 0; i -= 1) {
    z = (z << 1) ^ ((z >>> 7) * 0x11d)
    z ^= ((y >>> i) & 1) * x
  }
  return z
}

function rsDivisor(degree: number): number[] {
  const result = new Array<number>(degree).fill(0)
  result[degree - 1] = 1
  let root = 1
  for (let i = 0; i < degree; i += 1) {
    for (let j = 0; j < result.length; j += 1) {
      result[j] = gfMultiply(result[j] ?? 0, root)
      if (j + 1 < result.length) result[j] = (result[j] ?? 0) ^ (result[j + 1] ?? 0)
    }
    root = gfMultiply(root, 0x02)
  }
  return result
}

function rsRemainder(data: readonly number[], divisor: readonly number[]): number[] {
  const result = new Array<number>(divisor.length).fill(0)
  for (const byte of data) {
    const factor = byte ^ (result.shift() ?? 0)
    result.push(0)
    divisor.forEach((coefficient, index) => {
      result[index] = (result[index] ?? 0) ^ gfMultiply(coefficient, factor)
    })
  }
  return result
}

function bit(value: number, index: number): boolean {
  return ((value >>> index) & 1) !== 0
}

function pickVersion(byteLength: number): number {
  for (let version = 1; version <= QR_MAX_VERSION; version += 1) {
    const countBits = version <= 9 ? 8 : 16
    if (4 + countBits + byteLength * 8 <= qrVersionInfo(version).dataCodewords * 8) return version
  }
  throw new RangeError(`QR payload of ${byteLength} bytes exceeds the ${QR_MAX_BYTES} byte capacity`)
}

function buildCodewords(bytes: Uint8Array, version: number): number[] {
  const info = qrVersionInfo(version)
  const bits: number[] = []
  const push = (value: number, length: number): void => {
    for (let i = length - 1; i >= 0; i -= 1) bits.push((value >>> i) & 1)
  }
  push(0b0100, 4)
  push(bytes.length, version <= 9 ? 8 : 16)
  for (const byte of bytes) push(byte, 8)
  const capacity = info.dataCodewords * 8
  push(0, Math.min(4, capacity - bits.length))
  while (bits.length % 8 !== 0) bits.push(0)
  for (let pad = 0xec; bits.length < capacity; pad ^= 0xec ^ 0x11) push(pad, 8)
  const data: number[] = []
  for (let i = 0; i < bits.length; i += 8) {
    let value = 0
    for (let j = 0; j < 8; j += 1) value = (value << 1) | (bits[i + j] ?? 0)
    data.push(value)
  }
  const divisor = rsDivisor(info.ecCodewordsPerBlock)
  const dataBlocks: number[][] = []
  const ecBlocks: number[][] = []
  let offset = 0
  for (const length of info.blocks) {
    const block = data.slice(offset, offset + length)
    offset += length
    dataBlocks.push(block)
    ecBlocks.push(rsRemainder(block, divisor))
  }
  const result: number[] = []
  const longest = Math.max(...info.blocks)
  for (let i = 0; i < longest; i += 1) for (const block of dataBlocks) if (i < block.length) result.push(block[i] ?? 0)
  for (let i = 0; i < info.ecCodewordsPerBlock; i += 1) for (const block of ecBlocks) result.push(block[i] ?? 0)
  return result
}

function maskApplies(mask: number, row: number, column: number): boolean {
  switch (mask) {
    case 0: return (row + column) % 2 === 0
    case 1: return row % 2 === 0
    case 2: return column % 3 === 0
    case 3: return (row + column) % 3 === 0
    case 4: return (Math.floor(row / 2) + Math.floor(column / 3)) % 2 === 0
    case 5: return ((row * column) % 2) + ((row * column) % 3) === 0
    case 6: return (((row * column) % 2) + ((row * column) % 3)) % 2 === 0
    default: return (((row + column) % 2) + ((row * column) % 3)) % 2 === 0
  }
}

const FINDER_LIKE = ['10111010000', '00001011101']

function penaltyScore(modules: readonly (readonly boolean[])[], size: number): number {
  let result = 0
  const lines: string[] = []
  for (let y = 0; y < size; y += 1) lines.push((modules[y] ?? []).map(dark => (dark ? '1' : '0')).join(''))
  for (let x = 0; x < size; x += 1) lines.push(modules.map(row => (row[x] ? '1' : '0')).join(''))
  for (const line of lines) {
    let run = 1
    for (let i = 1; i <= size; i += 1) {
      if (i < size && line[i] === line[i - 1]) run += 1
      else {
        if (run >= 5) result += run - 2
        run = 1
      }
    }
    for (const pattern of FINDER_LIKE) {
      let at = line.indexOf(pattern)
      while (at !== -1) {
        result += 40
        at = line.indexOf(pattern, at + 1)
      }
    }
  }
  let dark = 0
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const value = modules[y]?.[x] === true
      if (value) dark += 1
      if (y < size - 1 && x < size - 1) {
        if (value === (modules[y]?.[x + 1] === true) && value === (modules[y + 1]?.[x] === true) && value === (modules[y + 1]?.[x + 1] === true)) result += 3
      }
    }
  }
  const total = size * size
  result += Math.max(0, Math.ceil(Math.abs(dark * 20 - total * 10) / total) - 1) * 10
  return result
}

/** Encode UTF-8 text. Throws RangeError when it does not fit version 10 at level M. */
export function encodeQr(text: string): QrCode {
  const bytes = new TextEncoder().encode(text)
  const version = pickVersion(bytes.length)
  const codewords = buildCodewords(bytes, version)
  const size = 17 + 4 * version
  const modules: boolean[][] = Array.from({ length: size }, () => new Array<boolean>(size).fill(false))
  const isFunction: boolean[][] = Array.from({ length: size }, () => new Array<boolean>(size).fill(false))
  const setFunction = (x: number, y: number, dark: boolean): void => {
    const row = modules[y]
    const marks = isFunction[y]
    if (!row || !marks || x < 0 || x >= size) return
    row[x] = dark
    marks[x] = true
  }

  for (let i = 0; i < size; i += 1) {
    setFunction(6, i, i % 2 === 0)
    setFunction(i, 6, i % 2 === 0)
  }
  for (const [cx, cy] of [[3, 3], [size - 4, 3], [3, size - 4]] as const) {
    for (let dy = -4; dy <= 4; dy += 1) {
      for (let dx = -4; dx <= 4; dx += 1) {
        const distance = Math.max(Math.abs(dx), Math.abs(dy))
        setFunction(cx + dx, cy + dy, distance !== 2 && distance !== 4)
      }
    }
  }
  const positions = ALIGNMENT[version] ?? []
  const last = positions.length - 1
  positions.forEach((cy, i) => {
    positions.forEach((cx, j) => {
      if ((i === 0 && j === 0) || (i === 0 && j === last) || (i === last && j === 0)) return
      for (let dy = -2; dy <= 2; dy += 1) for (let dx = -2; dx <= 2; dx += 1) setFunction(cx + dx, cy + dy, Math.max(Math.abs(dx), Math.abs(dy)) !== 1)
    })
  })

  const drawFormat = (mask: number): void => {
    const data = (0b00 << 3) | mask // error correction level M
    let remainder = data
    for (let i = 0; i < 10; i += 1) remainder = (remainder << 1) ^ ((remainder >>> 9) * 0x537)
    const bits = ((data << 10) | remainder) ^ 0x5412
    for (let i = 0; i <= 5; i += 1) setFunction(8, i, bit(bits, i))
    setFunction(8, 7, bit(bits, 6))
    setFunction(8, 8, bit(bits, 7))
    setFunction(7, 8, bit(bits, 8))
    for (let i = 9; i < 15; i += 1) setFunction(14 - i, 8, bit(bits, i))
    for (let i = 0; i < 8; i += 1) setFunction(size - 1 - i, 8, bit(bits, i))
    for (let i = 8; i < 15; i += 1) setFunction(8, size - 15 + i, bit(bits, i))
    setFunction(8, size - 8, true)
  }
  drawFormat(0)
  if (version >= 7) {
    let remainder = version
    for (let i = 0; i < 12; i += 1) remainder = (remainder << 1) ^ ((remainder >>> 11) * 0x1f25)
    const bits = (version << 12) | remainder
    for (let i = 0; i < 18; i += 1) {
      const a = size - 11 + (i % 3)
      const b = Math.floor(i / 3)
      setFunction(a, b, bit(bits, i))
      setFunction(b, a, bit(bits, i))
    }
  }

  let index = 0
  for (let right = size - 1; right >= 1; right -= 2) {
    if (right === 6) right = 5
    for (let vertical = 0; vertical < size; vertical += 1) {
      for (let j = 0; j < 2; j += 1) {
        const x = right - j
        const upward = ((right + 1) & 2) === 0
        const y = upward ? size - 1 - vertical : vertical
        if (!isFunction[y]?.[x] && index < codewords.length * 8) {
          const row = modules[y]
          if (row) row[x] = bit(codewords[index >>> 3] ?? 0, 7 - (index & 7))
          index += 1
        }
      }
    }
  }

  const applyMask = (mask: number): void => {
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const row = modules[y]
        if (row && !isFunction[y]?.[x] && maskApplies(mask, y, x)) row[x] = !row[x]
      }
    }
  }
  let bestMask = 0
  let bestScore = Number.POSITIVE_INFINITY
  for (let mask = 0; mask < 8; mask += 1) {
    applyMask(mask)
    drawFormat(mask)
    const score = penaltyScore(modules, size)
    if (score < bestScore) {
      bestScore = score
      bestMask = mask
    }
    applyMask(mask)
  }
  applyMask(bestMask)
  drawFormat(bestMask)
  return { version, size, mask: bestMask, modules, isFunction }
}

export function qrToSvg(qr: QrCode, options: { readonly quiet?: number; readonly dark?: string; readonly light?: string } = {}): string {
  const quiet = options.quiet ?? 4
  const dark = options.dark ?? '#000000'
  const light = options.light ?? '#ffffff'
  const dimension = qr.size + quiet * 2
  const parts: string[] = []
  for (let y = 0; y < qr.size; y += 1) {
    let x = 0
    while (x < qr.size) {
      if (qr.modules[y]?.[x] !== true) {
        x += 1
        continue
      }
      let end = x
      while (end < qr.size && qr.modules[y]?.[end] === true) end += 1
      parts.push(`M${x + quiet} ${y + quiet}h${end - x}v1h-${end - x}z`)
      x = end
    }
  }
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${dimension} ${dimension}" shape-rendering="crispEdges">` +
    `<rect width="${dimension}" height="${dimension}" fill="${light}"/><path d="${parts.join('')}" fill="${dark}"/></svg>`
  )
}

/**
 * Two module rows per text line. Light modules are drawn as filled blocks, so
 * the code is correct on the usual dark terminal background (dark modules are
 * the terminal background showing through).
 */
export function qrToTerminal(qr: QrCode, quiet = 2): string {
  const dark = (x: number, y: number): boolean => qr.modules[y]?.[x] === true
  const lines: string[] = []
  for (let y = -quiet; y < qr.size + quiet; y += 2) {
    let line = ''
    for (let x = -quiet; x < qr.size + quiet; x += 1) {
      const top = dark(x, y)
      const bottom = dark(x, y + 1)
      line += top ? (bottom ? ' ' : '\u2584') : bottom ? '\u2580' : '\u2588'
    }
    lines.push(line)
  }
  return lines.join('\n')
}

/** Render when the text fits, otherwise null (callers still have the URL). */
export function tryRenderQr(text: string): { readonly qrSvg: string | null; readonly qrTerminal: string | null } {
  try {
    const qr = encodeQr(text)
    return { qrSvg: qrToSvg(qr), qrTerminal: qrToTerminal(qr) }
  } catch (error) {
    if (error instanceof RangeError) return { qrSvg: null, qrTerminal: null }
    throw error
  }
}
