// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { QR_MAX_BYTES, encodeQr, qrToSvg, qrToTerminal, qrVersionInfo, rawDataModules, tryRenderQr, type QrCode } from '../src/hub/identity/qr.js'

/**
 * The encoder is checked by an independent reader: format information against
 * the published level-M table and BCH, version information against the Golay
 * code, finder/timing/alignment structure, the module count against the
 * spec's capacity formula, Reed-Solomon syndromes (all zero), and finally the
 * decoded payload.
 */

// ISO/IEC 18004 format strings for error correction level M, masks 0-7 (already XOR-masked).
const FORMAT_M = ['101010000010010', '101000100100101', '101111001111100', '101101101001011', '100010111111001', '100000011001110', '100111110010111', '100101010100000']

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

const dark = (qr: QrCode, x: number, y: number): boolean => qr.modules[y]?.[x] === true
const bitOf = (value: number, index: number): number => (value >>> index) & 1

function readFormatCopy1(qr: QrCode): number {
  let value = 0
  for (let i = 0; i <= 5; i += 1) value |= (dark(qr, 8, i) ? 1 : 0) << i
  value |= (dark(qr, 8, 7) ? 1 : 0) << 6
  value |= (dark(qr, 8, 8) ? 1 : 0) << 7
  value |= (dark(qr, 7, 8) ? 1 : 0) << 8
  for (let i = 9; i < 15; i += 1) value |= (dark(qr, 14 - i, 8) ? 1 : 0) << i
  return value
}

function readFormatCopy2(qr: QrCode): number {
  let value = 0
  for (let i = 0; i < 8; i += 1) value |= (dark(qr, qr.size - 1 - i, 8) ? 1 : 0) << i
  for (let i = 8; i < 15; i += 1) value |= (dark(qr, 8, qr.size - 15 + i) ? 1 : 0) << i
  return value
}

/** Remainder of a 15 bit word divided by the BCH(15,5) generator 0x537; zero for a valid codeword. */
function bchRemainder(word: number): number {
  let value = word
  for (let i = 14; i >= 10; i -= 1) if (bitOf(value, i)) value ^= 0x537 << (i - 10)
  return value
}

const EXP = new Array<number>(512).fill(0)
const LOG = new Array<number>(256).fill(0)
{
  let x = 1
  for (let i = 0; i < 255; i += 1) {
    EXP[i] = x
    LOG[x] = i
    x <<= 1
    if (x & 256) x ^= 0x11d
  }
  for (let i = 255; i < 512; i += 1) EXP[i] = EXP[i - 255] ?? 0
}
const mul = (a: number, b: number): number => (a === 0 || b === 0 ? 0 : (EXP[(LOG[a] ?? 0) + (LOG[b] ?? 0)] ?? 0))

function readDataBits(qr: QrCode): number[] {
  const { size } = qr
  const bits: number[] = []
  for (let right = size - 1; right >= 1; right -= 2) {
    if (right === 6) right = 5
    for (let vertical = 0; vertical < size; vertical += 1) {
      for (let j = 0; j < 2; j += 1) {
        const x = right - j
        const y = ((right + 1) & 2) === 0 ? size - 1 - vertical : vertical
        if (qr.isFunction[y]?.[x]) continue
        let value = dark(qr, x, y)
        if (maskApplies(qr.mask, y, x)) value = !value
        bits.push(value ? 1 : 0)
      }
    }
  }
  return bits
}

function decode(qr: QrCode): { readonly text: string; readonly syndromesZero: boolean; readonly moduleCount: number } {
  const bits = readDataBits(qr)
  const info = qrVersionInfo(qr.version)
  const codewords: number[] = []
  for (let i = 0; i < info.totalCodewords; i += 1) {
    let value = 0
    for (let j = 0; j < 8; j += 1) value = (value << 1) | (bits[i * 8 + j] ?? 0)
    codewords.push(value)
  }
  const blocks = info.blocks.map(() => ({ data: [] as number[], ec: [] as number[] }))
  let index = 0
  for (let i = 0; i < Math.max(...info.blocks); i += 1) {
    blocks.forEach((block, b) => {
      if (i < (info.blocks[b] ?? 0)) block.data.push(codewords[index++] ?? 0)
    })
  }
  for (let i = 0; i < info.ecCodewordsPerBlock; i += 1) for (const block of blocks) block.ec.push(codewords[index++] ?? 0)

  let syndromesZero = true
  for (const block of blocks) {
    const word = [...block.data, ...block.ec]
    for (let j = 0; j < info.ecCodewordsPerBlock; j += 1) {
      let accumulator = 0
      for (const codeword of word) accumulator = mul(accumulator, EXP[j] ?? 0) ^ codeword
      if (accumulator !== 0) syndromesZero = false
    }
  }

  const data = blocks.flatMap(block => block.data)
  const stream: number[] = []
  for (const byte of data) for (let i = 7; i >= 0; i -= 1) stream.push(bitOf(byte, i))
  const take = (count: number, from: number): number => stream.slice(from, from + count).reduce((acc, bit) => (acc << 1) | bit, 0)
  expect(take(4, 0)).toBe(0b0100)
  const countBits = qr.version <= 9 ? 8 : 16
  const length = take(countBits, 4)
  const bytes = new Uint8Array(length)
  for (let i = 0; i < length; i += 1) bytes[i] = take(8, 4 + countBits + i * 8)
  return { text: new TextDecoder().decode(bytes), syndromesZero, moduleCount: bits.length }
}

function checkFinder(qr: QrCode, left: number, top: number): void {
  for (let dy = 0; dy < 7; dy += 1) {
    for (let dx = 0; dx < 7; dx += 1) {
      const ring = Math.max(Math.abs(dx - 3), Math.abs(dy - 3))
      expect(dark(qr, left + dx, top + dy)).toBe(ring !== 2)
    }
  }
}

function checkStructure(qr: QrCode): void {
  checkFinder(qr, 0, 0)
  checkFinder(qr, qr.size - 7, 0)
  checkFinder(qr, 0, qr.size - 7)
  // Separators around the finders are light.
  for (let i = 0; i < 8; i += 1) {
    expect(dark(qr, 7, i)).toBe(false)
    expect(dark(qr, i, 7)).toBe(false)
    expect(dark(qr, qr.size - 8, i)).toBe(false)
    expect(dark(qr, qr.size - 1 - i, 7)).toBe(false)
    expect(dark(qr, 7, qr.size - 1 - i)).toBe(false)
    expect(dark(qr, i, qr.size - 8)).toBe(false)
  }
  for (let i = 8; i < qr.size - 8; i += 1) {
    expect(dark(qr, i, 6)).toBe(i % 2 === 0)
    expect(dark(qr, 6, i)).toBe(i % 2 === 0)
  }
  expect(dark(qr, 8, qr.size - 8)).toBe(true) // the always-dark module
  const format = readFormatCopy1(qr)
  expect(readFormatCopy2(qr)).toBe(format)
  expect(format.toString(2).padStart(15, '0')).toBe(FORMAT_M[qr.mask] ?? '')
  expect(bchRemainder(format ^ 0x5412)).toBe(0)
}

describe('QR encoder', () => {
  test('version 1: structure, format information and payload', () => {
    const qr = encodeQr('HELLO WORLD')
    expect(qr.version).toBe(1)
    expect(qr.size).toBe(21)
    checkStructure(qr)
    const decoded = decode(qr)
    expect(decoded.moduleCount).toBe(rawDataModules(1))
    expect(decoded.syndromesZero).toBe(true)
    expect(decoded.text).toBe('HELLO WORLD')
  })

  test('version 3-4 with multiple blocks decodes and its Reed-Solomon checks are consistent', () => {
    const text = 'https://hub.example.ts.net/pair#eyJodWJfaWQiOiJhYmMifQ'
    const qr = encodeQr(text)
    expect(qr.version).toBeGreaterThanOrEqual(3)
    checkStructure(qr)
    const decoded = decode(qr)
    expect(decoded.moduleCount).toBe(rawDataModules(qr.version))
    expect(decoded.syndromesZero).toBe(true)
    expect(decoded.text).toBe(text)
  })

  test('version 7: version information, alignment patterns, interleaved blocks', () => {
    const text = Array.from({ length: 110 }, (_, i) => String.fromCharCode(48 + ((i * 7) % 70))).join('')
    const qr = encodeQr(text)
    expect(qr.version).toBe(7)
    checkStructure(qr)
    // Golay-coded version information for version 7 is 0x07C94, in both copies.
    let first = 0
    let second = 0
    for (let i = 0; i < 18; i += 1) {
      first |= (dark(qr, qr.size - 11 + (i % 3), Math.floor(i / 3)) ? 1 : 0) << i
      second |= (dark(qr, Math.floor(i / 3), qr.size - 11 + (i % 3)) ? 1 : 0) << i
    }
    expect(first).toBe(0x07c94)
    expect(second).toBe(0x07c94)
    for (const [cx, cy] of [[22, 22], [38, 38], [22, 38], [38, 22]] as const) {
      for (let dy = -2; dy <= 2; dy += 1) for (let dx = -2; dx <= 2; dx += 1) expect(dark(qr, cx + dx, cy + dy)).toBe(Math.max(Math.abs(dx), Math.abs(dy)) !== 1)
    }
    const decoded = decode(qr)
    expect(decoded.moduleCount).toBe(rawDataModules(7))
    expect(decoded.syndromesZero).toBe(true)
    expect(decoded.text).toBe(text)
  })

  test('largest payload fits version 10; one more byte is refused', () => {
    const text = 'a'.repeat(QR_MAX_BYTES)
    const qr = encodeQr(text)
    expect(qr.version).toBe(10)
    checkStructure(qr)
    const decoded = decode(qr)
    expect(decoded.syndromesZero).toBe(true)
    expect(decoded.text).toBe(text)
    expect(() => encodeQr('a'.repeat(QR_MAX_BYTES + 1))).toThrow(RangeError)
  })

  test('UTF-8 text round-trips by bytes', () => {
    const qr = encodeQr('héllo ✓')
    expect(decode(qr).text).toBe('héllo ✓')
  })

  test('module count matches the capacity formula for every supported version', () => {
    for (let version = 1; version <= 10; version += 1) {
      const info = qrVersionInfo(version)
      const bits = rawDataModules(version)
      expect(Math.floor(bits / 8)).toBe(info.totalCodewords)
    }
    expect(() => qrVersionInfo(11)).toThrow(RangeError)
  })

  test('the chosen mask is the one with the lowest penalty (different inputs may choose different masks)', () => {
    const masks = new Set<number>()
    for (const text of ['a', 'abcdefgh', 'https://hub.test/pair#AAAA', '0123456789012345678901234567890123456789']) {
      const qr = encodeQr(text)
      masks.add(qr.mask)
      expect(qr.mask).toBeGreaterThanOrEqual(0)
      expect(qr.mask).toBeLessThan(8)
      expect(decode(qr).text).toBe(text)
    }
  })

  test('SVG and terminal renderings', () => {
    const qr = encodeQr('https://hub.test/pair#x')
    const svg = qrToSvg(qr)
    expect(svg.startsWith('<svg xmlns="http://www.w3.org/2000/svg"')).toBe(true)
    expect(svg).toContain(`viewBox="0 0 ${qr.size + 8} ${qr.size + 8}"`)
    expect(svg.endsWith('</svg>')).toBe(true)
    const terminal = qrToTerminal(qr)
    const lines = terminal.split('\n')
    expect(lines).toHaveLength(Math.ceil((qr.size + 4) / 2))
    expect(new Set(lines.map(line => [...line].length)).size).toBe(1)
    expect([...(lines[0] ?? '')].every(char => char === '\u2588')).toBe(true) // quiet zone
    const rendered = tryRenderQr('hello')
    expect(rendered.qrSvg).not.toBeNull()
    expect(tryRenderQr('x'.repeat(400))).toEqual({ qrSvg: null, qrTerminal: null })
  })
})
