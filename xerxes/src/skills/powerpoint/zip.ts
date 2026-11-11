// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { deflateRawSync, inflateRawSync } from 'node:zlib'

import { OfficeZipEntryTooLargeError, UnsupportedOfficeZipError } from './errors.js'
import { normalizeOfficePartName } from './package.js'

const CENTRAL_DIRECTORY_SIGNATURE = 0x0201_4b50
const END_OF_CENTRAL_DIRECTORY_SIGNATURE = 0x0605_4b50
const LOCAL_FILE_SIGNATURE = 0x0403_4b50
const ZIP_DEFLATE_METHOD = 8
const ZIP_STORED_METHOD = 0
const ZIP_VERSION_NEEDED = 20

/** Entry supplied to the native OOXML ZIP writer. */
export interface OfficeZipEntry {
  readonly data: Uint8Array
  readonly name: string
}

/** Create a deterministic, Deflate-compressed ZIP archive without a Python dependency. */
export function createOfficeZip(entries: readonly OfficeZipEntry[]): Uint8Array {
  if (entries.length > 0xffff) throw new UnsupportedOfficeZipError('ZIP64 archives are not supported')
  const writer = new ZipWriter()
  const centralEntries: CentralEntry[] = []
  const seen = new Set<string>()
  for (const entry of [...entries].sort((left, right) => left.name.localeCompare(right.name))) {
    const name = normalizeOfficePartName(entry.name)
    if (seen.has(name)) throw new UnsupportedOfficeZipError(`ZIP has duplicate entry: ${name}`)
    seen.add(name)
    if (entry.data.byteLength > 0xffff_ffff) throw new UnsupportedOfficeZipError(`ZIP entry is too large: ${name}`)
    const nameBytes = new TextEncoder().encode(name)
    if (nameBytes.byteLength > 0xffff) throw new UnsupportedOfficeZipError(`ZIP entry name is too long: ${name}`)
    const compressed = new Uint8Array(deflateRawSync(entry.data))
    const useDeflate = compressed.byteLength < entry.data.byteLength
    const payload = useDeflate ? compressed : entry.data
    if (payload.byteLength > 0xffff_ffff || writer.length > 0xffff_ffff) {
      throw new UnsupportedOfficeZipError('ZIP64 archives are not supported')
    }
    const method = useDeflate ? ZIP_DEFLATE_METHOD : ZIP_STORED_METHOD
    const crc = crc32(entry.data)
    const localOffset = writer.length
    writer.u32(LOCAL_FILE_SIGNATURE)
    writer.u16(ZIP_VERSION_NEEDED)
    writer.u16(0)
    writer.u16(method)
    writer.u16(0)
    writer.u16(0)
    writer.u32(crc)
    writer.u32(payload.byteLength)
    writer.u32(entry.data.byteLength)
    writer.u16(nameBytes.byteLength)
    writer.u16(0)
    writer.bytes(nameBytes)
    writer.bytes(payload)
    centralEntries.push({ crc, dataLength: entry.data.byteLength, localOffset, method, nameBytes, payloadLength: payload.byteLength })
  }
  const centralOffset = writer.length
  for (const entry of centralEntries) {
    writer.u32(CENTRAL_DIRECTORY_SIGNATURE)
    writer.u16(ZIP_VERSION_NEEDED)
    writer.u16(ZIP_VERSION_NEEDED)
    writer.u16(0)
    writer.u16(entry.method)
    writer.u16(0)
    writer.u16(0)
    writer.u32(entry.crc)
    writer.u32(entry.payloadLength)
    writer.u32(entry.dataLength)
    writer.u16(entry.nameBytes.byteLength)
    writer.u16(0)
    writer.u16(0)
    writer.u16(0)
    writer.u16(0)
    writer.u32(0)
    writer.u32(entry.localOffset)
    writer.bytes(entry.nameBytes)
  }
  const centralLength = writer.length - centralOffset
  if (centralOffset > 0xffff_ffff || centralLength > 0xffff_ffff) throw new UnsupportedOfficeZipError('ZIP64 archives are not supported')
  writer.u32(END_OF_CENTRAL_DIRECTORY_SIGNATURE)
  writer.u16(0)
  writer.u16(0)
  writer.u16(centralEntries.length)
  writer.u16(centralEntries.length)
  writer.u32(centralLength)
  writer.u32(centralOffset)
  writer.u16(0)
  return writer.finish()
}

/** Read stored and Deflate entries from a regular (non-ZIP64, non-encrypted) OOXML archive. */
export function readOfficeZip(archive: Uint8Array): ReadonlyMap<string, Uint8Array> {
  const zip = openOfficeZip(archive)
  const entries = new Map<string, Uint8Array>()
  for (const name of zip.names) entries.set(name, zip.read(name)!)
  return entries
}

/** An archive whose entries are inflated one at a time, only when asked for. */
export interface OfficeZipReader {
  /** Normalized part names of the file entries, in central-directory order. */
  readonly names: readonly string[]
  /**
   * Inflate one entry, or undefined when the archive has none by that name.
   * With `maxBytes`, an entry that would inflate past it throws
   * OfficeZipEntryTooLargeError before (or while) inflating, never after.
   */
  read(name: string, maxBytes?: number): Uint8Array | undefined
}

/**
 * Index an archive without inflating anything. Reading only the parts a
 * caller needs, each under a byte limit, keeps a small file that inflates to
 * gigabytes (a ZIP bomb) from exhausting memory.
 */
export function openOfficeZip(archive: Uint8Array): OfficeZipReader {
  const directory = readCentralDirectory(archive)
  // OPC part names compare case-insensitively; writers mostly agree on case, not always.
  const folded = new Map<string, DirectoryEntry>()
  for (const [name, entry] of directory) if (!folded.has(name.toLowerCase())) folded.set(name.toLowerCase(), entry)
  return {
    names: [...directory.keys()],
    read(name, maxBytes) {
      const normalized = normalizeOfficePartName(name)
      const entry = directory.get(normalized) ?? folded.get(normalized.toLowerCase())
      if (entry === undefined) return undefined
      if (maxBytes !== undefined && entry.uncompressedLength > maxBytes) throw new OfficeZipEntryTooLargeError(entry.name, maxBytes)
      const data = readLocalEntry(archive, entry.localOffset, entry.method, entry.compressedLength, entry.uncompressedLength, maxBytes === undefined ? undefined : { entry: entry.name, limit: maxBytes })
      if (crc32(data) !== entry.crc) throw new UnsupportedOfficeZipError(`ZIP CRC mismatch for ${entry.name}`)
      return data
    },
  }
}

interface DirectoryEntry {
  readonly compressedLength: number
  readonly crc: number
  readonly localOffset: number
  readonly method: number
  readonly name: string
  readonly uncompressedLength: number
}

function readCentralDirectory(archive: Uint8Array): ReadonlyMap<string, DirectoryEntry> {
  const endOffset = findEndOfCentralDirectory(archive)
  const disk = u16(archive, endOffset + 4)
  const centralDisk = u16(archive, endOffset + 6)
  const diskEntries = u16(archive, endOffset + 8)
  const totalEntries = u16(archive, endOffset + 10)
  const centralLength = u32(archive, endOffset + 12)
  const centralOffset = u32(archive, endOffset + 16)
  if (disk !== 0 || centralDisk !== 0 || diskEntries !== totalEntries) {
    throw new UnsupportedOfficeZipError('Multi-disk ZIP archives are not supported')
  }
  if (centralOffset + centralLength > archive.byteLength) throw new UnsupportedOfficeZipError('ZIP central directory is truncated')
  const entries = new Map<string, DirectoryEntry>()
  let cursor = centralOffset
  for (let index = 0; index < totalEntries; index += 1) {
    if (u32(archive, cursor) !== CENTRAL_DIRECTORY_SIGNATURE) throw new UnsupportedOfficeZipError('Invalid central directory entry')
    const flags = u16(archive, cursor + 8)
    const method = u16(archive, cursor + 10)
    const crc = u32(archive, cursor + 16)
    const compressedLength = u32(archive, cursor + 20)
    const uncompressedLength = u32(archive, cursor + 24)
    const nameLength = u16(archive, cursor + 28)
    const extraLength = u16(archive, cursor + 30)
    const commentLength = u16(archive, cursor + 32)
    const localOffset = u32(archive, cursor + 42)
    const entryLength = 46 + nameLength + extraLength + commentLength
    if (cursor + entryLength > archive.byteLength) throw new UnsupportedOfficeZipError('ZIP central directory entry is truncated')
    if ((flags & 1) !== 0) throw new UnsupportedOfficeZipError('Encrypted ZIP entries are not supported')
    const name = new TextDecoder().decode(archive.slice(cursor + 46, cursor + 46 + nameLength))
    cursor += entryLength
    if (name.endsWith('/')) continue
    const normalizedName = normalizeOfficePartName(name)
    if (entries.has(normalizedName)) throw new UnsupportedOfficeZipError(`ZIP has duplicate entry: ${normalizedName}`)
    entries.set(normalizedName, { compressedLength, crc, localOffset, method, name: normalizedName, uncompressedLength })
  }
  return entries
}

interface CentralEntry {
  readonly crc: number
  readonly dataLength: number
  readonly localOffset: number
  readonly method: number
  readonly nameBytes: Uint8Array
  readonly payloadLength: number
}

class ZipWriter {
  private readonly chunks: Uint8Array[] = []
  private byteLength = 0

  get length(): number {
    return this.byteLength
  }

  bytes(value: Uint8Array): void {
    const copy = new Uint8Array(value)
    this.chunks.push(copy)
    this.byteLength += copy.byteLength
  }

  finish(): Uint8Array {
    const output = new Uint8Array(this.byteLength)
    let offset = 0
    for (const chunk of this.chunks) {
      output.set(chunk, offset)
      offset += chunk.byteLength
    }
    return output
  }

  u16(value: number): void {
    const bytes = new Uint8Array(2)
    new DataView(bytes.buffer).setUint16(0, value, true)
    this.bytes(bytes)
  }

  u32(value: number): void {
    const bytes = new Uint8Array(4)
    new DataView(bytes.buffer).setUint32(0, value >>> 0, true)
    this.bytes(bytes)
  }
}

function findEndOfCentralDirectory(archive: Uint8Array): number {
  const minimum = Math.max(0, archive.byteLength - 65_557)
  for (let offset = archive.byteLength - 22; offset >= minimum; offset -= 1) {
    if (u32(archive, offset) === END_OF_CENTRAL_DIRECTORY_SIGNATURE) return offset
  }
  throw new UnsupportedOfficeZipError('ZIP end-of-central-directory record was not found')
}

function readLocalEntry(
  archive: Uint8Array,
  localOffset: number,
  method: number,
  compressedLength: number,
  uncompressedLength: number,
  bound?: { readonly entry: string; readonly limit: number },
): Uint8Array {
  if (localOffset + 30 > archive.byteLength || u32(archive, localOffset) !== LOCAL_FILE_SIGNATURE) {
    throw new UnsupportedOfficeZipError('ZIP local file header is invalid')
  }
  const localNameLength = u16(archive, localOffset + 26)
  const localExtraLength = u16(archive, localOffset + 28)
  const dataStart = localOffset + 30 + localNameLength + localExtraLength
  const dataEnd = dataStart + compressedLength
  if (dataEnd > archive.byteLength) throw new UnsupportedOfficeZipError('ZIP entry data is truncated')
  const compressed = archive.slice(dataStart, dataEnd)
  let data: Uint8Array
  if (method === ZIP_STORED_METHOD) {
    data = compressed
  } else if (method === ZIP_DEFLATE_METHOD) {
    try {
      // The declared size can lie; the inflater's own cap is what holds.
      data = new Uint8Array(bound === undefined ? inflateRawSync(compressed) : inflateRawSync(compressed, { maxOutputLength: Math.max(1, bound.limit) }))
    } catch (error) {
      if (bound !== undefined && typeof error === 'object' && error !== null && 'code' in error && error.code === 'ERR_BUFFER_TOO_LARGE') {
        throw new OfficeZipEntryTooLargeError(bound.entry, bound.limit)
      }
      throw new UnsupportedOfficeZipError(`Could not inflate ZIP entry: ${error instanceof Error ? error.message : String(error)}`)
    }
  } else {
    throw new UnsupportedOfficeZipError(`ZIP compression method ${method} is not supported`)
  }
  if (data.byteLength !== uncompressedLength) throw new UnsupportedOfficeZipError('ZIP entry size does not match central directory')
  return data
}

function u16(bytes: Uint8Array, offset: number): number {
  if (offset < 0 || offset + 2 > bytes.byteLength) throw new UnsupportedOfficeZipError('ZIP structure is truncated')
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint16(offset, true)
}

function u32(bytes: Uint8Array, offset: number): number {
  if (offset < 0 || offset + 4 > bytes.byteLength) throw new UnsupportedOfficeZipError('ZIP structure is truncated')
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(offset, true)
}

let crcTable: Uint32Array | undefined

function crc32(bytes: Uint8Array): number {
  const table = crcTable ?? buildCrcTable()
  crcTable = table
  let crc = 0xffff_ffff
  for (const byte of bytes) crc = table[(crc ^ byte) & 0xff]! ^ (crc >>> 8)
  return (crc ^ 0xffff_ffff) >>> 0
}

function buildCrcTable(): Uint32Array {
  const table = new Uint32Array(256)
  for (let index = 0; index < table.length; index += 1) {
    let value = index
    for (let bit = 0; bit < 8; bit += 1) value = (value & 1) === 0 ? value >>> 1 : (value >>> 1) ^ 0xedb8_8320
    table[index] = value >>> 0
  }
  return table
}
