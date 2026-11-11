// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { dlopen, FFIType, ptr, toArrayBuffer } from 'bun:ffi'
import { closeSync, constants, fstatSync, openSync, read as readFd } from 'node:fs'
import { lstat, realpath } from 'node:fs/promises'
import { isAbsolute } from 'node:path'

export class OversizedObservationFile extends Error {}

let native: ReturnType<typeof loadNative> | undefined
function loadNative() {
  if (process.platform !== 'darwin' && process.platform !== 'linux') throw new Error('Secure descriptor-anchored file host is unavailable on this platform')
  return dlopen(process.platform === 'darwin' ? '/usr/lib/libSystem.B.dylib' : 'libc.so.6', {
    openat: { args: [FFIType.i32, FFIType.ptr, FFIType.i32], returns: FFIType.i32 },
    fdopendir: { args: [FFIType.i32], returns: FFIType.ptr },
    readdir: { args: [FFIType.ptr], returns: FFIType.ptr },
    closedir: { args: [FFIType.ptr], returns: FFIType.i32 },
  })
}

/** Each component is opened relative to an already-open parent, with no symlink following. */
export class AnchoredWorkspaceFiles {
  constructor(private readonly root: string) {}

  private async open(path: string, directory: boolean): Promise<number> {
    native ??= loadNative()
    if (isAbsolute(path) || path.includes('\\') || /[\0\r\n]/.test(path) || path.length > 4096) throw new Error('Invalid anchored path')
    const parts = path === '.' ? [] : path.split('/')
    if (parts.some(part => part === '' || part === '.' || part === '..')) throw new Error('Invalid anchored path')
    const canonical = await realpath(this.root)
    const expected = await lstat(canonical)
    let fd = openSync(canonical, constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW | constants.O_NONBLOCK)
    try {
      const actual = fstatSync(fd)
      if (!actual.isDirectory() || actual.dev !== expected.dev || actual.ino !== expected.ino) throw new Error('Workspace anchor changed during open')
      native ??= loadNative()
      for (let index = 0; index < parts.length; index++) {
        const name = Buffer.from(`${parts[index]}\0`)
        const isDirectory = directory || index < parts.length - 1
        const next = native.symbols.openat(fd, ptr(name), constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK | (isDirectory ? constants.O_DIRECTORY : 0))
        if (next < 0) throw new Error('Secure workspace anchor refused a symlink or inaccessible component')
        closeSync(fd); fd = next
      }
      return fd
    } catch (error) { closeSync(fd); throw error }
  }

  async read(path: string, offset: number, maxBytes: number) {
    if (!Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > 65536) throw new Error('Invalid anchored read bounds')
    const fd = await this.open(path, false)
    try {
      const info = fstatSync(fd)
      if (!info.isFile() || info.nlink !== 1) throw new Error('Only regular, non-hardlinked anchored files can be read')
      if (info.size > 1024 * 1024) throw new OversizedObservationFile('File exceeds 1 MiB observation limit')
      const bytes = new Uint8Array(maxBytes)
      const count = await new Promise<number>((resolve, reject) => {
        readFd(fd, bytes, 0, maxBytes, offset, (error, count) => { if (error) reject(error); else resolve(count) })
      })
      return { bytes: bytes.subarray(0, count), totalBytes: info.size }
    } finally { closeSync(fd) }
  }

  async list(path: string) {
    const fd = await this.open(path, true)
    native ??= loadNative()
    const directory = native.symbols.fdopendir(fd)
    if (directory === null) { closeSync(fd); throw new Error('Unable to list workspace anchor') }
    const entries: { name: string; kind: 'directory' | 'file' | 'symlink' | 'other' }[] = []
    try {
      while (true) {
        const item = native.symbols.readdir(directory)
        if (item === null) break
        // Native dirent layout is platform ABI, not a user-controlled structure.
        const darwin = process.platform === 'darwin'
        const header = new DataView(toArrayBuffer(item, 0, darwin ? 21 : 19))
        const recordLength = header.getUint16(darwin ? 16 : 16, true)
        const nameOffset = darwin ? 21 : 19
        if (recordLength <= nameOffset || recordLength > 4096) throw new Error('Invalid native directory entry')
        const type = header.getUint8(darwin ? 20 : 18)
        const raw = new Uint8Array(toArrayBuffer(item, nameOffset, recordLength - nameOffset))
        const end = raw.indexOf(0)
        if (end < 0) throw new Error('Invalid native directory name')
        const name = new TextDecoder('utf-8', { fatal: true }).decode(raw.subarray(0, end))
        if (name === '.' || name === '..') continue
        const kind = type === 4 ? 'directory' : type === 8 ? 'file' : type === 10 ? 'symlink' : 'other'
        entries.push({ name, kind })
        if (entries.length > 10000) throw new Error('Directory observation exceeds 10000 entries')
      }
      return entries
    } finally { native.symbols.closedir(directory) }
  }
}
