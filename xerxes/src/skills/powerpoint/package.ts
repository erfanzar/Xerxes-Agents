// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { dirname, relative, resolve, sep } from 'node:path'

import { OfficePackageError, OfficePackagePartNotFoundError } from './errors.js'

/**
 * Boundary for an unpacked OOXML directory. Part names are slash-separated,
 * package-relative names such as `ppt/slides/slide1.xml`.
 */
export interface OfficePackagePort {
  deletePart(partName: string): Promise<void>
  hasPart(partName: string): Promise<boolean>
  listParts(): Promise<readonly string[]>
  readBytes(partName: string): Promise<Uint8Array>
  readText(partName: string): Promise<string>
  writeBytes(partName: string, contents: Uint8Array): Promise<void>
  writeText(partName: string, contents: string): Promise<void>
}

/** Output boundary used by packers without coupling transformation logic to disk. */
export interface OfficeBinaryOutputPort {
  writeBytes(path: string, contents: Uint8Array): Promise<void>
}

/** Native Bun-compatible adapter for an unpacked OOXML directory on disk. */
export class BunOfficePackageDirectory implements OfficePackagePort {
  readonly rootDirectory: string
  private readonly resolvedRoot: string

  constructor(rootDirectory: string) {
    if (typeof rootDirectory !== 'string' || !rootDirectory.trim()) {
      throw new TypeError('rootDirectory must be a non-empty string')
    }
    this.rootDirectory = rootDirectory
    this.resolvedRoot = resolve(rootDirectory)
  }

  async deletePart(partName: string): Promise<void> {
    await rm(this.partPath(partName), { force: true })
  }

  async hasPart(partName: string): Promise<boolean> {
    try {
      return (await stat(this.partPath(partName))).isFile()
    } catch (error) {
      if (isMissingPathError(error)) return false
      throw error
    }
  }

  async listParts(): Promise<readonly string[]> {
    try {
      return await collectParts(this.resolvedRoot, this.resolvedRoot)
    } catch (error) {
      if (isMissingPathError(error)) return []
      throw error
    }
  }

  async readBytes(partName: string): Promise<Uint8Array> {
    const partPath = this.partPath(partName)
    try {
      return new Uint8Array(await readFile(partPath))
    } catch (error) {
      if (isMissingPathError(error)) throw new OfficePackagePartNotFoundError(normalizeOfficePartName(partName))
      throw error
    }
  }

  async readText(partName: string): Promise<string> {
    const bytes = await this.readBytes(partName)
    return new TextDecoder().decode(bytes)
  }

  async writeBytes(partName: string, contents: Uint8Array): Promise<void> {
    const partPath = this.partPath(partName)
    await mkdir(dirname(partPath), { recursive: true })
    await writeFile(partPath, contents)
  }

  async writeText(partName: string, contents: string): Promise<void> {
    if (typeof contents !== 'string') throw new TypeError('contents must be a string')
    await this.writeBytes(partName, new TextEncoder().encode(contents))
  }

  private partPath(partName: string): string {
    const normalized = normalizeOfficePartName(partName)
    const path = resolve(this.resolvedRoot, ...normalized.split('/'))
    if (!isPathInside(path, this.resolvedRoot)) {
      throw new OfficePackageError(`Office part escapes package root: ${partName}`)
    }
    return path
  }
}

/** Native Bun-compatible disk writer for a final `.pptx`, `.docx`, or `.xlsx` archive. */
export const bunOfficeFileOutput: OfficeBinaryOutputPort = {
  async writeBytes(path: string, contents: Uint8Array): Promise<void> {
    if (typeof path !== 'string' || !path.trim()) throw new TypeError('output path must be a non-empty string')
    const outputPath = resolve(path)
    await mkdir(dirname(outputPath), { recursive: true })
    await writeFile(outputPath, contents)
  },
}

/** Normalize and validate a package-relative OOXML part name. */
export function normalizeOfficePartName(partName: string): string {
  if (typeof partName !== 'string' || !partName.trim()) {
    throw new TypeError('Office part name must be a non-empty string')
  }
  const source = partName.replaceAll('\\', '/').replace(/^\/+/, '')
  const parts: string[] = []
  for (const segment of source.split('/')) {
    if (!segment || segment === '.') continue
    if (segment === '..') {
      if (!parts.length) throw new OfficePackageError(`Office part escapes package root: ${partName}`)
      parts.pop()
      continue
    }
    if (segment.includes('\0')) throw new OfficePackageError('Office part name contains a NUL byte')
    parts.push(segment)
  }
  if (!parts.length) throw new TypeError('Office part name must identify a file')
  return parts.join('/')
}

/** Combine a package-relative directory and target while refusing path traversal. */
export function joinOfficePartName(directory: string, target: string): string {
  if (typeof target !== 'string' || !target.trim()) throw new TypeError('target must be a non-empty string')
  if (target.startsWith('/')) return normalizeOfficePartName(target)
  return normalizeOfficePartName(directory ? `${directory}/${target}` : target)
}

/** Return a package-relative parent directory, or the empty root directory. */
export function officePartDirectory(partName: string): string {
  const normalized = normalizeOfficePartName(partName)
  const separator = normalized.lastIndexOf('/')
  return separator < 0 ? '' : normalized.slice(0, separator)
}

/** Return the filename segment of a package-relative part. */
export function officePartBasename(partName: string): string {
  const normalized = normalizeOfficePartName(partName)
  return normalized.slice(normalized.lastIndexOf('/') + 1)
}

/** Resolve a relationship target relative to its owning `.rels` part. */
export function relationshipTargetPartName(relationshipPartName: string, target: string): string {
  const normalizedRelationship = normalizeOfficePartName(relationshipPartName)
  const segments = normalizedRelationship.split('/')
  const relsIndex = segments.lastIndexOf('_rels')
  if (relsIndex < 0 || relsIndex !== segments.length - 2 || !segments.at(-1)?.endsWith('.rels')) {
    throw new OfficePackageError(`Invalid OOXML relationship part: ${relationshipPartName}`)
  }
  return joinOfficePartName(segments.slice(0, relsIndex).join('/'), target)
}

async function collectParts(root: string, directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true })
  const results: string[] = []
  for (const entry of entries) {
    const path = resolve(directory, entry.name)
    if (entry.isDirectory()) {
      results.push(...await collectParts(root, path))
      continue
    }
    if (!entry.isFile()) continue
    const relativePath = relative(root, path).split(sep).join('/')
    results.push(normalizeOfficePartName(relativePath))
  }
  return results.sort()
}

function isPathInside(path: string, root: string): boolean {
  return path === root || path.startsWith(`${root}${sep}`)
}

function isMissingPathError(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}
