// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'

import type { OcrDocumentFilesystemPort } from './types.js'

/** Bun-native filesystem implementation for callers that want local image output. */
export const bunOcrDocumentFilesystem: OcrDocumentFilesystemPort = {
  async ensureDirectory(path: string): Promise<void> {
    await mkdir(path, { recursive: true })
  },
  join,
  async writeFile(path: string, bytes: Uint8Array): Promise<void> {
    await Bun.write(path, bytes)
  },
}
