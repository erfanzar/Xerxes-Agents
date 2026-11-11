// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { deflateSync } from 'node:zlib'

export function chunk(kind: string, data: Uint8Array): Buffer {
  const type = Buffer.from(kind); const result = Buffer.alloc(data.length + 12)
  result.writeUInt32BE(data.length, 0); type.copy(result, 4); result.set(data, 8)
  result.writeUInt32BE(Bun.hash.crc32(result.subarray(4, -4)), result.length - 4)
  return result
}
export function png(width = 390, height = 900, extra: readonly Buffer[] = [], pixels?: Uint8Array, unused = Buffer.alloc(0), color = 6): Buffer {
  const header = Buffer.alloc(13); header.writeUInt32BE(width, 0); header.writeUInt32BE(height, 4); header[8] = 8; header[9] = color
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', header), ...extra,
    chunk('IDAT', Buffer.concat([deflateSync(pixels ?? Buffer.alloc((width * (color === 6 ? 4 : 3) + 1) * height)), unused])), chunk('IEND', Buffer.alloc(0))])
}
export function captureEnvelope(bytes = png()) {
  return { kind: 'xerxes-ui-capture/v1', view_id: 'a'.repeat(32), mime: 'image/png', width: 390, height: 900,
    captured_at: 1234, sha256: new Bun.CryptoHasher('sha256').update(bytes).digest('hex'), png_base64: bytes.toString('base64') }
}
