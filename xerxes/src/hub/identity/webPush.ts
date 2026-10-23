// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createCipheriv, createECDH, createPrivateKey, createPublicKey, generateKeyPairSync, hkdfSync, randomBytes, sign } from 'node:crypto'

/**
 * Web Push primitives with node:crypto only:
 *  - RFC 8291 message encryption (`aes128gcm` content coding, single record);
 *  - RFC 8292 VAPID (an ES256 JWT in the `Authorization: vapid` header).
 */

/** Largest plaintext that keeps the whole POST body within the 4096 byte push-service limit. */
export const MAX_PUSH_PLAINTEXT_BYTES = 3993
const RECORD_SIZE = 4096

export function generateVapidPem(): string {
  const { privateKey } = generateKeyPairSync('ec', { namedCurve: 'prime256v1' })
  return privateKey.export({ type: 'pkcs8', format: 'pem' }).toString()
}

/** base64url of the uncompressed P-256 point, the `applicationServerKey` a browser subscribes with. */
export function vapidPublicKeyOf(privateKeyPem: string): string {
  const jwk = createPublicKey(createPrivateKey(privateKeyPem)).export({ format: 'jwk' })
  if (typeof jwk.x !== 'string' || typeof jwk.y !== 'string') throw new Error('VAPID key is not an EC key')
  return Buffer.concat([Buffer.from([0x04]), Buffer.from(jwk.x, 'base64url'), Buffer.from(jwk.y, 'base64url')]).toString('base64url')
}

export interface VapidJwtInput {
  readonly privateKeyPem: string
  /** Origin of the push service endpoint. */
  readonly audience: string
  /** `mailto:` or `https:` contact URI. */
  readonly subject: string
  /** Unix seconds; RFC 8292 caps the lifetime at 24 hours. */
  readonly expiresAtSeconds: number
}

export function createVapidJwt(input: VapidJwtInput): string {
  const encode = (value: unknown): string => Buffer.from(JSON.stringify(value), 'utf8').toString('base64url')
  const signingInput = `${encode({ typ: 'JWT', alg: 'ES256' })}.${encode({ aud: input.audience, exp: input.expiresAtSeconds, sub: input.subject })}`
  const signature = sign('sha256', Buffer.from(signingInput, 'utf8'), { key: createPrivateKey(input.privateKeyPem), dsaEncoding: 'ieee-p1363' })
  return `${signingInput}.${signature.toString('base64url')}`
}

export function vapidAuthorization(jwt: string, publicKey: string): string {
  return `vapid t=${jwt}, k=${publicKey}`
}

export interface EncryptWebPushInput {
  readonly payload: Uint8Array
  /** The subscription's `p256dh` key: 65-byte uncompressed P-256 point. */
  readonly p256dh: Uint8Array
  /** The subscription's `auth` secret: 16 bytes. */
  readonly auth: Uint8Array
  /** 16-byte salt; random by default. Tests may pin it. */
  readonly salt?: Uint8Array
}

/** Returns the complete request body: header (salt, record size, key id) followed by the ciphertext. */
export function encryptWebPush(input: EncryptWebPushInput): Uint8Array {
  if (input.payload.length > MAX_PUSH_PLAINTEXT_BYTES) throw new RangeError(`push payload exceeds ${MAX_PUSH_PLAINTEXT_BYTES} bytes`)
  if (input.p256dh.length !== 65 || input.p256dh[0] !== 0x04) throw new RangeError('p256dh must be a 65 byte uncompressed P-256 point')
  if (input.auth.length !== 16) throw new RangeError('auth secret must be 16 bytes')
  const salt = Buffer.from(input.salt ?? randomBytes(16))
  if (salt.length !== 16) throw new RangeError('salt must be 16 bytes')
  const sender = createECDH('prime256v1')
  const senderPublic = sender.generateKeys()
  const shared = sender.computeSecret(Buffer.from(input.p256dh))
  const keyInfo = Buffer.concat([Buffer.from('WebPush: info\0', 'utf8'), Buffer.from(input.p256dh), senderPublic])
  const ikm = Buffer.from(hkdfSync('sha256', shared, Buffer.from(input.auth), keyInfo, 32))
  const contentKey = Buffer.from(hkdfSync('sha256', ikm, salt, Buffer.from('Content-Encoding: aes128gcm\0', 'utf8'), 16))
  const nonce = Buffer.from(hkdfSync('sha256', ikm, salt, Buffer.from('Content-Encoding: nonce\0', 'utf8'), 12))
  const cipher = createCipheriv('aes-128-gcm', contentKey, nonce)
  // 0x02 marks the last (here: only) record.
  const encrypted = Buffer.concat([cipher.update(Buffer.concat([Buffer.from(input.payload), Buffer.from([0x02])])), cipher.final(), cipher.getAuthTag()])
  const header = Buffer.alloc(16 + 4 + 1)
  salt.copy(header, 0)
  header.writeUInt32BE(RECORD_SIZE, 16)
  header.writeUInt8(senderPublic.length, 20)
  return Buffer.concat([header, senderPublic, encrypted])
}
