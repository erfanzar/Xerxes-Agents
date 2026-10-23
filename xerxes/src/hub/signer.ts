// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createPrivateKey, createPublicKey, generateKeyPairSync, type KeyObject } from 'node:crypto'
import { link, lstat, readFile, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

import { publicKeyFingerprint, signHubAuth, type HubAuthClaims, type HubAuthEnvelope } from '../security/hubAuthEnvelope.js'
import { ensureHubDirectories, type HubPaths } from './paths.js'

/**
 * The Hub's Ed25519 application key. It signs the `hub_auth` envelope attached
 * to forwarded daemon RPCs. The private key never leaves this module: callers
 * get the public half, a fingerprint for enrollment, and `sign`.
 */
export interface HubSigner {
  /** Stable audience identifier derived from the public key. */
  readonly hubId: string
  readonly publicKeyPem: string
  /** sha256 hex of the SPKI DER encoding, shown during enrollment. */
  readonly fingerprint: string
  /** Sign claims. Refuses claims naming a different `hub_id`. */
  sign(claims: HubAuthClaims): HubAuthEnvelope
}

export const HUB_KEY_FILENAME = 'hub-key.pem'

export class HubSignerError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'HubSignerError'
  }
}

export function createHubSigner(privateKeyPem: string): HubSigner {
  let privateKey: KeyObject
  try {
    privateKey = createPrivateKey(privateKeyPem)
  } catch {
    throw new HubSignerError('Hub signing key is not a valid PEM private key')
  }
  if (privateKey.asymmetricKeyType !== 'ed25519') throw new HubSignerError('Hub signing key must be Ed25519')
  const publicKey = createPublicKey(privateKey)
  const fingerprint = publicKeyFingerprint(publicKey)
  const hubId = `hub-${fingerprint.slice(0, 16)}`
  return Object.freeze({
    hubId,
    publicKeyPem: publicKey.export({ type: 'spki', format: 'pem' }).toString(),
    fingerprint,
    sign(claims: HubAuthClaims): HubAuthEnvelope {
      if (claims.hub_id !== hubId) throw new HubSignerError('refusing to sign claims for a different hub')
      return signHubAuth(claims, privateKey)
    },
  })
}

function isMissing(error: unknown): boolean {
  return !!error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT'
}

/**
 * Load `<hub state>/hub-key.pem`, or create it with mode 0600. A key readable by
 * group or others is refused rather than silently tightened: it may already have
 * been exposed, and the operator should rotate it.
 */
export async function loadOrCreateHubSigner(options: { readonly paths: HubPaths; readonly keyPath?: string }): Promise<HubSigner> {
  await ensureHubDirectories(options.paths)
  const keyPath = options.keyPath ?? join(options.paths.stateDir, HUB_KEY_FILENAME)
  let existing = await inspect(keyPath)
  if (existing === 'missing') {
    const { privateKey } = generateKeyPairSync('ed25519')
    const pem = privateKey.export({ type: 'pkcs8', format: 'pem' }).toString()
    // Write privately, then publish atomically: a concurrent creator can never read a partial key.
    const temporary = `${keyPath}.${crypto.randomUUID()}.tmp`
    try {
      await writeFile(temporary, pem, { mode: 0o600, flag: 'wx' })
      try {
        await link(temporary, keyPath)
      } catch (error) {
        if (!(error && typeof error === 'object' && 'code' in error && error.code === 'EEXIST')) throw error
      }
    } finally {
      await rm(temporary, { force: true })
    }
    existing = await inspect(keyPath)
  }
  if (existing === 'missing') throw new HubSignerError('Hub signing key could not be created')
  return createHubSigner(await readFile(keyPath, 'utf8'))
}

async function inspect(path: string): Promise<'present' | 'missing'> {
  let info
  try {
    info = await lstat(path)
  } catch (error) {
    if (isMissing(error)) return 'missing'
    throw error
  }
  if (!info.isFile()) throw new HubSignerError('Hub signing key path must be a regular file, not a link or directory')
  if ((info.mode & 0o077) !== 0) {
    throw new HubSignerError(`Hub signing key ${path} is readable by other users; run chmod 600 on it or rotate it`)
  }
  return 'present'
}
