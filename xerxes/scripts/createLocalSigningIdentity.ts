// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Create, once, the self-signed code-signing certificate the desktop packaging
 * signs with when no Developer ID is available. macOS keys privacy grants
 * (Screen Recording, Accessibility) to the signing certificate instead of the
 * build hash, so they survive app updates on this Mac. The private key goes
 * straight into the login keychain, usable by codesign only; the temporary
 * files are removed. Running it again does nothing when the identity exists.
 */

import { randomBytes } from 'node:crypto'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { homedir, tmpdir } from 'node:os'
import { join } from 'node:path'

import { LOCAL_SIGNING_NAME, parseLocalSigningIdentity } from './packageDesktopMac.js'

async function output(argv: readonly string[]): Promise<string> {
  const proc = Bun.spawn([...argv], { stdout: 'pipe', stderr: 'pipe' })
  const [text, errors, code] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited])
  if (code !== 0) throw new Error(`${argv[0]} ${argv[1] ?? ''} failed (${code}): ${errors.trim().slice(0, 400)}`)
  return text
}

async function existing(): Promise<string | undefined> {
  return parseLocalSigningIdentity(await output(['security', 'find-identity', '-p', 'codesigning']))
}

export async function createLocalSigningIdentity(): Promise<string> {
  if (process.platform !== 'darwin') throw new Error('Local code signing applies to macOS only.')
  const found = await existing()
  if (found) return found
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-signing-'))
  try {
    const config = join(directory, 'identity.cnf'), key = join(directory, 'key.pem')
    const certificate = join(directory, 'certificate.pem'), bundle = join(directory, 'identity.p12')
    await writeFile(config, [
      '[req]', 'distinguished_name=dn', 'x509_extensions=ext', 'prompt=no',
      '[dn]', `CN=${LOCAL_SIGNING_NAME}`,
      '[ext]', 'basicConstraints=critical,CA:false', 'keyUsage=critical,digitalSignature', 'extendedKeyUsage=critical,codeSigning', '',
    ].join('\n'), { mode: 0o600 })
    await output(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', key, '-out', certificate, '-days', '3650', '-config', config])
    // Throwaway transport password for the temporary .p12 only.
    const password = randomBytes(18).toString('base64url')
    await output(['openssl', 'pkcs12', '-export', '-inkey', key, '-in', certificate, '-out', bundle, '-name', LOCAL_SIGNING_NAME, '-passout', `pass:${password}`])
    await output(['security', 'import', bundle, '-k', join(homedir(), 'Library', 'Keychains', 'login.keychain-db'), '-P', password, '-T', '/usr/bin/codesign'])
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
  const created = await existing()
  if (!created) throw new Error(`The certificate was imported but "${LOCAL_SIGNING_NAME}" is not listed as a code-signing identity.`)
  return created
}

if (import.meta.main) {
  const identity = await createLocalSigningIdentity()
  console.log(`Local signing identity ready: ${LOCAL_SIGNING_NAME} (${identity}). Rebuild with bun run --cwd xerxes build:installer.`)
}
