// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { LOCAL_SIGNING_NAME, parseLocalSigningIdentity } from '../scripts/packageDesktopMac.js'

test('the stable local signing identity is found in the full codesigning listing, untrusted or not', () => {
  const hash = 'A'.repeat(40)
  const listing = [
    'Policy: Code Signing',
    '  Matching identities',
    `  1) ${'B'.repeat(40)} "Apple Development: someone (TEAM)"`,
    `  2) ${hash} "${LOCAL_SIGNING_NAME}" (CSSMERR_TP_NOT_TRUSTED)`,
    '     2 identities found',
  ].join('\n')
  expect(parseLocalSigningIdentity(listing)).toBe(hash)
})

test('no local identity, or a similarly named one, means no local signing', () => {
  expect(parseLocalSigningIdentity('  Matching identities\n     0 identities found\n')).toBeUndefined()
  expect(parseLocalSigningIdentity(`  1) ${'C'.repeat(40)} "${LOCAL_SIGNING_NAME} copy"`)).toBeUndefined()
})
