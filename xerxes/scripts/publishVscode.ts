// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Publishes the VS Code extension in one step: builds every platform's
 * .vsix, publishes them all to the Visual Studio Marketplace, and attaches
 * them to the GitHub release for this version.
 *
 *   bun run publish:vscode                    # build, Marketplace, GitHub release
 *   bun run publish:vscode -- --skip-build    # reuse dist/*.vsix
 *   bun run publish:vscode -- --skip-github   # Marketplace only
 *   bun run publish:vscode -- --skip-marketplace
 *
 * The Marketplace needs a publish token for the publisher, once per machine:
 * `vsce login <publisher>` stores it, or set VSCE_PAT. This script never asks
 * for, prints or stores the token itself; without one it stops and says how
 * to add it. A version already on the Marketplace is skipped, not an error.
 */

import { existsSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'

const packageDirectory = join(import.meta.dir, '..')
const distDirectory = join(packageDirectory, 'dist')
const vsce = join(packageDirectory, 'node_modules', '@vscode', 'vsce', 'vsce')
const TARGETS = ['darwin-arm64', 'darwin-x64', 'linux-x64', 'linux-arm64', 'win32-x64'] as const

const args = new Set(process.argv.slice(2))
for (const arg of args) {
  if (!['--skip-build', '--skip-github', '--skip-marketplace'].includes(arg)) throw new Error(`Unknown option ${arg}`)
}

function run(command: readonly string[], options: { quiet?: boolean } = {}): number {
  const quiet = options.quiet ?? false
  const result = Bun.spawnSync([...command], { cwd: packageDirectory, stdout: quiet ? 'pipe' : 'inherit', stderr: quiet ? 'pipe' : 'inherit' })
  return result.exitCode
}

function must(command: readonly string[], what: string): void {
  if (run(command) !== 0) throw new Error(`${what} failed`)
}

const { version } = JSON.parse(await readFile(join(packageDirectory, 'package.json'), 'utf8')) as { version: string }

if (!args.has('--skip-build')) {
  must([process.execPath, 'run', 'build:runtime'], 'Runtime build')
  must([process.execPath, join(packageDirectory, 'scripts', 'buildDesktop.ts')], 'Renderer build')
  must([process.execPath, join(packageDirectory, 'scripts', 'buildVscode.ts'), '--all'], 'Extension build')
}

const packages = TARGETS.map(target => join(distDirectory, `xerxes-agents-${version}-${target}.vsix`))
const missing = packages.filter(path => !existsSync(path))
if (missing.length) throw new Error(`Missing ${missing.map(path => path.split('/').at(-1)).join(', ')}. Run without --skip-build.`)

// The publisher is whatever the build wrote into the manifest.
const manifest = JSON.parse(await readFile(join(distDirectory, 'vscode', TARGETS[0], 'package.json'), 'utf8')) as { publisher: string; name: string }

if (!args.has('--skip-marketplace')) {
  if (run([process.execPath, vsce, 'verify-pat', manifest.publisher], { quiet: true }) !== 0) {
    console.error(`
No Marketplace publish token for "${manifest.publisher}" on this machine. Add one once:

  1. https://dev.azure.com → User settings → Personal access tokens → New token
     Organization: All accessible organizations
     Scopes: Custom defined → Marketplace → Manage
  2. bun ${vsce} login ${manifest.publisher}   (paste the token)

Then run this again. Without a token you can upload the files in dist/ by hand at
https://marketplace.visualstudio.com/manage/publishers/${manifest.publisher}
`)
    process.exit(1)
  }
  // The preload lifts vsce's three-minute request timeout, which a slow
  // uplink outlasts while one platform package uploads; see vsceRequestTimeout.ts.
  must([process.execPath, '--preload', join(import.meta.dir, 'vsceRequestTimeout.ts'), vsce, 'publish', '--skip-duplicate', '--packagePath', ...packages], 'Marketplace publish')
  console.log(`Marketplace: https://marketplace.visualstudio.com/items?itemName=${manifest.publisher}.${manifest.name}`)
}

if (!args.has('--skip-github')) {
  const tag = `v${version}`
  if (run(['gh', 'release', 'view', tag], { quiet: true }) !== 0) throw new Error(`GitHub release ${tag} does not exist. Create it first, or pass --skip-github.`)
  must(['gh', 'release', 'upload', tag, ...packages, '--clobber'], 'GitHub release upload')
  console.log(`GitHub release ${tag}: ${packages.length} .vsix files attached`)
}
