// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, beforeEach, expect, it } from 'vitest'
import { chmod, mkdir, mkdtemp, rm, stat, symlink } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { remoteBunInstallScript } from '../lib/remoteBun.js'

// Only these host tools are visible to the script, so a host without curl
// (or without wget, unzip, a hashing tool) is simulated by leaving it out.
const BASE_TOOLS = ['sh', 'grep', 'cut', 'sed', 'mktemp', 'rm', 'mkdir', 'mv', 'chmod', 'cat', 'cp', 'basename', 'ldd', 'sysctl']
const RELEASES = 'https://github.com/oven-sh/bun/releases/latest/download'

let root = '', bin = '', home = '', fixtures = ''

const which = (tool: string) => Bun.spawnSync(['/bin/sh', '-c', `command -v ${tool}`]).stdout.toString().trim()
const executable = async (name: string, body: string) => {
  await Bun.write(join(bin, name), `#!/bin/sh\nset -eu\n${body}`)
  await chmod(join(bin, name), 0o755)
}
const tools = async (names: readonly string[]) => {
  for (const name of names) { const path = which(name); if (path) await symlink(path, join(bin, name)) }
}
/** A downloader that serves `fixtures/<basename of the URL>` and logs every URL it is asked for. */
const downloader = (name: 'curl' | 'wget') => executable(name, `out=''; url=''
while [ $# -gt 0 ]; do case "$1" in -o|-O) out=$2; shift 2 ;; https://*) url=$1; shift ;; *) shift ;; esac; done
echo "${name} $url" >> "$FIXTURES/calls"
[ -f "$FIXTURES/$(basename "$url")" ] || exit 22
cp "$FIXTURES/$(basename "$url")" "$out"
`)
const host = (system: string, machine: string) => executable('uname', `case "$1" in -s) echo ${system} ;; -m) echo ${machine} ;; esac\n`)

/** A real Bun release archive layout: `<asset>/bun` inside `<asset>.zip`, listed in SHASUMS256.txt. */
async function release(asset: string, binary = '#!/bin/sh\necho 1.3.99\n', sum?: string): Promise<void> {
  const stage = join(root, 'stage')
  await mkdir(join(stage, asset), { recursive: true })
  await Bun.write(join(stage, asset, 'bun'), binary)
  const zipped = Bun.spawnSync(['zip', '-qr', join(fixtures, `${asset}.zip`), asset], { cwd: stage })
  if (zipped.exitCode !== 0) throw new Error(zipped.stderr.toString())
  const digest = sum ?? createHash('sha256').update(new Uint8Array(await Bun.file(join(fixtures, `${asset}.zip`)).arrayBuffer())).digest('hex')
  await Bun.write(join(fixtures, 'SHASUMS256.txt'), `${'0'.repeat(64)}  bun-other.zip\n${digest}  ${asset}.zip\n`)
}

async function install(overrides = 'xerxes_has_avx2() { return 0; }\nxerxes_is_musl() { return 1; }\nxerxes_is_rosetta() { return 1; }'): Promise<{ code: number; output: string }> {
  const script = `${remoteBunInstallScript()}\n${overrides}\nxerxes_install_bun`
  const child = Bun.spawn([join(bin, 'sh'), '-c', script], { env: { HOME: home, PATH: bin, TMPDIR: root, FIXTURES: fixtures }, stdout: 'pipe', stderr: 'pipe' })
  const code = await child.exited
  return { code, output: (await new Response(child.stdout).text()).trim() }
}
const calls = async () => (await Bun.file(join(fixtures, 'calls')).exists() ? (await Bun.file(join(fixtures, 'calls')).text()).trim().split('\n') : [])

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), 'xerxes-remote-bun-'))
  bin = join(root, 'bin'); home = join(root, 'home'); fixtures = join(root, 'fixtures')
  await Promise.all([mkdir(bin), mkdir(home), mkdir(fixtures)])
  await tools(BASE_TOOLS)
  await host('Linux', 'x86_64')
})
afterEach(async () => { await rm(root, { recursive: true, force: true }) })

it('installs Bun on a host that has wget but no curl, verified against the published checksum', async () => {
  await tools(['unzip', 'shasum'])
  await downloader('wget')
  await release('bun-linux-x64')
  const result = await install()
  expect(result).toEqual({ code: 0, output: '' })
  const installed = join(home, '.bun/bin/bun')
  expect(await Bun.file(installed).text()).toBe('#!/bin/sh\necho 1.3.99\n')
  expect((await stat(installed)).mode & 0o777).toBe(0o755)
  expect(await calls()).toEqual([`wget ${RELEASES}/bun-linux-x64.zip`, `wget ${RELEASES}/SHASUMS256.txt`])
  expect(await Bun.file(join(home, '.bun/bin/bun.partial')).exists()).toBe(false)
})

it('prefers curl when both downloaders exist and replaces an outdated Bun in place', async () => {
  await tools(['unzip', 'shasum'])
  await downloader('curl'); await downloader('wget')
  await release('bun-linux-x64')
  await mkdir(join(home, '.bun/bin'), { recursive: true })
  await Bun.write(join(home, '.bun/bin/bun'), 'old')
  expect((await install()).code).toBe(0)
  expect(await Bun.file(join(home, '.bun/bin/bun')).text()).toContain('1.3.99')
  expect((await calls()).every(line => line.startsWith('curl '))).toBe(true)
})

it('refuses a download that does not match its checksum and leaves the existing Bun alone', async () => {
  await tools(['unzip', 'shasum'])
  await downloader('wget')
  await release('bun-linux-x64', '#!/bin/sh\necho tampered\n', 'f'.repeat(64))
  await mkdir(join(home, '.bun/bin'), { recursive: true })
  await Bun.write(join(home, '.bun/bin/bun'), 'old')
  expect(await install()).toEqual({ code: 1, output: 'The Bun download did not match its published checksum. Reconnect to try again.' })
  expect(await Bun.file(join(home, '.bun/bin/bun')).text()).toBe('old')
})

it('unpacks with bsdtar when unzip is missing', async () => {
  await tools(['bsdtar', 'shasum'])
  await downloader('wget')
  await release('bun-linux-x64')
  expect((await install()).code).toBe(0)
  expect(await Bun.file(join(home, '.bun/bin/bun')).exists()).toBe(true)
})

it('names exactly the tool to install when the host has no downloader or no unpacker', async () => {
  await tools(['unzip'])
  expect(await install()).toEqual({ code: 1, output: 'Install curl or wget on this host, then reconnect.' })
  await rm(join(bin, 'unzip'))
  await downloader('wget')
  expect(await install()).toEqual({ code: 1, output: 'Install unzip on this host, then reconnect.' })
  expect(await calls()).toEqual([])
})

it('reports an unreachable release host without installing anything', async () => {
  await tools(['unzip', 'shasum'])
  await downloader('wget')
  expect(await install()).toEqual({ code: 1, output: 'Could not download Bun. Check that this host can reach github.com.' })
  expect(await Bun.file(join(home, '.bun/bin/bun')).exists()).toBe(false)
})

it('chooses the official build for each host', async () => {
  await tools(['unzip', 'shasum'])
  await downloader('wget')
  const cases: Array<[string, string, string, string]> = [
    ['Linux', 'x86_64', '', 'bun-linux-x64'],
    ['Linux', 'x86_64', 'xerxes_has_avx2() { return 1; }', 'bun-linux-x64-baseline'],
    ['Linux', 'x86_64', 'xerxes_is_musl() { return 0; }', 'bun-linux-x64-musl'],
    ['Linux', 'x86_64', 'xerxes_is_musl() { return 0; }\nxerxes_has_avx2() { return 1; }', 'bun-linux-x64-musl-baseline'],
    ['Linux', 'aarch64', 'xerxes_has_avx2() { return 1; }', 'bun-linux-aarch64'],
    ['Linux', 'arm64', 'xerxes_is_musl() { return 0; }', 'bun-linux-aarch64-musl'],
    ['Darwin', 'arm64', '', 'bun-darwin-aarch64'],
    ['Darwin', 'x86_64', 'xerxes_is_rosetta() { return 0; }', 'bun-darwin-aarch64'],
    ['Darwin', 'x86_64', 'xerxes_has_avx2() { return 1; }', 'bun-darwin-x64-baseline'],
  ]
  for (const [system, machine, quirk, asset] of cases) {
    await host(system, machine)
    await rm(join(fixtures, 'calls'), { force: true })
    await release(asset)
    const result = await install(`xerxes_has_avx2() { return 0; }\nxerxes_is_musl() { return 1; }\nxerxes_is_rosetta() { return 1; }\n${quirk}`)
    expect({ asset, code: result.code }).toEqual({ asset, code: 0 })
    expect((await calls())[0]).toBe(`wget ${RELEASES}/${asset}.zip`)
  }
  await host('FreeBSD', 'amd64')
  expect(await install()).toEqual({ code: 1, output: 'Bun has no build for this host (FreeBSD amd64).' })
})
