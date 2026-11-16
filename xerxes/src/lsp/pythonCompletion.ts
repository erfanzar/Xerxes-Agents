// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Python completion for the desktop editor: basedpyright, run by a pinned
 * per-user Node build, both installed on first use under <home>/lsp/python.
 * Nothing is installed system-wide and the project's venv is only read
 * (its interpreter is handed to the server so installed libraries resolve).
 * Bun can run basedpyright, but measured 13–31 s per completion; Node answers
 * in milliseconds, so Node it is.
 */

import { chmod, mkdir, mkdtemp, rename, rm, stat, statfs } from 'node:fs/promises'
import { join } from 'node:path'

import { LspHost, type LspCompletionItem, type LspHostOptions } from './host.js'
import { LspManager } from './manager.js'

export const PYTHON_LSP_NODE_VERSION = 'v22.21.1'
export const BASEDPYRIGHT_VERSION = '1.40.2'
/** Official nodejs.org SHASUMS256 for the pinned version. */
const NODE_SHA256: Readonly<Record<string, string>> = {
  'darwin-arm64': 'c170d6554fba83d41d25a76cdbad85487c077e51fa73519e41ac885aa429d8af',
  'darwin-x64': '8e3dc89614debe66c2a6ad2313a1adb06eb37db6cd6c40d7de6f7d987f7d1afd',
  'linux-arm64': 'c86830dedf77f8941faa6c5a9c863bdfdd1927a336a46943decc06a38f80bfb2',
  'linux-x64': '219a152ea859861d75adea578bdec3dce8143853c13c5187f40c40e77b0143b2',
}
/** sha256 of the npm tarball; it matches the registry's published sha512 integrity. */
const BASEDPYRIGHT_SHA256 = '5d1073a0e61a1adc6b40b2bed4489017aa5c1336aa8e6fdcb7157b9bf26f3fab'
const DOWNLOAD_LIMIT = 100 * 1024 * 1024
/** Installed node binary (~107 MB) and server (~42 MB), plus the archives while unpacking. */
export const PYTHON_LSP_INSTALL_BYTES = 220 * 1024 * 1024
/** Never let the install leave less than this free. */
export const PYTHON_LSP_MIN_FREE_BYTES = 3 * 1024 ** 3

export interface PythonLspPorts {
  readonly fetch?: (url: string, init: { signal: AbortSignal }) => Promise<Response>
  readonly freeBytes?: (directory: string) => Promise<number>
  readonly extract?: (archive: string, destination: string, members: readonly string[]) => Promise<void>
  readonly platform?: string
  readonly arch?: string
  /** Test hook: replace the pinned digests. */
  readonly digests?: { readonly node?: Readonly<Record<string, string>>; readonly basedpyright?: string }
  /** Test hook: replace the language-server launcher. */
  readonly start?: (options: LspHostOptions, signal: AbortSignal) => Promise<LspHost>
}

export interface PythonLspPaths { readonly root: string; readonly target: string; readonly node: string; readonly server: string }

export function pythonLspPaths(home: string, platform: string = process.platform, arch: string = process.arch): PythonLspPaths {
  const target = `${platform}-${arch}`
  const root = join(home, 'lsp', 'python')
  return {
    root, target,
    node: join(root, `node-${PYTHON_LSP_NODE_VERSION}-${target}`, 'bin', 'node'),
    server: join(root, `basedpyright-${BASEDPYRIGHT_VERSION}`, 'package', 'langserver.index.js'),
  }
}

async function isFile(path: string): Promise<boolean> {
  try { return (await stat(path)).isFile() } catch { return false }
}

async function defaultFreeBytes(directory: string): Promise<number> {
  const info = await statfs(directory)
  return Number(info.bavail) * Number(info.bsize)
}

async function defaultExtract(archive: string, destination: string, members: readonly string[]): Promise<void> {
  const child = Bun.spawn(['tar', '-xzf', archive, '-C', destination, ...members], { stdin: 'ignore', stdout: 'ignore', stderr: 'ignore' })
  if (await child.exited !== 0) throw new Error('Could not unpack the downloaded archive')
}

async function download(fetchImpl: NonNullable<PythonLspPorts['fetch']>, url: string, path: string, digest: string, signal: AbortSignal): Promise<void> {
  const response = await fetchImpl(url, { signal })
  if (!response.ok || !response.body) throw new Error(`Download failed (HTTP ${response.status})`)
  const hasher = new Bun.CryptoHasher('sha256')
  const sink = Bun.file(path).writer()
  const reader = response.body.getReader()
  let size = 0
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      size += value.byteLength
      if (size > DOWNLOAD_LIMIT) throw new Error('Download is larger than expected; nothing was installed')
      hasher.update(value)
      sink.write(value)
    }
  } finally { await sink.end() }
  if (hasher.digest('hex') !== digest) throw new Error('Download checksum mismatch; nothing was installed')
}

/** Download, verify and unpack one archive into `final`, all-or-nothing. */
async function installArchive(root: string, url: string, digest: string, final: string, members: readonly string[], inner: string | undefined, ports: PythonLspPorts, signal: AbortSignal): Promise<void> {
  const stage = await mkdtemp(join(root, '.stage-'))
  try {
    const archive = join(stage, 'download.tgz'), out = join(stage, 'out')
    await download(ports.fetch ?? ((input, init) => fetch(input, init)), url, archive, digest, signal)
    await mkdir(out)
    await (ports.extract ?? defaultExtract)(archive, out, members)
    await rm(final, { recursive: true, force: true })
    await rename(inner ? join(out, inner) : out, final)
  } finally { await rm(stage, { recursive: true, force: true }) }
}

/** Install the pinned Node and basedpyright under <home>/lsp/python unless already there. */
export async function installPythonLsp(home: string, ports: PythonLspPorts = {}, signal: AbortSignal = AbortSignal.timeout(300_000)): Promise<PythonLspPaths> {
  const paths = pythonLspPaths(home, ports.platform, ports.arch)
  const nodeDigest = (ports.digests?.node ?? NODE_SHA256)[paths.target]
  if (!nodeDigest) throw new Error(`Python completion is not available on ${paths.target}`)
  const haveNode = await isFile(paths.node), haveServer = await isFile(paths.server)
  if (haveNode && haveServer) return paths
  await mkdir(paths.root, { recursive: true, mode: 0o700 })
  const free = await (ports.freeBytes ?? defaultFreeBytes)(paths.root)
  if (free - PYTHON_LSP_INSTALL_BYTES < PYTHON_LSP_MIN_FREE_BYTES) {
    throw new Error(`Python completion needs about ${Math.round(PYTHON_LSP_INSTALL_BYTES / 1024 ** 2)} MB and would leave less than 3 GB free (${(free / 1024 ** 3).toFixed(1)} GB free now). Nothing was installed.`)
  }
  if (!haveNode) {
    const name = `node-${PYTHON_LSP_NODE_VERSION}-${paths.target}`
    await installArchive(paths.root, `https://nodejs.org/dist/${PYTHON_LSP_NODE_VERSION}/${name}.tar.gz`, nodeDigest,
      join(paths.root, name), [`${name}/bin/node`], name, ports, signal)
    await chmod(paths.node, 0o755)
  }
  if (!haveServer) {
    await installArchive(paths.root, `https://registry.npmjs.org/basedpyright/-/basedpyright-${BASEDPYRIGHT_VERSION}.tgz`,
      ports.digests?.basedpyright ?? BASEDPYRIGHT_SHA256, join(paths.root, `basedpyright-${BASEDPYRIGHT_VERSION}`), [], undefined, ports, signal)
  }
  if (!await isFile(paths.node) || !await isFile(paths.server)) throw new Error('Python completion install is incomplete; retry')
  return paths
}

/** The project's own interpreter (venv first), so its installed libraries are discovered. */
export async function pythonInterpreter(root: string): Promise<string | undefined> {
  for (const candidate of ['.venv/bin/python', 'venv/bin/python', '.venv/Scripts/python.exe', 'venv/Scripts/python.exe']) {
    const path = join(root, candidate)
    if (await isFile(path)) return path
  }
  return Bun.which('python3') ?? Bun.which('python') ?? undefined
}

export function pythonConfiguration(python: string | undefined): Readonly<Record<string, unknown>> {
  // Open files only: a large repository is never analysed as a whole just to complete a name.
  const analysis = { diagnosticMode: 'openFilesOnly', typeCheckingMode: 'off', autoImportCompletions: true }
  return { python: python ? { pythonPath: python } : {}, basedpyright: { analysis }, 'basedpyright.analysis': analysis }
}

export interface PythonCompletionRequest { readonly filePath: string; readonly text: string; readonly line: number; readonly character: number }
export interface PythonCompletion {
  complete(root: string, request: PythonCompletionRequest, signal?: AbortSignal): Promise<{ readonly items: readonly LspCompletionItem[]; readonly incomplete: boolean }>
  close(): Promise<void>
}

/** One lazily installed server per workspace; a caller's cancellation never cancels the shared install. */
export function createPythonCompletion(home: string, ports: PythonLspPorts = {}): PythonCompletion {
  const paths = pythonLspPaths(home, ports.platform, ports.arch)
  let install: Promise<PythonLspPaths> | undefined
  const ensure = (): Promise<PythonLspPaths> => install ??= installPythonLsp(home, ports).catch(error => { install = undefined; throw error })
  const start = ports.start ?? LspHost.start
  const manager = new LspManager({ servers: [{
    name: 'python', command: paths.node, args: [paths.server, '--stdio'], languageId: 'python', extensions: ['.py', '.pyi'], timeoutMs: 60_000,
  }] }, async (options, signal) => {
    await ensure()
    return start({ ...options, configuration: pythonConfiguration(await pythonInterpreter(options.cwd)) }, signal)
  })
  return {
    complete: (root, request, signal) => manager.complete(root, request, signal),
    close: () => manager.close(),
  }
}
