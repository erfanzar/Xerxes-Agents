// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * `xerxes local <setup|start|stop|status|doctor>`: the user-facing surface of the
 * harness-managed local engine. All side effects go through `LocalCommandPorts`.
 * With `--json` exactly one JSON object is printed on stdout; progress and
 * diagnostics go to stderr.
 */
import { join } from 'node:path'

import { ValidationError } from '../core/errors.js'
import {
  catalogWithConfig,
  formatGiB,
  memoryBudget,
  parseLocalConfig,
  launchCtxSize,
  selectModel,
  type CatalogEntry,
  type LocalConfig,
} from './catalog.js'
import { DownloadError, ensureModelFile, useExistingModelFile, verifyModelFile, type FetchPort } from './download.js'
import type { HardwareProfile } from './hardware.js'
import { ServerError, type LocalServer, type ServerStatus } from './server.js'

export const LOCAL_PROFILE_NAME = 'bonsai-local'

export const LOCAL_USAGE = `Usage: xerxes local <command> [options]

Run a Bonsai model on this machine with llama.cpp's llama-server. Nothing runs until you ask.

Commands:
  setup   Probe hardware, pick a model, download it (verified), start the server, save the
          "${LOCAL_PROFILE_NAME}" provider profile.
            --model <id>         bonsai-9b | ternary-bonsai-2-27b (default: largest that fits)
            --model-file <path>  use an existing .gguf instead of downloading
            --yes                continue even if the model is larger than this machine's memory
            --json               print one JSON object
  start   Start the server with an already-downloaded model (--model, --model-file accepted).
  stop    Stop the server started by Xerxes.
  status  Show whether the server is running and healthy (exit 0 running, 1 not). --json
  doctor  Report hardware, chosen model, llama-server, model file and server health. --json

Download sources are never built in. Configure them in ~/.xerxes/daemon/config.json:
  { "runtime": { "local": { "models": { "bonsai-9b": { "url": "https://…", "sha256": "…" } } } } }
Other settings: runtime.local.llama_server_path, models_dir, port (default 11581), ctx_size
(llama-server --ctx-size; up to the model's own window, 262144 for Ternary Bonsai 2 27B).
`

export interface SaveLocalProfileInput {
  readonly name: string
  readonly baseUrl: string
  readonly model: string
  readonly provider: string
  readonly dataTrust: 'local'
  readonly promptProfile: 'lean'
  /** The --ctx-size llama-server runs with, so compaction and overflow rescue know the real window. */
  readonly contextLimit: number
}

export interface LocalCommandPorts {
  /** Probe the machine (wrap `probeHardware(systemHardwarePorts())` in production). */
  readonly hardware: () => Promise<HardwareProfile>
  readonly fetch: FetchPort
  readonly server: LocalServer
  saveProfile(input: SaveLocalProfileInput): Promise<void>
  stdout(text: string): void
  stderr(text: string): void
  /** `DaemonConfig.runtime`. */
  readonly config: Record<string, unknown>
  /** Defaults to `process.env`. */
  readonly env?: Record<string, string | undefined>
  /** Optional download confirmation; skipped by `--yes`. When absent the download proceeds. */
  readonly confirm?: (message: string) => Promise<boolean>
  readonly signal?: AbortSignal
}

interface ParsedArgs {
  readonly sub?: string
  readonly model?: string
  readonly modelFile?: string
  readonly yes: boolean
  readonly json: boolean
  readonly help: boolean
  readonly error?: string
}

function parseArgs(argv: readonly string[]): ParsedArgs {
  let sub: string | undefined
  let model: string | undefined
  let modelFile: string | undefined
  let yes = false
  let json = false
  let help = false
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i] ?? ''
    const eq = arg.startsWith('--') ? arg.indexOf('=') : -1
    const flag = eq > 0 ? arg.slice(0, eq) : arg
    const inline = eq > 0 ? arg.slice(eq + 1) : undefined
    const value = (): string | undefined => {
      if (inline !== undefined) return inline
      const next = argv[i + 1]
      if (next === undefined || next.startsWith('--')) return undefined
      i += 1
      return next
    }
    switch (flag) {
      case '--model': {
        const v = value()
        if (v === undefined || v === '') return { yes, json, help, error: '--model needs a model id' }
        model = v
        break
      }
      case '--model-file': {
        const v = value()
        if (v === undefined || v === '') return { yes, json, help, error: '--model-file needs a path' }
        modelFile = v
        break
      }
      case '--yes':
      case '-y':
        yes = true
        break
      case '--json':
        json = true
        break
      case '--help':
      case '-h':
        help = true
        break
      default:
        if (arg.startsWith('-')) return { yes, json, help, error: `Unknown option: ${arg}` }
        if (sub !== undefined) return { yes, json, help, error: `Unexpected argument: ${arg}` }
        sub = arg
    }
  }
  return {
    yes,
    json,
    help,
    ...(sub === undefined ? {} : { sub }),
    ...(model === undefined ? {} : { model }),
    ...(modelFile === undefined ? {} : { modelFile }),
  }
}

function describeError(error: unknown): { code: string; message: string } {
  if (error instanceof DownloadError || error instanceof ServerError) return { code: error.code, message: error.message }
  if (error instanceof ValidationError) return { code: 'INVALID_CONFIG', message: error.message }
  return { code: 'ERROR', message: error instanceof Error ? error.message : String(error) }
}

function hardwareJson(hw: HardwareProfile): Record<string, unknown> {
  return {
    platform: hw.platform,
    arch: hw.arch,
    total_mem_bytes: hw.totalMemBytes,
    apple_silicon: hw.appleSilicon,
    gpu: {
      vendor: hw.gpu.vendor,
      ...(hw.gpu.name === undefined ? {} : { name: hw.gpu.name }),
      ...(hw.gpu.vramBytes === undefined ? {} : { vram_bytes: hw.gpu.vramBytes }),
    },
  }
}

function hardwareLine(hw: HardwareProfile): string {
  const gpu =
    hw.gpu.vendor === 'none'
      ? 'no GPU detected'
      : `${hw.gpu.vendor} GPU${hw.gpu.name ? ` ${hw.gpu.name}` : ''}${hw.gpu.vramBytes === undefined ? '' : ` (${formatGiB(hw.gpu.vramBytes)} VRAM)`}`
  return `${hw.platform}/${hw.arch}, ${formatGiB(hw.totalMemBytes)} RAM, ${gpu}`
}

function statusJson(status: ServerStatus): Record<string, unknown> {
  return {
    running: status.running,
    healthy: status.healthy,
    stale: status.stale,
    ...(status.pid === undefined ? {} : { pid: status.pid }),
    ...(status.port === undefined ? {} : { port: status.port }),
    ...(status.model === undefined ? {} : { model: status.model }),
    ...(status.startedAt === undefined ? {} : { started_at: status.startedAt }),
    ...(status.logPath === undefined ? {} : { log_path: status.logPath }),
  }
}

function statusLine(status: ServerStatus): string {
  if (status.running) {
    return `running (pid ${status.pid}, port ${status.port}, ${status.healthy ? 'healthy' : 'not healthy yet'}) model ${status.model}`
  }
  return status.stale ? 'not running (a stale server record was found)' : 'not running'
}

type Chosen =
  | { readonly ok: true; readonly entry: CatalogEntry; readonly warning?: string }
  | { readonly ok: false; readonly code: string; readonly message: string }

function chooseEntry(
  modelId: string | undefined,
  catalog: readonly CatalogEntry[],
  hw: HardwareProfile,
  allowOversize: boolean,
): Chosen {
  if (modelId === undefined) {
    const selection = selectModel(hw, catalog)
    return selection.ok
      ? { ok: true, entry: selection.entry }
      : { ok: false, code: 'INSUFFICIENT_MEMORY', message: selection.reason }
  }
  const entry = catalog.find((candidate) => candidate.id === modelId)
  if (entry === undefined) {
    return {
      ok: false,
      code: 'UNKNOWN_MODEL',
      message: `Unknown model "${modelId}". Known models: ${catalog.map((c) => c.id).join(', ')}.`,
    }
  }
  const { budgetBytes } = memoryBudget(hw)
  if (budgetBytes < entry.minMemoryBytes) {
    const message = `${entry.id} needs ${formatGiB(entry.minMemoryBytes)} but this machine has ${formatGiB(budgetBytes)} usable.`
    return allowOversize
      ? { ok: true, entry, warning: `${message} Continuing because of --yes.` }
      : { ok: false, code: 'INSUFFICIENT_MEMORY', message: `${message} Pass --yes to try anyway, or pick a smaller model.` }
  }
  return { ok: true, entry }
}

function noSourceGuidance(entry: CatalogEntry): string {
  return (
    `No download source is configured for ${entry.id}. Xerxes ships no built-in model URLs or hashes. ` +
    `Add one to ~/.xerxes/daemon/config.json:\n` +
    `  { "runtime": { "local": { "models": { "${entry.id}": { "url": "https://…", "sha256": "<64 hex>" } } } } }\n` +
    `or pass an existing file with \`xerxes local setup --model-file /path/to/model.gguf\`.`
  )
}

class CommandFailure extends Error {
  readonly code: string
  readonly exit: number

  constructor(code: string, message: string, exit = 1) {
    super(message)
    this.code = code
    this.exit = exit
  }
}

export async function runLocalCommand(argv: string[], ports: LocalCommandPorts): Promise<number> {
  const args = parseArgs(argv)
  const env = ports.env ?? process.env

  const ok = (payload: Record<string, unknown>, human: string): number => {
    ports.stdout(args.json ? JSON.stringify({ ok: true, ...payload }) + '\n' : human.endsWith('\n') ? human : human + '\n')
    return 0
  }
  const fail = (code: string, message: string, exit: number, extra: Record<string, unknown> = {}): number => {
    if (args.json) {
      ports.stdout(JSON.stringify({ ok: false, error: { code, message }, ...extra }) + '\n')
      ports.stderr(message + '\n')
    } else {
      ports.stderr(message + '\n')
    }
    return exit
  }
  const progress = (text: string): void => ports.stderr(text + '\n')

  if (args.error !== undefined) return fail('USAGE', `${args.error}\n\n${LOCAL_USAGE}`, 2)
  if (args.help || args.sub === undefined || args.sub === 'help') {
    ports.stdout(LOCAL_USAGE)
    return 0
  }

  try {
    const cfg = parseLocalConfig(ports.config, env)
    const catalog = catalogWithConfig(cfg)

    switch (args.sub) {
      case 'setup':
        return await setup(args, cfg, catalog)
      case 'start':
        return await start(args, cfg, catalog)
      case 'stop':
        return await stop()
      case 'status':
        return await status()
      case 'doctor':
        return await doctor(args, cfg, catalog)
      default:
        return fail('USAGE', `Unknown command "${args.sub}".\n\n${LOCAL_USAGE}`, 2)
    }
  } catch (error) {
    if (error instanceof CommandFailure) return fail(error.code, error.message, error.exit)
    const described = describeError(error)
    return fail(described.code, described.message, described.code === 'INVALID_CONFIG' ? 2 : 1)
  }

  /** Resolve the model file for `start`: a user file, or the catalog file already on disk. */
  async function locateModel(
    parsed: ParsedArgs,
    cfg: LocalConfig,
    catalog: readonly CatalogEntry[],
    hw: HardwareProfile,
  ): Promise<{ entry: CatalogEntry; path: string }> {
    const chosen = chooseEntry(parsed.model, catalog, hw, parsed.yes)
    if (!chosen.ok) throw new CommandFailure(chosen.code, chosen.message)
    if (chosen.warning !== undefined) progress(chosen.warning)
    if (parsed.modelFile !== undefined) {
      const file = await useExistingModelFile(parsed.modelFile)
      return { entry: chosen.entry, path: file.path }
    }
    const path = join(cfg.modelsDir, chosen.entry.fileName)
    if (chosen.entry.source !== undefined) {
      const check = await verifyModelFile(path, chosen.entry.source)
      if (!check.present) {
        throw new CommandFailure('MODEL_MISSING', `${chosen.entry.id} is not downloaded (${path}). Run \`xerxes local setup\` first.`)
      }
      if (!check.verified) {
        throw new CommandFailure('MODEL_NOT_VERIFIED', `${path} does not match the configured sha256. Delete it and run \`xerxes local setup\` again.`)
      }
    } else {
      const check = await useExistingModelFile(path).catch(() => undefined)
      if (check === undefined) {
        throw new CommandFailure('MODEL_MISSING', `No model file at ${path}. ${noSourceGuidance(chosen.entry)}`)
      }
      progress(`Using ${path} without a configured sha256, so it is not verified.`)
    }
    return { entry: chosen.entry, path }
  }

  async function setup(parsed: ParsedArgs, cfg: LocalConfig, catalog: readonly CatalogEntry[]): Promise<number> {
    progress('Probing hardware…')
    const hw = await ports.hardware()
    progress(hardwareLine(hw))
    const chosen = chooseEntry(parsed.model, catalog, hw, parsed.yes)
    if (!chosen.ok) return fail(chosen.code, chosen.message, 1, { hardware: hardwareJson(hw) })
    if (chosen.warning !== undefined) progress(chosen.warning)
    const entry = chosen.entry
    progress(`Model: ${entry.id}`)

    let modelPath: string
    let downloaded = false
    if (parsed.modelFile !== undefined) {
      modelPath = (await useExistingModelFile(parsed.modelFile)).path
    } else {
      const source = entry.source
      if (source === undefined) return fail('NO_SOURCE', noSourceGuidance(entry), 1, { model: entry.id })
      const dest = join(cfg.modelsDir, entry.fileName)
      const existing = await verifyModelFile(dest, source)
      if (!existing.verified && ports.confirm !== undefined && !parsed.yes) {
        const size = source.bytes === undefined ? 'a large file' : formatGiB(source.bytes)
        const approved = await ports.confirm(`Download ${entry.id} (${size}) from ${new URL(source.url).host}?`)
        if (!approved) return fail('CANCELLED', 'Setup cancelled; nothing was downloaded.', 1)
      }
      let lastStep = -1
      const file = await ensureModelFile({
        destDir: cfg.modelsDir,
        fileName: entry.fileName,
        source,
        fetch: ports.fetch,
        ...(ports.signal === undefined ? {} : { signal: ports.signal }),
        onProgress: ({ receivedBytes, totalBytes }) => {
          if (totalBytes === undefined) return
          const step = Math.floor((receivedBytes / totalBytes) * 20)
          if (step === lastStep) return
          lastStep = step
          progress(`Downloading ${entry.id}: ${step * 5}% (${formatGiB(receivedBytes)} / ${formatGiB(totalBytes)})`)
        },
      })
      modelPath = file.path
      downloaded = !file.reused
      progress(file.reused ? 'Model already downloaded and verified.' : 'Download verified (sha256 ok).')
    }

    const current = await ports.server.status()
    let started = false
    let port = cfg.port
    if (current.running) {
      if (current.model !== modelPath) {
        return fail(
          'ALREADY_RUNNING',
          `A local server is already running with another model (${current.model}). Run \`xerxes local stop\` first.`,
          1,
        )
      }
      port = current.port ?? cfg.port
      progress(`Server already running on port ${port}.`)
    } else {
      progress(`Starting llama-server on 127.0.0.1:${cfg.port} (loading the model can take a while)…`)
      const result = await ports.server.start({ modelPath, port: cfg.port, ctxSize: launchCtxSize(entry, cfg) })
      port = result.port ?? cfg.port
      started = true
    }

    const baseUrl = `http://127.0.0.1:${port}/v1`
    await ports.saveProfile({
      name: LOCAL_PROFILE_NAME,
      baseUrl,
      model: entry.id,
      provider: 'custom',
      dataTrust: 'local',
      promptProfile: 'lean',
      contextLimit: launchCtxSize(entry, cfg),
    })
    return ok(
      {
        model: entry.id,
        model_path: modelPath,
        port,
        base_url: baseUrl,
        profile: LOCAL_PROFILE_NAME,
        downloaded,
        started,
      },
      `Local engine ready: ${entry.id} at ${baseUrl}\nSaved provider profile "${LOCAL_PROFILE_NAME}" (data trust: local, prompt profile: lean).`,
    )
  }

  async function start(parsed: ParsedArgs, cfg: LocalConfig, catalog: readonly CatalogEntry[]): Promise<number> {
    const current = await ports.server.status()
    if (current.running) {
      return ok({ already_running: true, ...statusJson(current) }, `Already running: ${statusLine(current)}`)
    }
    const hw = await ports.hardware()
    const { entry, path } = await locateModel(parsed, cfg, catalog, hw)
    progress(`Starting llama-server on 127.0.0.1:${cfg.port}…`)
    const result = await ports.server.start({ modelPath: path, port: cfg.port, ctxSize: launchCtxSize(entry, cfg) })
    return ok({ already_running: false, model_id: entry.id, ...statusJson(result) }, `Started: ${statusLine(result)}`)
  }

  async function stop(): Promise<number> {
    const result = await ports.server.stop()
    if (result.stopped) {
      return ok({ stopped: true, pid: result.pid, forced: result.forced }, `Stopped local server (pid ${result.pid}).`)
    }
    const text =
      result.reason === 'not-running'
        ? 'No local server is running.'
        : result.reason === 'stale'
          ? 'The recorded local server was already gone; cleaned up its record.'
          : `Pid ${result.pid} is no longer the Xerxes llama-server, so it was NOT signalled; its record was removed.`
    return ok({ stopped: false, reason: result.reason, ...(result.pid === undefined ? {} : { pid: result.pid }) }, text)
  }

  async function status(): Promise<number> {
    const current = await ports.server.status()
    const payload = statusJson(current)
    if (args.json) {
      ports.stdout(JSON.stringify({ ok: true, ...payload }) + '\n')
    } else {
      ports.stdout(statusLine(current) + '\n')
    }
    return current.running ? 0 : 1
  }

  async function doctor(parsed: ParsedArgs, cfg: LocalConfig, catalog: readonly CatalogEntry[]): Promise<number> {
    const hw = await ports.hardware()
    const selection = selectModel(hw, catalog)
    const binary = await ports.server.findBinary()
    const server = await ports.server.status()

    const entry = parsed.model === undefined ? (selection.ok ? selection.entry : undefined) : catalog.find((c) => c.id === parsed.model)
    let model: Record<string, unknown> = { id: null, present: false, verified: null, source_configured: false }
    let modelReady = false
    if (entry !== undefined) {
      const path = join(cfg.modelsDir, entry.fileName)
      if (entry.source === undefined) {
        const file = await useExistingModelFile(path).then(() => true, () => false)
        model = { id: entry.id, path, present: file, verified: null, source_configured: false }
        modelReady = file
      } else {
        const check = await verifyModelFile(path, entry.source)
        model = { id: entry.id, path, present: check.present, verified: check.verified, source_configured: true }
        modelReady = check.verified
      }
    }
    const ready = selection.ok && binary !== undefined && modelReady
    const payload = {
      ready,
      hardware: hardwareJson(hw),
      selection: selection.ok
        ? { ok: true, model: selection.entry.id, budget_bytes: selection.budgetBytes, basis: selection.basis }
        : { ok: false, reason: selection.reason, budget_bytes: selection.budgetBytes, basis: selection.basis },
      llama_server: { found: binary !== undefined, ...(binary === undefined ? {} : { path: binary }) },
      model,
      server: statusJson(server),
    }
    if (args.json) {
      ports.stdout(JSON.stringify({ ok: true, ...payload }) + '\n')
    } else {
      const lines = [
        `Hardware:     ${hardwareLine(hw)}`,
        `Model choice: ${selection.ok ? selection.entry.id : `none (${selection.reason})`}`,
        `llama-server: ${binary ?? 'NOT FOUND (see `xerxes local setup` for install options)'}`,
        `Model file:   ${
          entry === undefined
            ? 'n/a'
            : modelReady
              ? `ok (${String(model.path)})`
              : model.present === true
                ? `present but not verified (${String(model.path)})`
                : `missing (${String(model.path)})`
        }`,
        `Server:       ${statusLine(server)}`,
        ready ? 'Ready: run `xerxes local start`.' : 'Not ready yet: run `xerxes local setup`.',
      ]
      ports.stdout(lines.join('\n') + '\n')
    }
    return ready ? 0 : 1
  }
}
