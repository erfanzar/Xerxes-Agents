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
  parseCtxSize,
  parseEngine,
  parseLocalConfig,
  launchCtxSize,
  selectModel,
  type CatalogEntry,
  type LocalConfig,
  type LocalEngineName,
} from './catalog.js'
import { DownloadError, ensureModelFile, useExistingModelFile, verifyModelFile, type FetchPort } from './download.js'
import type { HardwareProfile } from './hardware.js'
import { LMS_INSTALL_HELP, type LmStudioEngine, type LmStudioModel } from './lmstudio.js'
import { windowFor, type LoadedWindows } from './loadedWindow.js'
import { ServerError, type LocalServer, type ServerStatus } from './server.js'

export const LOCAL_PROFILE_NAME = 'bonsai-local'

export const LOCAL_USAGE = `Usage: xerxes local <command> [options]

Run a Bonsai model on this machine with llama.cpp's llama-server or with LM Studio. Nothing
runs until you ask.

Commands:
  setup   Probe hardware, pick a model, start the engine with it, save the
          "${LOCAL_PROFILE_NAME}" provider profile with the window the engine loaded.
          llama-server: downloads the model first (verified).
          lmstudio: loads a model from LM Studio's library with one slot (--parallel 1).
            --engine <name>      llama-server (default) | lmstudio
            --model <id>         bonsai-9b | ternary-bonsai-2-27b (default: largest that fits)
            --model-file <path>  llama-server: use an existing .gguf instead of downloading
            --model-key <key>    lmstudio: load this LM Studio model key (see \`lms ls\`)
            --ctx-size <n>       context window to load (default: the model's launch default)
            --yes                continue even if the model is larger than this machine's memory
            --json               print one JSON object
  start   Start the engine with an already-downloaded model (same options as setup).
  stop    Stop what Xerxes started: the llama-server process, or the LM Studio model it
          loaded (and LM Studio's server, if Xerxes started it).
  status  Show whether the engine is running and healthy (exit 0 running, 1 not). --json
  doctor  Report hardware, chosen model, engine binary, model and server health. --json

Download sources for llama-server are never built in. Configure them in
~/.xerxes/daemon/config.json:
  { "runtime": { "local": { "models": { "bonsai-9b": { "url": "https://…", "sha256": "…" } } } } }
Other settings under runtime.local: engine (llama-server | lmstudio), llama_server_path,
lms_path, models_dir, port (llama-server default 11581; LM Studio keeps its own port unless
set), ctx_size (up to the model's own window, 262144 for Ternary Bonsai 2 27B).
Ternary Bonsai 2 27B's MLX pack runs only under LM Studio on Apple silicon (key
ternary-bonsai-2-27b-mlx); llama-server needs a GGUF build from Prism's llama.cpp fork.
`

export interface SaveLocalProfileInput {
  readonly name: string
  readonly baseUrl: string
  readonly model: string
  readonly provider: string
  readonly dataTrust: 'local'
  readonly promptProfile: 'lean'
  /** The window the engine loaded, so compaction and overflow rescue know the real window. */
  readonly contextLimit: number
}

export interface LocalCommandPorts {
  /** Probe the machine (wrap `probeHardware(systemHardwarePorts())` in production). */
  readonly hardware: () => Promise<HardwareProfile>
  readonly fetch: FetchPort
  /** The llama-server supervisor. */
  readonly server: LocalServer
  /** The LM Studio engine adapter; absent, `--engine lmstudio` is refused. */
  readonly lmstudio?: LmStudioEngine
  /** Ask a started engine for the window it loaded; absent, llama-server's --ctx-size is recorded. */
  readonly loadedWindow?: (baseUrl: string) => Promise<LoadedWindows>
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
  readonly engine?: LocalEngineName
  readonly model?: string
  readonly modelFile?: string
  readonly modelKey?: string
  readonly ctxSize?: number
  readonly yes: boolean
  readonly json: boolean
  readonly help: boolean
  readonly error?: string
}

function parseArgs(argv: readonly string[]): ParsedArgs {
  let sub: string | undefined
  let engine: LocalEngineName | undefined
  let model: string | undefined
  let modelFile: string | undefined
  let modelKey: string | undefined
  let ctxSize: number | undefined
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
      case '--model-key': {
        const v = value()
        if (v === undefined || v.trim() === '') return { yes, json, help, error: '--model-key needs an LM Studio model key' }
        modelKey = v.trim()
        break
      }
      case '--engine': {
        const v = value()
        try {
          engine = parseEngine(v, '--engine')
        } catch {
          return { yes, json, help, error: '--engine must be llama-server or lmstudio' }
        }
        break
      }
      case '--ctx-size': {
        const v = value()
        try {
          ctxSize = parseCtxSize(v, '--ctx-size')
        } catch (error) {
          return { yes, json, help, error: error instanceof Error ? error.message : '--ctx-size needs a number' }
        }
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
    ...(engine === undefined ? {} : { engine }),
    ...(model === undefined ? {} : { model }),
    ...(modelFile === undefined ? {} : { modelFile }),
    ...(modelKey === undefined ? {} : { modelKey }),
    ...(ctxSize === undefined ? {} : { ctxSize }),
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
    ...(status.contextLength === undefined ? {} : { context_length: status.contextLength }),
  }
}

function statusLine(status: ServerStatus, engine: LocalEngineName = 'llama-server'): string {
  if (status.running) {
    const where = engine === 'lmstudio' ? `LM Studio, port ${status.port}` : `pid ${status.pid}, port ${status.port}`
    const window = status.contextLength === undefined ? '' : `, ${status.contextLength}-token window`
    return `running (${where}, ${status.healthy ? 'healthy' : 'not healthy yet'}) model ${status.model}${window}`
  }
  if (!status.stale) return 'not running'
  return engine === 'lmstudio'
    ? 'not running (the model Xerxes loaded in LM Studio is gone)'
    : 'not running (a stale server record was found)'
}

function lmStudioMissingModel(key: string, library: readonly LmStudioModel[]): string {
  const keys = library.map((model) => model.modelKey)
  return (
    `LM Studio has no model with key "${key}". Download it in LM Studio, or pass --model-key with a key from \`lms ls\`` +
    (keys.length === 0 ? '.' : `: ${keys.join(', ')}.`)
  )
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
    const configured = parseLocalConfig(ports.config, env)
    const cfg: LocalConfig = args.ctxSize === undefined ? configured : { ...configured, ctxSize: args.ctxSize }
    const catalog = catalogWithConfig(cfg)
    // An explicit choice (flag, env, config) or, for stop/status, whichever engine has a record.
    const chosen = args.engine ?? cfg.engine
    const engine: LocalEngineName = chosen ?? 'llama-server'
    if (engine === 'llama-server' && args.modelKey !== undefined) {
      return fail('USAGE', '--model-key applies to --engine lmstudio; llama-server takes --model or --model-file.', 2)
    }
    if (engine === 'lmstudio' && args.modelFile !== undefined) {
      return fail('USAGE', "--model-file applies to llama-server; LM Studio loads from its own library (use --model-key).", 2)
    }

    switch (args.sub) {
      case 'setup':
        return engine === 'lmstudio' ? await setupLmStudio(args, cfg, catalog, lmStudio()) : await setup(args, cfg, catalog)
      case 'start':
        return engine === 'lmstudio' ? await startLmStudio(args, cfg, catalog, lmStudio()) : await start(args, cfg, catalog)
      case 'stop': {
        const recorded = await recordedEngine(chosen)
        return recorded.engine === 'lmstudio' ? await stopLmStudio(lmStudio()) : await stop()
      }
      case 'status': {
        const recorded = await recordedEngine(chosen)
        return report(recorded.status, recorded.engine)
      }
      case 'doctor':
        return engine === 'lmstudio' ? await doctorLmStudio(args, cfg, catalog, lmStudio()) : await doctor(args, cfg, catalog)
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
    // What llama-server actually loaded (/props); a server that was already
    // running may hold a different --ctx-size than this run's settings.
    const contextLimit = (await probedWindow(baseUrl, entry.id)) ?? launchCtxSize(entry, cfg)
    await ports.saveProfile({
      name: LOCAL_PROFILE_NAME,
      baseUrl,
      model: entry.id,
      provider: 'custom',
      dataTrust: 'local',
      promptProfile: 'lean',
      contextLimit,
    })
    return ok(
      {
        engine: 'llama-server',
        model: entry.id,
        model_path: modelPath,
        port,
        base_url: baseUrl,
        profile: LOCAL_PROFILE_NAME,
        context_length: contextLimit,
        downloaded,
        started,
      },
      `Local engine ready: ${entry.id} at ${baseUrl} (${contextLimit}-token window)\nSaved provider profile "${LOCAL_PROFILE_NAME}" (data trust: local, prompt profile: lean).`,
    )
  }

  async function probedWindow(baseUrl: string, model: string): Promise<number | undefined> {
    if (ports.loadedWindow === undefined) return undefined
    const result = await ports.loadedWindow(baseUrl)
    const window = windowFor(result, model)
    if (window === undefined) {
      progress(`Could not read the loaded window from the server (${result.ok ? `${model} is not loaded` : result.reason}); recording the launch setting.`)
    }
    return window
  }

  function lmStudio(): LmStudioEngine {
    if (ports.lmstudio === undefined) throw new CommandFailure('UNSUPPORTED', 'The LM Studio engine is not available in this host.')
    return ports.lmstudio
  }

  /** The engine stop/status act on: the explicit choice, else whichever engine Xerxes has a record for. */
  async function recordedEngine(explicit: LocalEngineName | undefined): Promise<{ engine: LocalEngineName; status: ServerStatus }> {
    if (explicit === 'lmstudio') return { engine: 'lmstudio', status: await lmStudio().status() }
    const llama = await ports.server.status()
    if (explicit === 'llama-server' || llama.running || llama.stale || ports.lmstudio === undefined) {
      return { engine: 'llama-server', status: llama }
    }
    const lms = await ports.lmstudio.status()
    return lms.running || lms.stale ? { engine: 'lmstudio', status: lms } : { engine: 'llama-server', status: llama }
  }

  /**
   * What LM Studio should load: a `--model-key` the user named, or the
   * catalog entry with an LM Studio build that fits this machine.
   */
  function lmStudioTarget(
    parsed: ParsedArgs,
    cfg: LocalConfig,
    catalog: readonly CatalogEntry[],
    hw: HardwareProfile,
  ): { readonly modelKey: string; readonly entry?: CatalogEntry; readonly contextLength?: number } {
    if (parsed.modelKey !== undefined) {
      const entry =
        parsed.model === undefined
          ? catalog.find((candidate) => candidate.lmstudio?.modelKey === parsed.modelKey)
          : catalog.find((candidate) => candidate.id === parsed.model)
      if (parsed.model !== undefined && entry === undefined) {
        throw new CommandFailure('UNKNOWN_MODEL', `Unknown model "${parsed.model}". Known models: ${catalog.map((c) => c.id).join(', ')}.`)
      }
      const contextLength = entry === undefined ? cfg.ctxSize : launchCtxSize(entry, cfg)
      return { modelKey: parsed.modelKey, ...(entry === undefined ? {} : { entry }), ...(contextLength === undefined ? {} : { contextLength }) }
    }
    if (parsed.model !== undefined) {
      const named = catalog.find((candidate) => candidate.id === parsed.model)
      if (named !== undefined && named.lmstudio === undefined) {
        throw new CommandFailure(
          'NO_LMSTUDIO_BUILD',
          `${named.id} has no LM Studio model key in the catalog. Pass --model-key with its key from \`lms ls\`.`,
        )
      }
    }
    const buildable = catalog.flatMap((entry) =>
      entry.lmstudio === undefined ? [] : [{ ...entry, minMemoryBytes: entry.lmstudio.minMemoryBytes }],
    )
    const chosen = chooseEntry(parsed.model, buildable, hw, parsed.yes)
    if (!chosen.ok) {
      throw new CommandFailure(chosen.code, `${chosen.message} A smaller LM Studio model can be loaded with --model-key <key from \`lms ls\`>.`)
    }
    if (chosen.warning !== undefined) progress(chosen.warning)
    const entry = chosen.entry
    const modelKey = entry.lmstudio?.modelKey
    if (modelKey === undefined) throw new CommandFailure('NO_LMSTUDIO_BUILD', `${entry.id} has no LM Studio model key.`)
    return { modelKey, entry, contextLength: launchCtxSize(entry, cfg) }
  }

  /** Resolve the target and check `lms` and the model key before anything is started. */
  async function prepareLmStudio(
    parsed: ParsedArgs,
    cfg: LocalConfig,
    catalog: readonly CatalogEntry[],
    lm: LmStudioEngine,
  ): Promise<{ readonly modelKey: string; readonly entry?: CatalogEntry; readonly contextLength?: number }> {
    progress('Probing hardware…')
    const hw = await ports.hardware()
    progress(hardwareLine(hw))
    const target = lmStudioTarget(parsed, cfg, catalog, hw)
    if ((await lm.findBinary()) === undefined) throw new CommandFailure('BINARY_NOT_FOUND', LMS_INSTALL_HELP)
    const library = await lm.listModels()
    const found = library.find((model) => model.modelKey === target.modelKey)
    if (found === undefined) throw new CommandFailure('MODEL_MISSING', lmStudioMissingModel(target.modelKey, library))
    if (target.contextLength !== undefined && found.maxContextLength !== undefined && target.contextLength > found.maxContextLength) {
      throw new CommandFailure(
        'INVALID_CONFIG',
        `The ${target.contextLength}-token window exceeds ${target.modelKey}'s ${found.maxContextLength}-token maximum.`,
        2,
      )
    }
    progress(`Model: ${target.modelKey}${target.entry === undefined ? '' : ` (${target.entry.label})`}`)
    return target
  }

  async function setupLmStudio(parsed: ParsedArgs, cfg: LocalConfig, catalog: readonly CatalogEntry[], lm: LmStudioEngine): Promise<number> {
    const target = await prepareLmStudio(parsed, cfg, catalog, lm)
    const current = await lm.status()
    let result: ServerStatus
    let started = false
    if (current.running) {
      if (current.model !== target.modelKey || (target.contextLength !== undefined && current.contextLength !== target.contextLength)) {
        return fail(
          'ALREADY_RUNNING',
          `Xerxes already has ${current.model} loaded in LM Studio${current.contextLength === undefined ? '' : ` with a ${current.contextLength}-token window`}. ` +
            'Run `xerxes local stop --engine lmstudio` first.',
          1,
        )
      }
      progress(`${target.modelKey} is already loaded in LM Studio on port ${current.port}.`)
      result = current
    } else {
      result = await lm.start({
        modelKey: target.modelKey,
        identifier: target.modelKey,
        ...(target.contextLength === undefined ? {} : { contextLength: target.contextLength }),
        ...(cfg.portExplicit ? { port: cfg.port } : {}),
        onProgress: progress,
      })
      started = true
    }
    const port = result.port
    const window = result.contextLength
    if (port === undefined || window === undefined) {
      throw new CommandFailure('LOAD_FAILED', 'LM Studio did not report the port and window of the loaded model.')
    }
    const baseUrl = `http://127.0.0.1:${port}/v1`
    await ports.saveProfile({
      name: LOCAL_PROFILE_NAME,
      baseUrl,
      model: target.modelKey,
      provider: 'custom',
      dataTrust: 'local',
      promptProfile: 'lean',
      contextLimit: window,
    })
    return ok(
      {
        engine: 'lmstudio',
        model: target.modelKey,
        ...(target.entry === undefined ? {} : { model_id: target.entry.id }),
        port,
        base_url: baseUrl,
        profile: LOCAL_PROFILE_NAME,
        context_length: window,
        started,
      },
      `Local engine ready: ${target.modelKey} in LM Studio at ${baseUrl} (${window}-token window, one slot)\n` +
        `Saved provider profile "${LOCAL_PROFILE_NAME}" (data trust: local, prompt profile: lean).`,
    )
  }

  async function startLmStudio(parsed: ParsedArgs, cfg: LocalConfig, catalog: readonly CatalogEntry[], lm: LmStudioEngine): Promise<number> {
    const current = await lm.status()
    if (current.running) {
      return ok({ already_running: true, engine: 'lmstudio', ...statusJson(current) }, `Already running: ${statusLine(current, 'lmstudio')}`)
    }
    const target = await prepareLmStudio(parsed, cfg, catalog, lm)
    const result = await lm.start({
      modelKey: target.modelKey,
      identifier: target.modelKey,
      ...(target.contextLength === undefined ? {} : { contextLength: target.contextLength }),
      ...(cfg.portExplicit ? { port: cfg.port } : {}),
      onProgress: progress,
    })
    return ok(
      { already_running: false, engine: 'lmstudio', ...(target.entry === undefined ? {} : { model_id: target.entry.id }), ...statusJson(result) },
      `Started: ${statusLine(result, 'lmstudio')}`,
    )
  }

  async function stopLmStudio(lm: LmStudioEngine): Promise<number> {
    const result = await lm.stop()
    if (result.stopped) {
      const text = result.unloaded
        ? `Unloaded ${result.model} from LM Studio${result.serverStopped ? " and stopped LM Studio's server" : ''}.`
        : `${result.model} was already loaded before Xerxes used it, so it stays loaded; Xerxes' record was removed.`
      return ok(
        { stopped: true, engine: 'lmstudio', model: result.model, unloaded: result.unloaded, server_stopped: result.serverStopped },
        text,
      )
    }
    const text =
      result.reason === 'not-running'
        ? 'Xerxes has no model loaded in LM Studio.'
        : 'The model Xerxes loaded in LM Studio was already gone; cleaned up its record.'
    return ok({ stopped: false, engine: 'lmstudio', reason: result.reason }, text)
  }

  async function doctorLmStudio(parsed: ParsedArgs, cfg: LocalConfig, catalog: readonly CatalogEntry[], lm: LmStudioEngine): Promise<number> {
    const hw = await ports.hardware()
    let target: { readonly modelKey: string; readonly entry?: CatalogEntry; readonly contextLength?: number } | undefined
    let selectionError: string | undefined
    try {
      target = lmStudioTarget(parsed, cfg, catalog, hw)
    } catch (error) {
      if (!(error instanceof CommandFailure)) throw error
      selectionError = error.message
    }
    const binary = await lm.findBinary()
    const library = binary === undefined ? undefined : await lm.listModels()
    const serverInfo = binary === undefined ? undefined : await lm.serverInfo()
    const found = target === undefined ? undefined : library?.find((model) => model.modelKey === target.modelKey)
    const server = await lm.status()
    const ready = target !== undefined && binary !== undefined && found !== undefined
    const payload = {
      engine: 'lmstudio',
      ready,
      hardware: hardwareJson(hw),
      selection:
        target === undefined
          ? { ok: false, reason: selectionError ?? 'no model' }
          : {
              ok: true,
              model_key: target.modelKey,
              ...(target.entry === undefined ? {} : { model: target.entry.id }),
              ...(target.contextLength === undefined ? {} : { context_length: target.contextLength }),
            },
      lms: { found: binary !== undefined, ...(binary === undefined ? {} : { path: binary }) },
      lmstudio_server: serverInfo === undefined ? null : { running: serverInfo.running, ...(serverInfo.port === undefined ? {} : { port: serverInfo.port }) },
      model:
        target === undefined
          ? { key: null, present: false }
          : {
              key: target.modelKey,
              present: found !== undefined,
              ...(found?.maxContextLength === undefined ? {} : { max_context_length: found.maxContextLength }),
            },
      server: statusJson(server),
    }
    if (args.json) {
      ports.stdout(JSON.stringify({ ok: true, ...payload }) + '\n')
    } else {
      const lines = [
        `Hardware:     ${hardwareLine(hw)}`,
        `Model choice: ${target === undefined ? `none (${selectionError ?? 'no model'})` : `${target.modelKey}${target.contextLength === undefined ? '' : `, ${target.contextLength}-token window`}`}`,
        `lms:          ${binary ?? `NOT FOUND (${LMS_INSTALL_HELP})`}`,
        `LM Studio:    ${serverInfo === undefined ? 'n/a' : serverInfo.running ? `server running on port ${serverInfo.port ?? '?'}` : 'server not running'}`,
        `Model:        ${target === undefined ? 'n/a' : found !== undefined ? `in LM Studio's library (${target.modelKey})` : `missing from LM Studio's library (${target.modelKey})`}`,
        `Xerxes load:  ${statusLine(server, 'lmstudio')}`,
        ready ? 'Ready: run `xerxes local setup --engine lmstudio`.' : 'Not ready yet.',
      ]
      ports.stdout(lines.join('\n') + '\n')
    }
    return ready ? 0 : 1
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

  function report(current: ServerStatus, engine: LocalEngineName): number {
    const payload = { ...statusJson(current), ...(engine === 'lmstudio' ? { engine } : {}) }
    if (args.json) {
      ports.stdout(JSON.stringify({ ok: true, ...payload }) + '\n')
    } else {
      ports.stdout(statusLine(current, engine) + '\n')
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
