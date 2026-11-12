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
import { huggingFaceUrl } from '../llms/modelCards.js'
import {
  catalogWithConfig,
  formatGiB,
  memoryBudget,
  parseCtxSize,
  parseEngine,
  parseLocalConfig,
  launchCtxSize,
  selectModel,
  type Catalog,
  type CatalogEntry,
  type LocalConfig,
  type LocalEngineName,
} from './catalog.js'
import { DownloadError, ensureModelFile, useExistingModelFile, verifyModelFile, type FetchPort } from './download.js'
import { checkLlamaCpp, checkMlxLm, parseLlamaCppVersion, parseMlxLmVersion, type VersionCheck } from './engineVersion.js'
import type { GgufAttention } from './ggufMetadata.js'
import type { HardwareProfile } from './hardware.js'
import { LMS_INSTALL_HELP, type LmStudioEngine, type LmStudioModel } from './lmstudio.js'
import { windowFor, type LoadedWindows } from './loadedWindow.js'
import { fitWindow } from './memoryFit.js'
import { BINARY_INSTALL_HELP, MLX_INSTALL_HELP, ServerError, type LocalServer, type ServerStatus } from './server.js'

export const LOCAL_PROFILE_NAME = 'bonsai-local'

export const NO_ENGINE_HELP =
  'No local engine was found. Install one, in the order Xerxes prefers them: llama.cpp (`brew install llama.cpp`, ' +
  'v0.7.0 or newer for Ternary Bonsai 2 27B), mlx-lm on Apple silicon (`uv tool install mlx-lm`, 0.32.2 or newer), ' +
  'or LM Studio (https://lmstudio.ai). Then run `xerxes local setup` again.'

export const LOCAL_USAGE = `Usage: xerxes local <command> [options]

Run a Bonsai model on this machine with llama.cpp's llama-server, mlx-lm's mlx_lm.server or
LM Studio. Nothing runs until you ask.

Commands:
  setup   Probe hardware, pick a model, start the engine with it, save the
          "${LOCAL_PROFILE_NAME}" provider profile with the window the engine loaded.
          llama-server and mlx: download the model first, each file checked by its sha256.
          lmstudio: loads a model from LM Studio's library with one slot (--parallel 1).
            --engine <name>      llama-server | mlx | lmstudio (default: the first one installed,
                                 in that order; mlx only on Apple silicon)
            --model <id>         bonsai-9b | ternary-bonsai-2-27b (default: largest that fits)
            --model-file <path>  llama-server: use an existing .gguf instead of downloading
            --model-key <key>    lmstudio: load this LM Studio model key (see \`lms ls\`)
            --ctx-size <n>       context window to load (default: sized from free memory)
            --yes                continue even if the model is larger than this machine's memory,
                                 or the engine's version is older than the model needs or unreadable
            --json               print one JSON object
  start   Start the engine with an already-downloaded model (same options as setup).
  stop    Stop what Xerxes started: the llama-server or mlx_lm.server process, or the LM Studio
          model it loaded (and LM Studio's server, if Xerxes started it).
  status  Show whether the engine is running and healthy (exit 0 running, 1 not). --json
  doctor  Report hardware, chosen model, engine binary and version, model and server health. --json

Ternary Bonsai 2 27B downloads from Prism's Hugging Face repos at a pinned commit: the Q2_0
GGUF for llama-server (stock llama.cpp v0.7.0 / build b11555 or newer) and the 2-bit MLX pack
for mlx_lm.server (mlx-lm 0.32.2 or newer). Other models need a source in
~/.xerxes/daemon/config.json:
  { "runtime": { "local": { "models": { "bonsai-9b": { "url": "https://…", "sha256": "…" } } } } }
Other settings under runtime.local: engine (llama-server | mlx | lmstudio), llama_server_path,
mlx_server_path, lms_path, models_dir, port (default 11581; LM Studio keeps its own port unless
set), ctx_size (up to the model's own window, 262144 for Ternary Bonsai 2 27B).
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
  /** The mlx_lm.server supervisor; absent, `--engine mlx` is refused. */
  readonly mlx?: LocalServer
  /** The LM Studio engine adapter; absent, `--engine lmstudio` is refused. */
  readonly lmstudio?: LmStudioEngine
  /** The model catalog; defaults to the model cards with the user's sources. */
  readonly catalog?: (config: LocalConfig) => Catalog
  /** Ask a started engine for the window it loaded; absent, llama-server's --ctx-size is recorded. */
  readonly loadedWindow?: (baseUrl: string) => Promise<LoadedWindows>
  /**
   * Size llama-server's window from free memory (`memoryFit.ts`) when no
   * ctx_size is set; absent, the catalog's fixed window is used.
   */
  readonly memoryFit?: {
    freeBytes(hw: HardwareProfile): Promise<number | undefined>
    /** The model file's attention shape; rejects when its header cannot be read. */
    attention(modelPath: string): Promise<GgufAttention>
    weightsBytes(modelPath: string): number
    /** An MLX model folder's attention shape, from its config.json; rejects when it cannot be read. */
    mlxAttention?(modelDir: string): Promise<GgufAttention>
  }
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
          return { yes, json, help, error: '--engine must be llama-server, mlx or lmstudio' }
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

function engineLabel(engine: LocalEngineName): string {
  return engine === 'mlx' ? 'mlx_lm.server' : engine === 'lmstudio' ? 'LM Studio' : 'llama-server'
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
    `No download source is known for ${entry.id}: its maker has published no GGUF build with a checksum. ` +
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
    const catalog = ports.catalog?.(cfg) ?? catalogWithConfig(cfg)
    // An explicit choice (flag, env, config) or, for stop/status, whichever engine has a record.
    const chosen = args.engine ?? cfg.engine

    switch (args.sub) {
      case 'setup':
      case 'start':
      case 'doctor': {
        const engine = chosen ?? await defaultEngine(args)
        if (engine !== 'lmstudio' && args.modelKey !== undefined) {
          return fail('USAGE', `--model-key applies to --engine lmstudio; ${engineLabel(engine)} takes --model.`, 2)
        }
        if (engine !== 'llama-server' && args.modelFile !== undefined) {
          return fail('USAGE', `--model-file applies to llama-server; ${engineLabel(engine)} ${engine === 'lmstudio' ? 'loads from its own library (use --model-key)' : 'downloads its model folder'}.`, 2)
        }
        if (chosen === undefined) progress(`Engine: ${engineLabel(engine)} (set --engine or runtime.local.engine to choose another)`)
        if (engine === 'lmstudio') {
          const lm = lmStudio()
          return args.sub === 'setup' ? await setupLmStudio(args, cfg, catalog, lm)
            : args.sub === 'start' ? await startLmStudio(args, cfg, catalog, lm)
              : await doctorLmStudio(args, cfg, catalog, lm)
        }
        if (engine === 'mlx') {
          const server = mlxServer()
          return args.sub === 'setup' ? await setupMlx(args, cfg, catalog, server)
            : args.sub === 'start' ? await startMlx(args, cfg, catalog, server)
              : await doctorMlx(args, cfg, catalog, server)
        }
        return args.sub === 'setup' ? await setup(args, cfg, catalog, chosen === undefined)
          : args.sub === 'start' ? await start(args, cfg, catalog)
            : await doctor(args, cfg, catalog)
      }
      case 'stop': {
        const recorded = await recordedEngine(chosen)
        return recorded.engine === 'lmstudio' ? await stopLmStudio(lmStudio())
          : await stop(recorded.engine === 'mlx' ? mlxServer() : ports.server)
      }
      case 'status': {
        const recorded = await recordedEngine(chosen)
        return report(recorded.status, recorded.engine)
      }
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

  /**
   * llama-server's --ctx-size: ctx_size when set, else fitted to the memory
   * free now, else the catalog's window. Says which, and why.
   */
  async function llamaWindow(entry: CatalogEntry, cfg: LocalConfig, modelPath: string, hw: HardwareProfile): Promise<number> {
    const fixed = launchCtxSize(entry, cfg)
    if (cfg.ctxSize !== undefined || ports.memoryFit === undefined) return fixed
    let shape: GgufAttention
    try {
      shape = await ports.memoryFit.attention(modelPath)
    } catch (error) {
      progress(`Using the ${fixed}-token window: ${error instanceof Error ? error.message : String(error)}`)
      return fixed
    }
    const free = await ports.memoryFit.freeBytes(hw)
    if (free === undefined) {
      progress(`Using the ${fixed}-token window: free memory could not be measured.`)
      return fixed
    }
    const fit = fitWindow({
      freeBytes: free,
      weightsBytes: ports.memoryFit.weightsBytes(modelPath),
      kvBytesPerToken: shape.kvBytesPerToken,
      maxWindow: entry.nativeContext ?? shape.contextLength ?? fixed,
    })
    progress(fit.fits
      ? `Window: ${fit.ctxSize} tokens, sized from ${formatGiB(free)} free (${Math.round(shape.kvBytesPerToken / 1024)} KiB of cache per token). Set runtime.local.ctx_size to choose another.`
      : `Only ${formatGiB(free)} is free, so the window is the ${fit.ctxSize}-token minimum and the model may not load. Close other apps, or set runtime.local.ctx_size.`)
    return fit.ctxSize
  }

  async function setup(parsed: ParsedArgs, cfg: LocalConfig, catalog: readonly CatalogEntry[], auto = false): Promise<number> {
    progress('Probing hardware…')
    const hw = await ports.hardware()
    progress(hardwareLine(hw))
    const chosen = chooseEntry(parsed.model, catalog, hw, parsed.yes)
    if (!chosen.ok) return fail(chosen.code, chosen.message, 1, { hardware: hardwareJson(hw) })
    if (chosen.warning !== undefined) progress(chosen.warning)
    const entry = chosen.entry
    progress(`Model: ${entry.id}`)
    // Before anything is downloaded: the engine must be installed and new enough.
    if ((await ports.server.findBinary()) === undefined) {
      throw new CommandFailure('BINARY_NOT_FOUND', auto ? NO_ENGINE_HELP : BINARY_INSTALL_HELP)
    }
    await requireLlamaCppVersion(entry, parsed)

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
    let ctxSize = launchCtxSize(entry, cfg)
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
      ctxSize = await llamaWindow(entry, cfg, modelPath, hw)
      const result = await ports.server.start({ modelPath, port: cfg.port, ctxSize })
      port = result.port ?? cfg.port
      started = true
    }

    const baseUrl = `http://127.0.0.1:${port}/v1`
    // What llama-server actually loaded (/props); a server that was already
    // running may hold a different --ctx-size than this run's settings.
    const contextLimit = (await probedWindow(baseUrl, entry.id)) ?? ctxSize
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

  function mlxServer(): LocalServer {
    if (ports.mlx === undefined) throw new CommandFailure('UNSUPPORTED', 'The mlx engine is not available in this host.')
    return ports.mlx
  }

  /**
   * The engine setup, start and doctor use when none is chosen: a flag that
   * belongs to one engine picks it; otherwise the first one installed, in the
   * order llama-server, mlx (Apple silicon only), LM Studio. With none
   * installed it is llama-server, whose setup says how to install each.
   */
  async function defaultEngine(parsed: ParsedArgs): Promise<LocalEngineName> {
    if (parsed.modelKey !== undefined) return 'lmstudio'
    if (parsed.modelFile !== undefined) return 'llama-server'
    if ((await ports.server.findBinary()) !== undefined) return 'llama-server'
    if (ports.mlx !== undefined && (await ports.hardware()).appleSilicon && (await ports.mlx.findBinary()) !== undefined) return 'mlx'
    if (ports.lmstudio !== undefined && (await ports.lmstudio.findBinary()) !== undefined) return 'lmstudio'
    return 'llama-server'
  }

  /** The engine stop/status act on: the explicit choice, else whichever engine Xerxes has a record for. */
  async function recordedEngine(explicit: LocalEngineName | undefined): Promise<{ engine: LocalEngineName; status: ServerStatus }> {
    if (explicit === 'lmstudio') return { engine: 'lmstudio', status: await lmStudio().status() }
    if (explicit === 'mlx') return { engine: 'mlx', status: await mlxServer().status() }
    const llama = await ports.server.status()
    if (explicit === 'llama-server' || llama.running || llama.stale) return { engine: 'llama-server', status: llama }
    if (ports.mlx !== undefined) {
      const mlx = await ports.mlx.status()
      if (mlx.running || mlx.stale) return { engine: 'mlx', status: mlx }
    }
    if (ports.lmstudio !== undefined) {
      const lms = await ports.lmstudio.status()
      if (lms.running || lms.stale) return { engine: 'lmstudio', status: lms }
    }
    return { engine: 'llama-server', status: llama }
  }

  /**
   * Refuse an engine older than the model needs, or one whose version cannot
   * be read, unless --yes: a llama.cpp build from before a model's runtime
   * loads the file and writes gibberish.
   */
  function requireVersion(label: string, result: VersionCheck, need: string, upgrade: string, parsed: ParsedArgs): void {
    if (result.ok) {
      progress(`${label} ${result.found}: new enough (this model needs ${need}).`)
      return
    }
    const problem = result.reason === 'too-old'
      ? `${label} ${result.found} is older than this model needs (${need}).`
      : `Could not read ${label}'s version; this model needs ${need}.`
    if (parsed.yes) {
      progress(`${problem} Continuing because of --yes.`)
      return
    }
    throw new CommandFailure(
      result.reason === 'too-old' ? 'ENGINE_TOO_OLD' : 'ENGINE_VERSION_UNKNOWN',
      `${problem} ${upgrade} If this build does run the model (Prism's own builds report other numbers), pass --yes.`,
    )
  }

  async function requireLlamaCppVersion(entry: CatalogEntry, parsed: ParsedArgs): Promise<void> {
    const minimum = entry.minLlamaCpp
    if (minimum === undefined) return
    const text = await ports.server.version()
    requireVersion(
      'llama-server',
      text === undefined ? { ok: false, reason: 'unknown' } : checkLlamaCpp(parseLlamaCppVersion(text), minimum),
      `llama.cpp build b${minimum.build} (release v${minimum.version}) or newer; older builds load it and write gibberish`,
      'Update llama.cpp (`brew upgrade llama.cpp`, or a release from https://github.com/ggml-org/llama.cpp/releases).',
      parsed,
    )
  }

  async function requireMlxLmVersion(entry: CatalogEntry, server: LocalServer, parsed: ParsedArgs): Promise<void> {
    const minimum = entry.mlx?.minMlxLm
    if (minimum === undefined) return
    const text = await server.version()
    requireVersion(
      'mlx-lm',
      checkMlxLm(text === undefined ? undefined : parseMlxLmVersion(text), minimum),
      `mlx-lm ${minimum} or newer`,
      'Update it with `uv tool upgrade mlx-lm` (or `pip install -U mlx-lm`).',
      parsed,
    )
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

  /** The model with an MLX build that fits this machine, or the one named with --model. */
  function mlxEntry(parsed: ParsedArgs, catalog: readonly CatalogEntry[], hw: HardwareProfile): CatalogEntry {
    if (!hw.appleSilicon && !parsed.yes) {
      throw new CommandFailure('UNSUPPORTED', 'mlx_lm.server runs on Apple silicon only. Use --engine llama-server on this machine.')
    }
    if (parsed.model !== undefined) {
      const named = catalog.find(candidate => candidate.id === parsed.model)
      if (named !== undefined && named.mlx === undefined) {
        throw new CommandFailure('NO_MLX_BUILD', `${named.id} has no MLX build in the catalog. Use --engine llama-server or lmstudio.`)
      }
    }
    const buildable = catalog.flatMap(entry => entry.mlx === undefined ? [] : [{ ...entry, minMemoryBytes: entry.mlx.minMemoryBytes }])
    const chosen = chooseEntry(parsed.model, buildable, hw, parsed.yes)
    if (!chosen.ok) throw new CommandFailure(chosen.code, chosen.message)
    if (chosen.warning !== undefined) progress(chosen.warning)
    return chosen.entry
  }

  /** The MLX model folder; mlx_lm.server knows the model by this path, so it is also the profile's model name. */
  function mlxDir(entry: CatalogEntry, cfg: LocalConfig): string {
    return join(cfg.modelsDir, entry.mlx!.dirName)
  }

  /** Download (or reuse) every file of the entry's MLX build, each checked against its pinned sha256. */
  async function ensureMlxFiles(entry: CatalogEntry, cfg: LocalConfig, parsed: ParsedArgs): Promise<boolean> {
    const build = entry.mlx!
    const dir = mlxDir(entry, cfg)
    const missing: string[] = []
    for (const file of build.files) {
      const check = await verifyModelFile(join(dir, file.path), { url: huggingFaceUrl(build, file.path), sha256: file.sha256, bytes: file.bytes })
      if (!check.verified) missing.push(file.path)
    }
    if (missing.length === 0) {
      progress('Model already downloaded and verified.')
      return false
    }
    if (ports.confirm !== undefined && !parsed.yes) {
      const bytes = build.files.filter(file => missing.includes(file.path)).reduce((sum, file) => sum + file.bytes, 0)
      const approved = await ports.confirm(`Download ${entry.id}'s MLX build (${formatGiB(bytes)}) from huggingface.co (${build.repo})?`)
      if (!approved) throw new CommandFailure('CANCELLED', 'Setup cancelled; nothing was downloaded.')
    }
    for (const file of build.files) {
      if (!missing.includes(file.path)) continue
      let lastStep = -1
      await ensureModelFile({
        destDir: dir,
        fileName: file.path,
        source: { url: huggingFaceUrl(build, file.path), sha256: file.sha256, bytes: file.bytes },
        fetch: ports.fetch,
        ...(ports.signal === undefined ? {} : { signal: ports.signal }),
        onProgress: ({ receivedBytes, totalBytes }) => {
          if (totalBytes === undefined || totalBytes < 100 * 1024 * 1024) return
          const step = Math.floor((receivedBytes / totalBytes) * 20)
          if (step === lastStep) return
          lastStep = step
          progress(`Downloading ${file.path}: ${step * 5}% (${formatGiB(receivedBytes)} / ${formatGiB(totalBytes)})`)
        },
      })
    }
    progress(`Download verified (sha256 ok for ${missing.length} file${missing.length === 1 ? '' : 's'}).`)
    return true
  }

  /** The window recorded for an MLX model: ctx_size, else fitted to free memory, else the catalog's. */
  async function mlxWindow(entry: CatalogEntry, cfg: LocalConfig, dir: string, hw: HardwareProfile): Promise<number> {
    const fixed = launchCtxSize(entry, cfg)
    const read = ports.memoryFit?.mlxAttention
    if (cfg.ctxSize !== undefined || ports.memoryFit === undefined || read === undefined) return fixed
    let shape: GgufAttention
    try {
      shape = await read(dir)
    } catch (error) {
      progress(`Using the ${fixed}-token window: ${error instanceof Error ? error.message : String(error)}`)
      return fixed
    }
    const free = await ports.memoryFit.freeBytes(hw)
    if (free === undefined) {
      progress(`Using the ${fixed}-token window: free memory could not be measured.`)
      return fixed
    }
    const fit = fitWindow({
      freeBytes: free,
      weightsBytes: entry.mlx!.files.reduce((sum, file) => sum + file.bytes, 0),
      kvBytesPerToken: shape.kvBytesPerToken,
      maxWindow: entry.nativeContext ?? shape.contextLength ?? fixed,
    })
    progress(fit.fits
      ? `Window: ${fit.ctxSize} tokens, sized from ${formatGiB(free)} free (${Math.round(shape.kvBytesPerToken / 1024)} KiB of cache per token). Set runtime.local.ctx_size to choose another.`
      : `Only ${formatGiB(free)} is free, so the window is the ${fit.ctxSize}-token minimum and the model may not load. Close other apps, or set runtime.local.ctx_size.`)
    return fit.ctxSize
  }

  async function setupMlx(parsed: ParsedArgs, cfg: LocalConfig, catalog: readonly CatalogEntry[], server: LocalServer): Promise<number> {
    progress('Probing hardware…')
    const hw = await ports.hardware()
    progress(hardwareLine(hw))
    const entry = mlxEntry(parsed, catalog, hw)
    progress(`Model: ${entry.id} (MLX)`)
    if ((await server.findBinary()) === undefined) throw new CommandFailure('BINARY_NOT_FOUND', MLX_INSTALL_HELP)
    await requireMlxLmVersion(entry, server, parsed)
    const downloaded = await ensureMlxFiles(entry, cfg, parsed)
    const dir = mlxDir(entry, cfg)

    const current = await server.status()
    let started = false
    let port = cfg.port
    let window = launchCtxSize(entry, cfg)
    if (current.running) {
      if (current.model !== dir) {
        return fail('ALREADY_RUNNING', `mlx_lm.server is already running with another model (${current.model}). Run \`xerxes local stop\` first.`, 1)
      }
      port = current.port ?? cfg.port
      progress(`mlx_lm.server already running on port ${port}.`)
    } else {
      window = await mlxWindow(entry, cfg, dir, hw)
      progress(`Starting mlx_lm.server on 127.0.0.1:${cfg.port} (loading the model can take a while)…`)
      const result = await server.start({ modelPath: dir, port: cfg.port, ctxSize: window })
      port = result.port ?? cfg.port
      started = true
    }
    const baseUrl = `http://127.0.0.1:${port}/v1`
    await ports.saveProfile({
      name: LOCAL_PROFILE_NAME,
      baseUrl,
      // mlx_lm.server loads any other model name as a new model, so requests must name this folder.
      model: dir,
      provider: 'custom',
      dataTrust: 'local',
      promptProfile: 'lean',
      contextLimit: window,
    })
    return ok(
      { engine: 'mlx', model: dir, model_id: entry.id, port, base_url: baseUrl, profile: LOCAL_PROFILE_NAME, context_length: window, downloaded, started },
      `Local engine ready: ${entry.id} in mlx_lm.server at ${baseUrl} (${window}-token window)\n` +
        `Saved provider profile "${LOCAL_PROFILE_NAME}" (data trust: local, prompt profile: lean).`,
    )
  }

  async function startMlx(parsed: ParsedArgs, cfg: LocalConfig, catalog: readonly CatalogEntry[], server: LocalServer): Promise<number> {
    const current = await server.status()
    if (current.running) {
      return ok({ already_running: true, engine: 'mlx', ...statusJson(current) }, `Already running: ${statusLine(current, 'mlx')}`)
    }
    const hw = await ports.hardware()
    const entry = mlxEntry(parsed, catalog, hw)
    const dir = mlxDir(entry, cfg)
    for (const file of entry.mlx!.files) {
      const check = await verifyModelFile(join(dir, file.path), { url: huggingFaceUrl(entry.mlx!, file.path), sha256: file.sha256, bytes: file.bytes })
      if (!check.present) throw new CommandFailure('MODEL_MISSING', `${entry.id}'s MLX build is not downloaded (${join(dir, file.path)} is missing). Run \`xerxes local setup --engine mlx\` first.`)
      if (!check.verified) throw new CommandFailure('MODEL_NOT_VERIFIED', `${join(dir, file.path)} does not match its pinned sha256. Delete it and run \`xerxes local setup --engine mlx\` again.`)
    }
    await requireMlxLmVersion(entry, server, parsed)
    progress(`Starting mlx_lm.server on 127.0.0.1:${cfg.port}…`)
    const result = await server.start({ modelPath: dir, port: cfg.port, ctxSize: await mlxWindow(entry, cfg, dir, hw) })
    return ok({ already_running: false, engine: 'mlx', model_id: entry.id, ...statusJson(result) }, `Started: ${statusLine(result, 'mlx')}`)
  }

  async function doctorMlx(parsed: ParsedArgs, cfg: LocalConfig, catalog: readonly CatalogEntry[], server: LocalServer): Promise<number> {
    const hw = await ports.hardware()
    let entry: CatalogEntry | undefined
    let selectionError: string | undefined
    try {
      entry = mlxEntry(parsed, catalog, hw)
    } catch (error) {
      if (!(error instanceof CommandFailure)) throw error
      selectionError = error.message
    }
    const binary = await server.findBinary()
    const versionText = binary === undefined ? undefined : await server.version()
    const version = versionText === undefined ? undefined : parseMlxLmVersion(versionText)
    const versionCheck = entry?.mlx === undefined ? undefined : checkMlxLm(version, entry.mlx.minMlxLm)
    let files = { present: 0, verified: 0, total: 0 }
    if (entry?.mlx !== undefined) {
      const dir = mlxDir(entry, cfg)
      files = { present: 0, verified: 0, total: entry.mlx.files.length }
      for (const file of entry.mlx.files) {
        const check = await verifyModelFile(join(dir, file.path), { url: huggingFaceUrl(entry.mlx, file.path), sha256: file.sha256, bytes: file.bytes })
        if (check.present) files.present += 1
        if (check.verified) files.verified += 1
      }
    }
    const status = await server.status()
    const ready = entry !== undefined && binary !== undefined && versionCheck?.ok === true && files.verified === files.total
    const payload = {
      engine: 'mlx',
      ready,
      hardware: hardwareJson(hw),
      selection: entry === undefined ? { ok: false, reason: selectionError ?? 'no model' } : { ok: true, model: entry.id },
      mlx_server: {
        found: binary !== undefined,
        ...(binary === undefined ? {} : { path: binary }),
        ...(version === undefined ? {} : { mlx_lm_version: version }),
        ...(versionCheck === undefined || binary === undefined ? {} : { new_enough: versionCheck.ok }),
      },
      model: entry === undefined ? { id: null, present: false } : {
        id: entry.id,
        label: entry.label,
        path: mlxDir(entry, cfg),
        files_verified: files.verified,
        files_total: files.total,
        ...(entry.mlx === undefined ? {} : { bytes: entry.mlx.files.reduce((sum, file) => sum + file.bytes, 0) }),
      },
      server: statusJson(status),
    }
    if (args.json) {
      ports.stdout(JSON.stringify({ ok: true, ...payload }) + '\n')
    } else {
      const lines = [
        `Hardware:      ${hardwareLine(hw)}`,
        `Model choice:  ${entry === undefined ? `none (${selectionError ?? 'no model'})` : `${entry.id} (MLX)`}`,
        `mlx_lm.server: ${binary === undefined ? `NOT FOUND (${MLX_INSTALL_HELP})` : `${binary}${version === undefined ? ' (version unreadable)' : ` (mlx-lm ${version}${versionCheck?.ok === false ? `, needs ${entry?.mlx?.minMlxLm} or newer` : ''})`}`}`,
        `Model files:   ${entry === undefined ? 'n/a' : `${files.verified} of ${files.total} verified (${mlxDir(entry, cfg)})`}`,
        `Server:        ${statusLine(status, 'mlx')}`,
        ready ? 'Ready: run `xerxes local start --engine mlx`.' : 'Not ready yet: run `xerxes local setup --engine mlx`.',
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
    await requireLlamaCppVersion(entry, parsed)
    progress(`Starting llama-server on 127.0.0.1:${cfg.port}…`)
    const result = await ports.server.start({ modelPath: path, port: cfg.port, ctxSize: await llamaWindow(entry, cfg, path, hw) })
    return ok({ already_running: false, model_id: entry.id, ...statusJson(result) }, `Started: ${statusLine(result)}`)
  }

  async function stop(server: LocalServer): Promise<number> {
    const label = engineLabel(server.engine)
    const engineField = server.engine === 'llama-server' ? {} : { engine: server.engine }
    const result = await server.stop()
    if (result.stopped) {
      return ok({ stopped: true, ...engineField, pid: result.pid, forced: result.forced }, `Stopped local ${label} (pid ${result.pid}).`)
    }
    const text =
      result.reason === 'not-running'
        ? 'No local server is running.'
        : result.reason === 'stale'
          ? 'The recorded local server was already gone; cleaned up its record.'
          : `Pid ${result.pid} is no longer the Xerxes ${label}, so it was NOT signalled; its record was removed.`
    return ok({ stopped: false, ...engineField, reason: result.reason, ...(result.pid === undefined ? {} : { pid: result.pid }) }, text)
  }

  function report(current: ServerStatus, engine: LocalEngineName): number {
    const payload = { ...statusJson(current), ...(engine === 'llama-server' ? {} : { engine }) }
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
    const versionText = binary === undefined ? undefined : await ports.server.version()
    const version = versionText === undefined ? undefined : parseLlamaCppVersion(versionText)
    const server = await ports.server.status()

    const entry = parsed.model === undefined ? (selection.ok ? selection.entry : undefined) : catalog.find((c) => c.id === parsed.model)
    const versionCheck = entry?.minLlamaCpp === undefined || version === undefined ? undefined : checkLlamaCpp(version, entry.minLlamaCpp)
    let model: Record<string, unknown> = { id: null, present: false, verified: null, source_configured: false }
    let modelReady = false
    if (entry !== undefined) {
      const path = join(cfg.modelsDir, entry.fileName)
      if (entry.source === undefined) {
        const file = await useExistingModelFile(path).then(() => true, () => false)
        model = { id: entry.id, label: entry.label, path, present: file, verified: null, source_configured: false }
        modelReady = file
      } else {
        const check = await verifyModelFile(path, entry.source)
        model = { id: entry.id, label: entry.label, path, present: check.present, verified: check.verified, source_configured: true, ...(entry.source.bytes === undefined ? {} : { bytes: entry.source.bytes }) }
        modelReady = check.verified
      }
    }
    const versionOk = entry?.minLlamaCpp === undefined || versionCheck?.ok === true
    const ready = selection.ok && binary !== undefined && modelReady && versionOk
    const payload = {
      engine: 'llama-server',
      ready,
      hardware: hardwareJson(hw),
      selection: selection.ok
        ? { ok: true, model: selection.entry.id, budget_bytes: selection.budgetBytes, basis: selection.basis }
        : { ok: false, reason: selection.reason, budget_bytes: selection.budgetBytes, basis: selection.basis },
      llama_server: {
        found: binary !== undefined,
        ...(binary === undefined ? {} : { path: binary }),
        ...(version?.build === undefined ? {} : { build: version.build }),
        ...(version?.version === undefined ? {} : { version: version.version }),
        ...(versionCheck === undefined ? {} : { new_enough: versionCheck.ok }),
      },
      model,
      server: statusJson(server),
    }
    if (args.json) {
      ports.stdout(JSON.stringify({ ok: true, ...payload }) + '\n')
    } else {
      const lines = [
        `Hardware:     ${hardwareLine(hw)}`,
        `Model choice: ${selection.ok ? selection.entry.id : `none (${selection.reason})`}`,
        `llama-server: ${binary === undefined
          ? 'NOT FOUND (see `xerxes local setup` for install options)'
          : `${binary}${version === undefined ? ' (version unreadable)' : ` (${version.version === undefined ? '' : `v${version.version}, `}build ${version.build ?? '?'})`}${versionCheck?.ok === false ? `, older than ${entry?.id} needs (b${entry?.minLlamaCpp?.build})` : ''}`}`,
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
