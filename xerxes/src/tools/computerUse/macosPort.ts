// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Buffer } from 'node:buffer'
import { existsSync } from 'node:fs'
import { unlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  ComputerUseSession,
  ComputerUseUnavailableError,
  type ActionResult,
  type CaptureRequest,
  type CaptureResult,
  type ClickRequest,
  type ComputerUsePort,
  type DragRequest,
  type KeyRequest,
  type MouseMoveRequest,
  type ScrollRequest,
  type SetValueRequest,
  type TextRequest,
} from './backend.js'
import {
  JXA_CLICK,
  JXA_CURSOR_POSITION,
  JXA_DRAG,
  JXA_FOCUS_APP,
  JXA_KEY_CHORD,
  JXA_LIST_APPS,
  JXA_MOUSE_MOVE,
  JXA_SCREEN_INFO,
  JXA_SCROLL,
  JXA_TYPE,
  KEY_CODES,
  MODIFIER_ALIASES,
} from './macosScripts.js'
import type { ComputerUseToolsOptions } from './tool.js'

// Vision models are billed per image token and lose click precision on very
// large screenshots, so captures are capped at a longest edge that current
// vision models ingest natively. 1568 keeps UI text legible while bounding
// per-turn token cost.
const DEFAULT_MAX_CAPTURE_EDGE = 1568
// Absolute paths for the three system binaries are pinned so the backend
// never depends on (or gets hijacked through) the caller's PATH.
const DEFAULT_SCREENCAPTURE = '/usr/sbin/screencapture'
const DEFAULT_SIPS = '/usr/bin/sips'
const DEFAULT_OSASCRIPT = '/usr/bin/osascript'
// A drag is posted as N interpolated mouse-dragged events because many
// AppKit and web views only recognize a drag when they observe continuous
// motion between press and release; a single teleport drop is ignored.
const DRAG_STEPS = 24
// CoreGraphics scroll events are expressed in line units while the public
// tool API takes pixel deltas. 40 px/line approximates one native macOS
// scroll line so a model's pixel intent lands at a natural scroll distance.
const PIXELS_PER_SCROLL_LINE = 40

// macOS reports missing Screen Recording / Accessibility grants through a
// grab-bag of opaque stderr strings from screencapture and osascript. This
// matcher recognizes those phrasings so failures can be upgraded from a
// cryptic OS error into actionable guidance.
const PERMISSION_PROBLEM = /assistive|accessibility|screen recording|not permitted|not allowed|denied|could not create image/i
// Appended to any error that looks permission-related: the only fix is a
// user grant in System Settings, so the hint names the exact pane.
const PERMISSION_HINT =
  'macOS blocked the action. Grant Screen Recording and Accessibility to the terminal app in System Settings > Privacy & Security, then retry.'

/** Result of one spawned system command. */
export interface MacOSCommandResult {
  readonly code: number
  readonly stderr: string
  readonly stdout: string
}

/** Injectable process boundary; production uses Bun.spawn, tests use fakes. */
export type MacOSCommandRunner = (argv: readonly string[], signal?: AbortSignal) => Promise<MacOSCommandResult>

export interface MacOSComputerUsePortOptions {
  readonly fileExists?: (path: string) => boolean
  readonly maxCaptureEdge?: number
  readonly osascriptPath?: string
  readonly platform?: string
  readonly readFile?: (path: string) => Promise<Uint8Array>
  readonly removeFile?: (path: string) => Promise<void>
  readonly runner?: MacOSCommandRunner
  readonly screencapturePath?: string
  readonly sipsPath?: string
  readonly sleep?: (ms: number, signal?: AbortSignal) => Promise<void>
  readonly tmpDir?: string
  readonly uniqueId?: () => string
}

interface ScreenInfo {
  readonly backingScale: number
  readonly logicalHeight: number
  readonly logicalWidth: number
}

async function defaultRunner(argv: readonly string[], signal?: AbortSignal): Promise<MacOSCommandResult> {
  const [command, ...args] = argv
  if (command === undefined) throw new Error('macOS command runner requires an executable')
  // No shell is involved: argv goes straight to the kernel, so no
  // model-produced value can be reinterpreted as shell syntax.
  const proc = Bun.spawn([command, ...args], { stderr: 'pipe', stdout: 'pipe' })
  // AbortSignal must reap the child; otherwise a cancelled turn would leave
  // screencapture/osascript running past the caller's intent.
  const onAbort = (): void => {
    try {
      proc.kill()
    } catch {
      // Already exited.
    }
  }
  signal?.addEventListener('abort', onAbort, { once: true })
  try {
    const [stdout, stderr, code] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
      proc.exited,
    ])
    return { code, stderr, stdout }
  } finally {
    signal?.removeEventListener('abort', onAbort)
  }
}

async function defaultReadFile(path: string): Promise<Uint8Array> {
  return new Uint8Array(await Bun.file(path).arrayBuffer())
}

async function defaultSleep(ms: number, signal?: AbortSignal): Promise<void> {
  if (ms <= 0 || signal?.aborted) return
  await new Promise<void>(resolve => {
    const timer = setTimeout(() => resolve(), ms)
    signal?.addEventListener('abort', () => {
      clearTimeout(timer)
      resolve()
    }, { once: true })
  })
}

/**
 * Zero-install macOS desktop backend.
 *
 * Vision: `screencapture` + `sips` downscale (the image the model sees is
 * scaled back to logical points before dispatching input, so model
 * coordinates are always in captured-image pixels). Input: CoreGraphics
 * events through JXA (`osascript -l JavaScript`) with values passed only via
 * argv. Element/AX addressing is intentionally absent; set_value and
 * element-only targets return actionable failures instead of pretending.
 */
export class MacOSComputerUsePort implements ComputerUsePort {
  // Ratio of captured-image pixels to logical screen points, refreshed on
  // every capture. The model reasons in the pixels of the image it was
  // shown, but CoreGraphics only accepts logical points; this scale is the
  // bridge between the two coordinate spaces. On Retina displays a raw
  // screenshot is backingScale times larger than the logical screen, and
  // the sips downscale shrinks it further, so this is almost never 1.
  private coordinateScale = 1
  // Screen geometry and backing scale effectively never change mid-session,
  // so the (relatively expensive) osascript probe is cached after first use.
  private screenInfoCache: ScreenInfo | undefined

  private readonly fileExists: (path: string) => boolean
  private readonly maxCaptureEdge: number
  private readonly osascript: string
  private readonly platform: string
  private readonly readFile: (path: string) => Promise<Uint8Array>
  private readonly removeFile: (path: string) => Promise<void>
  private readonly runner: MacOSCommandRunner
  private readonly screencapture: string
  private readonly sips: string
  private readonly sleep: (ms: number, signal?: AbortSignal) => Promise<void>
  private readonly tmpDir: string
  private readonly uniqueId: () => string

  constructor(options: MacOSComputerUsePortOptions = {}) {
    this.fileExists = options.fileExists ?? existsSync
    this.maxCaptureEdge = options.maxCaptureEdge ?? DEFAULT_MAX_CAPTURE_EDGE
    this.osascript = options.osascriptPath ?? DEFAULT_OSASCRIPT
    this.platform = options.platform ?? process.platform
    this.readFile = options.readFile ?? defaultReadFile
    this.removeFile = options.removeFile ?? (path => unlink(path).then(() => undefined, () => undefined))
    this.runner = options.runner ?? defaultRunner
    this.screencapture = options.screencapturePath ?? DEFAULT_SCREENCAPTURE
    this.sips = options.sipsPath ?? DEFAULT_SIPS
    this.sleep = options.sleep ?? defaultSleep
    this.tmpDir = options.tmpDir ?? tmpdir()
    this.uniqueId = options.uniqueId ?? (() => crypto.randomUUID())
  }

  isAvailable(): boolean {
    return (
      this.platform === 'darwin' &&
      this.fileExists(this.screencapture) &&
      this.fileExists(this.sips) &&
      this.fileExists(this.osascript)
    )
  }

  async capture(request: CaptureRequest, signal?: AbortSignal): Promise<CaptureResult> {
    if (request.mode === 'ax') {
      return { elements: [], height: 0, mode: 'ax', width: 0 }
    }
    // Unique temp name per capture so concurrent turns cannot clobber each
    // other's screenshot file.
    const path = join(this.tmpDir, `xerxes-cua-${this.uniqueId()}.png`)
    // Pipeline step 1: grab the raw framebuffer with screencapture (`-x`
    // silences the shutter sound). Requires the Screen Recording
    // permission; on Retina displays the PNG is backingScale times larger
    // than the logical screen.
    await this.mustRun([this.screencapture, '-x', '-t', 'png', path], signal)
    try {
      const pixels = await this.imageDimensions(path, signal)
      const screen = await this.screenInfo(pixels, signal)
      // Normalize Retina pixels into logical points first: CoreGraphics
      // input dispatch speaks logical points, so every downstream size
      // decision is made in that space, never in raw screenshot pixels.
      const logicalWidth = Math.max(1, Math.round(pixels.width / screen.backingScale))
      const logicalHeight = Math.max(1, Math.round(pixels.height / screen.backingScale))
      // Never upscale (min with 1): upscaling would spend image tokens
      // without giving the model any information it did not already have.
      const scale = Math.min(1, this.maxCaptureEdge / Math.max(logicalWidth, logicalHeight))
      const targetWidth = Math.max(1, Math.round(logicalWidth * scale))
      const targetHeight = Math.max(1, Math.round(logicalHeight * scale))
      // Pipeline step 2: downscale in place with sips. When the target
      // already equals the raw size (small non-Retina screen) the resize is
      // skipped to avoid paying a second process spawn for an identity
      // transform.
      if (targetWidth !== pixels.width || targetHeight !== pixels.height) {
        await this.mustRun([this.sips, '-z', String(targetHeight), String(targetWidth), path], signal)
      }
      // Pipeline step 3: base64-encode the (now bounded) PNG into the wire
      // format the model consumes.
      const bytes = await this.readFile(path)
      // Record the image-px -> logical-point ratio for this exact image so
      // click coordinates quoted in image pixels land on the right logical
      // point; see logicalPoint().
      this.coordinateScale = targetWidth / logicalWidth
      return {
        elements: [],
        height: targetHeight,
        mode: request.mode,
        pngB64: Buffer.from(bytes).toString('base64'),
        pngBytesLength: bytes.length,
        width: targetWidth,
      }
    } finally {
      // A screenshot contains the user's screen contents; never leave it on
      // disk past the turn that needed it.
      await this.removeFile(path)
    }
  }

  click(request: ClickRequest, signal?: AbortSignal): Promise<ActionResult> {
    return this.clickLike('click', request, 'left', request.clickCount, signal)
  }

  doubleClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): Promise<ActionResult> {
    return this.clickLike('double_click', request, 'left', 2, signal)
  }

  tripleClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): Promise<ActionResult> {
    return this.clickLike('triple_click', request, 'left', 3, signal)
  }

  rightClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): Promise<ActionResult> {
    return this.clickLike('right_click', request, 'right', 1, signal)
  }

  middleClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): Promise<ActionResult> {
    return this.clickLike('middle_click', request, 'middle', 1, signal)
  }

  mouseMove(request: MouseMoveRequest, signal?: AbortSignal): Promise<ActionResult> {
    const point = this.logicalPoint(request.x, request.y)
    return this.runJxaAction('mouse_move', JXA_MOUSE_MOVE, [String(point.x), String(point.y)], request.captureAfter, signal)
  }

  drag(request: DragRequest, signal?: AbortSignal): Promise<ActionResult> {
    const start = this.optionalPoint(request.startX, request.startY, request.startElement, 'drag start')
    if (start === undefined) return Promise.resolve(unavailable('drag', 'drag requires start_x/start_y coordinates on this backend'))
    const end = this.optionalPoint(request.endX, request.endY, request.endElement, 'drag end')
    if (end === undefined) return Promise.resolve(unavailable('drag', 'drag requires end_x/end_y coordinates on this backend'))
    const from = this.logicalPoint(start.x, start.y)
    const to = this.logicalPoint(end.x, end.y)
    return this.runJxaAction(
      'drag',
      JXA_DRAG,
      [String(from.x), String(from.y), String(to.x), String(to.y), String(DRAG_STEPS)],
      request.captureAfter,
      signal,
    )
  }

  async scroll(request: ScrollRequest, signal?: AbortSignal): Promise<ActionResult> {
    if (request.element !== undefined) {
      return unavailable('scroll', 'element-targeted scroll requires an accessibility-tree backend; pass x/y or scroll at the cursor')
    }
    if (request.x !== undefined && request.y !== undefined) {
      const point = this.logicalPoint(request.x, request.y)
      const moved = await this.runJxaAction('scroll', JXA_MOUSE_MOVE, [String(point.x), String(point.y)], false, signal)
      if (!moved.ok) return moved
    }
    // CoreGraphics wheel deltas are positive when content moves down, while
    // the tool API's dy follows the model's "scroll the view" intuition;
    // the sign flip reconciles the two conventions.
    const wheelY = -scrollLines(request.dy)
    const wheelX = -scrollLines(request.dx)
    return this.runJxaAction('scroll', JXA_SCROLL, [String(wheelY), String(wheelX)], request.captureAfter, signal)
  }

  type(request: TextRequest, signal?: AbortSignal): Promise<ActionResult> {
    return this.runJxaAction('type', JXA_TYPE, [request.text], request.captureAfter, signal)
  }

  key(request: KeyRequest, signal?: AbortSignal): Promise<ActionResult> {
    const chord = parseChord(request.key)
    if (chord === undefined) {
      return Promise.resolve(unavailable('key', `unknown key in chord "${request.key}"; use names like enter, tab, a, f5, or modifiers command+shift+...`))
    }
    const argv = [
      String(chord.keyCode),
      chord.cmd ? '1' : '0',
      chord.shift ? '1' : '0',
      chord.alt ? '1' : '0',
      chord.ctrl ? '1' : '0',
    ]
    return this.runJxaAction('key', JXA_KEY_CHORD, argv, request.captureAfter, signal)
  }

  setValue(request: SetValueRequest): Promise<ActionResult> {
    void request
    return Promise.resolve(unavailable(
      'set_value',
      'set_value needs an accessibility-tree backend, which this zero-install port does not have. Click the field by x/y coordinates and use type instead.',
    ))
  }

  async wait(ms: number, signal?: AbortSignal): Promise<ActionResult> {
    await this.sleep(ms, signal)
    return { action: 'wait', message: `waited ${ms}ms`, ok: true }
  }

  async listApps(signal?: AbortSignal): Promise<ActionResult> {
    const result = await this.runJxa(JXA_LIST_APPS, [], signal)
    if (!result.ok) return { action: 'list_apps', message: result.message, ok: false }
    const names = result.message.split(',').map(name => name.trim()).filter(Boolean)
    return {
      action: 'list_apps',
      message: names.join(', '),
      meta: { apps: names },
      ok: true,
    }
  }

  async focusApp(app: string, signal?: AbortSignal): Promise<ActionResult> {
    const result = await this.runJxa(JXA_FOCUS_APP, [app], signal)
    if (!result.ok) return { action: 'focus_app', message: result.message, ok: false }
    if (result.message.trim() !== 'ok') {
      return { action: 'focus_app', message: `no running app matches "${app}"`, ok: false }
    }
    return { action: 'focus_app', message: `focused ${app}`, ok: true }
  }

  async cursorPosition(signal?: AbortSignal): Promise<ActionResult> {
    const result = await this.runJxa(JXA_CURSOR_POSITION, [], signal)
    if (!result.ok) return { action: 'cursor_position', message: result.message, ok: false }
    const [x = 0, y = 0] = result.message.trim().split(',').map(Number)
    return {
      action: 'cursor_position',
      message: `cursor at ${Math.round(x)}, ${Math.round(y)}`,
      meta: { x: Math.round(x), y: Math.round(y) },
      ok: true,
    }
  }

  private clickLike(
    action: 'click' | 'double_click' | 'middle_click' | 'right_click' | 'triple_click',
    request: { readonly captureAfter: boolean; readonly element?: number; readonly x?: number; readonly y?: number },
    button: 'left' | 'middle' | 'right',
    count: number,
    signal?: AbortSignal,
  ): Promise<ActionResult> {
    const point = this.optionalPoint(request.x, request.y, request.element, action)
    if (point === undefined) {
      return Promise.resolve(unavailable(action, `${action} needs x/y coordinates on this backend; element addressing requires an accessibility-tree backend`))
    }
    const logical = this.logicalPoint(point.x, point.y)
    return this.runJxaAction(
      action,
      JXA_CLICK,
      [String(logical.x), String(logical.y), button, String(Math.max(1, Math.round(count)))],
      request.captureAfter,
      signal,
    )
  }

  private optionalPoint(
    x: number | undefined,
    y: number | undefined,
    element: number | undefined,
    label: string,
  ): { readonly x: number; readonly y: number } | undefined {
    void label
    if (x !== undefined && y !== undefined) return { x, y }
    // Element indices come from an accessibility-tree capture this
    // zero-install backend intentionally does not provide; callers get an
    // explicit unavailability error instead of a click at a guessed point.
    void element
    return undefined
  }

  // Convert a point the model quoted in captured-image pixels into the
  // logical points CoreGraphics requires. Without this mapping, every click
  // on a Retina or downscaled capture would land scale times away from the
  // intended target. The `|| 1` guard keeps dispatch sane when an action
  // arrives before the first capture has set the scale.
  private logicalPoint(x: number, y: number): { readonly x: number; readonly y: number } {
    const scale = this.coordinateScale || 1
    return { x: Math.round(x / scale), y: Math.round(y / scale) }
  }

  private async runJxaAction(
    action: string,
    script: string,
    args: readonly string[],
    captureAfter: boolean,
    signal?: AbortSignal,
  ): Promise<ActionResult> {
    const result = await this.runJxa(script, args, signal)
    if (!result.ok) return { action, message: result.message, ok: false }
    if (!captureAfter) return { action, message: 'ok', ok: true }
    try {
      const capture = await this.capture({ mode: 'vision' }, signal)
      return { action, capture, message: 'ok', ok: true }
    } catch (error) {
      return { action, message: `ok (capture_after failed: ${errorMessage(error)})`, ok: true }
    }
  }

  private async runJxa(
    script: string,
    args: readonly string[],
    signal?: AbortSignal,
  ): Promise<{ readonly message: string; readonly ok: boolean }> {
    try {
      // Injection safety is structural: after `-e script`, osascript treats
      // every remaining token as an entry in JXA's `argv`, never as source.
      // Model-supplied text (type payloads, app names) only ever travels
      // through argv and is never interpolated into the script source, so a
      // hostile payload cannot escape into the automation runtime.
      const result = await this.runner([this.osascript, '-l', 'JavaScript', '-e', script, ...args], signal)
      if (result.code !== 0) {
        return { message: permissionHint(result.stderr.trim() || `osascript exited ${result.code}`), ok: false }
      }
      return { message: result.stdout, ok: true }
    } catch (error) {
      return { message: permissionHint(errorMessage(error)), ok: false }
    }
  }

  private async mustRun(argv: readonly string[], signal?: AbortSignal): Promise<MacOSCommandResult> {
    let result: MacOSCommandResult
    try {
      result = await this.runner(argv, signal)
    } catch (error) {
      // Spawn-level failure (missing binary, sandbox refusal): surface it as
      // backend unavailability, upgraded with a hint when it matches a
      // known permission phrasing.
      throw new ComputerUseUnavailableError(permissionHint(errorMessage(error)), error)
    }
    if (result.code !== 0) {
      // screencapture exits non-zero when Screen Recording is denied; the
      // permission hint turns that opaque stderr into a fixable instruction
      // instead of a dead-end error.
      throw new ComputerUseUnavailableError(permissionHint(result.stderr.trim() || `${argv[0]} exited ${result.code}`))
    }
    return result
  }

  private async imageDimensions(path: string, signal?: AbortSignal): Promise<{ readonly height: number; readonly width: number }> {
    const result = await this.mustRun([this.sips, '-g', 'pixelWidth', '-g', 'pixelHeight', path], signal)
    const width = /pixelWidth:\s*(\d+)/.exec(result.stdout)
    const height = /pixelHeight:\s*(\d+)/.exec(result.stdout)
    if (!width?.[1] || !height?.[1]) {
      throw new ComputerUseUnavailableError(`could not read screenshot dimensions from sips output: ${result.stdout.trim()}`)
    }
    return { height: Number(height[1]), width: Number(width[1]) }
  }

  private async screenInfo(
    pixels: { readonly height: number; readonly width: number },
    signal?: AbortSignal,
  ): Promise<ScreenInfo> {
    if (this.screenInfoCache) return this.screenInfoCache
    try {
      const result = await this.runner([this.osascript, '-l', 'JavaScript', '-e', JXA_SCREEN_INFO], signal)
      const [width = 0, height = 0, scale = 0] = result.stdout.trim().split(',').map(Number)
      if (result.code === 0 && width > 0 && height > 0 && scale > 0) {
        this.screenInfoCache = { backingScale: scale, logicalHeight: height, logicalWidth: width }
        return this.screenInfoCache
      }
    } catch {
      // Fall through to the pixel-derived estimate below.
    }
    // The osascript screen probe is best-effort (automation consent prompts
    // can block it); a screenshot wider than 3000 px is effectively always
    // a Retina 2x framebuffer, so estimate from the image itself rather
    // than failing the whole capture.
    const backingScale = pixels.width >= 3000 ? 2 : 1
    this.screenInfoCache = { backingScale, logicalHeight: pixels.height, logicalWidth: pixels.width }
    return this.screenInfoCache
  }
}

// Convert a pixel delta into whole scroll lines. A sub-line remainder still
// counts as one line because CoreGraphics cannot express fractional line
// scrolls, and dropping it would make small model scrolls silent no-ops.
function scrollLines(deltaPixels: number): number {
  if (deltaPixels === 0) return 0
  return Math.sign(deltaPixels) * Math.max(1, Math.round(Math.abs(deltaPixels) / PIXELS_PER_SCROLL_LINE))
}

interface Chord {
  readonly alt: boolean
  readonly cmd: boolean
  readonly ctrl: boolean
  readonly keyCode: number
  readonly shift: boolean
}

// Parse "command+shift+p" style chords. The last segment is the key and
// everything before it must be a modifier; an unknown modifier fails the
// whole chord (undefined) instead of being silently dropped, so a typo like
// "cmd+shfit+p" can never fire an unintended plain keystroke at the desktop.
function parseChord(input: string): Chord | undefined {
  const parts = input.split('+').map(part => part.trim().toLowerCase()).filter(Boolean)
  if (!parts.length) return undefined
  const flags = { alt: false, cmd: false, ctrl: false, shift: false }
  for (const part of parts.slice(0, -1)) {
    const modifier = MODIFIER_ALIASES[part]
    if (modifier === undefined) return undefined
    flags[modifier] = true
  }
  const keyName = parts.at(-1) ?? ''
  const keyCode = KEY_CODES[keyName.length === 1 ? keyName : keyName.replace(/\s+/g, '')]
  return keyCode === undefined ? undefined : { ...flags, keyCode }
}

function unavailable(action: string, message: string): ActionResult {
  return { action, message, ok: false }
}

// Map macOS's many phrasings of "permission denied" onto a single hint
// naming the exact System Settings pane. Without this, the model (and the
// user) would see raw CoreGraphics/osascript errors with no path to a fix.
function permissionHint(message: string): string {
  return PERMISSION_PROBLEM.test(message) ? `${message}\n${PERMISSION_HINT}` : message
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/**
 * Build the CoreTools `computerUseTool` option.
 *
 * Desktop control is part of the baseline agent surface: it is enabled by
 * default whenever the zero-install macOS backend can run on this host. A
 * user opts out with `runtime.computer_use_enabled: false` or
 * `XERXES_COMPUTER_USE=off`. Force-enabling on an unsupported host
 * (`computer_use_enabled: true`) registers the tool anyway so calls fail
 * with an explicit unavailability error instead of a silent absence.
 */
export function createMacOSComputerUseToolOptions(
  settings: Readonly<Record<string, unknown>>,
  environment: Readonly<Record<string, string | undefined>> = process.env,
  hostAvailable?: () => boolean,
): ComputerUseToolsOptions | undefined {
  const flag = environment['XERXES_COMPUTER_USE']?.trim().toLowerCase()
  // Opt-out wins over everything: a user who said "no desktop control" must
  // never have the tool registered, regardless of host capability or any
  // other setting.
  const explicitlyDisabled = settings['computer_use_enabled'] === false
    || flag === '0' || flag === 'false' || flag === 'no' || flag === 'off'
  if (explicitlyDisabled) return undefined
  // A non-macos backend selection means another port owns this surface;
  // registering the macOS port too would shadow it.
  const backend = settings['computer_use_backend']
  if (backend !== undefined && backend !== 'macos') return undefined
  const explicitlyEnabled = settings['computer_use_enabled'] === true
    || flag === '1' || flag === 'true' || flag === 'yes' || flag === 'on'
  const available = hostAvailable?.() ?? new MacOSComputerUsePort().isAvailable()
  // Default-on semantics: with no explicit setting the tool registers
  // whenever the host can actually run it. Force-enable (`true`/`on`)
  // registers even on an unsupported host so calls fail loudly with an
  // unavailability error instead of the tool silently going missing.
  if (!explicitlyEnabled && !available) return undefined
  return { session: new ComputerUseSession({ port: new MacOSComputerUsePort() }) }
}
