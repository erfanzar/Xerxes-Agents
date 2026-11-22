// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The agent's app-wide status mark: Xerxes's minimalist architectural Derafsh
 * Kaviani bookplate (`size >= 36`) and 8-ray Kavian starburst (`size < 36`)
 * rendered via `<XerxesStatusMark />`.
 *
 * Motion runs on GPU-composited CSS keyframes that pause automatically when
 * `live === false`, once `settleMs` elapses, under `prefers-reduced-motion`,
 * or while the window is unfocused or occluded.
 *
 * The legacy `OrbEngine` and `wake()` exports remain available for headless
 * clock/preset unit tests (`desktopAgentOrb.test.tsx`).
 */

import { memo, useEffect, useState, type ReactElement } from 'react'
import { MODE_FRAMES, paintFrame, resolvePreset, type ModeKey, type ModeOpts, type OrbSize, type OrbState } from 'thinking-orbs/engine'

import {
  XerxesStatusMark,
  resolveXerxesStatusTone,
  type XerxesStatusState,
} from './XerxesSprites.js'

export type { OrbSize, OrbState }
export type AgentOrbState = XerxesStatusState
export type AgentOrbSize = OrbSize | number

const ORB_FPS = 30
/** Where a still orb is drawn: a representative pose, as the library's reduced-motion frame. */
const STILL_TIME = 0.6

function lightTheme(): boolean {
  return typeof document !== 'undefined' && document.documentElement.getAttribute('data-theme') === 'light'
}

function reducedMotion(): boolean {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches
}

export interface OrbOptions {
  readonly state: OrbState
  /** False draws one still frame; true animates until `settleMs` passes in one state. */
  readonly live: boolean
  readonly settleMs?: number | undefined
}

/** One orb's canvas and clock. Holds no timers; the shared ticker drives it. */
export class OrbEngine {
  readonly #canvas: HTMLCanvasElement
  readonly #size: OrbSize
  readonly #ratio: number
  #ctx: CanvasRenderingContext2D | null
  #options: OrbOptions
  #mode: ModeKey
  #speed: number
  #opts: ModeOpts
  #time = 0
  #stateAge = 0
  visible = true

  constructor(canvas: HTMLCanvasElement, size: OrbSize, options: OrbOptions, ratio = 1) {
    this.#canvas = canvas
    this.#size = size
    this.#ratio = Math.min(Math.max(ratio, 1), 2)
    canvas.width = canvas.height = Math.round(size * this.#ratio)
    this.#ctx = canvas.getContext('2d')
    this.#options = options
    const preset = resolvePreset(options.state, size)
    this.#mode = preset.mode
    this.#speed = preset.speed
    this.#opts = preset.opts
  }

  get mode(): ModeKey { return this.#mode }

  update(options: OrbOptions): void {
    if (options.state !== this.#options.state) {
      this.#stateAge = 0
      const preset = resolvePreset(options.state, this.#size)
      this.#mode = preset.mode
      this.#speed = preset.speed
      this.#opts = preset.opts
    }
    this.#options = options
  }

  /** Whether the next tick would change anything on screen. */
  wantsFrames(): boolean {
    if (!this.visible || !this.#options.live || reducedMotion()) return false
    const settle = this.#options.settleMs
    return settle === undefined || this.#stateAge * 1000 < settle
  }

  step(dt: number): void {
    this.#time += dt
    this.#stateAge += dt
    this.paint()
  }

  paint(): void {
    const ctx = this.#ctx
    if (!ctx) return
    const moving = this.#options.live && !reducedMotion()
    const t = moving ? this.#time * this.#speed : STILL_TIME
    ctx.setTransform(this.#ratio, 0, 0, this.#ratio, 0, 0)
    ctx.clearRect(0, 0, this.#size, this.#size)
    paintFrame(ctx, MODE_FRAMES[this.#mode](this.#size, t, this.#opts), !lightTheme())
  }

  dispose(): void {
    this.#ctx = null
  }
}

// ── Shared ticker ────────────────────────────────────────────────────

const mounted = new Set<OrbEngine>()
let timer: ReturnType<typeof setTimeout> | undefined
/** When a frame was requested and not yet run; a dropped one must not stop the ticker for good. */
let framePending = 0
const FRAME_LOST_MS = 1000
let lastTick = 0

function paused(): boolean {
  if (typeof document === 'undefined') return true
  const root = document.documentElement
  return document.hidden
    || root.getAttribute('data-perf-paused') === 'true'
    || root.hasAttribute('data-window-unfocused')
    || root.hasAttribute('data-occluded')
}

function tick(now: number): void {
  framePending = 0
  if (paused()) { lastTick = 0; return }
  const dt = lastTick ? Math.min((now - lastTick) / 1000, 0.1) : 1 / ORB_FPS
  lastTick = now
  for (const engine of mounted) if (engine.wantsFrames()) engine.step(dt)
  wake()
}

/** Arms one timer if any orb has frames to draw. Idempotent. */
export function wake(): void {
  // A frame requested while the window was hidden can be dropped; once it is
  // long overdue it no longer holds the ticker.
  if (framePending && performance.now() - framePending > FRAME_LOST_MS) framePending = 0
  if (timer !== undefined || framePending || paused()) return
  let any = false
  for (const engine of mounted) if (engine.wantsFrames()) { any = true; break }
  if (!any) { lastTick = 0; return }
  // A timer, not a rAF loop: rAF would wake the renderer at the display rate
  // even on the frames this skips.
  timer = setTimeout(() => {
    timer = undefined
    framePending = performance.now()
    requestAnimationFrame(tick)
  }, 1000 / ORB_FPS)
}

export interface AgentOrbProps {
  readonly state: AgentOrbState
  /** Rendered size in CSS px: 14–24 compact 8-ray Kavian star, 36–64 minimalist Derafsh Kaviani bookplate. */
  readonly size: AgentOrbSize
  readonly live?: boolean
  readonly settleMs?: number
  readonly className?: string
  readonly label?: string
}

export const AgentOrb = memo(function AgentOrb({ state, size, live = true, settleMs, className, label }: AgentOrbProps): ReactElement {
  const [settled, setSettled] = useState(false)

  useEffect(() => {
    setSettled(false)
    if (!live || settleMs === undefined) return
    if (settleMs <= 0) {
      setSettled(true)
      return
    }
    const handle = setTimeout(() => setSettled(true), settleMs)
    return () => clearTimeout(handle)
  }, [state, live, settleMs])

  const isLive = live && !settled
  const tone = resolveXerxesStatusTone(state)

  return (
    <span
      className={`agent-orb${className ? ` ${className}` : ''}`}
      data-state={state}
      data-tone={tone}
      data-live={isLive ? 'true' : 'false'}
      style={{ width: size, height: size }}
      role={label ? 'img' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      <XerxesStatusMark state={state} size={size} live={isLive} />
    </span>
  )
})
