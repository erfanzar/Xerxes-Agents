// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The agent's status mark: a dotted "thinking orb" whose motion says what the
 * turn is doing — scanning a globe while it searches, scrambling bands while
 * it reasons, a sash while it writes.
 *
 * The drawings are the hand-tuned presets from `thinking-orbs` (MIT, Jakub
 * Antalik), rendered through its raw painters rather than its component. The
 * component runs one requestAnimationFrame loop per orb at the display rate;
 * three orbs on a 120 Hz panel woke the renderer 360 times a second during a
 * turn. Here every orb shares one timer at ORB_FPS, and the timer stops while
 * the window is unfocused, hidden or covered by another workspace, while an
 * orb is off-screen, under reduced motion, and once an orb has settled.
 */

import { useEffect, useRef, type ReactElement } from 'react'
import { MODE_FRAMES, paintFrame, resolvePreset, type ModeKey, type ModeOpts, type OrbSize, type OrbState } from 'thinking-orbs/engine'

export type { OrbSize, OrbState }

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
let lastTick = 0
let watching = false

function paused(): boolean {
  if (typeof document === 'undefined') return true
  const root = document.documentElement
  return document.hidden || root.hasAttribute('data-window-unfocused') || root.hasAttribute('data-occluded')
}

function tick(now: number): void {
  timer = undefined
  if (paused()) { lastTick = 0; return }
  const dt = lastTick ? Math.min((now - lastTick) / 1000, 0.1) : 1 / ORB_FPS
  lastTick = now
  for (const engine of mounted) if (engine.wantsFrames()) engine.step(dt)
  wake()
}

/** Arms one timer if any orb has frames to draw. Idempotent. */
export function wake(): void {
  if (timer !== undefined || paused()) return
  let any = false
  for (const engine of mounted) if (engine.wantsFrames()) { any = true; break }
  if (!any) { lastTick = 0; return }
  // A timer, not a rAF loop: rAF would wake the renderer at the display rate
  // even on the frames this skips.
  timer = setTimeout(() => requestAnimationFrame(tick), 1000 / ORB_FPS)
}

function watchEnvironment(): void {
  if (watching || typeof document === 'undefined') return
  watching = true
  document.addEventListener('visibilitychange', wake)
  // Focus, occlusion and theme all live on the root element's attributes.
  new MutationObserver(() => {
    for (const engine of mounted) engine.paint()
    wake()
  }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-window-unfocused', 'data-occluded', 'data-theme'] })
}

let intersections: IntersectionObserver | null = null
const byCanvas = new WeakMap<Element, OrbEngine>()
function observeVisibility(canvas: HTMLCanvasElement, engine: OrbEngine): void {
  if (typeof IntersectionObserver === 'undefined') return
  intersections ??= new IntersectionObserver(entries => {
    for (const entry of entries) {
      const found = byCanvas.get(entry.target)
      if (found) found.visible = entry.isIntersecting
    }
    wake()
  })
  byCanvas.set(canvas, engine)
  intersections.observe(canvas)
}

export interface AgentOrbProps {
  readonly state: OrbState
  /** A tuned preset: 20 inline with text, 32 compact, 64 as the main mark. */
  readonly size: OrbSize
  readonly live?: boolean
  readonly settleMs?: number
  readonly className?: string
  readonly label?: string
}

export function AgentOrb({ state, size, live = true, settleMs, className, label }: AgentOrbProps): ReactElement {
  const canvas = useRef<HTMLCanvasElement>(null)
  const engine = useRef<OrbEngine | null>(null)

  useEffect(() => {
    const element = canvas.current
    if (!element) return
    watchEnvironment()
    const created = new OrbEngine(element, size, { state, live, settleMs }, globalThis.devicePixelRatio || 1)
    engine.current = created
    mounted.add(created)
    observeVisibility(element, created)
    created.paint()
    wake()
    return () => {
      mounted.delete(created)
      intersections?.unobserve(element)
      created.dispose()
      engine.current = null
    }
    // Size fixes the canvas and preset; state changes go through update().
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size])

  useEffect(() => {
    const current = engine.current
    if (!current) return
    current.update({ state, live, settleMs })
    // Repaint now: a paused ticker would otherwise leave the old state's
    // drawing up until focus returns.
    current.paint()
    wake()
  }, [state, live, settleMs])

  return <span className={`agent-orb${className ? ` ${className}` : ''}`} data-state={state} style={{ width: size, height: size }} role={label ? 'img' : undefined} aria-label={label} aria-hidden={label ? undefined : true}>
    <canvas ref={canvas} />
  </span>
}
