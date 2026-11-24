// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Clean, modern developer-tool vector icon system for Xerxes Agents:
 *
 * - `XerxesBrandEmblem` (`48×48`) & `AgentBrandGlyph provider="xerxes"` (`24×24`):
 *   Faceted Architectural X-Prism — four precision-cut interlocking geometric
 *   diagonal blades separated by a uniform negative-space cleft, rendered in
 *   flat tonal `currentColor` / `var(--x-title)` contrast (1.0 & 0.42) with
 *   zero frames, badges, crowns, wings, or gradients.
 * - `SimurghSprite` (`24×24`) + `drawSimurghSpriteOnCanvas`:
 *   Five crisp `1.5px` monochrome line icons on a transparent canvas
 *   (`coordinator` hub network, `worker` terminal prompt, `verifier` circle
 *   check, `narrator` summary lines, `monitor` activity pulse) with 100%
 *   Canvas 2D parity for GIF exports.
 * - `XerxesThinkingMark` & `XerxesStatusMark` (`24×24`):
 *   Minimal `1.75px` open-arc progress spinner (the sole animated primitive,
 *   GPU-composited and auto-paused under occlusion/blur/reduced-motion) and
 *   quiet semantic glyphs for `waiting`, `done`, `error`, and `idle`.
 *
 * 100% theme-dynamic via `--x-*` tokens (zero `#RRGGBB` hex literals).
 */

import { memo, useId, type ReactElement } from 'react'
import { usePerfPaused } from './pageVisibility.js'

export type SimurghSpriteRole = 'coordinator' | 'worker' | 'verifier' | 'narrator' | 'monitor'
export type SimurghSpriteState = 'working' | 'waiting' | 'stopped' | 'killed' | 'error'
export type AgentBrandProvider = 'xerxes' | 'claude-code' | 'antigravity' | 'codex' | 'acp'

export interface SimurghSpriteProps {
  readonly role: SimurghSpriteRole
  readonly state?: SimurghSpriteState
  readonly size?: number
  readonly animate?: boolean
  readonly className?: string
  readonly title?: string
}

export interface AgentBrandGlyphProps {
  readonly provider: AgentBrandProvider
  readonly size?: number
  readonly animate?: boolean
  readonly className?: string
}

export interface XerxesThinkingMarkProps {
  readonly size?: number
  readonly active?: boolean
  readonly className?: string
}

export type XerxesStatusState =
  | 'working'
  | 'thinking'
  | 'reasoning'
  | 'solving'
  | 'searching'
  | 'writing'
  | 'composing'
  | 'shaping'
  | 'weaving'
  | 'listening'
  | 'idle'
  | 'breathing'
  | 'ready'
  | 'connecting'
  | 'waiting'
  | 'approval'
  | 'needs'
  | 'error'
  | 'failed'
  | 'done'
  | 'success'

export type XerxesStatusTone = 'active' | 'idle' | 'waiting' | 'error' | 'done'

export interface XerxesStatusMarkProps {
  readonly state: XerxesStatusState
  readonly size?: number
  readonly live?: boolean
  readonly className?: string
  readonly label?: string
}

export const SIMURGH_ROLES: readonly SimurghSpriteRole[] = [
  'coordinator',
  'worker',
  'verifier',
  'narrator',
  'monitor',
]

const VERIFIER_NAME_PATTERN = /(verif|audit|review|test|check|qa)/i
const NARRATOR_NAME_PATTERN = /narrat/i
const MONITOR_NAME_PATTERN = /monitor/i
const COORDINATOR_NAME_PATTERN = /^coord(inator)?\b/i

/**
 * Resolves a swarm member's name and optional explicit role into one of the
 * five canonical Xerxes role icons.
 */
export function resolveSimurghRole(name: string, role?: string): SimurghSpriteRole {
  const normalizedRole = (role ?? '').trim().toLowerCase()
  if (normalizedRole === 'coordinator') return 'coordinator'
  if (normalizedRole === 'verifier') return 'verifier'
  if (normalizedRole === 'narrator') return 'narrator'
  if (normalizedRole === 'monitor') return 'monitor'

  const normalizedName = name.trim().toLowerCase()
  if (normalizedName === 'coordinator' || COORDINATOR_NAME_PATTERN.test(normalizedName)) {
    return 'coordinator'
  }
  if (VERIFIER_NAME_PATTERN.test(normalizedName)) return 'verifier'
  if (NARRATOR_NAME_PATTERN.test(normalizedName)) return 'narrator'
  if (MONITOR_NAME_PATTERN.test(normalizedName)) return 'monitor'

  return 'worker'
}

// ---------------------------------------------------------------------------
// 1. Five Clean Monochrome Dev-Tool Role Icons (24×24 SVG + Canvas 2D Parity)
// ---------------------------------------------------------------------------

const DEFAULT_ROLE_LABELS: Readonly<Record<SimurghSpriteRole, string>> = {
  coordinator: 'Coordinator',
  worker: 'Worker',
  verifier: 'Verifier',
  narrator: 'Narrator',
  monitor: 'Monitor',
}

interface RoleStrokeToken {
  readonly stroke: string
  readonly opacity: number
}

function resolveRoleStrokeToken(role: SimurghSpriteRole, state: SimurghSpriteState): RoleStrokeToken {
  if (state === 'killed' || state === 'stopped') {
    return { stroke: 'var(--x-meta, currentColor)', opacity: 0.62 }
  }
  if (state === 'error') {
    return { stroke: 'var(--x-failed, currentColor)', opacity: 1 }
  }
  if (state === 'waiting') {
    return role === 'coordinator'
      ? { stroke: 'var(--x-title, currentColor)', opacity: 0.86 }
      : { stroke: 'var(--x-meta, currentColor)', opacity: 0.92 }
  }
  // Working state: clean monochrome developer-tool tinting
  if (role === 'coordinator') {
    return { stroke: 'var(--x-title, currentColor)', opacity: 1 }
  }
  if (role === 'verifier') {
    return { stroke: 'var(--x-done, currentColor)', opacity: 1 }
  }
  return { stroke: 'var(--x-working, var(--x-title, currentColor))', opacity: 1 }
}

/**
 * Pure `1.5px` Lucide/Linear-style monochrome role icon rendered directly on a
 * transparent background (zero medallion rings, crowns, shields, or boxes).
 */
export const SimurghSprite = memo(function SimurghSprite(props: SimurghSpriteProps): ReactElement {
  const { role, state = 'working', size = 16, className, title } = props
  const label = title ?? DEFAULT_ROLE_LABELS[role]
  const token = resolveRoleStrokeToken(role, state)
  const classes = [
    'xerxes-simurgh-sprite',
    `xerxes-simurgh-${role}`,
    `xerxes-simurgh-state-${state}`,
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      role="img"
      aria-label={label}
      className={classes}
      data-simurgh-role={role}
      data-simurgh-state={state}
      style={{ display: 'inline-block', flexShrink: 0 }}
    >
      <title>{label}</title>
      <g
        fill="none"
        stroke={token.stroke}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={token.opacity}
      >
        {role === 'coordinator' ? (
          <>
            {/* Hub / orchestration network: top node branching to two lower nodes */}
            <circle cx="12" cy="5.5" r="2.5" />
            <circle cx="6" cy="18.5" r="2.5" />
            <circle cx="18" cy="18.5" r="2.5" />
            <path d="M 12 8 V 12" />
            <path d="M 6 16 V 13.5 C 6 12.67 6.67 12 7.5 12 H 16.5 C 17.33 12 18 12.67 18 13.5 V 16" />
          </>
        ) : role === 'worker' ? (
          <>
            {/* Clean terminal prompt (`>_`) */}
            <path d="M 4.5 7 L 10.5 12 L 4.5 17" />
            <path d="M 12.5 17 H 19.5" />
          </>
        ) : role === 'verifier' ? (
          <>
            {/* Minimal circle-check */}
            <circle cx="12" cy="12" r="8.5" />
            <path d="M 8.5 12.2 L 10.8 14.5 L 15.8 9.5" />
          </>
        ) : role === 'narrator' ? (
          <>
            {/* Minimal document / summary lines */}
            <path d="M 4.5 7 H 19.5" />
            <path d="M 4.5 12 H 16.5" />
            <path d="M 4.5 17 H 12.5" />
          </>
        ) : (
          <>
            {/* Activity pulse */}
            <path d="M 3 12 H 7 L 9.5 6 L 14.5 18 L 17 12 H 21" />
          </>
        )}
      </g>
    </svg>
  )
})

// ---------------------------------------------------------------------------
// Canvas 2D Monochrome Role Icon Renderer (100% Parity with 24×24 SVG UI)
// ---------------------------------------------------------------------------

interface CanvasRoleStroke {
  readonly stroke: string
  readonly alpha: number
}

function resolveCanvasRoleStroke(
  role: SimurghSpriteRole,
  state: SimurghSpriteState,
): CanvasRoleStroke {
  if (state === 'killed' || state === 'stopped') {
    return { stroke: 'rgb(138, 144, 156)', alpha: 0.62 }
  }
  if (state === 'error') {
    return { stroke: 'rgb(241, 76, 76)', alpha: 1 }
  }
  if (state === 'waiting') {
    return role === 'coordinator'
      ? { stroke: 'rgb(232, 234, 238)', alpha: 0.86 }
      : { stroke: 'rgb(154, 160, 172)', alpha: 0.92 }
  }
  if (role === 'verifier') {
    return { stroke: 'rgb(63, 185, 80)', alpha: 1 }
  }
  return { stroke: 'rgb(234, 236, 240)', alpha: 1 }
}

function traceCanvasCircle(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  r: number,
): void {
  ctx.beginPath()
  if (typeof ctx.arc === 'function') {
    ctx.arc(cx, cy, r, 0, Math.PI * 2)
    return
  }
  const segs = 20
  for (let i = 0; i <= segs; i += 1) {
    const a = (i / segs) * Math.PI * 2
    const px = cx + r * Math.cos(a)
    const py = cy + r * Math.sin(a)
    if (i === 0) ctx.moveTo(px, py)
    else ctx.lineTo(px, py)
  }
  ctx.closePath()
}

/**
 * Draws the exact canonical 24×24 monochrome `1.5px` role icon onto a 2D
 * canvas context so exported GIFs match the live SVG UI.
 */
export function drawSimurghSpriteOnCanvas(
  ctx: CanvasRenderingContext2D,
  role: SimurghSpriteRole,
  state: SimurghSpriteState,
  x: number,
  y: number,
  size: number,
  _frameIndex = 0,
): void {
  if (size <= 0) return
  const token = resolveCanvasRoleStroke(role, state)
  const scale = size / 24
  const px = (u: number): number => x + u * scale
  const py = (v: number): number => y + v * scale
  const strokeW = Math.max(1.0, 1.5 * scale)

  const hasSave = typeof ctx.save === 'function'
  const hasRestore = typeof ctx.restore === 'function'
  if (hasSave) ctx.save()

  ctx.strokeStyle = token.stroke
  ctx.globalAlpha = token.alpha
  ctx.lineWidth = strokeW
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  if (role === 'coordinator') {
    traceCanvasCircle(ctx, px(12), py(5.5), 2.5 * scale)
    ctx.stroke()
    traceCanvasCircle(ctx, px(6), py(18.5), 2.5 * scale)
    ctx.stroke()
    traceCanvasCircle(ctx, px(18), py(18.5), 2.5 * scale)
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(px(12), py(8))
    ctx.lineTo(px(12), py(12))
    ctx.moveTo(px(6), py(16))
    ctx.lineTo(px(6), py(13.2))
    ctx.quadraticCurveTo(px(6), py(12), px(7.5), py(12))
    ctx.lineTo(px(16.5), py(12))
    ctx.quadraticCurveTo(px(18), py(12), px(18), py(13.2))
    ctx.lineTo(px(18), py(16))
    ctx.stroke()
  } else if (role === 'worker') {
    ctx.beginPath()
    ctx.moveTo(px(4.5), py(7))
    ctx.lineTo(px(10.5), py(12))
    ctx.lineTo(px(4.5), py(17))
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(px(12.5), py(17))
    ctx.lineTo(px(19.5), py(17))
    ctx.stroke()
  } else if (role === 'verifier') {
    traceCanvasCircle(ctx, px(12), py(12), 8.5 * scale)
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(px(8.5), py(12.2))
    ctx.lineTo(px(10.8), py(14.5))
    ctx.lineTo(px(15.8), py(9.5))
    ctx.stroke()
  } else if (role === 'narrator') {
    ctx.beginPath()
    ctx.moveTo(px(4.5), py(7))
    ctx.lineTo(px(19.5), py(7))
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(px(4.5), py(12))
    ctx.lineTo(px(16.5), py(12))
    ctx.stroke()

    ctx.beginPath()
    ctx.moveTo(px(4.5), py(17))
    ctx.lineTo(px(12.5), py(17))
    ctx.stroke()
  } else {
    ctx.beginPath()
    ctx.moveTo(px(3), py(12))
    ctx.lineTo(px(7), py(12))
    ctx.lineTo(px(9.5), py(6))
    ctx.lineTo(px(14.5), py(18))
    ctx.lineTo(px(17), py(12))
    ctx.lineTo(px(21), py(12))
    ctx.stroke()
  }

  if (hasRestore) ctx.restore()
}

// ---------------------------------------------------------------------------
// 2. Clean Faceted Architectural X-Prism Brand Mark (48×48)
// ---------------------------------------------------------------------------

/**
 * Faceted Architectural X-Prism polygons in canonical `48×48` space.
 * Every diagonal edge locks to exact ±1.28 slope with flat horizontal top/bottom
 * terminals and an exact 2.83-unit negative-space cleft through the center.
 */
export const XERXES_PRISM_PRIMARY_PATHS: readonly string[] = [
  'M 6.5 8 L 16.5 8 L 25.6 19.65 L 20.6 26.05 Z',
  'M 41.5 40 L 31.5 40 L 22.4 28.35 L 27.4 21.95 Z',
]

export const XERXES_PRISM_SECONDARY_PATHS: readonly string[] = [
  'M 31.5 8 L 41.5 8 L 30.8 21.7 L 25.8 15.3 Z',
  'M 16.5 40 L 6.5 40 L 17.2 26.3 L 22.2 32.7 Z',
]

/**
 * Renders Xerxes's real bundled phoenix emblem (`./logo-128.png`) across both
 * Desktop and VS Code (`XerxesBrandEmblem` never falls back to a split-X SVG).
 */
export const XerxesBrandEmblem = memo(function XerxesBrandEmblem(props: {
  readonly size?: number
  readonly className?: string
  readonly animate?: boolean
  readonly motion?: 'smil' | 'css'
}): ReactElement {
  const { size = 40, className } = props
  const classes = ['xerxes-logo-mark', 'xerxes-brand-emblem', className ?? ''].filter(Boolean).join(' ')
  return (
    <img
      src="./logo-128.png"
      alt="Xerxes Agents"
      width={size}
      height={size}
      className={classes}
      data-brand-provider="xerxes"
      decoding="async"
      draggable={false}
      style={{
        display: 'inline-block',
        flexShrink: 0,
        objectFit: 'cover',
        borderRadius: Math.max(4, Math.round(size * 0.22)),
      }}
    />
  )
})

// ---------------------------------------------------------------------------
// 3. Minimal Dev-Tool Thinking & Status Marks (24×24)
// ---------------------------------------------------------------------------

/**
 * Clean minimal thinking indicator: a thin `1.75px` open-arc spinner inside a
 * faint hairline track while active, or a quiet `3.5px` dot when settled.
 */
export const XerxesThinkingMark = memo(function XerxesThinkingMark(
  props: XerxesThinkingMarkProps,
): ReactElement {
  const { size = 16, active = true, className } = props
  const perfPaused = usePerfPaused()
  const classes = [
    'xerxes-thinking-mark',
    active ? 'xerxes-thinking-active' : 'xerxes-thinking-settled',
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      role="img"
      aria-label={active ? 'Xerxes thinking' : 'Xerxes ready'}
      className={classes}
      data-thinking-active={active ? 'true' : 'false'}
      data-emblem="xerxes-progress"
      style={{ display: 'inline-block', flexShrink: 0 }}
    >
      {active ? (
        <>
          <circle
            cx="12"
            cy="12"
            r="8.5"
            fill="none"
            stroke="var(--x-hairline, currentColor)"
            strokeWidth="1.75"
          />
          <g className={perfPaused ? undefined : 'xerxes-mark-spin'}>
            <path
              d="M 12 3.5 A 8.5 8.5 0 0 1 19.36 16.25"
              fill="none"
              stroke="var(--x-working, var(--x-accent, currentColor))"
              strokeWidth="1.75"
              strokeLinecap="round"
            />
          </g>
        </>
      ) : (
        <circle cx="12" cy="12" r="3.5" fill="var(--x-meta, currentColor)" />
      )}
    </svg>
  )
})

/**
 * Maps every Xerxes status state (including legacy `OrbState` names) onto one
 * of the five canonical semantic visual tones.
 */
export function resolveXerxesStatusTone(state: XerxesStatusState): XerxesStatusTone {
  switch (state) {
    case 'idle':
    case 'breathing':
    case 'ready':
      return 'idle'
    case 'connecting':
    case 'waiting':
    case 'approval':
    case 'needs':
      return 'waiting'
    case 'error':
    case 'failed':
      return 'error'
    case 'done':
    case 'success':
      return 'done'
    default:
      return 'active'
  }
}

const TONE_COLORS: Readonly<Record<XerxesStatusTone, string>> = {
  active: 'var(--x-working, var(--x-accent, currentColor))',
  idle: 'var(--x-meta, currentColor)',
  waiting: 'var(--x-needs, currentColor)',
  error: 'var(--x-failed, currentColor)',
  done: 'var(--x-done, currentColor)',
}

/**
 * App-wide minimal semantic status mark (`AgentOrb`, rail, agents, supervisor):
 * - `active`: thin `1.75px` open-arc spinner inside a faint hairline track
 * - `waiting`: quiet `1.75px` hollow circle (`r=4.5`)
 * - `done`: minimal `1.75px` check stroke
 * - `error`: minimal `1.75px` cross stroke
 * - `idle`: quiet `3.5px` solid dot
 */
export function XerxesStatusMark(props: XerxesStatusMarkProps): ReactElement {
  const { state, size = 20, live = true, className, label } = props
  const tone = resolveXerxesStatusTone(state)
  const perfPaused = usePerfPaused()
  const color = TONE_COLORS[tone]
  const classes = ['xerxes-status-mark', `xerxes-status-mark--${tone}`, className ?? '']
    .filter(Boolean)
    .join(' ')

  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      role={label ? 'img' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      className={classes}
      data-emblem="xerxes-status"
      data-xerxes-status-tone={tone}
      data-xerxes-status-state={state}
      data-live={live ? 'true' : 'false'}
      style={{ display: 'inline-block', flexShrink: 0 }}
    >
      {tone === 'active' ? (
        <>
          <circle
            cx="12"
            cy="12"
            r="8.5"
            fill="none"
            stroke="var(--x-hairline, currentColor)"
            strokeWidth="1.75"
          />
          <g className={live && !perfPaused ? 'xerxes-mark-spin' : undefined}>
            <path
              d="M 12 3.5 A 8.5 8.5 0 0 1 19.36 16.25"
              fill="none"
              stroke={color}
              strokeWidth="1.75"
              strokeLinecap="round"
            />
          </g>
        </>
      ) : tone === 'waiting' ? (
        <circle
          cx="12"
          cy="12"
          r="4.5"
          fill="none"
          stroke={color}
          strokeWidth="1.75"
        />
      ) : tone === 'done' ? (
        <path
          d="M 6.5 12.5 L 10.2 16.2 L 17.5 8.5"
          fill="none"
          stroke={color}
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      ) : tone === 'error' ? (
        <path
          d="M 8 8 L 16 16 M 16 8 L 8 16"
          fill="none"
          stroke={color}
          strokeWidth="1.75"
          strokeLinecap="round"
        />
      ) : (
        <circle cx="12" cy="12" r="3.5" fill={color} />
      )}
    </svg>
  )
}

// ---------------------------------------------------------------------------
// 4. Optical Runtime Brand Glyphs (24×24)
// ---------------------------------------------------------------------------

const PROVIDER_LABELS: Readonly<Record<AgentBrandProvider, string>> = {
  xerxes: 'Xerxes',
  'claude-code': 'Claude Code',
  antigravity: 'Antigravity',
  codex: 'Codex',
  acp: 'ACP Agent',
}

/** Official Anthropic Claude sunburst path (`viewBox="0 0 24 24"`, Simple Icons). */
const CLAUDE_BRAND_PATH =
  'm4.714 15.956l4.718-2.648l.079-.23l-.08-.128h-.23l-.79-.048l-2.695-.073l-2.337-.097l-2.265-.122l-.57-.121l-.535-.704l.055-.353l.48-.321l.685.06l1.518.104l2.277.157l1.651.098l2.447.255h.389l.054-.158l-.133-.097l-.103-.098l-2.356-1.596l-2.55-1.688l-1.336-.972l-.722-.491L2 6.223l-.158-1.008l.656-.722l.88.06l.224.061l.893.686l1.906 1.476l2.49 1.833l.364.304l.146-.104l.018-.072l-.164-.274l-1.354-2.446l-1.445-2.49l-.644-1.032l-.17-.619a3 3 0 0 1-.103-.729L6.287.133L6.7 0l.995.134l.42.364l.619 1.415L9.735 4.14l1.555 3.03l.455.898l.243.832l.09.255h.159V9.01l.127-1.706l.237-2.095l.23-2.695l.08-.76l.376-.91l.747-.492l.583.28l.48.685l-.067.444l-.286 1.851l-.558 2.903l-.365 1.942h.213l.243-.242l.983-1.306l1.652-2.064l.728-.82l.85-.904l.547-.431h1.032l.759 1.129l-.34 1.166l-1.063 1.347l-.88 1.142l-1.263 1.7l-.79 1.36l.074.11l.188-.02l2.853-.606l1.542-.28l1.84-.315l.832.388l.09.395l-.327.807l-1.967.486l-2.307.462l-3.436.813l-.043.03l.049.061l1.548.146l.662.036h1.62l3.018.225l.79.522l.473.638l-.08.485l-1.213.62l-1.64-.389l-3.825-.91l-1.31-.329h-.183v.11l1.093 1.068l2.003 1.81l2.508 2.33l.127.578l-.321.455l-.34-.049l-2.204-1.657l-.85-.747l-1.925-1.62h-.127v.17l.443.649l2.343 3.521l.122 1.08l-.17.353l-.607.213l-.668-.122l-1.372-1.924l-1.415-2.168l-1.141-1.943l-.14.08l-.674 7.254l-.316.37l-.728.28l-.607-.461l-.322-.747l.322-1.476l.388-1.924l.316-1.53l.285-1.9l.17-.632l-.012-.042l-.14.018l-1.432 1.967l-2.18 2.945l-1.724 1.845l-.413.164l-.716-.37l.066-.662l.401-.589l2.386-3.036l1.439-1.882l.929-1.086l-.006-.158h-.055L4.138 18.56l-1.13.146l-.485-.456l.06-.746l.231-.243l1.907-1.312Z'

/** Official Google Antigravity arch silhouette (`viewBox="-0.8 0 25.6 24.5"`). */
const ANTIGRAVITY_BRAND_PATH =
  'M21.751 22.607c1.34 1.005 3.35.335 1.508-1.508C17.73 15.74 18.904 1 12.037 1S6.342 15.74.815 21.1c-2.01 2.009.167 2.511 1.507 1.506c5.192-3.517 4.857-9.714 9.715-9.714s4.522 6.197 9.714 9.715'

/** Official OpenAI blossom path (`viewBox="0 0 24 24"`, Simple Icons). */
const OPENAI_BRAND_PATH =
  'M22.282 9.821a6 6 0 0 0-.516-4.91a6.05 6.05 0 0 0-6.51-2.9A6.065 6.065 0 0 0 4.981 4.18a6 6 0 0 0-3.998 2.9a6.05 6.05 0 0 0 .743 7.097a5.98 5.98 0 0 0 .51 4.911a6.05 6.05 0 0 0 6.515 2.9A6 6 0 0 0 13.26 24a6.06 6.06 0 0 0 5.772-4.206a6 6 0 0 0 3.997-2.9a6.06 6.06 0 0 0-.747-7.073M13.26 22.43a4.48 4.48 0 0 1-2.876-1.04l.141-.081l4.779-2.758a.8.8 0 0 0 .392-.681v-6.737l2.02 1.168a.07.07 0 0 1 .038.052v5.583a4.504 4.504 0 0 1-4.494 4.494M3.6 18.304a4.47 4.47 0 0 1-.535-3.014l.142.085l4.783 2.759a.77.77 0 0 0 .78 0l5.843-3.369v2.332a.08.08 0 0 1-.033.062L9.74 19.95a4.5 4.5 0 0 1-6.14-1.646M2.34 7.896a4.5 4.5 0 0 1 2.366-1.973V11.6a.77.77 0 0 0 .388.677l5.815 3.354l-2.02 1.168a.08.08 0 0 1-.071 0l-4.83-2.786A4.504 4.504 0 0 1 2.34 7.872zm16.597 3.855l-5.833-3.387L15.119 7.2a.08.08 0 0 1 .071 0l4.83 2.791a4.494 4.494 0 0 1-.676 8.105v-5.678a.79.79 0 0 0-.407-.667m2.01-3.023l-.141-.085l-4.774-2.782a.78.78 0 0 0-.785 0L9.409 9.23V6.897a.07.07 0 0 1 .028-.061l4.83-2.787a4.5 4.5 0 0 1 6.68 4.66zm-12.64 4.135l-2.02-1.164a.08.08 0 0 1-.038-.057V6.075a4.5 4.5 0 0 1 7.375-3.453l-.142.08L8.704 5.46a.8.8 0 0 0-.393.681zm1.097-2.365l2.602-1.5l2.607 1.5v2.999l-2.597 1.5l-2.607-1.5Z'

/**
 * Clean optical brand glyphs for the ACP runtime switcher and agent identity
 * badges across Desktop and VS Code. `xerxes` renders the crisp Faceted
 * Architectural X-Prism recognizable down to 11×11; external runtimes render
 * their official vector brand marks (Claude, Google Antigravity, OpenAI Codex).
 */
export const AgentBrandGlyph = memo(function AgentBrandGlyph(
  props: AgentBrandGlyphProps,
): ReactElement {
  const { provider, size = 16, className } = props
  const svgUid = useId().replace(/[^a-zA-Z0-9_-]/g, '')
  const classes = [
    'xerxes-brand-glyph',
    `xerxes-brand-glyph-${provider}`,
    className ?? '',
  ]
    .filter(Boolean)
    .join(' ')

  if (provider === 'xerxes') {
    return (
      <img
        src="./logo-128.png"
        alt={PROVIDER_LABELS.xerxes}
        width={size}
        height={size}
        className={classes}
        data-brand-provider={provider}
        decoding="async"
        draggable={false}
        style={{
          display: 'block',
          flexShrink: 0,
          objectFit: 'cover',
          borderRadius: Math.max(3, Math.round(size * 0.22)),
        }}
      />
    )
  }

  if (provider === 'claude-code') {
    return (
      <svg
        viewBox="0 0 24 24"
        width={size}
        height={size}
        role="img"
        aria-label={PROVIDER_LABELS['claude-code']}
        className={classes}
        data-brand-provider={provider}
        style={{ display: 'block', flexShrink: 0 }}
      >
        <path d={CLAUDE_BRAND_PATH} fill="rgb(217, 119, 87)" />
      </svg>
    )
  }

  if (provider === 'antigravity') {
    const clipId = `agy-clip-${svgUid}`
    const greenId = `agy-g-${svgUid}`
    const yellowId = `agy-y-${svgUid}`
    const redId = `agy-r-${svgUid}`
    const blueId = `agy-b-${svgUid}`
    return (
      <svg
        viewBox="-0.5 0.8 25 24.2"
        width={size}
        height={size}
        role="img"
        aria-label={PROVIDER_LABELS.antigravity}
        className={classes}
        data-brand-provider={provider}
        style={{ display: 'block', flexShrink: 0 }}
      >
        <defs>
          <clipPath id={clipId}>
            <path d={ANTIGRAVITY_BRAND_PATH} />
          </clipPath>
          <radialGradient id={greenId} cx="16%" cy="54%" r="58%">
            <stop offset="0%" stopColor="rgb(0, 185, 92)" stopOpacity="1" />
            <stop offset="100%" stopColor="rgb(0, 185, 92)" stopOpacity="0" />
          </radialGradient>
          <radialGradient id={yellowId} cx="36%" cy="16%" r="50%">
            <stop offset="0%" stopColor="rgb(251, 188, 4)" stopOpacity="1" />
            <stop offset="100%" stopColor="rgb(251, 188, 4)" stopOpacity="0" />
          </radialGradient>
          <radialGradient id={redId} cx="78%" cy="30%" r="54%">
            <stop offset="0%" stopColor="rgb(252, 65, 61)" stopOpacity="1" />
            <stop offset="100%" stopColor="rgb(252, 65, 61)" stopOpacity="0" />
          </radialGradient>
          <radialGradient id={blueId} cx="70%" cy="82%" r="56%">
            <stop offset="0%" stopColor="rgb(49, 134, 255)" stopOpacity="0.96" />
            <stop offset="100%" stopColor="rgb(49, 134, 255)" stopOpacity="0" />
          </radialGradient>
        </defs>
        <g clipPath={`url(#${clipId})`}>
          <rect x="-1" y="0" width="26" height="26" fill="rgb(49, 134, 255)" />
          <rect x="-1" y="0" width="26" height="26" fill={`url(#${greenId})`} />
          <rect x="-1" y="0" width="26" height="26" fill={`url(#${yellowId})`} />
          <rect x="-1" y="0" width="26" height="26" fill={`url(#${redId})`} />
          <rect x="-1" y="0" width="26" height="26" fill={`url(#${blueId})`} />
        </g>
      </svg>
    )
  }

  if (provider === 'codex') {
    return (
      <svg
        viewBox="0 0 24 24"
        width={size}
        height={size}
        role="img"
        aria-label={PROVIDER_LABELS.codex}
        className={classes}
        data-brand-provider={provider}
        style={{ display: 'block', flexShrink: 0, color: 'var(--x-title, currentColor)' }}
      >
        <path d={OPENAI_BRAND_PATH} fill="currentColor" />
      </svg>
    )
  }

  // 'acp': minimalist two-node protocol link
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      role="img"
      aria-label={PROVIDER_LABELS.acp}
      className={classes}
      data-brand-provider={provider}
      style={{ display: 'block', flexShrink: 0 }}
    >
      <g fill="none" strokeLinecap="round" strokeLinejoin="round">
        <line
          x1="9.6"
          y1="12"
          x2="14.4"
          y2="12"
          stroke="var(--x-secondary, rgb(166, 162, 156))"
          strokeWidth="1.5"
          strokeDasharray="1.8 2.2"
        />
        <circle
          cx="6.4"
          cy="12"
          r="3.2"
          fill="var(--x-accent, rgb(196, 132, 96))"
          fillOpacity="0.12"
          stroke="var(--x-accent, rgb(196, 132, 96))"
          strokeWidth="1.5"
        />
        <circle
          cx="17.6"
          cy="12"
          r="3.2"
          fill="var(--x-title, rgb(236, 232, 224))"
          fillOpacity="0.08"
          stroke="var(--x-title, rgb(236, 232, 224))"
          strokeWidth="1.5"
        />
        <circle cx="6.4" cy="12" r="1.05" fill="var(--x-accent, rgb(196, 132, 96))" />
        <circle cx="17.6" cy="12" r="1.05" fill="var(--x-title, rgb(236, 232, 224))" />
      </g>
    </svg>
  )
})
