// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ReactElement } from 'react'

import { AgentBrandGlyph, type AgentBrandProvider } from './XerxesSprites.js'

/**
 * Minimum icon dimension (px) for the real bundled Xerxes phoenix PNG
 * (`0` — always use `./logo-128.png` across Desktop and VS Code so the real
 * Xerxes phoenix icon appears everywhere at every size).
 */
export const RASTER_BRAND_MIN_SIZE = 0

/**
 * Renders Xerxes's real bundled phoenix emblem (`./logo-128.png`) at every size
 * across both Desktop and VS Code webviews (`<base href=".../media/renderer/">`).
 */
export function XerxesLogoMark({
  size = 20,
  className,
  alt = 'Xerxes Agents',
  decorative = false,
}: {
  readonly size?: number
  readonly className?: string
  readonly alt?: string
  readonly decorative?: boolean
}): ReactElement {
  const classes = ['xerxes-logo-mark', 'xerxes-brand-emblem', className ?? ''].filter(Boolean).join(' ')
  return (
    <img
      src="./logo-128.png"
      alt={decorative ? '' : alt}
      aria-hidden={decorative ? true : undefined}
      width={size}
      height={size}
      className={classes}
      data-brand-provider="xerxes"
      decoding="async"
      draggable={false}
    />
  )
}

/**
 * Renders Xerxes's real bundled phoenix emblem (`./logo-128.png`) on Desktop for
 * the built-in `xerxes` runtime, and official vector provider logos for Claude
 * Code, Google Antigravity, OpenAI Codex, and custom ACP runtimes.
 */
export function RuntimeBrandIcon({
  provider,
  size = 16,
  className,
}: {
  readonly provider: AgentBrandProvider
  readonly size?: number
  readonly className?: string
}): ReactElement {
  if (provider === 'xerxes') {
    return <XerxesLogoMark size={size} {...(className ? { className } : {})} />
  }
  return (
    <AgentBrandGlyph
      provider={provider}
      size={size}
      animate={false}
      {...(className ? { className } : {})}
    />
  )
}
