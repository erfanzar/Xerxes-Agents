// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ReactElement } from 'react'

import { isVscodeHost } from './host.js'
import { AgentBrandGlyph, XerxesBrandEmblem, type AgentBrandProvider } from './XerxesSprites.js'

/**
 * Renders Xerxes's real bundled phoenix emblem (`./logo-128.png`) on Desktop,
 * falling back to the compact vector mark only inside VS Code webviews where
 * the desktop asset bundle is not served from the webview root.
 */
export function XerxesLogoMark({
  size = 20,
  className,
  alt = 'Xerxes',
  decorative = false,
}: {
  readonly size?: number
  readonly className?: string
  readonly alt?: string
  readonly decorative?: boolean
}): ReactElement {
  if (isVscodeHost()) {
    return (
      <XerxesBrandEmblem
        size={size}
        animate={false}
        {...(className ? { className } : {})}
      />
    )
  }
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
