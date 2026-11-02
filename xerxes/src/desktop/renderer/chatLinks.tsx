// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Links in the chat. An http, https or mailto link opens in the person's
 * default browser or mail app through the host (Electron main's
 * shell.openExternal, VS Code's env.openExternal), which checks the scheme
 * again. A link to a file path opens that file in the Files panel. Anything
 * else (javascript:, in-page anchors, unknown schemes) stays plain text with
 * its target as the tooltip, so a click can never navigate the app window.
 */

import { useState, type KeyboardEvent, type MouseEvent, type ReactElement, type ReactNode } from 'react'

import { externalUrlOf, fileUrlPath } from '../chatMediaShared.js'
import { useDesktopNavigation } from './DesktopPanels.js'
import { desktopError } from './desktopRpc.js'
import type { XerxesBridge } from './types.js'

export type LinkTarget =
  | { readonly kind: 'external'; readonly url: string }
  | { readonly kind: 'file'; readonly path: string }
  | { readonly kind: 'inert' }

const INERT: LinkTarget = { kind: 'inert' }
const SCHEME = /^[a-z][a-z0-9+.-]*:/i
const WINDOWS_DRIVE = /^[a-z]:[\\/]/i

function fileTarget(raw: string): LinkTarget {
  // `src/a.ts:42`, `src/a.ts:42:7` and `src/a.ts#L42` all open the file.
  let path = raw.replace(/#L\d+(?:-L?\d+)?$/, '').replace(/:\d+(?::\d+)?$/, '')
  try { path = decodeURIComponent(path) } catch { /* a literal % stays as written */ }
  return path && !/[\x00-\x1f\x7f]/.test(path) ? { kind: 'file', path } : INERT
}

/** What a markdown link points at, decided from its text alone. */
export function linkTargetOf(href: string): LinkTarget {
  const value = href.trim()
  if (!value || value.length > 4096 || /[\x00-\x1f\x7f]/.test(value)) return INERT
  if (value.startsWith('#') || value.startsWith('?') || value.startsWith('//')) return INERT
  if (/^(?:https?|mailto):/i.test(value)) {
    const url = externalUrlOf(value)
    return url ? { kind: 'external', url } : INERT
  }
  if (/^file:/i.test(value)) {
    const path = fileUrlPath(value)
    return path ? fileTarget(path) : INERT
  }
  if (SCHEME.test(value) && !WINDOWS_DRIVE.test(value)) return INERT
  return fileTarget(value)
}

/**
 * Open a web or mail link outside the app. The host re-validates; an older
 * bridge without the call falls back to window.open, which Electron's window
 * open handler also routes to the system browser.
 */
export async function openExternalLink(url: string, bridge: Pick<XerxesBridge, 'openExternal'> | undefined = typeof window === 'undefined' ? undefined : window.xerxes): Promise<void> {
  const clean = externalUrlOf(url)
  if (!clean) throw new Error('Only http, https and mailto links open outside the app')
  if (bridge?.openExternal) {
    if (await bridge.openExternal(clean) === false) throw new Error('The system could not open the link')
    return
  }
  if (typeof window !== 'undefined' && typeof window.open === 'function') {
    window.open(clean, '_blank', 'noopener,noreferrer')
    return
  }
  throw new Error('This app cannot open links')
}

/** One chat link: external, a workspace file, or inert text. Keyboard reachable either way it acts. */
export function ChatLink({ label, href, className }: { label: ReactNode; href: string; className?: string }): ReactElement {
  const navigate = useDesktopNavigation()
  const [failure, setFailure] = useState('')
  const target = linkTargetOf(href)
  const cls = className ? `md__link ${className}` : 'md__link'
  if (target.kind === 'external') {
    const open = (): void => {
      setFailure('')
      openExternalLink(target.url).catch(error => setFailure(desktopError(error) || 'could not open the link'))
    }
    return <>
      <a
        className={cls}
        href={href}
        title={href}
        rel="noreferrer noopener"
        data-failed={failure ? '' : undefined}
        onClick={(event: MouseEvent) => { event.preventDefault(); event.stopPropagation(); open() }}
        // A middle click would otherwise reach the window-open handler unchecked.
        onAuxClick={(event: MouseEvent) => { event.preventDefault(); event.stopPropagation(); if (event.button === 1) open() }}
      >{label}</a>
      {failure && <span className="md__link-error" role="status"> (could not open: {failure})</span>}
    </>
  }
  if (target.kind === 'file') {
    const open = (): void => navigate('files', target.path)
    // No href: a file link must never become a file:// navigation or window.open.
    return <a
      className={`${cls} md__link--file`}
      role="link"
      tabIndex={0}
      title={href}
      onClick={(event: MouseEvent) => { event.preventDefault(); event.stopPropagation(); open() }}
      onKeyDown={(event: KeyboardEvent) => { if (event.key === 'Enter') { event.preventDefault(); event.stopPropagation(); open() } }}
    >{label}</a>
  }
  return <span className="md__link md__link--inert" title={href}>{label}</span>
}

const BARE_URL = /\b(?:https?:\/\/|mailto:)[^\s<>"'`]+/gi

/** Trim prose punctuation and an unbalanced closing bracket off a bare URL. */
function trimBareUrl(raw: string): string {
  let url = raw
  for (;;) {
    const last = url.at(-1)
    if (last && '.,;:!?\'"*_'.includes(last)) { url = url.slice(0, -1); continue }
    if (last === ')' && (url.match(/\(/g)?.length ?? 0) < (url.match(/\)/g)?.length ?? 0)) { url = url.slice(0, -1); continue }
    if (last === ']' && (url.match(/\[/g)?.length ?? 0) < (url.match(/]/g)?.length ?? 0)) { url = url.slice(0, -1); continue }
    return url
  }
}

/** Prose with its bare http(s) and mailto URLs turned into links. */
export function autolinked(text: string, nextKey: () => number): ReactNode[] {
  const out: ReactNode[] = []
  let last = 0
  for (const match of text.matchAll(BARE_URL)) {
    const start = match.index ?? 0
    const url = trimBareUrl(match[0])
    if (!externalUrlOf(url)) continue
    if (start > last) out.push(text.slice(last, start))
    out.push(<ChatLink key={`u${nextKey()}`} label={url} href={url} />)
    last = start + url.length
  }
  if (!out.length) return [text]
  if (last < text.length) out.push(text.slice(last))
  return out
}
