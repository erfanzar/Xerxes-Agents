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
import { Icon } from './Icon.js'
import { store, type Snapshot } from './store.js'
import { isSwarmSidecarHref } from './swarmState.js'
import type { XerxesBridge } from './types.js'

export type LinkTarget =
  | { readonly kind: 'external'; readonly url: string }
  | { readonly kind: 'file'; readonly path: string }
  | { readonly kind: 'swarm' }
  | { readonly kind: 'conversation'; readonly conversationId: string }
  | { readonly kind: 'inert' }

export type ConversationResolution =
  | { readonly kind: 'swarm_member'; readonly memberName: string }
  | { readonly kind: 'subagent'; readonly agentId: string }
  | { readonly kind: 'swarm_board' }

const INERT: LinkTarget = { kind: 'inert' }
const SCHEME = /^[a-z][a-z0-9+.-]*:/i
const WINDOWS_DRIVE = /^[a-z]:[\\/]/i
const CONVERSATION_HREF_RE = /^conversation:\/\/([a-zA-Z0-9][a-zA-Z0-9_:-]{1,127})(?:\/?(?:[?#].*)?)?$/i

/** Matches Antigravity's `"Run /swarm board (default shortcut alt+s) to open the live board."` hint. */
export const SWARM_BOARD_SENTENCE_RE =
  /\bRun\s+`?\/(?:swarm|immortals)\s+(?:board|ui)`?(?:\s*\([^)\n]*(?:alt\+s|⌥s)[^)\n]*\))?\s+to\s+open\s+the\s+(?:live\s+)?board\.?/gi

/** True for inline slash hints like `/swarm board`, `/swarm ui`, `/immortals board`, `/immortals ui`. */
export function isSwarmBoardCommand(raw: string): boolean {
  return /^\/(?:swarm|immortals)\s+(?:board|ui)$/i.test(raw.trim())
}

/** Resolve a `conversation://<id>` target against live Immortals members first, then subagents. */
export function resolveConversationTarget(
  conversationId: string,
  snap: Pick<Snapshot, 'swarm' | 'fleet' | 'blocks'>,
): ConversationResolution {
  const needle = conversationId.trim().toLowerCase()
  if (needle) {
    const matchedMember = snap.swarm?.members.find(
      member =>
        member.conversation_id?.trim().toLowerCase() === needle
        || member.name.trim().toLowerCase() === needle,
    )
    if (matchedMember) {
      return { kind: 'swarm_member', memberName: matchedMember.name }
    }
    const matchedFleet = snap.fleet.find(
      row =>
        row.id.trim().toLowerCase() === needle
        || row.key.trim().toLowerCase() === needle
        || row.agentDetails?.requestKey?.trim().toLowerCase() === needle,
    )
    if (matchedFleet) {
      return { kind: 'subagent', agentId: matchedFleet.id }
    }
    for (const block of snap.blocks) {
      if (block.kind !== 'agents') continue
      const matchedAgent = block.members.find(
        member =>
          member.runtimeId?.trim().toLowerCase() === needle
          || member.key.trim().toLowerCase() === needle,
      )
      if (matchedAgent) {
        return { kind: 'subagent', agentId: matchedAgent.runtimeId || matchedAgent.key }
      }
    }
  }
  return { kind: 'swarm_board' }
}

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
  if (isSwarmSidecarHref(value)) return { kind: 'swarm' }
  const conversation = CONVERSATION_HREF_RE.exec(value)
  if (conversation?.[1]) return { kind: 'conversation', conversationId: conversation[1] }
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

export interface SwarmChatLinkDomProps {
  readonly className: string
  readonly role: string
  readonly tabIndex: number
  readonly title: string
  readonly onClick: (event: MouseEvent) => void
  readonly onKeyDown: (event: KeyboardEvent) => void
}

export function SwarmChatLink({
  label,
  cta = 'Open →',
  className,
  navigate,
  onOpenBoard = () => store.openSwarmBoard(),
}: {
  readonly label?: ReactNode
  readonly cta?: string | null
  readonly className?: string
  readonly navigate: (panel: 'swarm') => void
  readonly onOpenBoard?: () => void
}): ReactElement<SwarmChatLinkDomProps> {
  const cls = className ? `md__link ${className}` : 'md__link'
  const displayLabel: ReactNode =
    label === undefined ||
    label === null ||
    label === '' ||
    (typeof label === 'string' && (!label.trim() || isSwarmSidecarHref(label)))
      ? 'Open the Immortals UI'
      : label
  const labelText = typeof displayLabel === 'string' ? displayLabel.trim() : ''
  const showCta = cta !== null && cta !== '' && !labelText.endsWith('→')
  const open = (): void => {
    onOpenBoard()
    navigate('swarm')
  }
  return <a
    className={`${cls} md__link--swarm`}
    role="button"
    tabIndex={0}
    title="Open the Immortals Board"
    onClick={(event: MouseEvent) => { event.preventDefault(); event.stopPropagation(); open() }}
    onKeyDown={(event: KeyboardEvent) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        event.stopPropagation()
        open()
      }
    }}
  ><Icon name="tree" size={13} /><span className="md__link-swarm-label">{displayLabel}</span>{showCta && <span className="md__link-swarm-cta" aria-hidden="true">{cta}</span>}</a>
}

export function ConversationChatLink({
  conversationId,
  label,
  className,
  navigate,
  getSnapshot = () => store.getSnapshot(),
  onOpenSwarmMember = (name: string) => store.openSwarmBoard(name),
  onOpenConversation = (id: string) => { void store.openSwarmConversation(id) },
}: {
  readonly conversationId: string
  readonly label?: ReactNode
  readonly className?: string
  readonly navigate: (panel: 'swarm' | 'activity', filePath?: string) => void
  readonly getSnapshot?: () => Pick<Snapshot, 'swarm' | 'fleet' | 'blocks'>
  readonly onOpenSwarmMember?: (memberName: string) => void
  readonly onOpenConversation?: (conversationId: string) => void
}): ReactElement<SwarmChatLinkDomProps> {
  const cls = className ? `md__link ${className}` : 'md__link'
  const displayLabel: ReactNode =
    label === undefined ||
    label === null ||
    label === '' ||
    (typeof label === 'string' && (!label.trim() || CONVERSATION_HREF_RE.test(label.trim())))
      ? `Conversation ${conversationId.slice(0, 8)}`
      : label
  const open = (): void => {
    const resolved = resolveConversationTarget(conversationId, getSnapshot())
    if (resolved.kind === 'swarm_member') {
      onOpenSwarmMember(resolved.memberName)
      navigate('swarm')
      return
    }
    if (resolved.kind === 'subagent') {
      navigate('activity', resolved.agentId)
      return
    }
    onOpenConversation(conversationId)
    navigate('swarm')
  }
  return <a
    className={`${cls} md__link--swarm md__link--conversation`}
    role="button"
    tabIndex={0}
    title={`Inspect conversation ${conversationId}`}
    onClick={(event: MouseEvent) => { event.preventDefault(); event.stopPropagation(); open() }}
    onKeyDown={(event: KeyboardEvent) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        event.stopPropagation()
        open()
      }
    }}
  ><Icon name="agents" size={13} /><span className="md__link-swarm-label">{displayLabel}</span><span className="md__link-swarm-cta" aria-hidden="true">Inspect →</span></a>
}

/** One chat link: external, a workspace file, an Immortals Board sidecar link, a conversation link, or inert text. */
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
  if (target.kind === 'swarm') {
    return <SwarmChatLink label={label} {...(className ? { className } : {})} navigate={navigate} />
  }
  if (target.kind === 'conversation') {
    return <ConversationChatLink conversationId={target.conversationId} label={label} {...(className ? { className } : {})} navigate={navigate} />
  }
  return <span className="md__link md__link--inert" title={href}>{label}</span>
}

const AUTOLINK_RE =
  /(\b(?:https?:\/\/|mailto:|conversation:\/\/)[^\s<>"'`]+)|((?:^|[\s([{"'])\/(?:swarm|immortals)\s+(?:board|ui)\b)/gi

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

/** Prose with bare http(s), mailto, conversation:// URLs, and `/swarm board` hints turned into links. */
export function autolinked(text: string, nextKey: () => number): ReactNode[] {
  const out: ReactNode[] = []
  let last = 0
  for (const match of text.matchAll(AUTOLINK_RE)) {
    const start = match.index ?? 0
    const [raw, bareUrl, swarmSlash] = match
    if (swarmSlash) {
      const slashOffset = swarmSlash.indexOf('/')
      const tokenStart = start + slashOffset
      if (tokenStart > last) out.push(text.slice(last, tokenStart))
      out.push(<ChatLink key={`s${nextKey()}`} label="Open Immortals →" href="sidecar://immortals/board" />)
      last = start + raw.length
      continue
    }
    if (!bareUrl) continue
    const url = trimBareUrl(bareUrl)
    if (/^conversation:/i.test(url)) {
      if (linkTargetOf(url).kind !== 'conversation') continue
      if (start > last) out.push(text.slice(last, start))
      out.push(<ChatLink key={`c${nextKey()}`} label={url} href={url} />)
      last = start + url.length
      continue
    }
    if (!externalUrlOf(url)) continue
    if (start > last) out.push(text.slice(last, start))
    out.push(<ChatLink key={`u${nextKey()}`} label={url} href={url} />)
    last = start + url.length
  }
  if (!out.length) return [text]
  if (last < text.length) out.push(text.slice(last))
  return out
}
