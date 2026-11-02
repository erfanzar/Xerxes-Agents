// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Pictures in the chat: a markdown image in a reply, or an image a tool step
 * already carries. The page's CSP allows only `'self'` and `data:` images,
 * so a file is never referenced by path: the host (Electron main, the VS
 * Code extension) reads it through a validated call and returns a data URL.
 * Web images are never fetched; they show as a link that opens in the
 * browser. Anything that cannot be shown says why in one inline line.
 * Images are bounded, load when scrolled near, and open in a lightbox.
 */

import { createContext, memo, useContext, useEffect, useMemo, useRef, useState, type ReactElement } from 'react'
import { createPortal } from 'react-dom'

import {
  chatImageDataUrl, checkChatImagePath, externalUrlOf, fileUrlPath,
  type ChatImageResult,
} from '../chatMediaShared.js'
import { ChatLink } from './chatLinks.js'
import { useDialogFocus } from './dialogFocus.js'
import { desktopError } from './desktopRpc.js'
import { Icon } from './Icon.js'
import { store } from './store.js'
import { toolImagesOf } from './toolImages.js'
import type { ToolItem, XerxesBridge } from './types.js'

/** Where image paths resolve: the task's folder and the session whose runtime reads them. */
export interface ChatMediaScopeValue {
  readonly cwd?: string
  readonly sessionKey?: string
}

/** Set by a surface showing another task's text; otherwise the open task's folder is used. */
export const ChatMediaScope = createContext<ChatMediaScopeValue | null>(null)

export type ImageSource =
  | { readonly kind: 'data'; readonly url: string }
  | { readonly kind: 'web'; readonly url: string }
  | { readonly kind: 'file'; readonly path: string }
  | { readonly kind: 'refused'; readonly error: string }

const refused = (error: string): ImageSource => ({ kind: 'refused', error })
function fileSource(path: string): ImageSource {
  const checked = checkChatImagePath(path)
  return 'error' in checked ? refused(checked.error) : { kind: 'file', path: checked.path }
}

/** What an image reference is, decided from its text alone (no file or network access). */
export function imageSourceOf(raw: string): ImageSource {
  const src = raw.trim()
  if (/^data:/i.test(src)) {
    const result = chatImageDataUrl(src.replaceAll(/\s/g, ''))
    return result.ok ? { kind: 'data', url: result.url } : refused(result.error)
  }
  if (/^https?:/i.test(src)) {
    const url = externalUrlOf(src)
    return url ? { kind: 'web', url } : refused('the link is not a valid web address')
  }
  if (/^file:/i.test(src)) {
    const path = fileUrlPath(src)
    return path ? fileSource(path) : refused('only file URLs on this computer can be shown')
  }
  if (/^[a-z][a-z0-9+.-]*:/i.test(src) && !/^[a-z]:[\\/]/i.test(src)) return refused('only file paths, data URLs and web links can be images')
  return fileSource(src)
}

const CACHE_LIMIT = 24
const cache = new Map<string, ChatImageResult>()

/** Read a file image through the host. Never rejects: a failure is a result with its reason. */
export async function loadChatImage(
  path: string,
  scope: ChatMediaScopeValue,
  bridge: Pick<XerxesBridge, 'imageData'> | undefined = typeof window === 'undefined' ? undefined : window.xerxes,
): Promise<ChatImageResult> {
  const key = `${scope.sessionKey ?? ''}\0${scope.cwd ?? ''}\0${path}`
  const hit = cache.get(key)
  if (hit) return hit
  if (!bridge?.imageData) return { ok: false, error: 'this version of the app cannot read image files' }
  let result: ChatImageResult
  try {
    result = await bridge.imageData({ path, ...(scope.cwd ? { cwd: scope.cwd } : {}), ...(scope.sessionKey ? { session_key: scope.sessionKey } : {}) })
  } catch (error) {
    return { ok: false, error: desktopError(error) || 'the image could not be read' }
  }
  // Only successes are kept: a missing file may appear once the agent writes it.
  if (result.ok) {
    cache.set(key, result)
    while (cache.size > CACHE_LIMIT) { const oldest = cache.keys().next().value; if (oldest === undefined) break; cache.delete(oldest) }
  }
  return result
}

function scopeFromStore(): ChatMediaScopeValue {
  const snap = store.getSnapshot()
  return { ...(snap.cwd ? { cwd: snap.cwd } : {}), ...(snap.sessionKey ? { sessionKey: snap.sessionKey } : {}) }
}

function Unavailable({ reason }: { reason: string }): ReactElement {
  return <span className="chatimg__error" role="note">Image unavailable: {reason}</span>
}

function ImageLightbox({ url, alt, close }: { url: string; alt: string; close: () => void }): ReactElement {
  const ref = useRef<HTMLDivElement>(null)
  useDialogFocus(ref)
  useEffect(() => {
    const key = (event: KeyboardEvent): void => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      event.stopImmediatePropagation()
      close()
    }
    window.addEventListener('keydown', key, true)
    return () => window.removeEventListener('keydown', key, true)
  }, [close])
  const layer = (
    <div className="atelier chatimg-layer" role="presentation" onClick={close}>
      <div ref={ref} className="chatimg-dialog" role="dialog" aria-modal="true" aria-label={alt || 'Image'} tabIndex={-1} onClick={event => event.stopPropagation()}>
        <img src={url} alt={alt} decoding="async" />
        <div className="chatimg-dialog__bar">
          <span title={alt}>{alt}</span>
          <button type="button" aria-label="Close image" title="Close (Esc)" onClick={close}><Icon name="close" size={14} /></button>
        </div>
      </div>
    </div>
  )
  return typeof document === 'undefined' ? layer : createPortal(layer, document.body)
}

function Shown({ url, alt }: { url: string; alt: string }): ReactElement {
  const [open, setOpen] = useState(false)
  const [broken, setBroken] = useState(false)
  if (broken) return <Unavailable reason="the image data could not be decoded" />
  return (
    <span className="chatimg">
      <button type="button" className="chatimg__open" aria-label={alt ? `Enlarge image: ${alt}` : 'Enlarge image'} title={alt || 'Enlarge image'} onClick={event => { event.stopPropagation(); setOpen(true) }}>
        <img className="chatimg__img" src={url} alt={alt} loading="lazy" decoding="async" onError={() => setBroken(true)} />
      </button>
      {open && <ImageLightbox url={url} alt={alt} close={() => setOpen(false)} />}
    </span>
  )
}

/** A file image: read through the host once it nears the viewport. */
function FileImage({ path, alt }: { path: string; alt: string }): ReactElement {
  const given = useContext(ChatMediaScope)
  const anchor = useRef<HTMLSpanElement>(null)
  const [visible, setVisible] = useState(false)
  const [loaded, setLoaded] = useState<{ readonly path: string; readonly result: ChatImageResult } | null>(null)
  useEffect(() => {
    if (visible) return
    const node = anchor.current
    if (!node || typeof IntersectionObserver === 'undefined') { setVisible(true); return }
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) { setVisible(true); observer.disconnect() }
    }, { rootMargin: '400px' })
    observer.observe(node)
    return () => observer.disconnect()
  }, [visible])
  useEffect(() => {
    if (!visible) return
    let live = true
    void loadChatImage(path, given ?? scopeFromStore()).then(result => { if (live) setLoaded({ path, result }) })
    return () => { live = false }
  }, [visible, path, given?.cwd, given?.sessionKey])
  if (loaded?.path !== path) return <span ref={anchor} className="chatimg__loading" title={path}>Loading image{alt ? `: ${alt}` : ''}…</span>
  return loaded.result.ok ? <Shown url={loaded.result.url} alt={alt} /> : <Unavailable reason={loaded.result.error} />
}

/** One image reference from a reply or a tool step. */
export const ChatImage = memo(function ChatImage({ src, alt }: { src: string; alt: string }): ReactElement {
  const source = useMemo(() => imageSourceOf(src), [src])
  if (source.kind === 'data') return <Shown url={source.url} alt={alt} />
  if (source.kind === 'file') return <FileImage path={source.path} alt={alt} />
  if (source.kind === 'web') {
    return <span className="chatimg__web">Web image (not loaded): <ChatLink label={alt || source.url} href={source.url} /></span>
  }
  return <Unavailable reason={source.error} />
})

/** The pictures a tool step carries, under its output. Nothing when it has none. */
export function ToolImages({ item }: { item: Pick<ToolItem, 'name' | 'input' | 'output'> }): ReactElement | null {
  const images = toolImagesOf(item)
  if (!images.length) return null
  return (
    <div className="toolimages" role="group" aria-label={images.length === 1 ? 'Image' : `${images.length} images`}>
      {images.map(image => <ChatImage key={image.src.length > 256 ? `${image.src.length}:${image.src.slice(-64)}` : image.src} src={image.src} alt={image.alt} />)}
    </div>
  )
}
