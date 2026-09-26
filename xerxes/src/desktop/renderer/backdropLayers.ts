// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Keeps each surface's blurred background layer aligned with the window
 * background.
 *
 * The surfaces (toolbar, sidebar, conversation, rail) used `backdrop-filter`.
 * The browser has to re-blur a backdrop-filtered element across its whole
 * area whenever anything inside it repaints — every frame of a spinner, every
 * streamed token, the ticking clock — which kept the GPU process near 50% CPU
 * during a turn. Each surface now paints a static, pre-blurred copy of the
 * background in a layer behind its content (atelier.css). A static layer is
 * rasterized once; a spinner then redraws only its own few pixels.
 *
 * The layer shows the part of the background that sits under the surface, so
 * it needs two things the CSS cannot know on its own: where the surface is in
 * the window (`--bd-x`/`--bd-y` on each surface) and where the image lands
 * under `cover` (`--bd-img-*` on the root). Both change only on resize or when
 * a surface mounts, so this runs on those events, never per frame.
 */

const SURFACES = '.atelier .top, .atelier .side, .atelier .chat, .atelier .desktop-rail'

let natural: { readonly width: number; readonly height: number } | undefined
let imageSrc: string | undefined
let started = false
let scheduled = false
const observed = new WeakSet<Element>()
let resizeObserver: ResizeObserver | undefined

/** Where `background-size: cover; background-position: center` puts the image in the window. */
export function coverRect(image: { width: number; height: number }, viewport: { width: number; height: number }): { x: number; y: number; width: number; height: number } {
  const scale = Math.max(viewport.width / image.width, viewport.height / image.height)
  const width = image.width * scale, height = image.height * scale
  return { x: (viewport.width - width) / 2, y: (viewport.height - height) / 2, width, height }
}

function update(): void {
  scheduled = false
  const root = document.documentElement
  if (natural) {
    const cover = coverRect(natural, { width: window.innerWidth, height: window.innerHeight })
    root.style.setProperty('--bd-img-x', `${cover.x}px`)
    root.style.setProperty('--bd-img-y', `${cover.y}px`)
    root.style.setProperty('--bd-img-w', `${cover.width}px`)
    root.style.setProperty('--bd-img-h', `${cover.height}px`)
  }
  for (const surface of document.querySelectorAll<HTMLElement>(SURFACES)) {
    const rect = surface.getBoundingClientRect()
    surface.style.setProperty('--bd-x', `${rect.left}px`)
    surface.style.setProperty('--bd-y', `${rect.top}px`)
    if (!observed.has(surface)) { observed.add(surface); resizeObserver?.observe(surface) }
  }
}

function schedule(): void {
  if (scheduled) return
  scheduled = true
  requestAnimationFrame(update)
}

function start(): void {
  if (started || typeof window === 'undefined' || typeof ResizeObserver === 'undefined') return
  started = true
  // One surface changing size moves its neighbours (a collapsing sidebar
  // shifts the conversation and the rail), so any change re-measures all.
  resizeObserver = new ResizeObserver(schedule)
  window.addEventListener('resize', schedule)
  // Surfaces mount and unmount (the rail opens and closes). Streaming text
  // also mutates the tree constantly, so only additions that are, or
  // contain, a surface schedule anything.
  new MutationObserver(records => {
    for (const record of records) {
      for (const node of record.addedNodes) {
        if (node instanceof Element && (node.matches(SURFACES) || node.querySelector(SURFACES))) { schedule(); return }
      }
    }
  }).observe(document.body, { childList: true, subtree: true })
  schedule()
}

/** Called when the background changes: learn the image's size, then align. */
export function trackBackdropImage(src: string | undefined): void {
  start()
  imageSrc = src
  if (!src) { natural = undefined; schedule(); return }
  const image = new Image()
  image.onload = () => {
    if (imageSrc !== src) return
    natural = { width: image.naturalWidth, height: image.naturalHeight }
    schedule()
  }
  image.src = src
}
