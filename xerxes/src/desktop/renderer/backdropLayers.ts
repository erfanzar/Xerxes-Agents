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

const SURFACES = '.atelier .top, .atelier .side, .atelier .chat, .atelier .desktop-rail, .atelier .app__body'

let natural: { readonly width: number; readonly height: number } | undefined
let imageSrc: string | undefined
let loadedImage: HTMLImageElement | undefined
let bakedUrl: string | undefined
let bakedKey = ''
let bakeTimer: ReturnType<typeof setTimeout> | undefined
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
  root.style.setProperty('--bd-vw', `${window.innerWidth}px`)
  root.style.setProperty('--bd-vh', `${window.innerHeight}px`)
  for (const surface of document.querySelectorAll<HTMLElement>(SURFACES)) {
    const rect = surface.getBoundingClientRect()
    surface.style.setProperty('--bd-x', `${rect.left}px`)
    surface.style.setProperty('--bd-y', `${rect.top}px`)
    if (!observed.has(surface)) { observed.add(surface); resizeObserver?.observe(surface) }
  }
  scheduleBake()
}

/**
 * Blur the picture ONCE, into a bitmap the surfaces draw as a plain image.
 *
 * A CSS `filter` on a composited layer is re-run by the GPU on every frame
 * the window draws — and while a turn streams (or any spinner turns) that is
 * every frame, for every surface. The bitmap is rebuilt only when the image,
 * the blur level or the window size changes. Until it exists (or when blur
 * is off, or the background is a gradient) the CSS path draws instead.
 */
function scheduleBake(): void {
  clearTimeout(bakeTimer)
  // Resizes arrive in bursts; bake once the window settles.
  bakeTimer = setTimeout(() => { void bake() }, 150)
}

async function bake(): Promise<void> {
  const root = document.documentElement
  const radius = parseFloat(getComputedStyle(root).getPropertyValue('--x-panel-r')) || 0
  const image = loadedImage
  const width = window.innerWidth, height = window.innerHeight
  if (!image || !natural || radius <= 0 || !imageSrc || root.getAttribute('data-backdrop') !== 'image' || width <= 0 || height <= 0) {
    clearBaked()
    return
  }
  const key = `${imageSrc.length}:${imageSrc.slice(-48)}:${radius}:${width}x${height}`
  if (key === bakedKey && bakedUrl) return
  // A blurred picture has no fine detail: half resolution looks the same and
  // bakes four times faster.
  const scale = Math.min(0.5, 1400 / Math.max(width, height))
  const w = Math.max(1, Math.round(width * scale)), h = Math.max(1, Math.round(height * scale))
  const sharp = document.createElement('canvas')
  sharp.width = w; sharp.height = h
  const cover = coverRect(natural, { width, height })
  sharp.getContext('2d')?.drawImage(image, cover.x * scale, cover.y * scale, cover.width * scale, cover.height * scale)
  const blurred = document.createElement('canvas')
  blurred.width = w; blurred.height = h
  const context = blurred.getContext('2d')
  if (!context) return
  const r = radius * scale
  // Same look as the CSS path (blur + 180% saturation); drawn a little
  // oversized so the edges blur into picture, not into transparency.
  context.filter = `blur(${r}px) saturate(180%)`
  context.drawImage(sharp, -2 * r, -2 * r, w + 4 * r, h + 4 * r)
  // A data: URL, not a blob: one — the renderer's Content-Security-Policy
  // allows images from 'self' and data: only, and a blocked blob left each
  // surface drawing just its tint over the sharp picture: "blur does nothing".
  // A blurred picture has no detail left, so the JPEG is small.
  let url: string
  try {
    url = blurred.toDataURL('image/jpeg', 0.85)
  } catch {
    // A tainted canvas (a file:// picture read without the app's help):
    // keep the CSS path rather than fail.
    return
  }
  if (imageSrc === undefined || loadedImage !== image || !url.startsWith('data:image/')) return
  bakedUrl = url
  bakedKey = key
  root.style.setProperty('--x-backdrop-baked', `url("${url}")`)
  root.setAttribute('data-backdrop-baked', '')
}

function clearBaked(): void {
  const root = document.documentElement
  root.removeAttribute('data-backdrop-baked')
  root.style.removeProperty('--x-backdrop-baked')
  bakedUrl = undefined
  bakedKey = ''
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
      // Streamed text lands inside the conversation, where no surface ever
      // mounts; searching every added message subtree cost each token.
      if (record.target instanceof Element && record.target.closest('.stream')) continue
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
  // Every applyBackdrop lands here, so a new blur level re-bakes too.
  if (src && src === imageSrc && loadedImage) { schedule(); return }
  imageSrc = src
  loadedImage = undefined
  if (!src) { natural = undefined; clearBaked(); schedule(); return }
  const image = new Image()
  image.onload = () => {
    if (imageSrc !== src) return
    natural = { width: image.naturalWidth, height: image.naturalHeight }
    loadedImage = image
    schedule()
  }
  // A bundled file:// picture would taint the bake canvas; read it as a
  // data URL through the app instead (the CSS path covers the wait).
  const bundled = !src.startsWith('data:') && window.xerxes?.backgroundData
  if (bundled) void bundled(src).then(data => { if (imageSrc === src) image.src = data }).catch(() => { if (imageSrc === src) image.src = src })
  else image.src = src
}
