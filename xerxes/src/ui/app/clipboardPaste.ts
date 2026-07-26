// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { readClipboardImage, type ReadClipboardImageOptions } from '../lib/clipboardImage.js'
import { formatBytes, MAX_IMAGE_ATTACHMENT_BYTES } from '../lib/imageAttachment.js'

import { addAttachment } from './attachmentsStore.js'

/**
 * Paste-flow wiring between the clipboard-image reader and the pending
 * attachments store. Called from the `/paste` slash command and the composer
 * paste fallback when the clipboard carries no usable text; on success the
 * image becomes a pending attachment for the next submitted message, exactly
 * like `/image <path>`.
 */

export interface ClipboardPasteOutcome {
  /** True when a clipboard image is now queued as a pending attachment. */
  readonly attached: boolean
  /** Transcript message describing what happened. */
  readonly message: string
}

export async function attachClipboardImage(options: ReadClipboardImageOptions = {}): Promise<ClipboardPasteOutcome> {
  const result = await readClipboardImage(options)

  switch (result.kind) {
    case 'image': {
      const rejected = addAttachment(result.attachment)

      if (rejected) {
        return { attached: false, message: `error: ${rejected}` }
      }

      const dimensions = result.width && result.height ? `, ${result.width}x${result.height}` : ''

      return {
        attached: true,
        message: `📎 attached clipboard image (${formatBytes(result.attachment.size)}${dimensions}) — /image to review`
      }
    }

    case 'no-image':
      return { attached: false, message: 'nothing to paste — the clipboard has no usable text or image' }

    case 'not-image':
      return { attached: false, message: 'error: clipboard image data was not a png, jpeg, gif, or webp image' }

    case 'too-large':
      return {
        attached: false,
        message: `error: clipboard image is ${formatBytes(result.size)} — the per-image limit is ${formatBytes(MAX_IMAGE_ATTACHMENT_BYTES)}`
      }

    case 'unsupported':
      return {
        attached: false,
        message: `clipboard image paste is not supported on ${result.platform} — save the image to a file and attach it with /image <path>`
      }

    case 'error':
      return { attached: false, message: `error: could not read the clipboard image: ${result.message}` }
  }
}
