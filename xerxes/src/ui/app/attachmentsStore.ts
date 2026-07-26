// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { atom } from 'nanostores'

import { formatBytes, MAX_IMAGE_ATTACHMENTS_TOTAL_BYTES } from '../lib/imageAttachment.js'

/**
 * Pending image attachments for the composer's next submitted message.
 *
 * A nanostores atom (not composer React state) so the `/image` slash command,
 * the composer indicator line, and the submit path can all reach the same
 * list without prop drilling through useInputHandlers. Attachments apply to
 * exactly one `prompt.submit`: the submit path takes them and clears the
 * list, restoring them only when the daemon rejects the submit as busy.
 */
export interface PendingAttachment {
  readonly data: string
  readonly mediaType: string
  readonly name: string
  readonly path: string
  readonly size: number
}

export const $attachments = atom<PendingAttachment[]>([])

export const getAttachments = () => $attachments.get()

/** Add one attachment; returns an error message when the combined cap would be exceeded. */
export function addAttachment(item: PendingAttachment): string | null {
  const current = $attachments.get()

  if (current.some(existing => existing.path === item.path)) {
    return `${item.name} is already attached`
  }

  const total = current.reduce((sum, existing) => sum + existing.size, 0) + item.size

  if (total > MAX_IMAGE_ATTACHMENTS_TOTAL_BYTES) {
    return `attachments would total ${formatBytes(total)} — the per-message limit is ${formatBytes(MAX_IMAGE_ATTACHMENTS_TOTAL_BYTES)}`
  }

  $attachments.set([...current, item])

  return null
}

export function clearAttachments(): void {
  $attachments.set([])
}

/** Remove every pending attachment and return them for inclusion in one submit. */
export function takeAttachments(): PendingAttachment[] {
  const current = $attachments.get()

  if (current.length) {
    $attachments.set([])
  }

  return current
}

/** Put attachments back after a submit that the daemon never accepted. */
export function restoreAttachments(items: readonly PendingAttachment[]): void {
  if (items.length) {
    $attachments.set([...items, ...$attachments.get()])
  }
}

/** Total decoded bytes across pending attachments, for the composer indicator. */
export function attachmentsTotalBytes(items: readonly PendingAttachment[]): number {
  return items.reduce((sum, item) => sum + item.size, 0)
}

/** Wire entries for the `images` param of one `prompt.submit` request. */
export function promptSubmitImages(items: readonly PendingAttachment[]): { data: string; media_type: string }[] {
  return items.map(item => ({ data: item.data, media_type: item.mediaType }))
}
