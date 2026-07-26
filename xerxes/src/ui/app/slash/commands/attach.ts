// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { existsSync } from 'node:fs'

import { formatBytes, ImageAttachmentError, loadImageAttachment, resolveAttachmentPath } from '../../../lib/imageAttachment.js'
import {
  addAttachment,
  attachmentsTotalBytes,
  clearAttachments,
  getAttachments
} from '../../attachmentsStore.js'
import { runNativeSlash } from '../nativeSlash.js'
import type { SlashCommand } from '../types.js'

const usage = '/image <path…> attach · /image list · /image clear · /image <prompt> generate'

/**
 * `/image` is shared between two honest behaviors:
 *
 *   - when every argument resolves to an existing file, the files are loaded,
 *     magic-byte-sniffed, and queued as pending attachments for the next
 *     submitted message (sent as `images` on `turn.submit`);
 *   - otherwise the arguments are forwarded verbatim to the daemon's native
 *     `/image` generation command, preserving the pre-attachment behavior.
 *
 * `/image` with no arguments lists pending attachments; `/image clear` drops
 * them. `clear` therefore can no longer be a generation prompt — an accepted
 * ambiguity, since generation remains reachable with any longer prompt.
 */
export const attachCommands: SlashCommand[] = [
  {
    aliases: ['attach'],
    help: 'attach image files to the next message (or generate an image from a prompt)',
    name: 'image',
    usage,
    run: (arg, ctx) => {
      const trimmed = arg.trim()

      if (!trimmed || trimmed.toLowerCase() === 'list') {
        const pending = getAttachments()

        if (!pending.length) {
          return ctx.transcript.sys('no pending image attachments — add one with /image <path>')
        }

        return ctx.transcript.sys(
          [
            `pending image attachments (${formatBytes(attachmentsTotalBytes(pending))} total, next message only):`,
            ...pending.map(item => `  📎 ${item.name} · ${item.mediaType} · ${formatBytes(item.size)} · ${item.path}`)
          ].join('\n')
        )
      }

      if (trimmed.toLowerCase() === 'clear') {
        const count = getAttachments().length

        clearAttachments()

        return ctx.transcript.sys(count ? `cleared ${count} image attachment${count === 1 ? '' : 's'}` : 'no pending image attachments')
      }

      const cwd = ctx.ui.info?.cwd || process.cwd()
      const tokens = trimmed.split(/\s+/).filter(Boolean)
      const resolved = tokens.map(token => resolveAttachmentPath(token, cwd))

      if (!tokens.length || !resolved.every(path => existsSync(path))) {
        // Not (only) files: this is a generation prompt for the native daemon.
        return runNativeSlash(ctx, `image ${trimmed}`, 'Image')
      }

      void Promise.all(
        resolved.map(path =>
          loadImageAttachment(path).then(loaded => ({ loaded, path }), (error: unknown) => ({ error, path }))
        )
      ).then(results => {
        if (ctx.stale()) {
          return
        }

        for (const result of results) {
          if ('error' in result) {
            const message =
              result.error instanceof ImageAttachmentError ? result.error.message : `could not attach ${result.path}`

            ctx.transcript.sys(`error: ${message}`)

            continue
          }

          const { loaded } = result
          const rejected = addAttachment(loaded)

          ctx.transcript.sys(rejected ? `error: ${rejected}` : `📎 attached ${loaded.name} · ${loaded.mediaType} · ${formatBytes(loaded.size)}`)
        }
      })
    }
  }
]
