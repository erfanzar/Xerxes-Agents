// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { parseSlashCommand } from '../domain/slash.js'
import type { SlashExecResponse } from '../gatewayTypes.js'
import { promoteSlashToUserMessage } from '../lib/messages.js'
import { asCommandDispatch, rpcErrorMessage } from '../lib/rpc.js'
import { slashExecText } from '../lib/slashOutput.js'

import type { SlashHandlerContext } from './interfaces.js'
import { isSkillTurnCommand } from './skillCatalog.js'
import { findSlashCommand } from './slash/registry.js'
import type { SlashRunCtx } from './slash/types.js'
import { getUiState } from './uiStore.js'

export function createSlashHandler(ctx: SlashHandlerContext): (cmd: string) => boolean {
  const { gw } = ctx.gateway
  const { catalog } = ctx.local
  const { page, send, sys } = ctx.transcript

  const handler = (cmd: string): boolean => {
    const flight = ++ctx.slashFlightRef.current
    const ui = getUiState()
    const sid = ui.sid
    const parsed = parseSlashCommand(cmd)
    const argTail = parsed.arg ? ` ${parsed.arg}` : ''

    const stale = () => flight !== ctx.slashFlightRef.current || getUiState().sid !== sid

    const guarded =
      <T>(fn: (r: T) => void) =>
      (r: null | T): void => {
        if (!stale() && r) {
          fn(r)
        }
      }

    const guardedErr = (e: unknown) => {
      if (!stale()) {
        sys(`error: ${rpcErrorMessage(e)}`)
      }
    }

    const runCtx: SlashRunCtx = { ...ctx, flight, guarded, guardedErr, sid, stale, ui }

    const found = findSlashCommand(parsed.name)

    if (found) {
      found.run(parsed.arg, runCtx, cmd)

      return true
    }

    if (catalog?.canon) {
      const needle = `/${parsed.name}`.toLowerCase()
      const exact = Object.entries(catalog.canon).find(([alias]) => alias.toLowerCase() === needle)?.[1]

      if (exact) {
        if (exact.toLowerCase() !== needle) {
          return handler(`${exact}${argTail}`)
        }
      } else {
        const matches = [
          ...new Set(
            Object.entries(catalog.canon)
              .filter(([alias]) => alias.startsWith(needle))
              .map(([, canon]) => canon)
          )
        ]

        if (matches.length === 1 && matches[0]!.toLowerCase() !== needle) {
          return handler(`${matches[0]}${argTail}`)
        }

        if (matches.length > 1) {
          sys(`ambiguous command: ${matches.slice(0, 6).join(', ')}${matches.length > 6 ? ', …' : ''}`)

          return true
        }
      }
    }

    if (isSkillTurnCommand(catalog, cmd)) {
      ctx.transcript.setHistoryItems(items => promoteSlashToUserMessage(items, cmd.trim()))
    }

    gw.request<SlashExecResponse>('slash.exec', { command: cmd.slice(1), session_id: sid })
      .then(r => {
        if (stale()) {
          return
        }

        const text = slashExecText(r)
        if (!text) {
          return
        }

        const long = text.length > 180 || text.split('\n').filter(Boolean).length > 2

        long ? page(text, parsed.name[0]!.toUpperCase() + parsed.name.slice(1)) : sys(text)
      })
      .catch(() => {
        gw.request('command.dispatch', { arg: parsed.arg, name: parsed.name, session_id: sid })
          .then((raw: unknown) => {
            if (stale()) {
              return
            }

            const slashText = slashExecText(raw as SlashExecResponse)
            if (slashText) {
              const long = slashText.length > 180 || slashText.split('\n').filter(Boolean).length > 2

              long ? page(slashText, parsed.name[0]!.toUpperCase() + parsed.name.slice(1)) : sys(slashText)
              return
            }

            const d = asCommandDispatch(raw)

            if (!d) {
              return
            }

            if (d.type === 'exec' || d.type === 'plugin') {
              const output = d.output?.trim()
              return output ? sys(output) : undefined
            }

            if (d.type === 'alias') {
              return handler(`/${d.target}${argTail}`)
            }

            if (d.type === 'skill') {
              sys(`⚡ loading skill: ${d.name}`)

              return d.message?.trim() ? send(d.message) : sys(`/${parsed.name}: skill payload missing message`)
            }

            if (d.type === 'send') {
              if (d.notice?.trim()) {
                sys(d.notice)
              }
              return d.message?.trim() ? send(d.message) : sys(`/${parsed.name}: empty message`)
            }

            if (d.type === 'prefill') {
              // /undo returns prefill: drop the backed-up message text into
              // the composer so the user can edit and resubmit, instead of
              // submitting it immediately like 'send'.
              if (d.notice?.trim()) {
                sys(d.notice)
              }
              if (d.message) {
                ctx.composer.setInput(d.message)
              }
              return
            }
          })
          .catch(guardedErr)
      })

    return true
  }

  return handler
}
