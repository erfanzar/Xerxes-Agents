// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { patchOverlayState } from '../../overlayStore.js'
import { askSideQuestion } from '../../sideQuestionStore.js'
import type { SlashCommand } from '../types.js'

export const sideQuestionCommands: SlashCommand[] = [
  {
    name: 'btw',
    aliases: ['side'],
    group: 'session',
    help: 'ask an ephemeral side question without interrupting or polluting the conversation [question]',
    usage: '/btw [question]',
    run: (arg, ctx) => {
      const question = arg.trim()
      patchOverlayState({ sideQuestion: true })
      if (!question) {
        return
      }
      void askSideQuestion(ctx.gateway.rpc, {
        sessionId: ctx.sid,
        question
      })
    }
  }
]
