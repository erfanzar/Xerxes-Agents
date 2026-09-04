// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { NO_CONFIRM_DESTRUCTIVE } from '../../../config/env.js'
import { dailyFortune, randomFortune } from '../../../content/fortunes.js'
import { HOTKEYS } from '../../../content/hotkeys.js'
import { isSectionName, nextDetailsMode, parseDetailsMode, SECTION_NAMES } from '../../../domain/details.js'
import type {
  ConfigGetValueResponse,
  ConfigSetResponse,
  NativeUpdateStatusResponse,
  SessionGoalResponse,
  SessionSaveResponse,
  SessionStatusResponse,
  SessionSteerResponse,
  SessionTitleResponse,
  SessionUndoResponse
} from '../../../gatewayTypes.js'
import { COPY_USAGE, copyableMessages, copyTextToClipboard, formatCopyOutcome, resolveCopyArg } from '../../../lib/copyText.js'
import { applyMouseTracking, forceRedraw } from '../../../lib/terminalRuntime.opentui.js'
import { configureDetectedTerminalKeybindings, configureTerminalKeybindings } from '../../../lib/terminalSetup.js'
import type { MouseTrackingMode } from '../../../lib/terminalTypes.js'
import type { Msg, PanelSection } from '../../../types.js'
import type { StatusBarMode } from '../../interfaces.js'
import { patchOverlayState } from '../../overlayStore.js'
import { patchUiState } from '../../uiStore.js'
import type { SlashCommand, SlashRunCtx } from '../types.js'

/**
 * Persist a display preference and surface daemon failures as a transcript
 * note. A swallowed rejection leaves the daemon and the TUI believing
 * different settings with no signal to the user.
 */
const persistDisplayPref = (ctx: SlashRunCtx, key: string, value: string, label: string): void => {
  ctx.gateway.rpc<ConfigSetResponse>('config.set', { key, value }).catch(error => {
    if (ctx.stale()) {
      return
    }
    const detail = error instanceof Error ? error.message : String(error)
    ctx.transcript.sys(`${label}: could not save preference (${detail})`)
  })
}

const flagFromArg = (arg: string, current: boolean): boolean | null => {
  if (!arg) {
    return !current
  }

  const mode = arg.trim().toLowerCase()

  if (mode === 'on') {
    return true
  }

  if (mode === 'off') {
    return false
  }

  if (mode === 'toggle') {
    return !current
  }

  return null
}

// `/mouse` toggles between full tracking and off when called bare so the
// old binary muscle-memory still works. Explicit presets (wheel / buttons /
// all) target the tmux-friendly hover-free subsets.
const MOUSE_MODE_ALIASES: Record<string, MouseTrackingMode> = {
  all: 'all',
  any: 'all',
  button: 'buttons',
  buttons: 'buttons',
  click: 'buttons',
  full: 'all',
  off: 'off',
  on: 'all',
  scroll: 'wheel',
  wheel: 'wheel'
}

const mouseModeFromArg = (arg: string, current: MouseTrackingMode): MouseTrackingMode | null => {
  if (!arg || arg.trim().toLowerCase() === 'toggle') {
    return current === 'off' ? 'all' : 'off'
  }

  return MOUSE_MODE_ALIASES[arg.trim().toLowerCase()] ?? null
}

const RESET_WORDS = new Set(['reset', 'clear', 'default'])
const CYCLE_WORDS = new Set(['cycle', 'toggle'])

const DETAILS_USAGE =
  'usage: /details [hidden|collapsed|expanded|cycle]  or  /details <section> [hidden|collapsed|expanded|reset]'

const DETAILS_SECTION_USAGE = 'usage: /details <section> [hidden|collapsed|expanded|reset]'

const BUN_UPDATE_NEXT_STEPS = [
  'Preview: bun run xerxes update --dry-run --spec <package-or-source-spec>',
  'Apply:   bun run xerxes update --apply --spec <package-or-source-spec>'
]

/** Format the read-only native daemon update status without implying that an update ran. */
export function formatBunUpdateStatus(response: NativeUpdateStatusResponse): string {
  const summary = response.summary?.trim() || 'local update status was unavailable'
  const steps = response.next_steps?.filter(step => step.trim()) ?? BUN_UPDATE_NEXT_STEPS
  const command = response.command?.trim() || 'bun run xerxes update'

  return [
    'Bun update status',
    summary,
    '',
    'No update was run from the TUI.',
    `Status command: ${command}`,
    ...steps
  ].join('\n')
}

// Fixed pager width for /help — the pager overlay has no live cols at
// command-build time, so this is a reasonable general-purpose width rather
// than an exact terminal fit (same limitation the raw-text /status page
// already has).
const HELP_PAGE_WIDTH = 88

// Grouped, dashed-divider layout (label left, command/hotkey right-aligned)
// matching Grok Build's floating "Commands" palette, rendered through the
// same floating pager overlay /status already uses instead of appending an
// inline transcript panel that scrolls out of view.
function formatCommandsPage(sections: PanelSection[]): string {
  const lines: string[] = []

  for (const section of sections) {
    if (lines.length) {
      lines.push('')
    }

    if (section.title) {
      const header = section.title.toUpperCase()
      lines.push(`${header} ${'─'.repeat(Math.max(1, HELP_PAGE_WIDTH - header.length - 1))}`)
    }

    for (const [cmd, desc] of section.rows ?? []) {
      const gap = Math.max(1, HELP_PAGE_WIDTH - desc.length - cmd.length)
      lines.push(`${desc}${' '.repeat(gap)}${cmd}`)
    }

    for (const item of section.items ?? []) {
      lines.push(item)
    }

    if (section.text) {
      lines.push(section.text)
    }
  }

  return lines.join('\n')
}

export const coreCommands: SlashCommand[] = [
  {
    help: 'list commands + hotkeys',
    name: 'help',
    run: (_arg, ctx) => {
      const sections: PanelSection[] = (ctx.local.catalog?.categories ?? []).map(cat => ({
        rows: cat.pairs,
        title: cat.name
      }))

      if (ctx.local.catalog?.skillCount) {
        sections.push({ text: `${ctx.local.catalog.skillCount} skill commands available — /skills to browse` })
      }

      sections.push(
        {
          rows: [
            ['/details [hidden|collapsed|expanded|cycle]', 'set global agent detail visibility mode'],
            [
              '/details <section> [hidden|collapsed|expanded|reset]',
              'override one section (thinking/tools/activity; agents use /agents)'
            ],
            ['/fortune [random|daily]', 'show a random or daily local fortune']
          ],
          title: 'TUI'
        },
        { rows: HOTKEYS, title: 'Hotkeys' }
      )

      ctx.transcript.page(formatCommandsPage(sections), ctx.ui.theme.brand.helpHeader)
    }
  },

  {
    aliases: ['exit'],
    help: 'exit xerxes',
    name: 'quit',
    run: (_arg, ctx) => ctx.session.die()
  },

  {
    help: 'show Bun update status (does not update or exit)',
    name: 'update',
    run: (_arg, ctx) => {
      ctx.transcript.sys('checking Bun update status…')
      ctx.gateway
        .rpc<NativeUpdateStatusResponse>('runtime.update_status', {})
        .then(
          ctx.guarded<NativeUpdateStatusResponse>(result => {
            ctx.transcript.page(formatBunUpdateStatus(result), 'Bun Update')
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    aliases: ['scroll'],
    help: 'set mouse tracking preset [on|off|toggle|wheel|buttons|all]',
    name: 'mouse',
    run: (arg, ctx) => {
      const current = ctx.ui.mouseTracking
      const next = mouseModeFromArg(arg, current)

      if (next === null) {
        return ctx.transcript.sys('usage: /mouse [on|off|toggle|wheel|buttons|all]')
      }

      patchUiState({ mouseTracking: next })
      persistDisplayPref(ctx, 'mouse', next, 'mouse')
      // Actually apply it: the state used to change while the renderer kept
      // its boot-time capture, so /mouse off never gave the terminal its
      // drags back. off hands select-copy to the terminal; every capture
      // preset enables the renderer switch (the FFI is binary).
      applyMouseTracking(next)

      queueMicrotask(() => ctx.transcript.sys(`mouse tracking ${next}`))
    }
  },

  {
    aliases: ['new'],
    help: 'start a new session',
    name: 'clear',
    run: (arg, ctx, cmd) => {
      if (ctx.session.guardBusySessionSwitch('switch sessions')) {
        return
      }

      const isNew = cmd.startsWith('/new')
      const requestedTitle = isNew ? arg.trim() : ''

      const commit = () => {
        patchUiState({ status: 'forging session…' })
        ctx.session.newSession(isNew ? 'new session started' : undefined, requestedTitle || undefined)
      }

      if (NO_CONFIRM_DESTRUCTIVE) {
        return commit()
      }

      patchOverlayState({
        confirm: {
          cancelLabel: 'No, keep going',
          confirmLabel: isNew ? 'Yes, start a new session' : 'Yes, clear the session',
          danger: true,
          detail: 'This ends the current conversation and clears the transcript.',
          onConfirm: commit,
          title: isNew ? 'Start a new session?' : 'Clear the current session?'
        }
      })
    }
  },

  {
    help: 'force a full UI repaint',
    name: 'redraw',
    run: (_arg, ctx) => {
      forceRedraw(process.stdout)
      ctx.transcript.sys('ui redrawn')
    }
  },

  {
    help: 'show live session info',
    name: 'status',
    run: (_arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys('no active session')
      }

      ctx.gateway
        .rpc<SessionStatusResponse>('session.status', { session_id: ctx.sid })
        .then(ctx.guarded<SessionStatusResponse>(r => ctx.transcript.page(r.output || '(no status)', 'Status')))
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'set or view the goal for a long-running task',
    name: 'goal',
    // Rendering happens daemon-side so the terminal, the bridge and every
    // channel describe one goal with one vocabulary. This end only carries the
    // words.
    run: (arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys('no active session')
      }

      ctx.gateway
        .rpc<SessionGoalResponse>('session.goal', { input: arg, session_id: ctx.sid })
        .then(
          ctx.guarded<SessionGoalResponse>(r => {
            ctx.transcript.page(r?.text || 'no goal state returned', 'Goal')
          })
        )
        .catch(ctx.guardedErr)
    },
    usage: '/goal [<objective>|clear|edit <objective>|pause|resume]'
  },

  {
    help: 'select a remote machine to work on',
    name: 'machine',
    run: (_arg, ctx) => {
      // The TUI opens the picker overlay; the daemon just acknowledges.
      ctx.gateway.rpc('slash', { command: '/machine' }).catch(ctx.guardedErr)
      // Open the picker locally — the daemon's response is a no-op.
      patchOverlayState({ machinePicker: true })
    },
    usage: '/machine'
  },

  {
    help: 'set or show current session title',
    name: 'title',
    run: (arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys('no active session')
      }

      const title = arg.trim()

      if (!arg) {
        ctx.gateway
          .rpc<SessionTitleResponse>('session.title', { session_id: ctx.sid })
          .then(
            ctx.guarded<SessionTitleResponse>(r => {
              const current = (r?.title ?? '').trim()
              ctx.transcript.sys(current ? `title: ${current}` : 'no title set')
            })
          )
          .catch(ctx.guardedErr)

        return
      }

      if (!title) {
        return ctx.transcript.sys('usage: /title <your session title>')
      }

      ctx.gateway
        .rpc<SessionTitleResponse>('session.title', { session_id: ctx.sid, title })
        .then(
          ctx.guarded<SessionTitleResponse>(r => {
            const next = (r?.title ?? title).trim()
            const suffix = r?.pending ? ' (queued while session initializes)' : ''
            ctx.transcript.sys(`session title set: ${next}${suffix}`)
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    aliases: ['transcript-compact'],
    help: 'toggle compact transcript display',
    name: 'ui-compact',
    run: (arg, ctx) => {
      const next = flagFromArg(arg, ctx.ui.compact)

      if (next === null) {
        return ctx.transcript.sys('usage: /ui-compact [on|off|toggle]')
      }

      patchUiState({ compact: next })
      persistDisplayPref(ctx, 'compact', next ? 'on' : 'off', 'ui-compact')

      queueMicrotask(() => ctx.transcript.sys(`transcript compact display ${next ? 'on' : 'off'}`))
    }
  },

  {
    aliases: ['detail'],
    help: 'control agent detail visibility (global or per-section)',
    name: 'details',
    run: (arg, ctx) => {
      const { gateway, transcript, ui } = ctx

      if (!arg) {
        gateway
          .rpc<ConfigGetValueResponse>('config.get', { key: 'details_mode' })
          .then(r => {
            if (ctx.stale()) {
              return
            }

            const mode = parseDetailsMode(r?.value) ?? ui.detailsMode
            patchUiState({ detailsMode: mode, detailsModeCommandOverride: false })

            const overrides = SECTION_NAMES.filter(s => ui.sections[s])
              .map(s => `${s}=${ui.sections[s]}`)
              .join(' ')

            transcript.sys(`details: ${mode}${overrides ? `  (${overrides})` : ''}`)
          })
          .catch(() => !ctx.stale() && transcript.sys(`details: ${ui.detailsMode}`))

        return
      }

      const [first, second] = arg.trim().toLowerCase().split(/\s+/)

      if (second && isSectionName(first)) {
        const reset = RESET_WORDS.has(second)
        const mode = reset ? null : parseDetailsMode(second)

        if (!reset && !mode) {
          return transcript.sys(DETAILS_SECTION_USAGE)
        }

        const { [first]: _drop, ...rest } = ui.sections

        patchUiState({ sections: mode ? { ...rest, [first]: mode } : rest })
        persistDisplayPref(ctx, `details_mode.${first}`, mode ?? '', `details ${first}`)
        transcript.sys(`details ${first}: ${mode ?? 'reset'}`)

        return
      }

      const next = CYCLE_WORDS.has(first ?? '') ? nextDetailsMode(ui.detailsMode) : parseDetailsMode(first)

      if (!next) {
        return transcript.sys(DETAILS_USAGE)
      }

      const sections = Object.fromEntries(SECTION_NAMES.map(section => [section, next]))

      patchUiState({ detailsMode: next, detailsModeCommandOverride: true, sections })
      persistDisplayPref(ctx, 'details_mode', next, 'details')
      transcript.sys(`details: ${next}`)
    }
  },

  {
    help: 'local fortune',
    name: 'fortune',
    run: (arg, ctx) => {
      const key = arg.trim().toLowerCase()

      if (!arg || key === 'random') {
        return ctx.transcript.sys(randomFortune())
      }

      if (['daily', 'stable', 'today'].includes(key)) {
        return ctx.transcript.sys(dailyFortune(ctx.sid))
      }

      ctx.transcript.sys('usage: /fortune [random|daily]')
    }
  },

  {
    help: 'copy transcript text: /copy [n] · user [n] · last · all (bare = message picker)',
    name: 'copy',
    run: async (arg, ctx) => {
      const { sys } = ctx.transcript

      if (!arg.trim() && ctx.composer.hasSelection) {
        const text = await ctx.composer.selection.copySelection()

        if (text) {
          return sys(`copied ${text.length} characters`)
        }

        return sys(`clipboard copy failed — ${COPY_USAGE}`)
      }

      const resolution = resolveCopyArg(arg, copyableMessages(ctx.local.getHistoryItems()))

      if (resolution.kind === 'usage') {
        return sys(COPY_USAGE)
      }

      if (resolution.kind === 'empty') {
        return sys(resolution.message)
      }

      if (resolution.kind === 'picker') {
        return patchOverlayState({ copyPicker: { items: resolution.items } })
      }

      const text = resolution.text

      void copyTextToClipboard(text)
        .then(outcome => {
          if (!ctx.stale()) {
            sys(formatCopyOutcome(outcome))
          }
        })
        .catch(error => {
          if (!ctx.stale()) {
            sys(`copy failed: ${String(error)}`)
          }
        })
    }
  },

  {
    help: 'attach clipboard image',
    name: 'paste',
    run: (arg, ctx) => (arg ? ctx.transcript.sys('usage: /paste') : ctx.composer.paste())
  },

  {
    help: 'open the git diff viewer (F7)',
    name: 'diff',
    run: (arg, ctx) => {
      if (arg) {
        return ctx.transcript.sys('usage: /diff')
      }

      patchOverlayState({ diff: true })
    }
  },

  {
    aliases: ['shells'],
    help: 'open the terminal viewer (F8)',
    name: 'terminals',
    run: (arg, ctx) => {
      if (arg) {
        return ctx.transcript.sys('usage: /terminals')
      }

      patchOverlayState({ terminals: true })
    }
  },

  {
    help: 'configure IDE terminal keybindings for multiline + undo/redo',
    name: 'terminal-setup',
    run: (arg, ctx) => {
      const target = arg.trim().toLowerCase()

      if (target && !['auto', 'cursor', 'vscode', 'windsurf'].includes(target)) {
        return ctx.transcript.sys('usage: /terminal-setup [auto|vscode|cursor|windsurf]')
      }

      const runner =
        !target || target === 'auto'
          ? configureDetectedTerminalKeybindings()
          : configureTerminalKeybindings(target as 'cursor' | 'vscode' | 'windsurf')

      void runner
        .then(result => {
          if (ctx.stale()) {
            return
          }

          ctx.transcript.sys(result.message)

          if (result.success && result.requiresRestart) {
            ctx.transcript.sys('restart the IDE terminal for the new keybindings to take effect')
          }
        })
        .catch(error => {
          if (!ctx.stale()) {
            ctx.transcript.sys(`terminal setup failed: ${String(error)}`)
          }
        })
    }
  },

  {
    help: 'view gateway logs',
    name: 'logs',
    run: (arg, ctx) => {
      const text = ctx.gateway.gw.getLogTail(Math.min(80, Math.max(1, parseInt(arg, 10) || 20)))

      text ? ctx.transcript.page(text, 'Logs') : ctx.transcript.sys('no gateway logs')
    }
  },

  {
    help: 'view current transcript (user + assistant messages)',
    name: 'history',
    run: (arg, ctx) => {
      // The CLI-side `/history` runs in a detached slash-worker subprocess
      // that never sees the TUI's turns — it only surfaces whatever was
      // persisted before this process started.  Render the TUI's own
      // transcript so `/history` actually reflects what the user just did.
      const items = ctx.local.getHistoryItems().filter(m => m.role === 'user' || m.role === 'assistant')

      if (!items.length) {
        return ctx.transcript.sys('no conversation yet')
      }

      const preview = Math.max(80, parseInt(arg, 10) || 400)

      const lines = items.map((m, i) => {
        const tag = m.role === 'user' ? `You #${i + 1}` : `Xerxes #${i + 1}`
        const body = m.text.trim() || (m.tools?.length ? `(${m.tools.length} tool calls)` : '(empty)')
        const clipped = body.length > preview ? `${body.slice(0, preview).trimEnd()}…` : body

        return `[${tag}]\n${clipped}`
      })

      ctx.transcript.page(lines.join('\n\n'), 'History')
    }
  },

  {
    help: 'save the current transcript to JSON',
    name: 'save',
    run: (_arg, ctx) => {
      const hasConversation = ctx.local
        .getHistoryItems()
        .some(m => m.role === 'user' || m.role === 'assistant' || m.role === 'tool')

      if (!hasConversation) {
        return ctx.transcript.sys('no conversation yet')
      }

      if (!ctx.sid) {
        return ctx.transcript.sys('no active session — nothing to save')
      }

      ctx.gateway
        .rpc<SessionSaveResponse>('session.save', { session_id: ctx.sid })
        .then(
          ctx.guarded<SessionSaveResponse>(r => {
            const file = r?.file

            if (file) {
              ctx.transcript.sys(`conversation saved to: ${file}`)
            } else {
              ctx.transcript.sys('failed to save')
            }
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    aliases: ['sb'],
    help: 'status bar position (on|off|top|bottom)',
    name: 'statusbar',
    run: (arg, ctx) => {
      const mode = arg.trim().toLowerCase()
      const toggle: StatusBarMode = ctx.ui.statusBar === 'off' ? 'top' : 'off'

      const next: null | StatusBarMode =
        !mode || mode === 'toggle'
          ? toggle
          : mode === 'on' || mode === 'top'
            ? 'top'
            : mode === 'off' || mode === 'bottom'
              ? mode
              : null

      if (!next) {
        return ctx.transcript.sys('usage: /statusbar [on|off|top|bottom|toggle]')
      }

      patchUiState({ statusBar: next })
      persistDisplayPref(ctx, 'statusbar', next, 'statusbar')

      queueMicrotask(() => ctx.transcript.sys(`status bar ${next}`))
    }
  },

  {
    aliases: ['q'],
    help: 'inspect or enqueue a message',
    name: 'queue',
    run: (arg, ctx) => {
      if (!arg) {
        return ctx.transcript.sys(`${ctx.composer.queueRef.current.length} queued message(s)`)
      }

      ctx.composer.enqueue(arg)
      ctx.transcript.sys(`queued: "${arg.slice(0, 50)}${arg.length > 50 ? '…' : ''}"`)
    }
  },

  {
    help: 'inject a message after the next tool call (no interrupt)',
    name: 'steer',
    run: (arg, ctx) => {
      const payload = arg?.trim() ?? ''

      if (!payload) {
        return ctx.transcript.sys('usage: /steer <prompt>')
      }

      // If the agent isn't running, fall back to the queue so the user's
      // message isn't lost — identical semantics to the gateway handler.
      if (!ctx.ui.busy || !ctx.sid) {
        ctx.composer.enqueue(payload)
        ctx.transcript.sys(
          `no active turn — queued for next: "${payload.slice(0, 50)}${payload.length > 50 ? '…' : ''}"`
        )

        return
      }

      ctx.gateway
        .rpc<SessionSteerResponse>('session.steer', { session_id: ctx.sid, text: payload })
        .then(
          ctx.guarded<SessionSteerResponse>(r => {
            if (r?.status === 'queued') {
              ctx.transcript.sys(
                `steer queued — arrives after next tool call: "${payload.slice(0, 50)}${payload.length > 50 ? '…' : ''}"`
              )
            } else {
              ctx.transcript.sys('steer rejected')
            }
          })
        )
        .catch(ctx.guardedErr)
    }
  },

  {
    help: 'undo last exchange',
    name: 'undo',
    run: (_arg, ctx) => {
      if (!ctx.sid) {
        return ctx.transcript.sys('nothing to undo')
      }

      ctx.gateway.rpc<SessionUndoResponse>('session.undo', { session_id: ctx.sid }).then(
        ctx.guarded<SessionUndoResponse>(r => {
          if ((r.removed ?? 0) > 0) {
            ctx.transcript.setHistoryItems((prev: Msg[]) => ctx.transcript.trimLastExchange(prev))
            ctx.transcript.sys(`undid ${r.removed} messages`)
          } else {
            ctx.transcript.sys('nothing to undo')
          }
        })
      ).catch(ctx.guardedErr)
    }
  },

  {
    help: 'retry last user message',
    name: 'retry',
    run: (_arg, ctx) => {
      const last = ctx.local.getLastUserMsg()

      if (!last) {
        return ctx.transcript.sys('nothing to retry')
      }

      if (!ctx.sid) {
        return ctx.transcript.send(last)
      }

      ctx.gateway.rpc<SessionUndoResponse>('session.undo', { session_id: ctx.sid }).then(
        ctx.guarded<SessionUndoResponse>(r => {
          if ((r.removed ?? 0) <= 0) {
            return ctx.transcript.sys('nothing to retry')
          }

          ctx.transcript.setHistoryItems((prev: Msg[]) => ctx.transcript.trimLastExchange(prev))
          ctx.transcript.send(last)
        })
      ).catch(ctx.guardedErr)
    }
  }
]
