// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { isMac, isRemoteShell } from '../lib/platform.js'

const action = isMac ? 'Cmd' : 'Ctrl'
const paste = isMac ? 'Cmd' : 'Alt'

const copyHotkeys: [string, string][] = isMac
  ? [
      ['Cmd+C', 'copy selection'],
      ['Ctrl+C', 'interrupt / clear draft / press twice to exit']
    ]
  : isRemoteShell()
    ? [
        ['Cmd+C', 'copy selection when forwarded by the terminal'],
        ['Ctrl+C', 'copy selection / interrupt / clear draft / press twice to exit']
      ]
    : [['Ctrl+C', 'copy selection / interrupt / clear draft / press twice to exit']]

export const HOTKEYS: [string, string][] = [
  ...copyHotkeys,
  ['Ctrl+O / /copy', 'copy last assistant message; /copy picks any message (user or Xerxes)'],
  ['Ctrl+T', 'expand / collapse all thinking blocks (click a thinking header to toggle one)'],
  ['F6 / F7 / F8', 'agents panel / git diff viewer / terminals Xerxes is running'],
  ['Shift+Cmd/Ctrl+←/→', 'resize the F6/F7/F8 panel width (Option works where Cmd is intercepted)'],
  [action + '+D', 'exit'],
  [action + '+G / Alt+G', 'open $EDITOR (Alt+G fallback for VSCode/Cursor)'],
  [action + '+L', 'redraw / repaint'],
  ['Ctrl+V / /paste', 'smart paste: clipboard text, or attach a clipboard image'],
  [paste + '+V', 'paste text via the terminal (terminals deliver text only)'],
  ['Tab', 'apply completion'],
  ['↑/↓', 'completions / queue edit / history'],
  ['Ctrl+X', 'open live session switcher (deletes queued message while editing)'],
  [action + '+A/E', 'home / end of line'],
  [action + '+Z / ' + action + '+Y', 'undo / redo input edits'],
  [action + '+W', 'delete word'],
  [action + '+U/K', 'delete to start / end'],
  [action + '+←/→', 'jump word'],
  ['Home/End', 'start / end of line'],
  ['Shift+Enter / Alt+Enter', 'insert newline'],
  ['\\+Enter', 'multi-line continuation (fallback)'],
  ['!<cmd>', 'run a shell command (e.g. !ls, !git status)'],
  ['{!<cmd>}', 'interpolate shell output inline (e.g. "branch is {!git branch --show-current}")']
]
