// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { atom, computed } from 'nanostores'

import { MOUSE_TRACKING } from '../config/env.js'
import { sectionMode } from '../domain/details.js'
import { ZERO } from '../domain/usage.js'
import { DEFAULT_THEME, themeForMode } from '../theme.js'

import { DEFAULT_INDICATOR_STYLE, type UiState } from './interfaces.js'

const buildUiState = (): UiState => ({
  bgTasks: new Set(),
  busy: false,
  busyInputMode: 'queue',
  compact: false,
  detailsMode: 'collapsed',
  detailsModeCommandOverride: false,
  indicatorStyle: DEFAULT_INDICATOR_STYLE,
  info: null,
  liveSessionCount: 0,
  inlineDiffs: true,
  mouseTracking: MOUSE_TRACKING,
  notice: null,
  pasteCollapseLines: 5,
  pasteCollapseChars: 2000,
  sections: {},
  sessionTitle: '',
  showCost: false,
  showReasoning: false,
  sid: null,
  status: 'ready',
  statusBar: 'top',
  streaming: true,
  theme: DEFAULT_THEME,
  usage: ZERO
})

export const $uiState = atom<UiState>(buildUiState())

export const $uiTheme = computed($uiState, state => themeForMode(state.theme, state.info?.mode))
export const $uiSessionId = computed($uiState, state => state.sid)
// A primitive snapshot prevents transcript rows from re-rendering on every
// unrelated status/usage tick while still making /details changes immediate.
export const $uiDetailVisibility = computed($uiState, state =>
  (['thinking', 'tools', 'subagents'] as const)
    .map(name =>
      name === 'subagents'
        ? false
        : sectionMode(name, state.detailsMode, state.sections, state.detailsModeCommandOverride) !== 'hidden'
    )
    .map(Boolean)
    .join(':')
)

export const getUiState = () => $uiState.get()

export const patchUiState = (next: Partial<UiState> | ((state: UiState) => UiState)) =>
  $uiState.set(typeof next === 'function' ? next($uiState.get()) : { ...$uiState.get(), ...next })

export const resetUiState = () => $uiState.set(buildUiState())
