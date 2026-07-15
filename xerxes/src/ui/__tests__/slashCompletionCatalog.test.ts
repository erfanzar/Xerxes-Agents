// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { mergeCompletionItems, slashCompletionsFromCatalog } from '../hooks/useCompletion.js'
import type { SlashCatalog } from '../types.js'

const catalog: SlashCatalog = {
  canon: {
    '/deepscan': '/deepscan',
    '/eternal-army': '/eternal-army',
    '/help': '/help',
    '/provider': '/provider'
  },
  categories: [
    { name: 'core', pairs: [['/help', 'show help']] },
    {
      name: 'project skills',
      pairs: [
        ['/deepscan', 'deep codebase scan'],
        ['/eternal-army', 'swarm of subagents']
      ]
    }
  ],
  pairs: [
    ['/help', 'show help'],
    ['/provider', 'pick a model'],
    ['/deepscan', 'deep codebase scan'],
    ['/eternal-army', 'swarm of subagents']
  ],
  skillCount: 2,
  sub: {}
}

describe('slash catalog completions', () => {
  it('shows loaded skills in the bare slash menu before core commands', () => {
    expect(slashCompletionsFromCatalog('/', catalog).slice(0, 3)).toEqual([
      { display: 'deepscan', meta: 'deep codebase scan', text: '/deepscan' },
      { display: 'eternal-army', meta: 'swarm of subagents', text: '/eternal-army' },
      { display: 'help', meta: 'show help', text: '/help' }
    ])
  })

  it('filters skills by typed slash prefix', () => {
    expect(slashCompletionsFromCatalog('/dee', catalog)).toEqual([
      { display: 'deepscan', meta: 'deep codebase scan', text: '/deepscan' }
    ])
  })

  it('dedupes daemon completions after catalog skills', () => {
    const local = slashCompletionsFromCatalog('/', catalog)
    const remote = [
      { display: 'help', meta: 'Show help', text: 'help' },
      { display: 'tools', meta: 'List tools', text: 'tools' }
    ]

    expect(mergeCompletionItems(local, remote).map(item => item.display)).toEqual([
      'deepscan',
      'eternal-army',
      'help',
      'provider',
      'tools'
    ])
  })
})
