// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { parseHistoryPage } from '../../client/hubParsers.js'
import { reduceHub } from '../../client/hubStore.js'
import type { Block } from '../../client/sessionFold.js'
import { BlockList } from '../components/BlockList.js'
import { createTranscriptScroll } from '../transcriptScroll.js'
import { liveState } from './helpers.js'
import { render } from './renderHelpers.js'

const tool = (id: string, state: 'working' | 'done' | 'error' = 'done', name = 'exec_command'): Block => ({ kind: 'tool', id, callId: id, name, summary: 'bun test', state, output: 'output' })
const plain = (html: string): string => html.replaceAll('<!-- -->', '')

describe('tool rows and tool groups', () => {
  it('folds consecutive commands into one collapsed group whose every call still expands on its own', () => {
    const html = plain(renderToStaticMarkup(<BlockList blocks={[tool('1'), tool('2', 'error'), tool('3'), tool('4', 'error')]} prefix="test" />))
    expect(html.match(/class="fold tool-group"/g)).toHaveLength(1)
    expect(html).toContain('<summary>Ran 4 commands<span class="tool-state tool-error"> · 2 failed</span></summary>')
    expect(html.match(/class="fold tool-row"/g)).toHaveLength(4)
    expect(html.match(/>Ran</g)).toHaveLength(4)
    expect(html.match(/<code class="tool-target">bun test<\/code>/g)).toHaveLength(4)
    expect(html).not.toMatch(/<details[^>]*\sopen[\s=>]/)
    expect(html.match(/>output<\/pre>/g)).toHaveLength(4)
  })

  it('groups every run of two or more tools of any kind, breaks it around other blocks, and leaves a lone call as a row', () => {
    const html = plain(renderToStaticMarkup(<BlockList blocks={[
      tool('1'), tool('2', 'working', 'ReadFile'), tool('3', 'error', 'GrepTool'), tool('5', 'done', 'GlobTool'),
      { kind: 'assistant', id: 'reply', text: 'The full reply', streaming: false },
      { kind: 'user', id: 'user', text: 'The full question', images: 0, steer: false },
      tool('6', 'done', 'ReadFile'), tool('7', 'done', 'GrepTool'),
      { kind: 'thinking', id: 'thought', text: 'Reasoning', streaming: false },
      tool('4'),
    ]} prefix="test" />))
    expect(html.match(/class="fold tool-group"/g)).toHaveLength(2)
    expect(html).toContain('Ran 1 command, read 1 file, 2 searches')
    expect(html).toContain('Read 1 file, 1 search')
    expect(html).toContain(' · running')
    expect(html).toContain('1 failed')
    expect(html).toContain('aria-label="Tool steps"')
    // 4 + 2 grouped rows, and the lone command after the thought stays a plain row.
    expect(html.match(/class="fold tool-row"/g)).toHaveLength(7)
    expect(html).toContain('<div class="md block-md"><p>The full reply</p></div>')
    expect(html).toContain('<p class="block-text">The full question</p>')
    expect(html).toContain('<summary>Thought it through</summary>')
    expect(html).not.toMatch(/<details[^>]*\sopen[\s=>]/)
  })

  it('names command checks apart from commands and never shows a bare JSON brace as the target', () => {
    const check: Block = { kind: 'tool', id: 'c', callId: 'c', name: 'check_command', summary: '', state: 'working', args: '{\n  "proc_id": "p3",\n  "wait_ms": 60000\n}' }
    const html = plain(renderToStaticMarkup(<BlockList blocks={[check, tool('1')]} prefix="test" />))
    expect(html).toContain('Ran 1 command, 1 command check')
    expect(html).toContain('>Checking a command<')
    expect(html).toContain('<code class="tool-target">proc_id=p3 wait_ms=60000</code>')
    expect(html).not.toContain('<code class="tool-target">{</code>')
    expect(html).not.toContain('tool-args is-command">{')
  })

  it('renders exhausted older history instead of hiding it, newest last, and offers fallback while more exists', () => {
    let state = reduceHub(liveState(), { type: 'select', workspaceId: 'w1', sessionId: 's1' })
    const page = parseHistoryPage({ history: { actions: [
      { id: '0:message', messages: [{ role: 'user', content: 'Old question' }] },
      { id: '1:message', messages: [{ role: 'assistant', content: 'Newest reply' }] },
    ], has_more: false } })
    state = reduceHub(state, { type: 'historyLoaded', workspaceId: 'w1', sessionId: 's1', page, throughIds: [] })
    const html = plain(render(state, '/w/w1/s/s1'))
    expect(html).toContain('Old question')
    expect(html.indexOf('Newest reply')).toBeGreaterThan(html.indexOf('Old question'))
    expect(html).not.toContain('Load earlier messages')
    expect(html).not.toContain('Nothing recorded')
    state = { ...state, selected: { ...state.selected!, hasOlder: true } }
    expect(plain(render(state, '/w/w1/s/s1'))).toContain('Load earlier messages')
  })
})

describe('transcript document scroll policy', () => {
  function fixture() {
    let y = 0
    let height = 2000
    let positions = [{ id: 'a', y: 100 }, { id: 'b', y: 1000 }, { id: 'c', y: 1600 }]
    const scrolls: number[] = []
    const scroll = createTranscriptScroll({
      y: () => y, height: () => height, viewport: () => 500, top: () => 200 - y,
      blocks: () => positions.map(item => ({ id: item.id, top: item.y - y })),
      scrollTo: value => { y = value; scrolls.push(value) },
    })
    return { scroll, scrolls, y: () => y, move: (value: number) => { y = value; return scroll.moved() },
      grow: (delta: number) => { height += delta },
      prepend: (delta: number) => { height += delta; positions = positions.map(item => ({ ...item, y: item.y + delta })) },
    }
  }

  it('opens at the bottom and follows new content while pinned', () => {
    const f = fixture()
    f.scroll.changed()
    expect(f.y()).toBe(1500)
    expect(f.move(1500)).toBe(false)
    f.grow(300)
    f.scroll.changed()
    expect(f.y()).toBe(1800)
  })

  it('does not yank a reader down, and only requests older content near the top while scrolling upward', () => {
    const f = fixture()
    f.scroll.changed()
    expect(f.move(800)).toBe(false)
    f.grow(300)
    f.scroll.changed()
    expect(f.y()).toBe(800)
    expect(f.move(260)).toBe(true)
    expect(f.move(280)).toBe(false)
  })

  it('preserves the visible block when history prepends, even if height changes elsewhere too', () => {
    const f = fixture()
    f.scroll.changed()
    f.move(260)
    f.scroll.beforePrepend()
    f.prepend(700)
    f.grow(200)
    f.scroll.changed()
    expect(f.y()).toBe(960)
  })
})
