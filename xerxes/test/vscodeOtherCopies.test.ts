// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { otherCopies } from '../src/vscode/otherCopies.js'

const chat = { contributes: { views: { xerxes: [{ type: 'webview', id: 'xerxes.chat', name: 'Xerxes' }] } } }

test('another installed copy of the chat view is found, by any ID but this one', () => {
  const installed = [
    { id: 'erfanzar.xerxes-agents', packageJSON: chat },
    { id: 'xsimurgh.xerxes-agents', packageJSON: chat },
    { id: 'erfanzar.xerxes-acp', packageJSON: { contributes: { views: { 'xerxes-acp': [{ id: 'xerxes-acp.chatView' }] } } } },
    { id: 'ms-python.python', packageJSON: { contributes: {} } },
  ]
  expect(otherCopies('erfanzar.xerxes-agents', 'xerxes.chat', installed)).toEqual(['xsimurgh.xerxes-agents'])
  expect(otherCopies('ERFANZAR.Xerxes-Agents', 'xerxes.chat', installed)).toEqual(['xsimurgh.xerxes-agents'])
})

test('malformed manifests are not mistaken for a copy', () => {
  const installed = [
    { id: 'a.none', packageJSON: null },
    { id: 'b.string', packageJSON: 'xerxes.chat' },
    { id: 'c.views-object', packageJSON: { contributes: { views: { xerxes: { id: 'xerxes.chat' } } } } },
    { id: 'd.null-entry', packageJSON: { contributes: { views: { xerxes: [null, 'xerxes.chat'] } } } },
  ]
  expect(otherCopies('erfanzar.xerxes-agents', 'xerxes.chat', installed)).toEqual([])
})
