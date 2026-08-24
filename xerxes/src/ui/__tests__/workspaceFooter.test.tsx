// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import { testRender } from '@opentui/react/test-utils'
import { describe, expect, it } from 'vitest'

import { WorkspaceFooter } from '../opentui/appChrome.js'
import { DEFAULT_THEME } from '../theme.js'

const t = DEFAULT_THEME

const render = async (props: { cwdLabel?: string; providerModel?: string; rightLabel?: string } = {}) => {
  const session = await testRender(
    <WorkspaceFooter
      cwdLabel={'cwdLabel' in props ? props.cwdLabel! : '~/Projects/Xerxes-Agents (main)'}
      providerModel={props.providerModel}
      rightLabel={props.rightLabel}
      t={t}
    />,
    { height: 6, width: 100 }
  )
  await session.flush()
  return session
}

describe('WorkspaceFooter provider health', () => {
  it('vouches for turns with a green provider-ready dot when a model is set', async () => {
    const session = await render({
      providerModel: 'claude-opus-4-6',
      rightLabel: 'F6 agents · F7 diff · F8 terminals'
    })
    try {
      const out = session.captureCharFrame()
      expect(out).toContain('provider ready')
      expect(out).not.toContain('no model')
      expect(out).toContain('F6 agents')
    } finally {
      session.renderer.destroy()
    }
  })

  it('flags a configured-but-empty model with the warn no-model hint', async () => {
    const session = await render({ providerModel: '', rightLabel: 'F6 agents' })
    try {
      const out = session.captureCharFrame()
      expect(out).toContain('no model · /provider')
      expect(out).not.toContain('provider ready')
    } finally {
      session.renderer.destroy()
    }
  })

  it('renders the segment on the home screen without hotkey hints', async () => {
    const session = await render({ providerModel: 'claude-opus-4-6' })
    try {
      expect(session.captureCharFrame()).toContain('provider ready')
    } finally {
      session.renderer.destroy()
    }
  })

  it('stays hidden while the daemon info payload has not loaded', async () => {
    const session = await render()
    try {
      // No providerModel at all: neither state may flash during boot.
      const out = session.captureCharFrame()
      expect(out).not.toContain('provider ready')
      expect(out).not.toContain('no model')
    } finally {
      session.renderer.destroy()
    }
  })

  it('keeps rendering nothing when there is nothing to say', async () => {
    const session = await render({ cwdLabel: '' })
    try {
      const out = session.captureCharFrame()
      expect(out).not.toContain('✦')
    } finally {
      session.renderer.destroy()
    }
  })
})
