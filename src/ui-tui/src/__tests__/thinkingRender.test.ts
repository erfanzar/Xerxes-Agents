import { describe, expect, it } from 'vitest'

import { LIVE_RENDER_MAX_CHARS } from '../config/limits.js'
import { thinkingRenderText } from '../components/thinking.js'

describe('thinkingRenderText', () => {
  it('does not live-tail truncate expanded thinking', () => {
    const head = 'important first analysis line'
    const body = Array.from({ length: LIVE_RENDER_MAX_CHARS + 200 }, (_, i) => String(i % 10)).join('')
    const text = `${head}\n${body}`
    const rendered = thinkingRenderText(text, 'full')

    expect(rendered).toContain(head)
    expect(rendered).not.toContain('[showing live tail')
    expect(rendered.length).toBeGreaterThan(LIVE_RENDER_MAX_CHARS)
  })

  it('still uses compact previews for truncated mode', () => {
    const rendered = thinkingRenderText('a '.repeat(500), 'truncated')

    expect(rendered.length).toBeLessThanOrEqual(160)
  })
})
