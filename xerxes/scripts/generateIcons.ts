// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Regenerates `src/desktop/renderer/Icon.tsx` from Phosphor Icons.
 *
 * The renderer inlines its icon geometry instead of depending on an icon
 * package: the desktop app loads from `file://`, ships every byte it draws,
 * and needs exactly the few dozen glyphs listed in `MAP` — not the 1,512
 * Phosphor publishes. Inlining keeps the bundle honest and means an icon
 * never changes shape because a transitive dependency moved.
 *
 * This is deliberately not wired into `bun run build`. Icons are a design
 * decision, not a build artifact — regenerate on purpose, look at the
 * result, commit it.
 *
 * Usage:
 *   npm pack @phosphor-icons/core@2.1.1
 *   tar xzf phosphor-icons-core-2.1.1.tgz
 *   bun scripts/generateIcons.ts ./package/assets/regular
 *
 * Phosphor Icons is MIT licensed (c) 2023 Phosphor Icons.
 * https://github.com/phosphor-icons/core
 */

import { readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

/**
 * Local name → Phosphor asset basename.
 *
 * The local names are the app's own vocabulary and are what call sites use,
 * so they outlive any particular icon library: swapping Phosphor for another
 * set is a change to the right-hand column only. Keep a name here even if it
 * currently has one call site — deleting it is a rename across the renderer.
 */
const MAP: ReadonlyArray<readonly [local: string, phosphor: string]> = [
  ['sidebar', 'sidebar-simple'], ['expand', 'corners-out'], ['collapse', 'corners-in'],
  ['search', 'magnifying-glass'], ['plus', 'plus'], ['close', 'x'], ['tools', 'wrench'],
  ['folder', 'folder'], ['file', 'file-text'], ['chevron', 'caret-right'],
  ['terminal', 'terminal-window'], ['shield', 'shield-check'], ['plan', 'list-checks'],
  ['clock', 'clock'], ['settings', 'gear-six'], ['activity', 'pulse'],
  ['changes', 'git-diff'], ['agent', 'robot'], ['arrow', 'arrow-right'],
  ['caretDown', 'caret-down'], ['check', 'check'], ['checkCircle', 'check-circle'],
  ['warning', 'warning'], ['error', 'x-circle'], ['info', 'info'], ['spark', 'sparkle'],
  ['stop', 'stop-circle'], ['send', 'paper-plane-tilt'], ['copy', 'copy'], ['trash', 'trash'],
  ['branch', 'git-branch'], ['user', 'user'], ['chat', 'chat-circle'], ['book', 'book-open'],
  ['database', 'database'], ['cloud', 'cloud'], ['dots', 'dots-three'], ['mic', 'microphone'],
  ['play', 'play'], ['pause', 'pause'], ['arrowUp', 'arrow-up'], ['download', 'download-simple'],
  ['bug', 'bug'], ['flask', 'flask'], ['lightning', 'lightning'], ['stack', 'stack-simple'],
  ['tree', 'tree-structure'], ['archive', 'archive'], ['pin', 'push-pin'], ['eye', 'eye'],
  ['brain', 'brain'], ['note', 'note-pencil'], ['hourglass', 'hourglass'],
  ['arrowDown', 'arrow-down'], ['retry', 'arrow-clockwise'], ['half', 'square-half'], ['spinner', 'circle-notch'],
]

/**
 * Phosphor's regular weight is a filled outline, so a glyph can be several
 * `<path>` elements. Concatenating their `d` attributes is safe here because
 * they all share one fill rule and one colour; it keeps the component down
 * to a single `<path>` per icon.
 */
function pathsOf(svg: string, asset: string): string {
  const found = [...svg.matchAll(/<path d="([^"]+)"/g)].map(match => match[1])
  if (found.length === 0) throw new Error(`no <path> in ${asset}.svg — did the upstream asset format change?`)
  return found.join(' ')
}

async function main(): Promise<void> {
  const dir = process.argv[2]
  if (!dir) throw new Error('usage: bun scripts/generateIcons.ts <path to @phosphor-icons/core assets/regular>')

  const entries: string[] = []
  for (const [local, asset] of MAP) {
    const svg = await readFile(join(dir, `${asset}.svg`), 'utf8')
    entries.push(`  /* ${asset} */ ${local}: '${pathsOf(svg, asset)}',`)
  }

  const file = `// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The app's icon set. GENERATED — edit \`scripts/generateIcons.ts\` instead.
 *
 * Geometry is Phosphor Icons (regular weight), MIT licensed, from
 * \`@phosphor-icons/core@2.1.1\` — https://github.com/phosphor-icons/core.
 *
 * These are filled outlines on a 256 grid, not 1.5px strokes on a 20 grid
 * like the hand-drawn set they replace. That matters for two reasons: the
 * weight stays optically constant at every size instead of going spindly as
 * the icon grows, and shapes that were previously approximated with a few
 * line segments (the gear, the robot, the diff) are now actually drawn.
 */

import type { ReactElement } from 'react'

const paths = {
${entries.join('\n')}
} as const

export type IconName = keyof typeof paths

export function Icon({ name, size = 18 }: { name: IconName; size?: number }): ReactElement {
  return (
    <svg
      className="app-icon"
      width={size}
      height={size}
      viewBox="0 0 256 256"
      fill="currentColor"
      aria-hidden="true"
      focusable="false"
    >
      <path d={paths[name]} />
    </svg>
  )
}
`
  const target = new URL('../src/desktop/renderer/Icon.tsx', import.meta.url).pathname
  await writeFile(target, file)
  console.log(`wrote ${target} — ${MAP.length} icons`)
}

await main()
