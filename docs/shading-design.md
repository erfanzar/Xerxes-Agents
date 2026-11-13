# Shading, redrawn

Status: plan for the 0.8.40 redesign of the desktop Shading style. Written with
Anthropic's `frontend-design` skill (`xerxes/skills/frontend-design`). This file
comes before the code changes. Each change is checked against it.

## The brief, in one paragraph

**Subject.** Xerxes is a control desk for one engineer supervising many coding
agents on several machines: a Mac, TPU hosts and a Hub server.

**Primary job.** Answer "what is running, what needs me, what broke" at a
glance, then get out of the way of reading and writing.

**Audience.** One technical person who keeps the app open all day, mostly in
dark mode, often for hours.

**Name.** The product's name and mark are Persian (Xerxes, the Simurgh), so the
visual vocabulary comes from that world rather than from a generic dev-tool kit.

## Starting point (screenshot of the preview fixture, 2026-11-13)

- Neutral near-black (#0a0a0a) with grey cards. This is generated-design
  default #2 in the skill.
- Three bordered boxes stacked above the composer: the goal bar, the workspace
  bar and the input.
- Meta strings joined with middle dots: `This Mac · xerxes-workspace`,
  `3 working · 1 failed · 8`, `Agents · Auto`.
- Monospace for small data labels: inspector numbers, `(fixture/reviewer)`
  under each agent.
- A grey dot is the only cue for a working task. Live state, the one thing that
  matters, has no more weight than a footer label.

## Tokens

### Colour

The palette is lapis and faience: the glazed tile of Persian architecture.
There is one deep ground, one warm text colour, and three glaze colours that
are used only for state.

| Role | Dark | Light |
|---|---|---|
| Ground, lapis night / parchment | `#0e1424` | `#f7f3ea` |
| Chrome (sidebar, inspector) | `#0b101d` | `#efe9dc` |
| Raised (cards, popovers) | `#162036` | `#fffdf8` |
| Selected | `#1e2a45` | `#e6dfcf` |
| Text, parchment / ink | `#f2ebdd` | `#14192b` |
| Meta text | `#9aa3b8` | `#555b6a` |
| Working, turquoise faience | `#4fd1c5` | `#0f6f68` |
| Needs you, saffron | `#f2b544` | `#7f5300` |
| Failed, pomegranate | `#f2727a` | `#b0222c` |
| Done, pistachio | `#8ccb7e` | `#2e6a29` |

Primary actions are drawn in parchment on the dark ground and in lapis
`#1f3a8a` on the light ground. Nothing else is filled.

### Type

- One family for the UI: the system sans (SF Pro), with `tabular-nums` for
  every number, so numbers line up without falling back to monospace.
- A second, clearly different family for exactly two places: the task title in
  the header and the welcome wordmark. These use `ui-serif` (New York on
  macOS), which matches the old-manuscript feel of the name.
- Monospace only for real code, commands, paths and model ids.
- Scale (13 px base): 11.5 / 13 / 14.5 / 18 / 26. There are two weights, 400
  and 600.

### Layout

```
┌ sidebar ─────┬ conversation ───────────────────────┬ inspector ─┐
│ ▎Task (live)  │ Serif task title     state pill     │ State      │
│  Task         │                                      │ numbers in │
│  Task         │  prose column ≤ 72ch, left aligned   │ a 2-col    │
│               │                                      │ table      │
│               │ ┌ one composer ─────────────────────┐│            │
│ Connected     │ │ goal · workspace as inline chips  ││ Agents:    │
│ Settings      │ │ input                             ││ rule per   │
└───────────────┴─┴────────────────────────────────────┘┴────────────┘
```

- Live state gets a 2 px turquoise rule on the left edge of the task row. A
  waiting task gets a saffron rule and a failed one a pomegranate rule. Idle
  tasks get nothing. This is where the design spends its boldness.
- The goal and workspace bars join the composer as one surface with a quiet
  header row, instead of three separate bordered boxes.
- The inspector reads as a table (label on the left, tabular value on the
  right) with hairline section breaks, and no boxes inside boxes.
- Failure cards keep the pomegranate border. The raw provider JSON goes in a
  monospace block below the readable first line.
- The update banner stays a floating bar. It must never be a drag region.

### Principles

1. Live state is the only loud thing. Everything else is parchment on lapis at
   rest.
2. Structure comes from alignment and hairlines, not from cards.
3. Numbers line up (`tabular-nums`). Monospace means "this is code".
4. Sentence case everywhere. No all-caps eyebrows.
5. Motion only answers an action, plus one slow breathing glow on a working
   rule. That glow is turned off under `prefers-reduced-motion`.

## Review against the skill's list of generated-design defaults

| Default | This plan |
|---|---|
| Cream background with a terracotta accent | No. The light theme is parchment, but the accent is lapis, and the warm hues are reserved for state. |
| Near-black with one acid accent | Replaced. The ground is a coloured lapis, and there are three state glazes, not one decorative accent. |
| Broadsheet layout with zero border radius | No. The panes are flat, but controls keep a 6–10 px radius. |
| SaaS card kit | Removed from the inspector and the composer stack. |
| All-caps eyebrows, dotted meta, monospace labels | Removed: sentence case, real sentences, and monospace only for code. |

**Revised after review.** The first draft used turquoise as the general accent
for links and buttons. That is default #2 again (one bright accent on a dark
ground). Turquoise now means "working" and nothing else.

## Guardrails that stay

- Every selector is scoped to `:root[data-style='shading']:not(#shading)`, and
  keyframes are prefixed `shading-`.
- WCAG AA (4.5:1) for every text token on every surface, in both themes. The
  test measures this.
- Visible keyboard focus, `prefers-reduced-motion`, and narrow widths down to
  620 px.
