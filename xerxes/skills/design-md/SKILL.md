---
name: design-md
description: Author, validate, and export Google's DESIGN.md design-token spec files.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [design, design-system, tokens, ui, accessibility, wcag]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/creative/design-md/SKILL.md
---

# DESIGN.md Skill

DESIGN.md is Google's open spec (`google-labs-code/design.md`, Apache-2.0) for
describing a visual identity to coding agents. One file combines:

- **YAML front matter** — machine-readable design tokens (normative values)
- **Markdown body** — human-readable rationale in canonical sections

Tokens give exact values; prose tells agents why those values exist and how to
apply them. An official CLI (`@google/design.md` on npm) lints structure and
WCAG contrast, diffs versions, and exports to Tailwind or W3C DTCG JSON. This
skill covers authoring the file by hand and validating it with that CLI when
Node/npx is available.

## When to use

- The user asks for a DESIGN.md file, design tokens, or a design system spec
- They want consistent UI/brand across multiple projects or tools
- They paste an existing DESIGN.md and ask to lint, diff, export, or extend it
- They want WCAG contrast validation on a color palette

For ready-made brand looks use `popular-web-designs`; for the *process and
taste* of designing a one-off HTML artifact use `claude-design`. This skill is
for the formal spec file itself.

## File anatomy

```md
---
version: alpha
name: Heritage
description: Architectural minimalism meets journalistic gravitas.
colors:
  primary: "#1A1C1E"
  secondary: "#6C7278"
  tertiary: "#B8422E"
  neutral: "#F7F5F2"
typography:
  h1:
    fontFamily: Public Sans
    fontSize: 3rem
    fontWeight: 700
    lineHeight: 1.1
    letterSpacing: "-0.02em"
rounded:
  sm: 4px
  md: 8px
spacing:
  md: 16px
components:
  button-primary:
    backgroundColor: "{colors.tertiary}"
    textColor: "#FFFFFF"
    rounded: "{rounded.sm}"
    padding: 12px
---

## Overview
Rationale prose...

## Colors
- Primary (#1A1C1E): deep ink for headlines and core text.

## Typography
Public Sans everywhere except small all-caps labels...

## Components
`button-primary` is the only high-emphasis action on a page...
```

## Token types

| Type | Format | Example |
|------|--------|---------|
| Color | any CSS color | `"#1A1C1E"`, `"oklch(62% 0.18 250)"` |
| Dimension | number + unit (`px`, `em`, `rem`) | `48px`, `"-0.02em"` |
| Token reference | `{path.to.token}` | `{colors.primary}` |
| Typography | object with `fontFamily`, `fontSize`, `fontWeight`, `lineHeight`, `letterSpacing`, `fontFeature`, `fontVariation` | see above |

Component property whitelist: `backgroundColor`, `textColor`, `typography`,
`rounded`, `padding`, `size`, `height`, `width`. Variants (hover, active,
pressed) are separate sibling entries (`button-primary-hover`), never nested.

## Canonical section order

Sections are optional, but present ones appear in this order; consumers reject
duplicates and expect the order:

1. Overview (alias: Brand & Style)
2. Colors
3. Typography
4. Layout (alias: Layout & Spacing)
5. Elevation & Depth (alias: Elevation)
6. Shapes
7. Components
8. Do's and Don'ts

Unknown sections are preserved; unknown token names are accepted if the value
type is valid; unknown component properties produce a warning.

## Workflow: authoring a new DESIGN.md

1. Ask for (or infer) brand tone, accent color, and typography direction. If
   the user supplied a site, image, or vibe, translate it into the token shape.
2. Write `DESIGN.md` in the project root with the file write tool. Always
   include `name:` and `colors:`; other sections are optional but encouraged.
3. Use token references (`{colors.primary}`) in `components:` instead of
   re-typing hex values, keeping the palette single-source.
4. Lint (below) and fix broken references or WCAG failures before returning.
5. For an existing project, also write Tailwind or DTCG exports next to the
   file (`tailwind.theme.json`, `tokens.json`) when requested.

## Workflow: lint / diff / export

The CLI is `@google/design.md`, run through npx with no global install:

```bash
npx -y @google/design.md lint DESIGN.md
npx -y @google/design.md diff DESIGN.md DESIGN-v2.md
npx -y @google/design.md export --format json-tailwind DESIGN.md > tailwind.theme.json
npx -y @google/design.md export --format css-tailwind DESIGN.md > theme.css
npx -y @google/design.md export --format dtcg DESIGN.md > tokens.json
```

Run these through the shell/terminal tool. `lint` exits 1 on errors (warnings
alone exit 0). `export` succeeds regardless of lint findings in the source, so
run `lint` separately to gate on them. On Windows the `design.md` bin name can
collide with the `.md` file association; use
`npx -y -p @google/design.md designmd lint DESIGN.md`.

### Lint rule reference (CLI 0.3.0)

- `broken-ref` (error) — `{colors.missing}` points at a non-existent token
- `contrast-ratio` (warning) — component text/background below WCAG AA (4.5:1)
- `missing-primary`, `missing-typography` (warnings) — colors defined but a
  primary or typography token absent
- `orphaned-tokens` (warning) — color tokens never referenced by a component
- `section-order`, `unknown-key` (warnings) — ordering or near-typo keys
- `token-summary`, `missing-sections` (info)

When the user cares about accessibility, call WCAG findings out explicitly in
the summary.

## Pitfalls

- **Do not nest component variants.** `button-primary-hover` as a sibling key,
  not `button-primary.hover`.
- **Quote hex colors and negative dimensions.** `letterSpacing: -0.02em`
  parses as a YAML flow; write `"-0.02em"`.
- **Section order matters** even though the linter only warns; reorder prose to
  the canonical list before saving.
- **Typography sub-property typos are silently dropped** in exports
  (`fontwight:` vanishes); double-check names against the schema.
- **`version: alpha` is the current spec version**; the spec may change.
- **Token references resolve by dotted path.** `{colors.primary}` works;
  `{primary}` does not.

## Verification

Run `lint` when npx is available and report the findings. Without Node
available, validate by hand: every token reference resolves to a defined path,
hex values are quoted, variants are siblings, sections follow the canonical
order, and component text/background pairs meet 4.5:1 contrast. State which
validation was actually performed.

---

Adapted from the `design-md` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Hermes Agent contributors.
