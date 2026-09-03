---
name: baoyu-infographic
description: Create infographics from any layout (21 options) x style (21 options) combination.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [infographic, visual-summary, creative]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/creative/baoyu-infographic/SKILL.md
---

# Infographic Generator

Two dimensions: **layout** (information structure) x **style** (visual
aesthetics). Freely combine any layout with any style. The deliverable in
Xerxes is an artifact the agent authors directly: a standalone SVG or HTML
infographic (plus the structured content and rationale files), not a call to a
hosted image generator.

## When to use

Trigger this skill when the user asks for an infographic, visual summary, or
information graphic, or uses terms like "信息图" or "可视化". The user provides
content (text, file path, URL, or topic) and optionally specifies layout,
style, aspect ratio, or language.

## Options

| Option | Values |
|--------|--------|
| Layout | 21 options (see Layout gallery), default: `bento-grid` |
| Style | 21 options (see Style gallery), default: `craft-handmade` |
| Aspect | landscape (16:9), portrait (9:16), square (1:1), or any custom W:H |
| Language | en, zh, ja, etc. |

## Layout gallery

| Layout | Best for |
|--------|----------|
| `linear-progression` | Timelines, processes, tutorials |
| `binary-comparison` | A vs B, before-after, pros-cons |
| `comparison-matrix` | Multi-factor comparisons |
| `hierarchical-layers` | Pyramids, priority levels |
| `tree-branching` | Categories, taxonomies |
| `hub-spoke` | Central concept with related items |
| `structural-breakdown` | Exploded views, cross-sections |
| `bento-grid` | Multiple topics, overview (default) |
| `iceberg` | Surface vs hidden aspects |
| `bridge` | Problem-solution |
| `funnel` | Conversion, filtering |
| `isometric-map` | Spatial relationships |
| `dashboard` | Metrics, KPIs |
| `periodic-table` | Categorized collections |
| `comic-strip` | Narratives, sequences |
| `story-mountain` | Plot structure, tension arcs |
| `jigsaw` | Interconnected parts |
| `venn-diagram` | Overlapping concepts |
| `winding-roadmap` | Journey, milestones |
| `circular-flow` | Cycles, recurring processes |
| `dense-modules` | High-density modules, data-rich guides |

## Style gallery

`craft-handmade` (hand-drawn paper craft, default), `claymation`, `kawaii`,
`storybook-watercolor`, `chalkboard`, `cyberpunk-neon`, `bold-graphic` (comic,
halftone), `aged-academia` (vintage sepia), `corporate-memphis` (flat vector),
`technical-schematic` (blueprint), `origami`, `pixel-art`, `ui-wireframe`,
`subway-map`, `ikea-manual` (minimal line art), `knolling` (organized
flat-lay), `lego-brick`, `pop-laboratory` (lab-precision blueprint grid),
`morandi-journal` (warm doodle tones), `retro-pop-grid` (1970s Swiss pop),
`hand-drawn-edu` (macaron pastels, stick figures).

## Recommended combinations

| Content type | Layout + style |
|--------------|----------------|
| Timeline / history | `linear-progression` + `craft-handmade` |
| Step-by-step | `linear-progression` + `ikea-manual` |
| A vs B | `binary-comparison` + `corporate-memphis` |
| Hierarchy | `hierarchical-layers` + `craft-handmade` |
| Conversion | `funnel` + `corporate-memphis` |
| Cycles | `circular-flow` + `craft-handmade` |
| Technical | `structural-breakdown` + `technical-schematic` |
| Metrics | `dashboard` + `corporate-memphis` |
| Educational | `bento-grid` + `chalkboard` |
| Journey | `winding-roadmap` + `storybook-watercolor` |
| Technical guide | `dense-modules` + `pop-laboratory` |
| Educational diagram | `hub-spoke` + `hand-drawn-edu` |

## Core principles

- Preserve source data faithfully: no summarization, paraphrasing, or altered
  statistics. "73% increase" stays "73% increase".
- Strip any credentials, API keys, tokens, or secrets before including source
  content in any output file.
- Define learning objectives before structuring content.
- Structure for visual communication: headlines, labels, visual elements.
- Each section conveys one clear concept; do not mix styles across a piece.

## Workflow

1. **Analyze content.** Save the source to
   `infographic/{topic-slug}/source.md` with the file write tool (back up any
   existing file as `source-backup-YYYYMMDD-HHMMSS.md`). Analyze topic, data
   type, complexity, tone, audience, and language; detect design instructions
   in the user's input; save the analysis to `analysis.md`.
2. **Generate structured content** → `structured-content.md`: title and
   learning objectives; per-section key concept, verbatim content, visual
   element, and text labels; all statistics and quotes copied exactly. Markdown
   only, no new information.
3. **Recommend combinations.** Propose 3-5 layout x style combinations based
   on data structure, tone, and audience, with rationale.
4. **Confirm options.** Ask the user which combination, which aspect ratio,
   and (only if the source language differs) which language for the text.
5. **Author the artifact.** Build a standalone SVG (or a self-contained HTML
   page embedding the SVG) that realizes the chosen layout's information
   structure in the chosen style's visual language, at the confirmed aspect
   ratio. Write it to `infographic/{topic-slug}/infographic.svg` (or `.html`).
   Apply the style consistently: one palette, one line weight, one texture
   family across every section.
6. **Report.** Topic, layout, style, aspect, language, output path, files
   created.

## Verification

- Open the SVG/HTML with a screenshot or vision-capable tool when available
  and check: all source data points present and exact, labels legible at the
  target size, sections follow the layout's structure, style is consistent.
- Without a visual check, verify structurally: every structured-content
  section maps to a rendered region and every statistic string matches the
  source character for character. Say which verification was performed.
- Keep the file self-contained: no remote fonts or images unless the user
  supplies them.

## Pitfalls

1. **Data integrity first** — never round, soften, or reinterpret a statistic.
2. **Strip secrets** before writing any output file.
3. **One concept per section**; overloading kills readability.
4. **Style consistency** — do not mix styles mid-piece.
5. **Legible density** — `dense-modules` still needs whitespace rhythm; scale
   type to the smallest plausible viewing size, not to full-screen zoom.

---

Adapted from the `baoyu-infographic` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright 宝玉 (JimLiu).
