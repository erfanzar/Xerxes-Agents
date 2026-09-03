---
name: claude-design
description: Design one-off HTML artifacts (landing pages, decks, prototypes) in a terminal-only environment.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [design, html, prototype, ux, ui, creative]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/creative/claude-design/SKILL.md
---

# Design: One-off HTML Artifacts

Produce designed artifacts (landing pages, prototypes, decks, component labs,
motion studies) as complete local files. Act as an expert designer working with
the user as the manager. There is no hosted design UI here: the deliverable is
a self-contained HTML file the agent authors directly, with the exact on-disk
path reported at the end.

Check for sibling design skills first: `popular-web-designs` (ready-to-paste
design systems for known brands), `design-md` (formal design-token spec files),
`excalidraw` (hand-drawn diagrams). Use this skill when the deliverable is a
rendered artifact, not a token file or a brand clone.

## When to use

Landing and teaser pages, high-fidelity prototypes, interactive mockups,
visual option boards, component explorations, HTML slide decks, motion
studies, onboarding flows, dashboard concepts, redesigns based on screenshots
or UI kits. Not for DESIGN.md token authoring (use `design-md`).

## Start from context, not vibes

Before designing, look for source context: brand docs, product screenshots,
repo components, token files, UI kits, prior mockups, copy, constraints. If a
repo exists, read the theme, token, stylesheet, layout, and component files
that define the visual vocabulary before inventing UI. If context is missing
and fidelity matters, ask a few short focused questions instead of producing a
generic mockup; skip questions when the brief is clear.

## Surface-first: commit before tokens

Most design slop is compositional, not cosmetic. Before writing any color or
type scale, commit to exactly one surface archetype and state it in one line:

1. **Monitor** — watching state change (dashboards, status pages): density and
   glanceability, no marketing framing.
2. **Operate** — taking action (consoles, admin panels, queues): action
   affordances and selection state dominate.
3. **Compare** — weighing options (pricing, spec tables): aligned columns,
   one differentiator emphasized.
4. **Configure** — setting things up (settings, wizards): progressive
   disclosure, clear save/validation states, low decoration.
5. **Decide / Learn** — being convinced or taught (landing, docs): one idea
   per section; the only surface where a hero is usually correct.
6. **Explore** — browsing (galleries, catalogs): filters, result grids,
   zoom/peek are the composition.
7. **Command / Inspect** — keyboard-driven or drilling into one object:
   speed and focus over breadth.

If a screen spans two, name the primary one. The hero-plus-three-cards
composition is correct for Decide/Learn only.

## Workflow

1. **Understand the brief:** what, for whom, what artifact, what is locked.
2. **Gather context** from docs, screenshots, and repo files.
3. **Commit to a surface** (above).
4. **Define the artifact's system:** colors, type, spacing, radii, elevation,
   motion posture, component treatment, interaction rules.
5. **Choose the format:** static comparison canvas, clickable prototype,
   fixed-size deck, component lab, or animation study.
6. **Build the artifact.** Prefer one self-contained HTML file with embedded
   CSS and JS. If the user asked for code in an existing repo, use the repo's
   actual stack and components instead of a standalone artifact. For major
   revisions keep prior versions (`Name v2.html`) or use in-page toggles.
7. **Verify:** confirm the file exists, run available syntax checks, and if
   browser/screenshot tools are available, open it and check for console
   errors at the primary viewport. Run the slop self-audit below and repair
   only what it flags. Say exactly what was and was not verified.
8. **Report briefly:** exact path, what it contains, caveats, next decision.

## Artifact standards

- CSS variables for tokens, CSS grid for layout, real focus and hover states,
  `prefers-reduced-motion` handling for non-trivial motion, responsive
  behavior unless the format is fixed-size, semantic HTML where practical.
- Mobile hit targets at least 44px; print text at least 12pt; deck text at
  24px or larger at 1920x1080.
- Decks: fixed 1920x1080 canvas scaled to the viewport, keyboard navigation,
  visible slide count, 1-2 background colors, sparse slides solved with layout
  and scale rather than filler text.
- Prototypes: make the primary path clickable and include default, hover,
  loading, empty, error, and success states where relevant.
- Variations: default to three (conservative, strong-fit, divergent) exploring
  layout, hierarchy, density, and interaction model, not color swaps alone.
  When the user picks a direction, consolidate.
- React from CDN only when state complexity warrants it; pin exact versions
  and give global style objects specific names.

## Slop diagnostic: score before you fix

Before polishing, audit the artifact and score it 0-10 on these tells; then
repair in the register each one calls for:

1. tech gradient on everything; 2. default indigo accent nobody chose;
3. feature-tile grid of equal-weight icon+heading+sentence rows; 4. accent
rail strips on cards; 5. glassmorphism with no depth system; 6. oversized
vanity statistics; 7. icon topper above every heading; 8. everything centered
with no composition; 9. default typeface used by default; 10. composition
mismatched to the surface (hero on a dashboard).

Tells 3, 8, 10 call for re-layout, not recolor. Tells 1, 2, 9 call for
re-coloring and re-typesetting. Tells 4, 5, 6, 7 call for deleting decoration
and replacing it with real hierarchy. Do not declare done while compositional
tells still fire.

## Content discipline and pitfalls

- Every element must earn its place. No fake metrics, placeholder
  testimonials, generic feature grids, or invented claims. Ask before adding
  sections or copy that would change strategy.
- Do not recreate a company's proprietary UI or branded screens; extract
  general principles (density without clutter, command-first interaction,
  monochrome plus one accent) and transform them into an original design.
- Use motion to clarify state, not to decorate; never loop without purpose.
- Use real supplied imagery; otherwise clean placeholders or typography, not
  elaborate fake SVG illustrations pretending to be product shots.
- Never claim browser verification that did not happen.

---

Adapted from the `claude-design` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright BadTechBandit.
