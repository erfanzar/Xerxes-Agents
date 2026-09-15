---
name: Xerxes Desktop
description: Compact developer-agent workspace following the approved coded future preview.
colors:
  x-screen: "#11131a"
  x-chrome: "#171a24"
  x-selected: "#2a2f40"
  x-title: "#f0f0f3"
  x-prose: "#dedee3"
  x-secondary: "#c2c3ca"
  x-meta: "#a1a3ae"
  x-accent: "#b4bdff"
  x-focus: "#b4bdff"
  x-hairline: "#ffffff12"
  mac-toolbar: "#141620f5"
  glass-navigation: "#141620f5"
  glass-overlay: "#1c202cf5"
  glass-edge: "#ffffff20"
  glass-highlight: "#ffffff26"
  x-screen-light: "#fafafa"
  x-chrome-light: "#eeeff1"
  x-selected-light: "#e0e2e6"
  x-title-light: "#242529"
  x-prose-light: "#38393e"
  x-secondary-light: "#555862"
  x-meta-light: "#686b75"
  x-accent-light: "#0068d7"
  x-focus-light: "#007aff"
  x-hairline-light: "#00000014"
  mac-toolbar-light: "#ffffff1f"
  glass-navigation-light: "#ffffff1f"
  glass-overlay-light: "#fafafac7"
  glass-edge-light: "#ffffffb3"
  glass-highlight-light: "#ffffffd9"
typography:
  display:
    fontFamily: "Didot, 'Bodoni MT', 'Times New Roman', serif"
    fontSize: "clamp(48px,21cqi,160px)"
    fontWeight: 400
    lineHeight: 1.05
    letterSpacing: "-.04em"
  body:
    fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', sans-serif"
    fontSize: "14px"
    lineHeight: 1.85
  mono:
    fontFamily: "'SF Mono', Menlo, monospace"
rounded:
  r-sm: "4px"
  r-md: "6px"
  r-lg: "8px"
  overlay: "16px"
  composer: "14px"
spacing:
  space-1: "4px"
  space-2: "8px"
  space-3: "12px"
  space-4: "16px"
  space-6: "24px"
components:
  button-primary:
    backgroundColor: "{colors.x-title}"
    textColor: "{colors.x-screen}"
    rounded: "{rounded.r-md}"
    padding: "7px 10px"
  input:
    backgroundColor: "{colors.x-screen}"
    textColor: "{colors.x-title}"
    rounded: "{rounded.r-md}"
    padding: "10px"
  navigation:
    height: "30px"
    rounded: "{rounded.r-sm}"
    padding: "6px 8px"
  overlay:
    backgroundColor: "{colors.glass-overlay}"
    rounded: "{rounded.overlay}"
  composer:
    rounded: "{rounded.composer}"
    padding: "12px"
---

# Design System: Xerxes Desktop

## Overview

**Creative North Star: "The task-centered Xerxes workspace"**

Preserve charcoal surfaces, periwinkle actions, restrained separators, and the expressive serif XERXES identity. The current scoped presentation follows Claude desktop Code view: readable conversation, expandable evidence, and persistent task context.

**Key Characteristics:**
- Compact controls and continuous reading surfaces.
- Visible task context with expandable tools and agent history.
- Keyboard access and responsive space for the conversation.

### Approved direction — September 14

The user approved the coded preview at ../xerxes-future-preview/index.html. This supersedes the earlier strongly translucent macOS material treatment: dark charcoal and cool ink surfaces, pale periwinkle selection and focus, restrained separators, and the serif XERXES identity.

Implementation uses existing runtime components and shared tokens in atelier.css. The sidebar includes a small serif wordmark while the large welcome wordmark remains. Conversation and composer share an 870px maximum width; prose retains a narrower reading measure. Goal, current task, and queued messages occupy the top of the composer dock with separators, outside the scrolling transcript. File and diff workspaces retain their wider dedicated layout. User-resized sidebar and inspector widths persist.

The toolbar is 48px high. The conversation header uses the same 48px height. Reading text is 14px with 1.85 line height. The composer dock has a 14px radius, bounded multiline input, and a compact control row. Menus retain viewport clamping and keyboard navigation. Light mode, reduced transparency, increased contrast, and reduced motion retain their existing fallbacks.

The CSS token definitions are authoritative; frontmatter records the main design tokens. Native inspection and current validation are recorded in ../xerxes-desktop-verification/approved/.

### Visual refinement

The follow-up polish uses softer charcoal surfaces, brighter secondary text, 15px conversation text at 1.75 line height, underlined sidebar tabs, flat to-do rows, and a 16px composer dock with a subtly separated task header. Existing font-size preferences still control navigation. Sidebar rows use a 36px minimum; completed task rows retain explicit status without nested pills. Native and fixture evidence: ../xerxes-desktop-verification/approved/polished-*.png.

### Desktop task presentation — September 2026

The current reference is Claude desktop Code view's task-oriented hierarchy, interpreted within Xerxes's existing palette and serif identity. Tool rows have explicit keyboard-operable disclosure controls and retain full results, failure details, and raw data. Active and failed delegated work stays visible; completed agents live in expandable history. Welcome starters cover research, planning, and building to reflect equal priority for coding and general agent work. This is a scoped presentation update, not a claim of feature parity or a daemon migration.

## Layout

The activity inspector opens by default when a workspace is available and the window can fit the visible navigation width, configured inspector width, and at least 320px of conversation. Below 850px, navigation starts hidden and has a separate toggle; opening it does not rewrite the saved wide-window sidebar preference. An explicit inspector open or close choice survives resizing and session changes. When the inspector cannot fit beside the conversation, opening it gives task context the available width.

Session statistics stay above agents. Their disclosure starts open, with compact two-column values (10px by 16px gaps), and can be collapsed. Goals, current work, and queued instructions remain above input, outside the scrolling transcript.

## Components

### Expandable activity

**The Evidence Access Rule.** Concise tool rows retain keyboard-operable disclosure and full underlying results. Failed execution exposes its error in the summary; expanded content retains command, output, standard error, and raw input/result. Output has a bounded scroll region (320px), an expanded view (70vh), and text wrapping controls where applicable.

### Agents and history

Failed and attention-needed agents precede working agents. Stopped and unknown states remain in the current roster. Only successful completion moves an agent into the initially collapsed Completed agents history; opening history reveals individually expandable records. Short previews use two lines, while expanded records retain the available summary, progress, files, usage, and tool details.

### Task context and continuation

The composer goal row opens activity. When activity already shows the objective, the row reads View goal details to avoid repeating a long objective; when it is hidden, the objective returns to the shortcut. Current work and queued instructions remain separately readable.

**The Truthful Continuation Rule.** Saved work without a displayed conversation receives Continue this task, with Review activity and a plan/todos action when those exist. The copy does not claim that saved work is still running. New-task starters include research, planning, and building.

## Do's and Don'ts

- Do preserve the established palette and serif identity while improving task hierarchy.
- Do keep full evidence reachable from concise activity summaries.
- Do retain failed, stopped, and unknown agent states outside completed history.
- Don't describe this scoped presentation update as full GUI/TUI parity or a global-daemon migration.

<!-- Scoped documentation note: existing frontmatter and historical material descriptions were preserved. Earlier palette, body-size, radius, and sidecar material drift were not reconciled in this hierarchy-only pass. The current renderer CSS remains the implementation evidence. -->
