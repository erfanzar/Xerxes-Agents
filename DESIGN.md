---
name: Xerxes Desktop
description: Compact developer-agent workspace with macOS-inspired glass materials.
colors:
  x-screen: "#202124"
  x-chrome: "#28292d"
  x-selected: "#3b3d43"
  x-title: "#f0f0f3"
  x-prose: "#dedee3"
  x-secondary: "#c2c3ca"
  x-meta: "#a1a3ae"
  x-accent: "#79b8ff"
  x-focus: "#79b8ff"
  x-hairline: "#ffffff12"
  mac-toolbar: "#20202014"
  glass-navigation: "#20202014"
  glass-overlay: "#292929b8"
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
    fontSize: "15px"
    lineHeight: 1.65
  mono:
    fontFamily: "'SF Mono', Menlo, monospace"
rounded:
  r-sm: "4px"
  r-md: "6px"
  r-lg: "8px"
  overlay: "16px"
  composer: "18px"
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

**Creative North Star: "macOS-inspired glass workspace"**

Xerxes is a Bun/Electron desktop for working with developer agents. The approved direction is macOS-inspired glass: translucent navigation, softly highlighted controls, and an opaque reading surface within a compact continuous workspace. The serif XERXES wordmark remains the signature.

This material refinement supersedes the earlier no-glass direction. It uses Electron native vibrancy and CSS translucent materials; it does not implement Apple’s SwiftUI Liquid Glass refraction API.

**Key Characteristics:**

- Translucent navigation and elevated overlays.
- Opaque conversation, code, and input surfaces for reading.
- Compact controls, restrained blue accents, and the serif XERXES wordmark.

Source of truth: xerxes/src/desktop/renderer/atelier.css and xerxes/src/desktop/main.ts. This document records the approved material scope. Review evidence consists of four native captures in the sibling xerxes-desktop-verification/liquid/ directory: workspace, menu, dark settings, and light settings. That evidence supports this bounded material pass, not an entire-app quality or accessibility claim.

## Colors

The palette pairs cool neutral surfaces with a restrained blue interaction accent. Unsuffixed frontmatter colors describe dark mode; -light entries describe light mode. Runtime components bind to the existing CSS properties so theme changes propagate automatically.

### Primary

Blue accent indicates selection and activity; the focus token supplies keyboard outlines. Existing success, warning, and failure semantic colors remain defined in the stylesheet.

### Neutral

Screen supplies the opaque reading ground. Navigation and toolbar materials reveal the native backdrop. Overlay is denser for menus and dialogs; edge and highlight define their soft illuminated boundary. Title, prose, secondary, and metadata colors preserve text hierarchy.

## Typography

Use the system sans stack for controls and conversation text, the monospace stack for source and terminal output, and the display stack for the XERXES welcome wordmark. The body role records the final conversation and Markdown rule. User-message text retains its 14px size with a 1.65 line height; the composer draft uses 14px with a 24px line height. The responsive wordmark stays within its conversation container and reduces to clamp(48px,18cqi,112px) below a 650px window height. Do not extend the serif treatment to dense navigation or settings labels.

## Layout

Preserve the continuous workspace: a 44px top bar, a default 240px navigation panel, flexible conversation, and a default 340px inspector. Native macOS traffic lights remain system-owned, with renderer space reserved at the top left. The window starts at 1560 × 980 and has a 760 × 560 minimum.

Keep the reading column bounded at 760px and the composer at 1080px. Navigation actions use a 30px minimum; session rows use a 32px minimum with 8px corners. Both follow the four-pixel spacing rhythm. Below 1000px, toolbar labels and connection text hide and content insets tighten. Below 850px, an open inspector takes the conversation space. A conversation container below 520px simplifies its header. Catalog forms collapse below 700px; sheet navigation wraps below 650px. These are existing layout behaviors, not new mobile-product commitments.

## Elevation & Depth

Electron uses an under-window vibrancy material on macOS with visual effect state following window activation. The transparent renderer root exposes that material through the toolbar and navigation panels. Conversation and composer surroundings retain their opaque screen background. Other platforms use the configured solid window background.

CSS combines a 24px blur with 110% saturation. Navigation has broad translucency; overlays have a denser fill, highlighted edge, and ambient shadow. The composer layers the control gradient over chrome and uses a smaller shadow. Selected sessions and segments receive the same subtle control highlight. Exact gradient, shadow, focus, and motion expressions are in the sidecar and runtime CSS.

Reduced transparency or increased contrast replaces navigation, toolbar, and overlay colors with solids, disables blur and control gradients, and paints the renderer root opaque. Unsupported backdrop filtering falls back to solid chrome surfaces. Reduced motion disables animations and transitions. These are implemented fallbacks; no system-wide accessibility certification is implied.

## Shapes

Small controls retain compact corners from the radius scale. Floating overlays use the overlay radius; the composer has a softer, larger outline. The navigation and inspector remain flush to the workspace. The send control is circular. Preserve these differences instead of giving every surface the same rounded-card silhouette.

## Components

- **Buttons:** primary sheet actions invert title and screen colors; quiet actions use transparent backgrounds and selected-color hover. Keyboard focus uses a two-pixel focus outline with a two-pixel offset. Disabled buttons reduce opacity and use the default cursor.
- **Inputs:** opaque screen fill, subtle hairline border, compact radius, and system text. The composer textarea stays visually integrated within its highlighted container; focus within changes its border to the focus color.
- **Navigation:** compact rows, quiet metadata, highlighted current session, and selected segments. Material belongs to the surrounding panel; rows remain readable within it.
- **Chips:** compact transparent context controls with metadata text; hover raises text emphasis and adds the control fill.
- **Containers:** menus, palettes, modal sheets, and model selectors share dense overlay material and the larger ambient shadow. Inline content retains the quieter existing treatment.
- **Composer:** highlighted chrome surface, subtle inner rim, small ambient shadow, and bounded width. Keep enough room for a growing draft and the control row.

## Do's and Don'ts

- Do reuse the theme-specific material tokens for navigation, toolbar, overlays, and the composer.
- Do retain the opaque reading surface and visible keyboard focus.
- Do check both themes and the reduced-transparency fallback after material changes.
- Don’t describe the Electron/CSS implementation as Apple’s native SwiftUI Liquid Glass.
- Don’t turn every message or content block into a floating glass card.
- Don’t treat the four material screenshots as verification of every application flow.

## Workspace chrome placement

Activity and background-work counts share the top toolbar action. Connection and runtime-version status belong beside workspace controls in the sidebar footer; a compact toolbar fallback appears only with the sidebar hidden. Details open in a viewport-contained popover. Toolbar action groups use an 18px radius and individual toolbar buttons 14px; these intentional small-control shapes differ from rectangular content surfaces.
