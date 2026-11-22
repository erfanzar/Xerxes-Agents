---
name: generative-ui
description: Render rich interactive HTML widgets, live 2D/3D visualizations, architecture diagrams, physics simulations, and interactive controls inline in chat or in the Artifacts panel with zero external CDN dependencies.
version: 1.0.0
author: Xerxes-Agents
tags: [ui, 3d, visualization, html, canvas, interactive, diagrams, artifacts]
---

# Generative UI & Interactive 3D Rendering

## Metadata

- License: Apache-2.0

Use this skill whenever an interactive visual, live 3D scene, chart, architecture diagram, simulator, or custom control panel explains or demonstrates a concept better than static prose.

## Two Ways to Render Interactive UI

### 1. Saved HTML Artifact (recommended for rich widgets, 3D explorers, and dashboards)

1. Call `ArtifactTool` with `action: "create"`, `kind: "html"`, a clear `title`, `embed: true` (for compact inline previews), and self-contained HTML `content`.
2. To display the artifact inline right inside your chat reply, put this tag on its own line:

```markdown
<xerxes-embed artifact="<artifact_id>"></xerxes-embed>
```

You can also pass an optional `title` attribute or use `<agent-embed src="artifact://<artifact_id>"></agent-embed>`.

3. For tall dashboards, wide tables, or full-screen explorers (`> 450px` tall), set `embed: false` (or omit it) and omit the inline embed tag so the user opens it full-height in the Artifacts panel.

### 2. Quick Inline Fenced Widgets

For lightweight self-contained visuals directly in your message without creating an artifact, use a fenced code block with language `widget`, `html-widget`, or `3d` (with an optional title after the language):

````markdown
```widget Latency vs Concurrency
<div class="x-col" style="padding: 12px">
  <div class="x-row x-between">
    <label class="x-muted">Workers <input id="w" class="x-range" type="range" min="1" max="32" value="8" /></label>
    <span id="out" class="x-pill x-mono">8 workers</span>
  </div>
  <canvas id="c" style="width:100%;height:180px"></canvas>
</div>
<script>
  const w = document.getElementById('w'), out = document.getElementById('out');
  const draw = () => { out.textContent = w.value + ' workers'; };
  w.addEventListener('input', draw);
  draw();
</script>
```
````

Standard ```` ```html ```` code blocks always render as syntax-highlighted source code and never execute scripts.

## Offline Sandbox & Theme System

Every widget executes inside a strict sandboxed iframe (`sandbox="allow-scripts"`) with an offline Content Security Policy (`default-src 'none'; script-src 'unsafe-inline' data:; style-src 'unsafe-inline'; img-src data: blob:; font-src data:;`).

- **Zero external CDNs**: Do not load external `<script src="...">`, `<link rel="stylesheet" href="...">`, or remote fonts. Everything must be inline HTML, CSS, SVG, Canvas, WebGL/WebGL2, or `window.Xerxes3D`. Attach DOM event handlers via `addEventListener` inside `<script>` tags rather than inline HTML `onclick="..."` attributes.
- **Semantic CSS variables** (automatically updated on dark/light switch):
  - Surfaces: `--background`, `--content`, `--card`, `--sidebar`
  - Borders: `--card-border`, `--border`
  - Text: `--foreground`, `--muted-foreground`, `--placeholder`, `--code-foreground`
  - Accents & status: `--primary`, `--primary-foreground`, `--secondary`, `--secondary-foreground`, `--accent`, `--link`, `--success`, `--warning`, `--error`
  - Chart series: `--chart-1`, `--chart-2`, `--chart-3`, `--chart-4`, `--chart-5`, `--chart-6`
- **Built-in utility classes**: `.x-row`, `.x-col`, `.x-between`, `.x-grid`, `.x-card`, `.x-btn`, `.x-btn--primary`, `.x-btn.is-active`, `.x-pill`, `.x-range`, `.x-mono`, `.x-muted`, `.x-canvas-wrap`.
- **Live dark/light theme sync**: Never hardcode dark-only or light-only text/surface colors on `:root`. Read CSS variables via `getComputedStyle(document.documentElement)` or check `document.documentElement.classList.contains('light')`, and re-render canvases on:

```js
window.addEventListener('themechange', draw)
```

## Built-In Interactive 3D Engine (`window.Xerxes3D`)

Every widget iframe automatically ships with `window.Xerxes3D`, a zero-dependency 60fps HiDPI 3D engine with Phong key/rim lighting, specular highlights, depth-sorted particles, 3D-projected floating labels, and built-in orbit camera controls (drag to orbit, wheel to zoom, double-click to reset). Raw `<canvas>` WebGL/WebGL2, Canvas 2D, SVG, and CSS 3D also work natively.

### `Xerxes3D.createScene(canvas, options)` Reference

- **Options**: `{ background?: 'transparent' | 'cosmic' | 'card' | string, glowBg?: boolean, azimuth?: number, elevation?: number, distance?: number, target?: [x,y,z], fov?: number, autoRotate?: boolean | number, interactive?: boolean, rimColor?: string }`
- **Primitives** (each returns a mutable node with `.position`, `.rotation`, `.scale`, `.color`, `.colorB`, `.opacity`, `.wireframe`, `.visible`):
  - `box({ size, width, height, depth, color, wireframe })`
  - `sphere({ radius, segments, rings, color, colorB })`
  - `torus({ radius, tube, radialSegments, tubularSegments, color, colorB })`
  - `torusKnot({ radius, tube, p, q, tubularSegments, radialSegments, color, colorB })`
  - `icosahedron({ radius, detail, color, wireframe })`
  - `cylinder({ radiusTop, radiusBottom, height, segments, openEnded, color })`
  - `cone({ radius, height, segments, color })`
  - `plane({ width, height, segmentsX, segmentsY, color })`
  - `mesh({ vertices, faces, colors }, opts)`
- **Helpers & Overlays**:
  - `grid({ size, divisions, y, color, opacity })`
  - `axes({ size })`
  - `particles(pointsOrConfig, { color, size, glow, opacity })`
  - `line(points, { color, width, opacity, closed, dashed })`
  - `label(pos, text, { color, bg, fontSize, dot, offset })`
- **Camera & Lifecycle**:
  - `setCamera({ azimuth, elevation, distance, target, fov })`, `getCamera()`, `resetCamera()`
  - `animate((timeSec, dtSec) => { ... })`, `render()`, `resize()`, `dispose()`

### Example: Interactive 3D Orbital Knot

````markdown
```3d Orbital Torus Knot
<div class="x-col" style="padding: 10px 12px 12px">
  <div class="x-row x-between">
    <span class="x-muted">Drag to orbit · Scroll to zoom · Double-click to reset</span>
    <label class="x-row x-muted">
      Speed <input id="spd" class="x-range" type="range" min="0" max="2.5" step="0.1" value="0.8" />
    </label>
  </div>
  <div class="x-canvas-wrap">
    <canvas id="scene" style="width:100%;height:300px"></canvas>
  </div>
</div>
<script>
  const speedInput = document.getElementById('spd');
  const world = Xerxes3D.createScene('#scene', {
    background: 'cosmic',
    distance: 4.8,
    elevation: 0.36,
    autoRotate: 0.35,
  });
  world.grid({ size: 7, divisions: 10, y: -1.45, opacity: 0.25 });
  const knot = world.torusKnot({
    radius: 1.05,
    tube: 0.27,
    p: 2,
    q: 3,
    color: 'var(--primary)',
    colorB: 'var(--accent)',
    shininess: 36,
    specular: 0.65,
  });
  const ringPoints = [];
  for (let i = 0; i < 64; i++) {
    const a = (i / 64) * Math.PI * 2;
    const r = 2.05 + 0.22 * Math.sin(i * 3);
    ringPoints.push({
      pos: [Math.cos(a) * r, 0.35 * Math.sin(a * 2), Math.sin(a) * r],
      color: i % 3 === 0 ? 'var(--accent)' : 'var(--chart-2)',
      size: i % 5 === 0 ? 4.4 : 2.6,
    });
  }
  const dust = world.particles(ringPoints, { glow: true });
  world.label([0, 1.55, 0], 'Torus Knot (2, 3)');
  world.animate((t) => {
    const k = Number(speedInput.value);
    knot.rotation[1] = t * 0.55 * k;
    knot.rotation[0] = Math.sin(t * 0.4 * k) * 0.25;
    dust.rotation[1] = -t * 0.28 * k;
  });
</script>
```
````

## Layout & Sizing Rules

- **Inline height budget**: Target `<= 450px` total height for inline chat widgets (the host clamps inline frames to `120–520px`). Keep controls compact and flush with the chat column.
- **Never size inline widgets with viewport units**: Do not use `100vh`, `min-height: 100vh`, `h-screen`, or `height: 100%` on top-level containers of inline widgets — inline iframe height is measured from your content, so viewport-relative roots collapse or loop. Set explicit pixel height on `<canvas>` (e.g. `height: 300px`) and use padding for spacing.
- **Standalone panel artifacts**: When creating a full-pane dashboard or multi-tab explorer (`embed: false`), use `background: var(--background)` and group sections inside `.x-card` containers.
