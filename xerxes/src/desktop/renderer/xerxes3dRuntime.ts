// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Self-contained, zero-dependency interactive 3D rendering engine injected into
 * sandboxed Xerxes Generative UI widget iframes as `window.Xerxes3D`.
 *
 * Uses HiDPI 2D Canvas painter's-algorithm projection with Blinn-Phong key +
 * rim lighting, specular highlights, subtle glow halos, depth-sorted particles,
 * 3D-projected floating labels, and smooth drag/wheel/double-click orbit camera
 * controls. Works 100% offline in Electron, VS Code webviews, and software
 * headless environments with zero WebGL driver crash risk, while leaving raw
 * WebGL/WebGL2, SVG, and CSS 3D natively available in the same iframe.
 */

export const XERXES_3D_RUNTIME_SCRIPT = `(function (global) {
  'use strict';
  if (global.Xerxes3D && global.Xerxes3D.__version === '1.0') return;

  var PI = Math.PI;
  var TAU = PI * 2;

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  function vecAdd(a, b) {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  }

  function vecSub(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  }

  function vecScale(a, s) {
    return [a[0] * s, a[1] * s, a[2] * s];
  }

  function vecDot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  function vecCross(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  }

  function vecLen(a) {
    return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  }

  function vecNorm(a) {
    var l = vecLen(a);
    return l > 1e-8 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 1, 0];
  }

  function readThemeTokens() {
    var root = global.document && global.document.documentElement;
    var light = !!(root && root.classList && root.classList.contains('light'));
    var cs = root && global.getComputedStyle ? global.getComputedStyle(root) : null;
    function tok(name, darkFallback, lightFallback) {
      var raw = cs ? (cs.getPropertyValue(name) || '').trim() : '';
      return raw || (light ? lightFallback : darkFallback);
    }
    return {
      light: light,
      background: tok('--background', '#0f0f13', '#fafaf9'),
      card: tok('--card', '#18181f', '#f4f4f5'),
      border: tok('--border', 'rgba(255,255,255,0.12)', 'rgba(17,24,39,0.14)'),
      foreground: tok('--foreground', '#ececf1', '#18181b'),
      muted: tok('--muted-foreground', '#9ca3af', '#52525b'),
      primary: tok('--primary', '#a855f7', '#7e22ce'),
      accent: tok('--accent', '#fbbf24', '#d97706'),
      chart1: tok('--chart-1', '#a855f7', '#7e22ce'),
      chart2: tok('--chart-2', '#38bdf8', '#0284c7'),
      chart3: tok('--chart-3', '#fbbf24', '#d97706'),
      chart4: tok('--chart-4', '#34d399', '#059669'),
    };
  }

  function parseColor(input, theme) {
    var s = typeof input === 'string' ? input.trim() : '';
    if (!s) s = theme.primary;
    if (s.indexOf('var(') === 0) {
      var mVar = /var\\(\\s*(--[a-zA-Z0-9_-]+)/.exec(s);
      if (mVar && global.document && global.getComputedStyle) {
        var resolved = global.getComputedStyle(global.document.documentElement).getPropertyValue(mVar[1]).trim();
        if (resolved) s = resolved;
      }
    }
    if (s.charAt(0) === '#') {
      var hex = s.slice(1);
      if (hex.length === 3) {
        return [
          parseInt(hex[0] + hex[0], 16),
          parseInt(hex[1] + hex[1], 16),
          parseInt(hex[2] + hex[2], 16),
          1,
        ];
      }
      if (hex.length === 6 || hex.length === 8) {
        var a = hex.length === 8 ? parseInt(hex.slice(6, 8), 16) / 255 : 1;
        return [
          parseInt(hex.slice(0, 2), 16),
          parseInt(hex.slice(2, 4), 16),
          parseInt(hex.slice(4, 6), 16),
          a,
        ];
      }
    }
    var mRgb = /^rgba?\\(\\s*([\\d.]+)\\s*,\\s*([\\d.]+)\\s*,\\s*([\\d.]+)(?:\\s*,\\s*([\\d.]+))?\\s*\\)$/i.exec(s);
    if (mRgb) {
      return [
        clamp(Number(mRgb[1]), 0, 255),
        clamp(Number(mRgb[2]), 0, 255),
        clamp(Number(mRgb[3]), 0, 255),
        mRgb[4] !== undefined ? clamp(Number(mRgb[4]), 0, 1) : 1,
      ];
    }
    return [168, 85, 247, 1];
  }

  function mixRgb(a, b, t) {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
      a[3] + (b[3] - a[3]) * t,
    ];
  }

  function rgbaCss(r, g, b, a) {
    return 'rgba(' + Math.round(clamp(r, 0, 255)) + ',' + Math.round(clamp(g, 0, 255)) + ',' + Math.round(clamp(b, 0, 255)) + ',' + clamp(a, 0, 1).toFixed(3) + ')';
  }

  function rotateEuler(v, rot) {
    var x = v[0], y = v[1], z = v[2];
    var rx = rot[0] || 0, ry = rot[1] || 0, rz = rot[2] || 0;
    if (rx) {
      var cx = Math.cos(rx), sx = Math.sin(rx);
      var y1 = y * cx - z * sx;
      var z1 = y * sx + z * cx;
      y = y1; z = z1;
    }
    if (ry) {
      var cy = Math.cos(ry), sy = Math.sin(ry);
      var x2 = x * cy + z * sy;
      var z2 = -x * sy + z * cy;
      x = x2; z = z2;
    }
    if (rz) {
      var cz = Math.cos(rz), sz = Math.sin(rz);
      var x3 = x * cz - y * sz;
      var y3 = x * sz + y * cz;
      x = x3; y = y3;
    }
    return [x, y, z];
  }

  function transformPoint(p, pos, rot, scale) {
    var sx = Array.isArray(scale) ? scale[0] : scale;
    var sy = Array.isArray(scale) ? scale[1] : scale;
    var sz = Array.isArray(scale) ? scale[2] : scale;
    var r = rotateEuler([p[0] * sx, p[1] * sy, p[2] * sz], rot);
    return [r[0] + pos[0], r[1] + pos[1], r[2] + pos[2]];
  }

  function orientFaceOutward(verts, tri, refPoint) {
    var a = verts[tri[0]], b = verts[tri[1]], c = verts[tri[2]];
    var ab = vecSub(b, a);
    var ac = vecSub(c, a);
    var n = vecCross(ab, ac);
    var center = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3];
    var out = vecSub(center, refPoint);
    return vecDot(n, out) < 0 ? [tri[0], tri[2], tri[1]] : tri;
  }

  // ── Procedural Geometry Builders ─────────────────────────────────────────

  function buildBox(opts) {
    var s = opts.size !== undefined ? opts.size : 1;
    var hx = (opts.width !== undefined ? opts.width : s) * 0.5;
    var hy = (opts.height !== undefined ? opts.height : s) * 0.5;
    var hz = (opts.depth !== undefined ? opts.depth : s) * 0.5;
    var vertices = [
      [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
      [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ];
    var raw = [
      [4, 5, 6], [4, 6, 7],
      [1, 0, 3], [1, 3, 2],
      [7, 6, 2], [7, 2, 3],
      [0, 1, 5], [0, 5, 4],
      [5, 1, 2], [5, 2, 6],
      [0, 4, 7], [0, 7, 3],
    ];
    var faces = raw.map(function (f) { return orientFaceOutward(vertices, f, [0, 0, 0]); });
    return { vertices: vertices, faces: faces };
  }

  function buildSphere(opts) {
    var r = opts.radius !== undefined ? opts.radius : 1;
    var segs = Math.max(8, Math.round(opts.segments || 22));
    var rings = Math.max(6, Math.round(opts.rings || 14));
    var vertices = [];
    var faces = [];
    var uvT = [];
    for (var iy = 0; iy <= rings; iy++) {
      var v = iy / rings;
      var phi = v * PI;
      var sinP = Math.sin(phi);
      var cosP = Math.cos(phi);
      for (var ix = 0; ix < segs; ix++) {
        var u = ix / segs;
        var theta = u * TAU;
        vertices.push([r * sinP * Math.cos(theta), r * cosP, r * sinP * Math.sin(theta)]);
      }
    }
    for (var j = 0; j < rings; j++) {
      for (var i = 0; i < segs; i++) {
        var next = (i + 1) % segs;
        var a = j * segs + i;
        var b = j * segs + next;
        var c = (j + 1) * segs + next;
        var d = (j + 1) * segs + i;
        var t = (j + 0.5) / rings;
        if (j > 0) {
          faces.push(orientFaceOutward(vertices, [a, b, c], [0, 0, 0]));
          uvT.push(t);
        }
        if (j < rings - 1) {
          faces.push(orientFaceOutward(vertices, [a, c, d], [0, 0, 0]));
          uvT.push(t);
        }
      }
    }
    return { vertices: vertices, faces: faces, faceT: uvT };
  }

  function buildTorus(opts) {
    var R = opts.radius !== undefined ? opts.radius : 1.05;
    var tube = opts.tube !== undefined ? opts.tube : 0.32;
    var tSegs = Math.max(12, Math.round(opts.tubularSegments || 36));
    var rSegs = Math.max(6, Math.round(opts.radialSegments || 16));
    var vertices = [];
    var spines = [];
    for (var i = 0; i < tSegs; i++) {
      var u = (i / tSegs) * TAU;
      var cu = Math.cos(u), su = Math.sin(u);
      spines.push([R * cu, 0, R * su]);
      for (var j = 0; j < rSegs; j++) {
        var v = (j / rSegs) * TAU;
        var cv = Math.cos(v), sv = Math.sin(v);
        var rad = R + tube * cv;
        vertices.push([rad * cu, tube * sv, rad * su]);
      }
    }
    var faces = [];
    var faceT = [];
    for (var i2 = 0; i2 < tSegs; i2++) {
      var iNext = (i2 + 1) % tSegs;
      var sp = vecScale(vecAdd(spines[i2], spines[iNext]), 0.5);
      var grad = 0.5 - 0.5 * Math.cos((i2 / tSegs) * TAU);
      for (var j2 = 0; j2 < rSegs; j2++) {
        var jNext = (j2 + 1) % rSegs;
        var a = i2 * rSegs + j2;
        var b = iNext * rSegs + j2;
        var c = iNext * rSegs + jNext;
        var d = i2 * rSegs + jNext;
        faces.push(orientFaceOutward(vertices, [a, b, c], sp));
        faceT.push(grad);
        faces.push(orientFaceOutward(vertices, [a, c, d], sp));
        faceT.push(grad);
      }
    }
    return { vertices: vertices, faces: faces, faceT: faceT };
  }

  function buildTorusKnot(opts) {
    var R = opts.radius !== undefined ? opts.radius : 1.1;
    var tube = opts.tube !== undefined ? opts.tube : 0.27;
    var p = opts.p !== undefined ? opts.p : 2;
    var q = opts.q !== undefined ? opts.q : 3;
    var tSegs = Math.max(24, Math.round(opts.tubularSegments || 112));
    var rSegs = Math.max(6, Math.round(opts.radialSegments || 12));

    function knotPos(u) {
      var r = R * (0.66 + 0.31 * Math.cos(q * u));
      return [
        r * Math.cos(p * u),
        r * Math.sin(p * u),
        R * 0.38 * Math.sin(q * u),
      ];
    }

    var vertices = [];
    var spines = [];
    var eps = 1e-4;
    for (var i = 0; i < tSegs; i++) {
      var u = (i / tSegs) * TAU;
      var pos = knotPos(u);
      var prev = knotPos(u - eps);
      var next = knotPos(u + eps);
      var T = vecNorm(vecSub(next, prev));
      var B = vecNorm(vecCross(T, pos));
      var N = vecNorm(vecCross(B, T));
      spines.push(pos);
      for (var j = 0; j < rSegs; j++) {
        var v = (j / rSegs) * TAU;
        var cv = Math.cos(v), sv = Math.sin(v);
        vertices.push([
          pos[0] + tube * (cv * N[0] + sv * B[0]),
          pos[1] + tube * (cv * N[1] + sv * B[1]),
          pos[2] + tube * (cv * N[2] + sv * B[2]),
        ]);
      }
    }
    var faces = [];
    var faceT = [];
    for (var i2 = 0; i2 < tSegs; i2++) {
      var iNext = (i2 + 1) % tSegs;
      var sp = vecScale(vecAdd(spines[i2], spines[iNext]), 0.5);
      var grad = 0.5 + 0.5 * Math.sin((i2 / tSegs) * TAU * q);
      for (var j2 = 0; j2 < rSegs; j2++) {
        var jNext = (j2 + 1) % rSegs;
        var a = i2 * rSegs + j2;
        var b = iNext * rSegs + j2;
        var c = iNext * rSegs + jNext;
        var d = i2 * rSegs + jNext;
        faces.push(orientFaceOutward(vertices, [a, b, c], sp));
        faceT.push(grad);
        faces.push(orientFaceOutward(vertices, [a, c, d], sp));
        faceT.push(grad);
      }
    }
    return { vertices: vertices, faces: faces, faceT: faceT };
  }

  function buildIcosahedron(opts) {
    var r = opts.radius !== undefined ? opts.radius : 1;
    var detail = clamp(Math.round(opts.detail || 0), 0, 2);
    var t = (1 + Math.sqrt(5)) / 2;
    var verts = [
      [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
      [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
      [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ].map(function (v) { return vecScale(vecNorm(v), r); });
    var faces = [
      [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
      [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
      [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    for (var d = 0; d < detail; d++) {
      var cache = {};
      var nextFaces = [];
      var mid = function (i1, i2) {
        var key = i1 < i2 ? i1 + '_' + i2 : i2 + '_' + i1;
        if (cache[key] !== undefined) return cache[key];
        var v1 = verts[i1], v2 = verts[i2];
        var m = vecScale(vecNorm([(v1[0] + v2[0]) * 0.5, (v1[1] + v2[1]) * 0.5, (v1[2] + v2[2]) * 0.5]), r);
        var idx = verts.length;
        verts.push(m);
        cache[key] = idx;
        return idx;
      };
      for (var k = 0; k < faces.length; k++) {
        var f = faces[k];
        var ab = mid(f[0], f[1]);
        var bc = mid(f[1], f[2]);
        var ca = mid(f[2], f[0]);
        nextFaces.push([f[0], ab, ca], [f[1], bc, ab], [f[2], ca, bc], [ab, bc, ca]);
      }
      faces = nextFaces;
    }
    faces = faces.map(function (f) { return orientFaceOutward(verts, f, [0, 0, 0]); });
    return { vertices: verts, faces: faces };
  }

  function buildCylinder(opts) {
    var rTop = opts.radiusTop !== undefined ? opts.radiusTop : (opts.radius !== undefined ? opts.radius : 0.6);
    var rBot = opts.radiusBottom !== undefined ? opts.radiusBottom : (opts.radius !== undefined ? opts.radius : 0.6);
    var h = opts.height !== undefined ? opts.height : 1.6;
    var segs = Math.max(6, Math.round(opts.segments || 22));
    var half = h * 0.5;
    var vertices = [];
    var faces = [];
    for (var i = 0; i < segs; i++) {
      var a = (i / segs) * TAU;
      var ca = Math.cos(a), sa = Math.sin(a);
      vertices.push([rTop * ca, half, rTop * sa]);
      vertices.push([rBot * ca, -half, rBot * sa]);
    }
    var topCenter = vertices.length;
    vertices.push([0, half, 0]);
    var botCenter = vertices.length;
    vertices.push([0, -half, 0]);
    for (var i2 = 0; i2 < segs; i2++) {
      var next = (i2 + 1) % segs;
      var t0 = i2 * 2, b0 = i2 * 2 + 1;
      var t1 = next * 2, b1 = next * 2 + 1;
      if (rTop > 1e-4) faces.push(orientFaceOutward(vertices, [t0, b0, t1], [0, 0, 0]));
      if (rBot > 1e-4) faces.push(orientFaceOutward(vertices, [b0, b1, t1], [0, 0, 0]));
      if (!opts.openEnded) {
        if (rTop > 1e-4) faces.push(orientFaceOutward(vertices, [topCenter, t1, t0], [0, 0, 0]));
        if (rBot > 1e-4) faces.push(orientFaceOutward(vertices, [botCenter, b0, b1], [0, 0, 0]));
      }
    }
    return { vertices: vertices, faces: faces };
  }

  function buildCone(opts) {
    return buildCylinder({
      radiusTop: 0,
      radiusBottom: opts.radius !== undefined ? opts.radius : 0.78,
      height: opts.height !== undefined ? opts.height : 1.6,
      segments: opts.segments || 22,
      openEnded: !!opts.openEnded,
    });
  }

  function buildPlane(opts) {
    var w = opts.width !== undefined ? opts.width : 2;
    var h = opts.height !== undefined ? opts.height : 2;
    var sx = Math.max(1, Math.round(opts.segmentsX || 1));
    var sy = Math.max(1, Math.round(opts.segmentsY || 1));
    var vertices = [];
    var faces = [];
    for (var iy = 0; iy <= sy; iy++) {
      var z = (iy / sy - 0.5) * h;
      for (var ix = 0; ix <= sx; ix++) {
        var x = (ix / sx - 0.5) * w;
        vertices.push([x, 0, z]);
      }
    }
    var stride = sx + 1;
    for (var j = 0; j < sy; j++) {
      for (var i = 0; i < sx; i++) {
        var a = j * stride + i;
        var b = a + 1;
        var c = a + stride + 1;
        var d = a + stride;
        faces.push([a, d, c], [a, c, b]);
      }
    }
    return { vertices: vertices, faces: faces };
  }

  // ── Scene Engine ─────────────────────────────────────────────────────────

  function createScene(canvasOrSelector, options) {
    var opts = options || {};
    var canvas = typeof canvasOrSelector === 'string'
      ? (global.document && global.document.querySelector(canvasOrSelector))
      : canvasOrSelector;
    if (!canvas || typeof canvas.getContext !== 'function') {
      throw new Error('Xerxes3D.createScene requires a valid <canvas> element or selector');
    }
    var ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Xerxes3D could not acquire a 2D canvas context');
    }

    var objects = [];
    var theme = readThemeTokens();
    var width = 600;
    var height = 340;
    var dpr = 1;
    var disposed = false;
    var rafId = 0;
    var animCb = null;
    var animStart = 0;
    var lastFrame = 0;
    var autoRotateSpeed = opts.autoRotate === true ? 0.42 : (typeof opts.autoRotate === 'number' ? opts.autoRotate : 0);

    var initialCam = {
      azimuth: opts.azimuth !== undefined ? opts.azimuth : 0.62,
      elevation: opts.elevation !== undefined ? opts.elevation : 0.34,
      distance: opts.distance !== undefined ? opts.distance : 5.2,
      target: Array.isArray(opts.target) ? opts.target.slice(0, 3) : [0, 0, 0],
      fov: opts.fov !== undefined ? opts.fov : 42,
    };
    var cam = {
      azimuth: initialCam.azimuth,
      elevation: initialCam.elevation,
      distance: initialCam.distance,
      target: initialCam.target.slice(),
      fov: initialCam.fov,
    };

    var dragging = false;
    var lastPtrX = 0;
    var lastPtrY = 0;
    var interactive = opts.interactive !== false;
    if (interactive && canvas.style) {
      canvas.style.cursor = 'grab';
      canvas.style.touchAction = 'none';
    }

    function resize() {
      if (disposed) return;
      var rect = typeof canvas.getBoundingClientRect === 'function' ? canvas.getBoundingClientRect() : null;
      var w = (rect && rect.width > 0 ? rect.width : canvas.clientWidth) || canvas.width || 600;
      var h = (rect && rect.height > 0 ? rect.height : canvas.clientHeight) || canvas.height || 340;
      dpr = Math.min(global.devicePixelRatio || 1, 2);
      width = Math.max(1, Math.round(w));
      height = Math.max(1, Math.round(h));
      var targetW = Math.round(width * dpr);
      var targetH = Math.round(height * dpr);
      if (canvas.width !== targetW) canvas.width = targetW;
      if (canvas.height !== targetH) canvas.height = targetH;
      if (typeof ctx.setTransform === 'function') {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
    }

    function makeNode(kind, data, nodeOpts) {
      var o = nodeOpts || {};
      var node = {
        kind: kind,
        data: data,
        visible: o.visible !== false,
        position: Array.isArray(o.position) ? o.position.slice(0, 3) : [0, 0, 0],
        rotation: Array.isArray(o.rotation) ? o.rotation.slice(0, 3) : [0, 0, 0],
        scale: o.scale !== undefined ? o.scale : 1,
        color: o.color,
        colorB: o.colorB,
        wireframe: !!o.wireframe,
        wireframeColor: o.wireframeColor,
        strokeWidth: o.strokeWidth !== undefined ? o.strokeWidth : 0.65,
        opacity: o.opacity !== undefined ? clamp(o.opacity, 0, 1) : 1,
        doubleSided: !!o.doubleSided,
        shininess: o.shininess !== undefined ? o.shininess : 28,
        specular: o.specular !== undefined ? o.specular : 0.5,
        emissive: o.emissive !== undefined ? o.emissive : 0.08,
        glow: o.glow,
        size: o.size,
        width: o.width,
        closed: !!o.closed,
        dashed: !!o.dashed,
        text: o.text,
        bg: o.bg,
        fontSize: o.fontSize || 11,
        dot: o.dot !== false,
        offset: Array.isArray(o.offset) ? o.offset : [0, -12],
      };
      return node;
    }

    function add(node) {
      if (node && objects.indexOf(node) === -1) objects.push(node);
      return node;
    }

    function remove(node) {
      var idx = objects.indexOf(node);
      if (idx >= 0) objects.splice(idx, 1);
    }

    function clear() {
      objects.length = 0;
    }

    function addMeshPrimitive(geom, o) {
      return add(makeNode('mesh', geom, o));
    }

    function box(o) { return addMeshPrimitive(buildBox(o || {}), o); }
    function sphere(o) { return addMeshPrimitive(buildSphere(o || {}), o); }
    function torus(o) { return addMeshPrimitive(buildTorus(o || {}), o); }
    function torusKnot(o) { return addMeshPrimitive(buildTorusKnot(o || {}), o); }
    function icosahedron(o) { return addMeshPrimitive(buildIcosahedron(o || {}), o); }
    function cylinder(o) { return addMeshPrimitive(buildCylinder(o || {}), o); }
    function cone(o) { return addMeshPrimitive(buildCone(o || {}), o); }
    function plane(o) {
      var po = Object.assign({ doubleSided: true }, o || {});
      return addMeshPrimitive(buildPlane(po), po);
    }
    function mesh(geometry, o) {
      var geom = {
        vertices: (geometry && geometry.vertices) || [],
        faces: (geometry && geometry.faces) || [],
        faceColors: geometry && geometry.colors,
        faceT: geometry && geometry.faceT,
      };
      return addMeshPrimitive(geom, o);
    }

    function line(points, o) {
      return add(makeNode('line', { points: points || [] }, o));
    }

    function grid(o) {
      var go = o || {};
      var size = go.size !== undefined ? go.size : 8;
      var divs = Math.max(2, Math.round(go.divisions || 12));
      var y = go.y !== undefined ? go.y : -1.5;
      var half = size * 0.5;
      var step = size / divs;
      var segments = [];
      for (var i = 0; i <= divs; i++) {
        var v = -half + i * step;
        segments.push([[-half, y, v], [half, y, v]]);
        segments.push([[v, y, -half], [v, y, half]]);
      }
      return add(makeNode('segments', { segments: segments }, Object.assign({
        color: go.color || theme.border,
        opacity: go.opacity !== undefined ? go.opacity : (theme.light ? 0.55 : 0.32),
        width: go.width || 1,
      }, go)));
    }

    function axes(o) {
      var ao = o || {};
      var len = ao.size !== undefined ? ao.size : 2.2;
      var nodes = [
        line([[0, 0, 0], [len, 0, 0]], { color: '#f87171', width: 1.8, opacity: 0.85 }),
        line([[0, 0, 0], [0, len, 0]], { color: '#34d399', width: 1.8, opacity: 0.85 }),
        line([[0, 0, 0], [0, 0, len]], { color: '#38bdf8', width: 1.8, opacity: 0.85 }),
      ];
      return nodes;
    }

    function particles(pointsOrOpts, o) {
      var list = [];
      var cfg = o || {};
      if (Array.isArray(pointsOrOpts)) {
        list = pointsOrOpts;
      } else if (pointsOrOpts && typeof pointsOrOpts === 'object') {
        cfg = pointsOrOpts;
        if (Array.isArray(pointsOrOpts.points)) {
          list = pointsOrOpts.points;
        } else {
          var count = Math.max(1, Math.round(pointsOrOpts.count || 60));
          var rad = pointsOrOpts.radius !== undefined ? pointsOrOpts.radius : 2.6;
          for (var i = 0; i < count; i++) {
            var phi = Math.acos(1 - 2 * ((i + 0.5) / count));
            var theta = PI * (1 + Math.sqrt(5)) * i;
            var rr = rad * (0.7 + 0.3 * Math.sin(i * 1.618));
            list.push([
              rr * Math.sin(phi) * Math.cos(theta),
              rr * Math.cos(phi) * 0.72,
              rr * Math.sin(phi) * Math.sin(theta),
            ]);
          }
        }
      }
      return add(makeNode('particles', { points: list }, Object.assign({
        color: cfg.color || theme.accent,
        size: cfg.size !== undefined ? cfg.size : 3.2,
        glow: cfg.glow !== false,
        opacity: cfg.opacity !== undefined ? cfg.opacity : 0.92,
      }, cfg)));
    }

    function label(pos, text, o) {
      return add(makeNode('label', { pos: pos || [0, 0, 0] }, Object.assign({
        text: String(text || ''),
        color: (o && o.color) || theme.foreground,
      }, o || {})));
    }

    function setCamera(next) {
      if (!next) return;
      if (next.azimuth !== undefined) cam.azimuth = next.azimuth;
      if (next.elevation !== undefined) cam.elevation = clamp(next.elevation, -1.48, 1.48);
      if (next.distance !== undefined) cam.distance = clamp(next.distance, 1.1, 45);
      if (Array.isArray(next.target)) cam.target = next.target.slice(0, 3);
      if (next.fov !== undefined) cam.fov = clamp(next.fov, 15, 110);
      render();
    }

    function getCamera() {
      return {
        azimuth: cam.azimuth,
        elevation: cam.elevation,
        distance: cam.distance,
        target: cam.target.slice(),
        fov: cam.fov,
      };
    }

    function resetCamera() {
      setCamera(initialCam);
    }

    function paintBackground() {
      ctx.clearRect(0, 0, width, height);
      var bg = opts.background;
      if (!bg || bg === 'transparent') {
        if (opts.glowBg) {
          var grd0 = ctx.createRadialGradient(width * 0.5, height * 0.46, 8, width * 0.5, height * 0.5, Math.max(width, height) * 0.6);
          grd0.addColorStop(0, theme.light ? 'rgba(126,34,206,0.08)' : 'rgba(168,85,247,0.14)');
          grd0.addColorStop(1, 'rgba(0,0,0,0)');
          ctx.fillStyle = grd0;
          ctx.fillRect(0, 0, width, height);
        }
        return;
      }
      if (bg === 'cosmic') {
        var grd = ctx.createRadialGradient(width * 0.5, height * 0.44, 10, width * 0.5, height * 0.5, Math.max(width, height) * 0.68);
        if (theme.light) {
          grd.addColorStop(0, '#f4f0ff');
          grd.addColorStop(0.55, '#f8f7fc');
          grd.addColorStop(1, '#f0eff5');
        } else {
          grd.addColorStop(0, '#1e1336');
          grd.addColorStop(0.52, '#110e1c');
          grd.addColorStop(1, '#09090e');
        }
        ctx.fillStyle = grd;
        ctx.fillRect(0, 0, width, height);
        return;
      }
      ctx.fillStyle = bg === 'card' ? theme.card : bg;
      ctx.fillRect(0, 0, width, height);
    }

    function render() {
      if (disposed) return;
      paintBackground();

      var cosE = Math.cos(cam.elevation);
      var sinE = Math.sin(cam.elevation);
      var cosA = Math.cos(cam.azimuth);
      var sinA = Math.sin(cam.azimuth);
      var camPos = [
        cam.target[0] + cam.distance * cosE * sinA,
        cam.target[1] + cam.distance * sinE,
        cam.target[2] + cam.distance * cosE * cosA,
      ];
      var forward = vecNorm(vecSub(cam.target, camPos));
      var worldUp = Math.abs(forward[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0];
      var right = vecNorm(vecCross(forward, worldUp));
      var up = vecCross(right, forward);
      var viewSpan = Math.min(height, width * 0.92);
      var focal = (viewSpan * 0.5) / Math.tan((cam.fov * PI) / 360);
      var cx = width * 0.5;
      var cy = height * 0.5;

      function project(worldPt) {
        var rel = vecSub(worldPt, camPos);
        var vz = vecDot(rel, forward);
        if (vz <= 0.12) return null;
        var vx = vecDot(rel, right);
        var vy = vecDot(rel, up);
        var inv = focal / vz;
        return [cx + vx * inv, cy - vy * inv, vz, inv];
      }

      // Camera-anchored key + rim lights so shapes remain vividly sculpted at every orbit angle.
      var keyLight = vecNorm(vecAdd(vecAdd(vecScale(right, 0.58), vecScale(up, 0.72)), vecScale(forward, -0.52)));
      var rimLight = vecNorm(vecAdd(vecAdd(vecScale(right, -0.74), vecScale(up, 0.34)), vecScale(forward, 0.42)));
      var rimTint = parseColor(opts.rimColor || theme.accent, theme);
      var ambient = theme.light ? 0.36 : 0.25;

      var drawList = [];

      for (var nIdx = 0; nIdx < objects.length; nIdx++) {
        var node = objects[nIdx];
        if (!node.visible) continue;

        if (node.kind === 'mesh') {
          var verts = node.data.vertices;
          var faces = node.data.faces;
          var faceT = node.data.faceT;
          var faceColors = node.data.faceColors;
          var worldVerts = new Array(verts.length);
          var projVerts = new Array(verts.length);
          for (var vi = 0; vi < verts.length; vi++) {
            var wp = transformPoint(verts[vi], node.position, node.rotation, node.scale);
            worldVerts[vi] = wp;
            projVerts[vi] = project(wp);
          }
          var baseA = parseColor(node.color || theme.primary, theme);
          var baseB = node.colorB ? parseColor(node.colorB, theme) : null;
          var wfColor = node.wireframeColor ? parseColor(node.wireframeColor, theme) : null;

          for (var fi = 0; fi < faces.length; fi++) {
            var tri = faces[fi];
            var p0 = projVerts[tri[0]], p1 = projVerts[tri[1]], p2 = projVerts[tri[2]];
            if (!p0 || !p1 || !p2) continue;
            var w0 = worldVerts[tri[0]], w1 = worldVerts[tri[1]], w2 = worldVerts[tri[2]];
            var edge1 = vecSub(w1, w0);
            var edge2 = vecSub(w2, w0);
            var normal = vecNorm(vecCross(edge1, edge2));
            var center = [(w0[0] + w1[0] + w2[0]) / 3, (w0[1] + w1[1] + w2[1]) / 3, (w0[2] + w1[2] + w2[2]) / 3];
            var viewDir = vecNorm(vecSub(camPos, center));
            var facing = vecDot(normal, viewDir);
            if (facing <= 0) {
              if (!node.doubleSided && !node.wireframe && node.opacity >= 0.99) continue;
              normal = vecScale(normal, -1);
            }

            var col = baseA;
            if (faceColors && faceColors[fi]) {
              col = parseColor(faceColors[fi], theme);
            } else if (baseB && faceT) {
              col = mixRgb(baseA, baseB, faceT[fi] || 0);
            }

            var diff = Math.max(0, vecDot(normal, keyLight));
            var rim = Math.max(0, vecDot(normal, rimLight));
            var halfVec = vecNorm(vecAdd(keyLight, viewDir));
            var spec = Math.pow(Math.max(0, vecDot(normal, halfVec)), node.shininess) * node.specular;
            var lightMul = ambient + node.emissive + diff * 0.78;
            var rCol = col[0] * lightMul + rimTint[0] * rim * 0.24 + 255 * spec;
            var gCol = col[1] * lightMul + rimTint[1] * rim * 0.24 + 255 * spec;
            var bCol = col[2] * lightMul + rimTint[2] * rim * 0.24 + 255 * spec;
            var alpha = col[3] * node.opacity;
            var depth = (p0[2] + p1[2] + p2[2]) / 3;

            drawList.push({
              kind: 'tri',
              z: depth,
              p0: p0,
              p1: p1,
              p2: p2,
              fill: node.wireframe ? null : rgbaCss(rCol, gCol, bCol, alpha),
              stroke: node.wireframe
                ? rgbaCss(wfColor ? wfColor[0] : rCol, wfColor ? wfColor[1] : gCol, wfColor ? wfColor[2] : bCol, alpha)
                : wfColor
                  ? rgbaCss(wfColor[0], wfColor[1], wfColor[2], wfColor[3] * node.opacity)
                  : rgbaCss(rCol, gCol, bCol, alpha),
              lineWidth: node.wireframe ? Math.max(0.8, node.strokeWidth) : (wfColor ? node.strokeWidth : 0.55),
            });
          }
        } else if (node.kind === 'segments') {
          var segs = node.data.segments;
          var sCol = parseColor(node.color || theme.border, theme);
          var sCss = rgbaCss(sCol[0], sCol[1], sCol[2], sCol[3] * node.opacity);
          for (var si = 0; si < segs.length; si++) {
            var aW = transformPoint(segs[si][0], node.position, node.rotation, node.scale);
            var bW = transformPoint(segs[si][1], node.position, node.rotation, node.scale);
            var aP = project(aW);
            var bP = project(bW);
            if (!aP || !bP) continue;
            drawList.push({
              kind: 'line',
              z: (aP[2] + bP[2]) * 0.5,
              a: aP,
              b: bP,
              stroke: sCss,
              width: node.width || 1,
              dashed: node.dashed,
            });
          }
        } else if (node.kind === 'line') {
          var pts = node.data.points;
          var lCol = parseColor(node.color || theme.primary, theme);
          var lCss = rgbaCss(lCol[0], lCol[1], lCol[2], lCol[3] * node.opacity);
          var total = node.closed ? pts.length : pts.length - 1;
          for (var li = 0; li < total; li++) {
            var pa = transformPoint(pts[li], node.position, node.rotation, node.scale);
            var pb = transformPoint(pts[(li + 1) % pts.length], node.position, node.rotation, node.scale);
            var ppA = project(pa);
            var ppB = project(pb);
            if (!ppA || !ppB) continue;
            drawList.push({
              kind: 'line',
              z: (ppA[2] + ppB[2]) * 0.5,
              a: ppA,
              b: ppB,
              stroke: lCss,
              width: node.width || 1.6,
              dashed: node.dashed,
            });
          }
        } else if (node.kind === 'particles') {
          var pList = node.data.points;
          var defaultPCol = parseColor(node.color || theme.accent, theme);
          var baseSize = node.size !== undefined ? node.size : 3.2;
          for (var pi = 0; pi < pList.length; pi++) {
            var entry = pList[pi];
            var rawPt = Array.isArray(entry) ? entry : entry.pos;
            if (!rawPt) continue;
            var pw = transformPoint(rawPt, node.position, node.rotation, node.scale);
            var pp = project(pw);
            if (!pp) continue;
            var pCol = (!Array.isArray(entry) && entry.color) ? parseColor(entry.color, theme) : defaultPCol;
            var pSz = (!Array.isArray(entry) && entry.size !== undefined) ? entry.size : baseSize;
            var rPx = clamp((pSz * pp[3]) / 120, 1.1, 14);
            drawList.push({
              kind: 'particle',
              z: pp[2],
              x: pp[0],
              y: pp[1],
              r: rPx,
              col: pCol,
              alpha: pCol[3] * node.opacity,
              glow: node.glow !== false,
            });
          }
        } else if (node.kind === 'label') {
          var lw = transformPoint(node.data.pos, node.position, node.rotation, node.scale);
          var lp = project(lw);
          if (lp) {
            drawList.push({
              kind: 'label',
              z: lp[2] - 0.05,
              x: lp[0],
              y: lp[1],
              text: node.text,
              color: node.color || theme.foreground,
              bg: node.bg,
              fontSize: node.fontSize,
              dot: node.dot,
              offset: node.offset,
            });
          }
        }
      }

      // Painter's algorithm: draw farthest items first.
      drawList.sort(function (a, b) { return b.z - a.z; });

      for (var di = 0; di < drawList.length; di++) {
        var item = drawList[di];
        if (item.kind === 'tri') {
          ctx.beginPath();
          ctx.moveTo(item.p0[0], item.p0[1]);
          ctx.lineTo(item.p1[0], item.p1[1]);
          ctx.lineTo(item.p2[0], item.p2[1]);
          ctx.closePath();
          if (item.fill) {
            ctx.fillStyle = item.fill;
            ctx.fill();
          }
          if (item.stroke && item.lineWidth > 0) {
            ctx.strokeStyle = item.stroke;
            ctx.lineWidth = item.lineWidth;
            ctx.lineJoin = 'round';
            ctx.stroke();
          }
        } else if (item.kind === 'line') {
          ctx.save();
          ctx.beginPath();
          if (item.dashed && ctx.setLineDash) ctx.setLineDash([4, 4]);
          ctx.moveTo(item.a[0], item.a[1]);
          ctx.lineTo(item.b[0], item.b[1]);
          ctx.strokeStyle = item.stroke;
          ctx.lineWidth = item.width;
          ctx.lineCap = 'round';
          ctx.stroke();
          ctx.restore();
        } else if (item.kind === 'particle') {
          var c = item.col;
          if (item.glow && typeof ctx.createRadialGradient === 'function') {
            var halo = ctx.createRadialGradient(item.x, item.y, 0, item.x, item.y, item.r * 3.1);
            halo.addColorStop(0, rgbaCss(c[0], c[1], c[2], item.alpha * 0.72));
            halo.addColorStop(0.42, rgbaCss(c[0], c[1], c[2], item.alpha * 0.22));
            halo.addColorStop(1, rgbaCss(c[0], c[1], c[2], 0));
            ctx.fillStyle = halo;
            ctx.beginPath();
            ctx.arc(item.x, item.y, item.r * 3.1, 0, TAU);
            ctx.fill();
          }
          ctx.fillStyle = rgbaCss(
            Math.min(255, c[0] + 55),
            Math.min(255, c[1] + 55),
            Math.min(255, c[2] + 55),
            item.alpha
          );
          ctx.beginPath();
          ctx.arc(item.x, item.y, item.r, 0, TAU);
          ctx.fill();
        } else if (item.kind === 'label') {
          var lx = item.x + item.offset[0];
          var ly = item.y + item.offset[1];
          if (item.dot) {
            ctx.fillStyle = item.color;
            ctx.beginPath();
            ctx.arc(item.x, item.y, 2.5, 0, TAU);
            ctx.fill();
          }
          ctx.font = '500 ' + item.fontSize + 'px -apple-system, BlinkMacSystemFont, system-ui, sans-serif';
          var mText = ctx.measureText ? ctx.measureText(item.text) : { width: item.text.length * 6.5 };
          var tw = mText.width;
          var padX = 7;
          var boxH = item.fontSize + 8;
          var bx = lx - tw * 0.5 - padX;
          var by = ly - boxH * 0.5;
          ctx.fillStyle = item.bg || (theme.light ? 'rgba(255,255,255,0.88)' : 'rgba(15,15,22,0.82)');
          ctx.strokeStyle = theme.border;
          ctx.lineWidth = 1;
          ctx.beginPath();
          if (typeof ctx.roundRect === 'function') {
            ctx.roundRect(bx, by, tw + padX * 2, boxH, 6);
          } else {
            ctx.rect(bx, by, tw + padX * 2, boxH);
          }
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = item.color;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(item.text, lx, ly + 0.5);
        }
      }
    }

    var hostPaused = false;
    var animActive = false;

    function isLoopPaused() {
      return disposed || hostPaused || (typeof document !== 'undefined' && document.hidden === true);
    }

    function stopRaf() {
      if (rafId && typeof global.cancelAnimationFrame === 'function') {
        global.cancelAnimationFrame(rafId);
      }
      rafId = 0;
    }

    function scheduleNextFrame() {
      if (!animActive || isLoopPaused() || rafId) return;
      if (typeof global.requestAnimationFrame === 'function') {
        rafId = global.requestAnimationFrame(stepFrame);
      }
    }

    function stepFrame(now) {
      rafId = 0;
      if (isLoopPaused()) {
        lastFrame = 0;
        return;
      }
      if (!animStart) animStart = now;
      var dt = lastFrame ? Math.min((now - lastFrame) / 1000, 0.08) : 0.016;
      lastFrame = now;
      var t = (now - animStart) / 1000;
      if (!dragging && autoRotateSpeed) {
        cam.azimuth += autoRotateSpeed * dt;
      }
      if (typeof animCb === 'function') {
        animCb(t, dt, api);
      }
      render();
      scheduleNextFrame();
    }

    function animate(callback) {
      animCb = typeof callback === 'function' ? callback : null;
      animActive = true;
      stopRaf();
      animStart = 0;
      lastFrame = 0;
      // Render initial frame synchronously so headless/static captures never see an empty canvas.
      if (animCb) animCb(0, 0.016, api);
      render();
      scheduleNextFrame();
      return api;
    }

    function onVisibilityOrPause() {
      if (isLoopPaused()) {
        stopRaf();
        lastFrame = 0;
      } else {
        scheduleNextFrame();
      }
    }

    function onHostMessage(event) {
      var data = event && event.data;
      if (!data || typeof data !== 'object' || data.type !== 'xerxes-widget-pause') return;
      hostPaused = data.paused === true;
      onVisibilityOrPause();
    }

    // ── Pointer & Wheel Orbit Controls ─────────────────────────────────────

    function onPointerDown(e) {
      if (!interactive) return;
      dragging = true;
      lastPtrX = e.clientX;
      lastPtrY = e.clientY;
      if (canvas.style) canvas.style.cursor = 'grabbing';
      if (typeof canvas.setPointerCapture === 'function' && e.pointerId !== undefined) {
        try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
      }
    }

    function onPointerMove(e) {
      if (!dragging) return;
      var dx = e.clientX - lastPtrX;
      var dy = e.clientY - lastPtrY;
      lastPtrX = e.clientX;
      lastPtrY = e.clientY;
      cam.azimuth -= dx * 0.0085;
      cam.elevation = clamp(cam.elevation + dy * 0.0075, -1.48, 1.48);
      if (!rafId) render();
    }

    function onPointerUp(e) {
      if (!dragging) return;
      dragging = false;
      if (canvas.style) canvas.style.cursor = 'grab';
      if (typeof canvas.releasePointerCapture === 'function' && e.pointerId !== undefined) {
        try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
      }
    }

    function onWheel(e) {
      if (!interactive) return;
      if (typeof e.preventDefault === 'function') e.preventDefault();
      var factor = 1 + clamp(e.deltaY, -120, 120) * 0.0014;
      cam.distance = clamp(cam.distance * factor, 1.2, 40);
      if (!rafId) render();
    }

    function onDblClick() {
      if (!interactive) return;
      resetCamera();
    }

    function onThemeChange() {
      theme = readThemeTokens();
      render();
    }

    if (typeof canvas.addEventListener === 'function') {
      canvas.addEventListener('pointerdown', onPointerDown);
      canvas.addEventListener('pointermove', onPointerMove);
      canvas.addEventListener('pointerup', onPointerUp);
      canvas.addEventListener('pointercancel', onPointerUp);
      canvas.addEventListener('wheel', onWheel, { passive: false });
      canvas.addEventListener('dblclick', onDblClick);
    }
    if (typeof global.addEventListener === 'function') {
      global.addEventListener('themechange', onThemeChange);
      global.addEventListener('message', onHostMessage);
    }
    if (typeof document !== 'undefined' && typeof document.addEventListener === 'function') {
      document.addEventListener('visibilitychange', onVisibilityOrPause);
    }

    var resizeObserver = null;
    if (typeof global.ResizeObserver === 'function') {
      resizeObserver = new global.ResizeObserver(function () {
        resize();
        render();
      });
      try { resizeObserver.observe(canvas); } catch (_) {}
    }

    function dispose() {
      if (disposed) return;
      disposed = true;
      animActive = false;
      stopRaf();
      if (typeof global.removeEventListener === 'function') {
        global.removeEventListener('themechange', onThemeChange);
        global.removeEventListener('message', onHostMessage);
      }
      if (typeof document !== 'undefined' && typeof document.removeEventListener === 'function') {
        document.removeEventListener('visibilitychange', onVisibilityOrPause);
      }
      if (resizeObserver) {
        try { resizeObserver.disconnect(); } catch (_) {}
        resizeObserver = null;
      }
      if (typeof canvas.removeEventListener === 'function') {
        canvas.removeEventListener('pointerdown', onPointerDown);
        canvas.removeEventListener('pointermove', onPointerMove);
        canvas.removeEventListener('pointerup', onPointerUp);
        canvas.removeEventListener('pointercancel', onPointerUp);
        canvas.removeEventListener('wheel', onWheel);
        canvas.removeEventListener('dblclick', onDblClick);
      }
      if (typeof global.removeEventListener === 'function') {
        global.removeEventListener('themechange', onThemeChange);
      }
    }

    resize();

    var api = {
      scene: { objects: objects, camera: cam },
      add: add,
      remove: remove,
      clear: clear,
      box: box,
      sphere: sphere,
      torus: torus,
      torusKnot: torusKnot,
      icosahedron: icosahedron,
      cylinder: cylinder,
      cone: cone,
      plane: plane,
      grid: grid,
      axes: axes,
      particles: particles,
      line: line,
      mesh: mesh,
      label: label,
      setCamera: setCamera,
      getCamera: getCamera,
      resetCamera: resetCamera,
      animate: animate,
      render: render,
      resize: resize,
      dispose: dispose,
    };
    return api;
  }

  global.Xerxes3D = {
    __version: '1.0',
    createScene: createScene,
  };
})(typeof window !== 'undefined' ? window : globalThis);`
