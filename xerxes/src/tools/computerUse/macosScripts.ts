// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * JXA (JavaScript for Automation) and AppleScript sources used by the macOS
 * computer-use port. Every script receives untrusted values exclusively
 * through `argv`; nothing is interpolated into the source, so a hostile app
 * name or type-text payload cannot escape into the automation runtime.
 */

/**
 * Click, double/triple-click, right-click, middle-click.
 *
 * argv: [x, y, button, count] — point in logical points, button name
 * ('left' | 'right' | 'middle'), click count.
 *
 * Requires the Accessibility permission: macOS refuses CGEventPost to the
 * HID event tap from unapproved processes. The clickState field is set per
 * iteration because AppKit distinguishes single/double/triple clicks by
 * clickState, not by timing alone — a synthetic double-click without
 * clickState 2 registers as two unrelated singles.
 */
export const JXA_CLICK = `
function run(argv) {
  ObjC.import('CoreGraphics');
  var p = $.CGPointMake(Number(argv[0]), Number(argv[1]));
  var buttons = {
    left:   [$.kCGEventLeftMouseDown,  $.kCGEventLeftMouseUp,  $.kCGMouseButtonLeft],
    right:  [$.kCGEventRightMouseDown, $.kCGEventRightMouseUp, $.kCGMouseButtonRight],
    middle: [$.kCGEventOtherMouseDown, $.kCGEventOtherMouseUp, $.kCGMouseButtonCenter]
  };
  var b = buttons[argv[2]];
  if (!b) { throw new Error('unknown button: ' + argv[2]); }
  var n = Math.max(1, Number(argv[3]));
  for (var i = 1; i <= n; i++) {
    var down = $.CGEventCreateMouseEvent(null, b[0], p, b[2]);
    $.CGEventSetIntegerValueField(down, $.kCGMouseEventClickState, i);
    $.CGEventPost($.kCGHIDEventTap, down);
    var up = $.CGEventCreateMouseEvent(null, b[1], p, b[2]);
    $.CGEventSetIntegerValueField(up, $.kCGMouseEventClickState, i);
    $.CGEventPost($.kCGHIDEventTap, up);
    if (i < n) { delay(0.08); }
  }
}
`

/**
 * Move the cursor without pressing a button. argv: [x, y] in logical points.
 *
 * Requires Accessibility (it posts a CG event). Also used to position the
 * pointer before a coordinate-targeted scroll, because scroll wheel events
 * land on whatever is under the cursor at post time.
 */
export const JXA_MOUSE_MOVE = `
function run(argv) {
  ObjC.import('CoreGraphics');
  var p = $.CGPointMake(Number(argv[0]), Number(argv[1]));
  var event = $.CGEventCreateMouseEvent(null, $.kCGEventMouseMoved, p, $.kCGMouseButtonLeft);
  $.CGEventPost($.kCGHIDEventTap, event);
}
`

/**
 * Drag from start to end in N interpolated steps.
 * argv: [x1, y1, x2, y2, steps] — logical points plus interpolation count.
 *
 * Requires Accessibility. The motion is emitted as discrete dragged events
 * with small delays because drop targets (file managers, canvases, web
 * drag-and-drop) track hover during the drag; an instant teleport from
 * press to release is rejected by most of them.
 */
export const JXA_DRAG = `
function run(argv) {
  ObjC.import('CoreGraphics');
  var x1 = Number(argv[0]), y1 = Number(argv[1]);
  var x2 = Number(argv[2]), y2 = Number(argv[3]);
  var steps = Math.max(1, Number(argv[4]));
  var down = $.CGEventCreateMouseEvent(null, $.kCGEventLeftMouseDown, $.CGPointMake(x1, y1), $.kCGMouseButtonLeft);
  $.CGEventPost($.kCGHIDEventTap, down);
  delay(0.05);
  for (var i = 1; i <= steps; i++) {
    var p = $.CGPointMake(x1 + (x2 - x1) * i / steps, y1 + (y2 - y1) * i / steps);
    var moved = $.CGEventCreateMouseEvent(null, $.kCGEventLeftMouseDragged, p, $.kCGMouseButtonLeft);
    $.CGEventPost($.kCGHIDEventTap, moved);
    delay(0.01);
  }
  var up = $.CGEventCreateMouseEvent(null, $.kCGEventLeftMouseUp, $.CGPointMake(x2, y2), $.kCGMouseButtonLeft);
  $.CGEventPost($.kCGHIDEventTap, up);
}
`

/**
 * Scroll wheel event in line units. argv: [wheelY, wheelX] (signed).
 *
 * Requires Accessibility. Line units (kCGScrollEventUnitLine) are used
 * instead of pixel units so scrolling matches native trackpad behavior in
 * AppKit lists and browsers; pixel-unit scrolls move far too little per
 * event to be useful to a model.
 */
export const JXA_SCROLL = `
function run(argv) {
  ObjC.import('CoreGraphics');
  var event = $.CGEventCreateScrollWheelEvent(null, $.kCGScrollEventUnitLine, 2, Number(argv[0]), Number(argv[1]));
  $.CGEventPost($.kCGHIDEventTap, event);
}
`

/**
 * Type Unicode text as one keyboard event payload. argv: [text].
 *
 * Requires Accessibility. The whole string is attached to a single key
 * event via CGEventKeyboardSetUnicodeString instead of synthesizing
 * per-character keystrokes, because emoji, CJK, and accented characters
 * have no virtual key codes and per-key simulation would mangle them.
 */
export const JXA_TYPE = `
function run(argv) {
  ObjC.import('CoreGraphics');
  var down = $.CGEventCreateKeyboardEvent(null, 0, true);
  $.CGEventKeyboardSetUnicodeString(down, argv[0]);
  $.CGEventPost($.kCGHIDEventTap, down);
  var up = $.CGEventCreateKeyboardEvent(null, 0, false);
  $.CGEventPost($.kCGHIDEventTap, up);
}
`

/**
 * Key chord by virtual key code and modifier flags.
 * argv: [keycode, cmd, shift, alt, ctrl] — flags as literal '1' or '0'.
 *
 * Requires Accessibility. Flags are string tokens rather than booleans
 * because JXA argv is string-only; the strict '1' comparison keeps any
 * unexpected value safely "off" instead of truthy-coercing garbage into a
 * stuck modifier.
 */
export const JXA_KEY_CHORD = `
function run(argv) {
  ObjC.import('CoreGraphics');
  var flags = 0;
  if (argv[1] === '1') { flags |= $.kCGEventFlagMaskCommand; }
  if (argv[2] === '1') { flags |= $.kCGEventFlagMaskShift; }
  if (argv[3] === '1') { flags |= $.kCGEventFlagMaskAlternate; }
  if (argv[4] === '1') { flags |= $.kCGEventFlagMaskControl; }
  var code = Number(argv[0]);
  var down = $.CGEventCreateKeyboardEvent(null, code, true);
  $.CGEventSetFlags(down, flags);
  $.CGEventPost($.kCGHIDEventTap, down);
  var up = $.CGEventCreateKeyboardEvent(null, code, false);
  $.CGEventSetFlags(up, flags);
  $.CGEventPost($.kCGHIDEventTap, up);
}
`

/**
 * Current cursor location in logical points. Prints "x,y".
 *
 * No permissions required: reading event state via CGEventGetLocation is
 * not gated by Accessibility the way posting events is, so this works even
 * before the user grants anything — useful for diagnosing why input is
 * failing.
 */
export const JXA_CURSOR_POSITION = `
function run(argv) {
  ObjC.import('CoreGraphics');
  var p = $.CGEventGetLocation($.CGEventCreate(null));
  return '' + p.x + ',' + p.y;
}
`

/**
 * Logical screen size and backing scale factor of the main screen.
 * Prints "width,height,scale".
 *
 * No permissions required (NSScreen reads are ungated). The backing scale
 * is what lets the port translate raw Retina screenshot pixels into the
 * logical points CoreGraphics input dispatch expects; without it every
 * coordinate would be off by 2x on Retina hardware.
 */
export const JXA_SCREEN_INFO = `
function run(argv) {
  ObjC.import('AppKit');
  var screen = $.NSScreen.mainScreen;
  return '' + screen.frame.size.width + ',' + screen.frame.size.height + ',' + screen.backingScaleFactor;
}
`

/**
 * Names of regular (foreground-capable) running apps. Prints a CSV list.
 *
 * No permissions required. Filters on activationPolicy 0 (regular apps)
 * because agents and background daemons cannot be focused and would only
 * pad the list with targets focus_app can never activate.
 */
export const JXA_LIST_APPS = `
function run(argv) {
  ObjC.import('AppKit');
  var apps = $.NSWorkspace.sharedWorkspace.runningApplications;
  var names = [];
  for (var i = 0; i < apps.count; i++) {
    var app = apps.objectAtIndex(i);
    if (Number(app.activationPolicy) === 0 && app.localizedName) {
      names.push(ObjC.unwrap(app.localizedName));
    }
  }
  return names.join(', ');
}
`

/**
 * Bring an app to the front by localized name or bundle identifier.
 * argv: [nameOrBundleId], matched case-insensitively. Prints "ok" or
 * "not found".
 *
 * No permissions required (NSWorkspace activation is not Accessibility
 * gated). Bundle identifiers are accepted in addition to display names so
 * a model can target apps whose localized name it cannot guess.
 * activateWithOptions(2) is NSApplicationActivateIgnoringOtherApps, needed
 * because the default policy may not raise the target above the current
 * frontmost or a fullscreen app.
 */
export const JXA_FOCUS_APP = `
function run(argv) {
  ObjC.import('AppKit');
  var target = ('' + argv[0]).toLowerCase();
  var apps = $.NSWorkspace.sharedWorkspace.runningApplications;
  for (var i = 0; i < apps.count; i++) {
    var app = apps.objectAtIndex(i);
    var name = app.localizedName ? ('' + ObjC.unwrap(app.localizedName)).toLowerCase() : '';
    var bundle = app.bundleIdentifier ? ('' + ObjC.unwrap(app.bundleIdentifier)).toLowerCase() : '';
    if (name === target || bundle === target) {
      app.activateWithOptions(2);
      return 'ok';
    }
  }
  return 'not found';
}
`

/**
 * Modifier aliases accepted in key chords, mapped to JXA flag positions.
 * Models and users variously say "cmd", "command", "meta", or "super" for
 * the same physical key; accepting the aliases up front keeps a chord from
 * failing over vocabulary rather than intent.
 */
export const MODIFIER_ALIASES: Readonly<Record<string, 'alt' | 'cmd' | 'ctrl' | 'shift'>> = Object.freeze({
  alt: 'alt',
  cmd: 'cmd',
  command: 'cmd',
  control: 'ctrl',
  ctrl: 'ctrl',
  meta: 'cmd',
  opt: 'alt',
  option: 'alt',
  shift: 'shift',
  super: 'cmd',
})

/**
 * macOS virtual key codes for chord keys (letters, digits, and named keys).
 * These are layout-independent hardware codes (HIToolbox kVK_* constants),
 * not characters: CGEventCreateKeyboardEvent requires a key code, so chord
 * dispatch goes through this table rather than through the character a key
 * happens to produce on the user's layout.
 */
export const KEY_CODES: Readonly<Record<string, number>> = Object.freeze({
  '0': 29,
  '1': 18,
  '2': 19,
  '3': 20,
  '4': 21,
  '5': 23,
  '6': 22,
  '7': 26,
  '8': 28,
  '9': 25,
  a: 0,
  b: 11,
  backspace: 51,
  c: 8,
  d: 2,
  delete: 51,
  down: 125,
  e: 14,
  enter: 36,
  escape: 53,
  f: 3,
  f1: 122,
  f10: 109,
  f11: 103,
  f12: 111,
  f2: 120,
  f3: 99,
  f4: 118,
  f5: 96,
  f6: 97,
  f7: 98,
  f8: 100,
  f9: 101,
  forwarddelete: 117,
  g: 5,
  h: 4,
  home: 115,
  i: 34,
  j: 38,
  k: 40,
  l: 37,
  left: 123,
  m: 46,
  n: 45,
  o: 31,
  p: 35,
  pagedown: 121,
  pageup: 116,
  q: 12,
  r: 15,
  return: 36,
  right: 124,
  s: 1,
  space: 49,
  t: 17,
  tab: 48,
  u: 32,
  up: 126,
  v: 9,
  w: 13,
  x: 7,
  y: 16,
  z: 6,
})
