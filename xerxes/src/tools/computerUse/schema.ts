// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ToolDefinition } from '../../types/toolCalls.js'

/**
 * Compact model-agnostic schema for privileged desktop control.
 *
 * The tool is not registered by default. A host must explicitly inject a
 * ComputerUseSession backed by a real, approved desktop automation service.
 */
export const COMPUTER_USE_SCHEMA = {
  name: 'computer_use',
  description: [
    'Drive a configured privileged desktop backend: capture accessibility-aware screenshots, click, type, scroll, drag, and focus apps.',
    'Prefer action="capture" with mode="som", then address an element by its 1-based element index rather than pixel coordinates.',
    'This tool is unavailable unless the host explicitly configures a real ComputerUsePort; it never substitutes browser automation or cursor simulation.',
  ].join(' '),
  parameters: {
    type: 'object',
    additionalProperties: false,
    properties: {
      action: {
        type: 'string',
        enum: [
          'capture',
          'click',
          'double_click',
          'right_click',
          'middle_click',
          'drag',
          'scroll',
          'type',
          'key',
          'set_value',
          'wait',
          'list_apps',
          'focus_app',
        ],
      },
      mode: {
        type: 'string',
        enum: ['som', 'vision', 'ax'],
        default: 'som',
        description: 'Capture mode: screenshot plus numbered elements, screenshot only, or accessibility tree only.',
      },
      app: {
        type: 'string',
        description: 'Target app name or bundle identifier for capture or focus_app.',
      },
      max_elements: {
        type: 'integer',
        minimum: 1,
        maximum: 1000,
        default: 100,
        description: 'Maximum accessibility elements included in a capture response.',
      },
      element: {
        type: 'integer',
        minimum: 1,
        description: '1-based element index from the most recent capture.',
      },
      x: { type: 'integer', minimum: 0, description: 'Logical pixel X coordinate.' },
      y: { type: 'integer', minimum: 0, description: 'Logical pixel Y coordinate.' },
      start_element: { type: 'integer', minimum: 1, description: '1-based drag start element index.' },
      start_x: { type: 'integer', minimum: 0, description: 'Logical pixel drag start X coordinate.' },
      start_y: { type: 'integer', minimum: 0, description: 'Logical pixel drag start Y coordinate.' },
      end_element: { type: 'integer', minimum: 1, description: '1-based drag end element index.' },
      end_x: { type: 'integer', minimum: 0, description: 'Logical pixel drag end X coordinate.' },
      end_y: { type: 'integer', minimum: 0, description: 'Logical pixel drag end Y coordinate.' },
      dx: { type: 'integer', default: 0, description: 'Horizontal scroll delta; positive moves right.' },
      dy: { type: 'integer', default: 0, description: 'Vertical scroll delta; positive moves down.' },
      text: { type: 'string', description: 'Text to type for action="type".' },
      key: {
        type: 'string',
        description: 'Keyboard key or chord for action="key", for example Enter, command+a, or shift+tab.',
      },
      value: { type: 'string', description: 'Value to set for action="set_value".' },
      ms: {
        type: 'integer',
        minimum: 0,
        maximum: 300000,
        default: 1000,
        description: 'Wait duration in milliseconds, up to five minutes.',
      },
      capture_after: {
        type: 'boolean',
        default: false,
        description: 'Ask the configured backend for a post-action capture when it supports one.',
      },
    },
    required: ['action'],
  },
} as const

export const COMPUTER_USE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: COMPUTER_USE_SCHEMA,
}
