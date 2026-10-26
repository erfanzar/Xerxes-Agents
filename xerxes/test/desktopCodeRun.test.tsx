// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { CodeRunContext, commandOf, isRunnableBlock } from '../src/desktop/renderer/codeRun.js'
import { Markdown } from '../src/desktop/renderer/markdown.js'

const reply = '```bash\nbun run --cwd xerxes signing:local\n```\n\n```ts\nconst x = 1\n```'

test('shell code blocks in an agent reply offer Run beside Copy; other languages only Copy', () => {
  const html = renderToStaticMarkup(createElement(CodeRunContext.Provider, { value: async () => {} }, createElement(Markdown, { text: reply })))
  expect(html.match(/aria-label="Run in terminal"/g)).toHaveLength(1)
  expect(html.match(/aria-label="Copy code"/g)).toHaveLength(2)
})

test('without a runner (release notes, plans, VS Code) code blocks keep only Copy', () => {
  const html = renderToStaticMarkup(createElement(Markdown, { text: reply }))
  expect(html).not.toContain('Run in terminal')
  expect(html).toContain('Copy code')
})

test('runnable blocks and the command a transcript block sends', () => {
  expect(isRunnableBlock('', 'ls')).toBe(true)
  expect(isRunnableBlock('Bash', 'ls')).toBe(true)
  expect(isRunnableBlock('python', 'print(1)')).toBe(false)
  expect(isRunnableBlock('sh', '   ')).toBe(false)
  expect(commandOf('$ cd xerxes\n$ bun test\n')).toBe('cd xerxes\nbun test')
  // A plain script keeps every line, including ones that happen to start with "$ ".
  expect(commandOf('echo hi\n$ not a prompt')).toBe('echo hi\n$ not a prompt')
})
