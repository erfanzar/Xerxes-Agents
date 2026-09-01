// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { WorkspacePathError } from '../src/tools/pathSafety.js'
import { generateImageTool, GENERATE_IMAGE_DEFINITION } from '../src/tools/imageGen.js'
import { rm } from 'node:fs/promises'
import { join } from 'node:path'

const PNG_BYTES = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00]
const PNG_BASE64 = Buffer.from(PNG_BYTES).toString('base64')

function imageResponse(text: string, imageCount = 1): Response {
  return new Response(JSON.stringify({
    id: 'resp-tool-1',
    choices: [{
      message: {
        role: 'assistant',
        content: text,
        images: Array.from({ length: imageCount }, () => ({
          image_url: { url: `data:image/png;base64,${PNG_BASE64}` },
        })),
      },
    }],
    usage: { prompt_tokens: 5, completion_tokens: 100 },
  }))
}

function toolOptions(workspaceRoot: string, fetchImplementation: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>) {
  return {
    resolveApiKey: () => 'sk-test',
    workspaceRoot,
    fetch: fetchImplementation,
  }
}

/** Wrap a simple handler in an injectable fetch signature. */
function stubFetch(handler: (url: string, init?: RequestInit) => Response) {
  return async (input: RequestInfo | URL, init?: RequestInit) => handler(String(input), init)
}

test('generate_image saves images inside the workspace and reports usage', async () => {
  const root = await mkdtemp()
  try {
    const calls: { url?: string; init?: RequestInit }[] = []
    const result = await generateImageTool(
      { prompt: 'a cat in space', model: 'openai/gpt-image-2' },
      toolOptions(root, stubFetch((url, init) => {
        calls.push({ ...(init === undefined ? {} : { init }), url })
        return imageResponse('here is your cat')
      })),
    )
    expect(calls[0]?.url).toBe('https://openrouter.ai/api/v1/chat/completions')
    expect(result.model).toBe('openrouter/openai/gpt-image-2')
    expect(result.text).toBe('here is your cat')
    expect(result.stop_reason).toBe('stop')
    expect(result.response_id).toBe('resp-tool-1')
    const usage = result.usage as Record<string, number>
    expect(usage.input_tokens).toBe(5)
    expect(usage.total_tokens).toBe(105)

    const saved = (result.saved as { bytes: number; media_type: string; path: string }[])
    expect(saved.length).toBe(1)
    expect(saved[0]?.media_type).toBe('image/png')
    expect(saved[0]?.bytes).toBe(PNG_BYTES.length)
    // The default destination is a relative images/ path slug from the route id.
    expect(saved[0]?.path.startsWith('images/openai-gpt-image-2-')).toBe(true)
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('generate_image honors explicit output paths and multiple images', async () => {
  const root = await mkdtemp()
  try {
    const result = await generateImageTool(
      { prompt: 'two variants', output_path: 'assets/cover.png' },
      toolOptions(root, stubFetch(() => imageResponse('', 2))),
    )
    const saved = result.saved as { bytes: number; media_type: string; path: string }[]
    // With an explicit path and multiple images, every image gets an ordered
    // suffix so nothing overwrites the caller's file or each other.
    expect(saved).toEqual([
      { bytes: PNG_BYTES.length, media_type: 'image/png', path: 'assets/cover-1.png' },
      { bytes: PNG_BYTES.length, media_type: 'image/png', path: 'assets/cover-2.png' },
    ])
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('generate_image rejects workspace escapes and missing credentials', async () => {
  const root = await mkdtemp()
  try {
    await expect(generateImageTool(
      { prompt: 'x', output_path: '../outside.png' },
      toolOptions(root, stubFetch(() => imageResponse(''))),
    )).rejects.toBeInstanceOf(WorkspacePathError)

    await expect(generateImageTool(
      { prompt: 'x' },
      { ...toolOptions(root, stubFetch(() => imageResponse(''))), resolveApiKey: () => '' },
    )).rejects.toThrow(/no image API key configured/)
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('generate_image surfaces provider failures as typed errors', async () => {
  const root = await mkdtemp()
  try {
    await expect(generateImageTool(
      { prompt: 'x' },
      toolOptions(root, stubFetch(() => new Response('{"error":{"message":"quota"}}', { status: 402 }))),
    )).rejects.toThrow(/402/)
    await expect(generateImageTool(
      { prompt: 'x' },
      toolOptions(root, stubFetch(() => imageResponse('text only, no images', 0))),
    )).rejects.toThrow(/produced no images/)
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('the tool definition is well-formed for the registry', () => {
  expect(GENERATE_IMAGE_DEFINITION.function.name).toBe('generate_image')
  const parameters = GENERATE_IMAGE_DEFINITION.function.parameters as { required?: string[] }
  expect(parameters.required).toEqual(['prompt'])
})

async function mkdtemp(): Promise<string> {
  const root = join('/tmp', `xerxes-image-gen-test-${Date.now()}-${Math.random().toString(36).slice(2)}`)
  await Bun.write(join(root, '.keep'), '')
  return root
}
