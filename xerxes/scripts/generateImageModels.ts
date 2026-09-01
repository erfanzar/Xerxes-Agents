// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Generate the Xerxes image-model catalog from pi-ai's generated data.
 *
 * Reads `dist/image-models.generated.js` from the installed
 * `@earendil-works/pi-ai` package (the file is not on the package export map,
 * so it is loaded by resolved path) and emits a deterministic, sorted JSON
 * catalog beside the image registry. `--check` fails when the committed file
 * is stale.
 */

import piPackage from '@earendil-works/pi-ai/package.json' with { type: 'json' }
import { pathToFileURL } from 'node:url'

interface GeneratedImageModel {
  readonly id: string
  readonly name: string
  readonly api: string
  readonly provider: string
  readonly baseUrl: string
  readonly input: readonly string[]
  readonly output: readonly string[]
  readonly cost: {
    readonly input: number
    readonly output: number
    readonly cacheRead: number
    readonly cacheWrite: number
  }
  readonly headers?: Readonly<Record<string, string>>
}

interface GeneratedImageCatalog {
  readonly source: {
    readonly package: string
    readonly version: string
    readonly generated_at?: string
  }
  readonly providers: Readonly<Record<string, readonly GeneratedImageModel[]>>
}

const resolvedPackage = import.meta.resolveSync('@earendil-works/pi-ai/package.json')
const generatedUrl = new URL('./dist/image-models.generated.js', pathToFileURL(resolvedPackage))
const { IMAGE_MODELS } = await import(generatedUrl.href) as {
  IMAGE_MODELS: Readonly<Record<string, Readonly<Record<string, GeneratedImageModel>>>>
}
const { getBuiltinModelDataGeneratedAt } = await import('@earendil-works/pi-ai/providers/all') as {
  getBuiltinModelDataGeneratedAt?: () => number | undefined
}
// ISO-stamped like generateModelCatalog.ts so the value is human-readable.
const rawGeneratedAt = getBuiltinModelDataGeneratedAt?.()
const builtinGeneratedAt = rawGeneratedAt === undefined ? undefined : new Date(rawGeneratedAt).toISOString()

const providers: Record<string, GeneratedImageModel[]> = Object.create(null)
for (const [provider, models] of Object.entries(IMAGE_MODELS)) {
  const entries = Object.values(models).map(model => ({
    api: model.api,
    baseUrl: model.baseUrl,
    cost: {
      cacheRead: model.cost.cacheRead,
      cacheWrite: model.cost.cacheWrite,
      input: model.cost.input,
      output: model.cost.output,
    },
    id: model.id,
    ...(model.headers && Object.keys(model.headers).length ? { headers: model.headers } : {}),
    input: [...model.input],
    name: model.name,
    output: [...model.output],
    provider: model.provider,
  }))
  // Deterministic output: provider order and model order are both sorted, and
  // every array is cloned so the JSON depends only on the source data.
  providers[provider] = entries.sort((left, right) => left.id.localeCompare(right.id))
}

const catalog: GeneratedImageCatalog = {
  source: {
    package: '@earendil-works/pi-ai',
    version: piPackage.version,
    ...(builtinGeneratedAt === undefined ? {} : { generated_at: builtinGeneratedAt }),
  },
  providers: Object.fromEntries(
    Object.entries(providers).sort(([left], [right]) => left.localeCompare(right)),
  ),
}

const destination = new URL('../src/images/imageModels.generated.json', import.meta.url)
const serialized = `${JSON.stringify(catalog, null, 2)}\n`
if (process.argv.includes('--check')) {
  const existing = await Bun.file(destination).text()
  if (existing !== serialized) {
    throw new Error('Pi image-model catalog is stale; run bun run generate:image-catalog')
  }
  console.log(`Pi image-model catalog matches pi-ai ${piPackage.version}`)
} else {
  await Bun.write(destination, serialized)
  const count = Object.values(providers).reduce((sum, models) => sum + models.length, 0)
  console.log(`Generated ${destination.pathname} from pi-ai ${piPackage.version} (${count} image models)`)
}
