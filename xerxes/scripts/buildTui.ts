// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
// Build the sole OpenTUI entry point into the canonical TUI artifact.
import { mkdir } from 'node:fs/promises'
import { dirname, isAbsolute, resolve } from 'node:path'

const root = resolve(import.meta.dir, '..')
const outputOverride = process.env.XERXES_TUI_BUILD_OUT?.trim()
const out = outputOverride
  ? isAbsolute(outputOverride)
    ? outputOverride
    : resolve(root, outputOverride)
  : resolve(root, 'dist/ui/entry.js')

const stubDevtools = {
  name: 'stub-react-devtools-core',
  setup(build: Bun.PluginBuilder) {
    build.onResolve({ filter: /^react-devtools-core$/ }, () => ({
      path: 'react-devtools-core',
      namespace: 'stub-devtools'
    }))
    build.onLoad({ filter: /.*/, namespace: 'stub-devtools' }, () => ({
      contents: 'export default { initialize() {}, connectToDevTools() {} }',
      loader: 'js'
    }))
  }
}

async function buildBundle(options: {
  entrypoint: string
  out: string
  plugins: Bun.BunPlugin[]
  label: string
  external?: string[]
}): Promise<void> {
  await mkdir(dirname(options.out), { recursive: true })
  const result = await Bun.build({
    entrypoints: [options.entrypoint],
    target: 'node',
    format: 'esm',
    outfile: options.out,
    jsx: { runtime: 'automatic', importSource: 'react' },
    define: { 'process.env.NODE_ENV': '"production"', 'process.env.DEV': '"false"' },
    legalComments: 'none',
    minify: true,
    plugins: options.plugins,
    external: options.external,
    // Some transitive deps use CommonJS `require(...)` at runtime. ESM bundles
    // don't get a `require` binding automatically, so we inject one.
    banner: "import { createRequire as __cr } from 'node:module'; const require = __cr(import.meta.url);"
  })

  if (!result.success) {
    for (const log of result.logs) {
      console.error(log.message)
    }

    throw new Error(`failed to build the ${options.label} bundle`)
  }

  const entryArtifact = result.outputs.find(artifact => artifact.kind === 'entry-point')

  if (!entryArtifact) {
    throw new Error(`the ${options.label} build did not produce an entry-point artifact`)
  }

  // Keep the generated module shebang-free. Release launchers invoke this
  // module explicitly through Bun, so the artifact must not select a second
  // runtime when executed from a package or Nix store.
  const body = await entryArtifact.text()
  const output = body.startsWith('#!') ? body.slice(body.indexOf('\n') + 1) : body
  await Bun.write(options.out, output)

  console.log(`built ${options.out}`)
}

// @opentui/core's platform-detection module dynamically imports whichever
// of these 8 packages matches process.platform/arch at runtime — only the
// one optionalDependency actually installed for the current machine
// exists on disk, so the bundler must not try to statically resolve the
// other 7 branches (same treatment esbuild/sharp-style native packages
// need). Runtime `import()` resolution still finds the real installed one
// — but ONLY if @opentui/core itself also stays external: bun installs
// the platform package as a *sibling* inside @opentui/core's own private
// node_modules (node_modules/.bun/@opentui+core@.../node_modules/), not
// hoisted anywhere a bundled dist/ file could reach by walking up parent
// directories. Inlining @opentui/core's own code into the bundle would
// sever that relative path. The canonical dist/ui/entry.js therefore needs
// node_modules present at runtime. That's an inherent consequence of
// shipping a native binary, not a bug: `bun build --compile` (embeds
// native deps into a real single-file executable) is the distribution path
// when a self-contained binary is required.
const OPENTUI_EXTERNAL = [
  // React MUST be external here: @opentui/react (also external) carries its
  // own react-reconciler that resolves React from node_modules at runtime.
  // If React were bundled into entry.js instead, the app tree would
  // use one React instance while the reconciler used another — the two
  // ReactSharedInternals dispatchers diverge and every useState throws
  // "null is not an object (evaluating 'H.useState')". One shared runtime
  // instance keeps the dispatcher valid.
  'react',
  'react/jsx-runtime',
  'react/jsx-dev-runtime',
  'react-reconciler',
  '@opentui/core',
  '@opentui/react',
  '@opentui/core-darwin-x64',
  '@opentui/core-darwin-arm64',
  '@opentui/core-linux-x64',
  '@opentui/core-linux-x64-musl',
  '@opentui/core-linux-arm64',
  '@opentui/core-linux-arm64-musl',
  '@opentui/core-win32-x64',
  '@opentui/core-win32-arm64'
]

await buildBundle({
  entrypoint: resolve(root, 'src/ui/opentui/entry.tsx'),
  out,
  plugins: [stubDevtools],
  label: 'Xerxes OpenTUI',
  external: OPENTUI_EXTERNAL
})
